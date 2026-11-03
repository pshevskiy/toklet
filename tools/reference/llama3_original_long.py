"""Explicit frozen original-FP32 long consistency probe, not adoption quality."""
import argparse
import gc
import importlib.metadata
import json
import os
from pathlib import Path
import resource
import shutil
import sys
import time

# Import fixes numerical provider thread counts before NumPy initialization.
import llama3_original_short as short
import llama3_original_fp32 as graph
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'bench'))
import prepare_llama3_original_quality as inputs

LIMIT = 6*1024**3


def resource_gate(output):
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if peak > LIMIT or shutil.disk_usage(output).free < 1024**3:
        raise ValueError('Reference memory/disk reserve exceeded')
    return peak


def produce(model, folder, manifest_sha, output):
    output = Path(output); output.mkdir()
    report = dict(format='toklet-llama3-original-long-source-v1', passed=False, cases=[],
                  input_manifest_sha256=manifest_sha, full_quality_qualified=False,
                  precision='original BF16 widened to FP32; FP32 graph; stable FP64 NLL')
    reader = None
    try:
        if importlib.metadata.version('numpy') != '2.3.2' or sys.byteorder != 'little':
            raise ValueError('Reference runtime differs')
        _, ids = inputs.load(folder, manifest_sha)
        paths = [Path(__file__), Path(graph.__file__), Path(short.__file__), Path(inputs.__file__),
                 Path(inputs.parent.__file__), graph.ROOT/'tools/reference/llama3_loss1024.py',
                 graph.ROOT/'tools/bench/acquire_llama3_original.py']
        report['source_sha256'] = {str(p):short.sha(p) for p in paths}
        provider = {str(p):short.sha(p) for p in sorted(Path(np.__file__).resolve().parent.rglob('*'))
                    if p.is_file() and p.suffix in ['.py','.so','.dylib'] and '__pycache__' not in p.parts}
        pins = output/'runtime-pins.json'; pins.write_text(json.dumps(provider,indent=2)+'\n')
        report['runtime_pins'] = dict(file=pins.name, sha256=short.sha(pins), entries=len(provider))
        report['runtime'] = dict(python=sys.version, numpy=np.__version__,
                                threads={n:os.environ[n] for n in ['VECLIB_MAXIMUM_THREADS','OPENBLAS_NUM_THREADS','OMP_NUM_THREADS']})
        report['original_acquisition_sha256'] = short.sha(Path(model)/'acquisition.json')
        resource_gate(output); started = time.monotonic(); reader = graph.Reader(model)
        batch = graph.quality_forward(reader, ids[:1025], [n-1 for n in inputs.PREFIXES])
        resource_gate(output)
        for index, n in enumerate(inputs.PREFIXES):
            print('original prefix', n, flush=True)
            target = ids[n]
            prefix = graph.quality_forward(reader, ids[:n], [n-1])[0]
            selected = batch[index]
            row = dict(prefix_tokens=n, logit_row=n-1, target_position=n, target_id=target,
                       consistency=short.comparison(selected, prefix, target), arrays={})
            for label, values in [('batch', selected), ('prefix', prefix)]:
                raw = values.astype('<f4',copy=False).tobytes()
                path = output/f'prefix-{n}-{label}.f32le'; path.write_bytes(raw)
                row['arrays'][label] = dict(file=path.name, bytes=len(raw), sha256=short.sha(path))
            report['cases'].append(row)
            resource_gate(output); del prefix; gc.collect()
        reader.close(); reader = None
        if not all(case['consistency']['passed'] for case in report['cases']):
            raise ValueError('Frozen long source consistency gate failed')
        if any(short.sha(Path(p)) != value for p,value in provider.items()):
            raise ValueError('Reference runtime changed')
        if any(short.sha(Path(p)) != value for p,value in report['source_sha256'].items()):
            raise ValueError('Reference source changed')
        inputs.load(folder, manifest_sha)
        report.update(passed=True, peak_rss_bytes=resource_gate(output), elapsed_seconds=time.monotonic()-started)
    except BaseException as error:
        report['error'] = repr(error)
        report['peak_rss_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    finally:
        if reader is not None:
            try: reader.close()
            except BaseException as error: report['close_error'] = repr(error)
        (output/'source.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    return report


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model',type=Path);parser.add_argument('inputs',type=Path)
    parser.add_argument('manifest_sha');parser.add_argument('new_output',type=Path)
    args=parser.parse_args()
    result=produce(args.model,args.inputs,args.manifest_sha,args.new_output)
    print(json.dumps(result,indent=2,allow_nan=False))
    raise SystemExit(0 if result['passed'] else 1)

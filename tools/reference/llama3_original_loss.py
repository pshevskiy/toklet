"""Collect all frozen original-FP32 loss targets, one bounded block at a time."""
import argparse
import gc
import importlib.metadata
import json
import math
from pathlib import Path
import sys
import time
import llama3_original_long as long


def produce(model, input_folder, manifest_sha, output):
    output = Path(output); output.mkdir()
    report = dict(format='toklet-llama3-quality-loss-original-v1', completed=False,
                  manifest_sha256=manifest_sha, full_quality_qualified=False, blocks=[])
    reader = None
    try:
        if importlib.metadata.version('numpy') != '2.3.2' or sys.byteorder != 'little':
            raise ValueError('Reference runtime differs')
        manifest, ids = long.inputs.load(input_folder, manifest_sha)
        graph=long.graph; short=long.short
        report['model'] = dict(repository=graph.artifact.META, revision=graph.artifact.META_REVISION,
                               weight_sha256=graph.artifact.WEIGHT_SHA)
        report['profile'] = 'original BF16 widened to FP32; FP32 graph; stable FP64 NLL'
        paths=[Path(__file__),Path(long.__file__),Path(graph.__file__),Path(short.__file__),
               Path(long.inputs.__file__),Path(long.inputs.parent.__file__),
               graph.ROOT/'tools/reference/llama3_loss1024.py',graph.ROOT/'tools/bench/acquire_llama3_original.py']
        report['source_sha256']={str(p):short.sha(p) for p in paths}
        provider={str(p):short.sha(p) for p in sorted(Path(long.np.__file__).resolve().parent.rglob('*'))
                  if p.is_file() and p.suffix in ['.py','.so','.dylib'] and '__pycache__' not in p.parts}
        pins=output/'runtime-pins.json';pins.write_text(json.dumps(provider,indent=2)+'\n')
        report['runtime_pins']=dict(file=pins.name,sha256=short.sha(pins),entries=len(provider))
        report['original_acquisition_sha256']=short.sha(Path(model)/'acquisition.json')
        report['runtime']=dict(python=sys.version,numpy=long.np.__version__,threads={n:long.os.environ[n] for n in ['VECLIB_MAXIMUM_THREADS','OPENBLAS_NUM_THREADS','OMP_NUM_THREADS']})
        long.resource_gate(output);started=time.monotonic();reader=graph.Reader(model)
        losses=[]
        with (output/'targets.jsonl').open('x') as journal:
            for block in manifest['blocks']:
                print('original block',block['block'],flush=True)
                tokens=ids[block['input_start']:block['input_end']]
                targets=block['score_rows'][1]
                logits=graph.quality_forward(reader,tokens,list(range(targets)))
                if logits.shape != (targets,128256):
                    raise ValueError('Original loss shape differs')
                long.resource_gate(output)
                for local in range(targets):
                    index=block['input_start']+local+1
                    record=short.row_statistics(logits[local],ids[index])
                    record.update(block=block['block'],local_row=local,global_target_index=index)
                    journal.write(json.dumps(record,separators=(',',':'),allow_nan=False)+'\n')
                    losses.append(record['nll'])
                journal.flush();del logits;gc.collect();long.resource_gate(output)
                report['blocks'].append(dict(block=block['block'],targets=targets,input_positions=len(tokens)))
        reader.close();reader=None
        if len(losses)!=10000:raise ValueError('Original target count differs')
        if any(short.sha(Path(p))!=v for p,v in provider.items()):raise ValueError('Runtime changed')
        if any(short.sha(Path(p))!=v for p,v in report['source_sha256'].items()):raise ValueError('Source changed')
        long.inputs.load(input_folder,manifest_sha)
        report.update(completed=True,targets=10000,mean_nll=math.fsum(losses)/10000,
                      targets_sha256=short.sha(output/'targets.jsonl'),peak_rss_bytes=long.resource_gate(output),
                      elapsed_seconds=time.monotonic()-started)
    except BaseException as error:
        report.update(error=repr(error),peak_rss_bytes=long.resource.getrusage(long.resource.RUSAGE_SELF).ru_maxrss)
    finally:
        if reader is not None:
            try:reader.close()
            except BaseException as error:report['close_error']=repr(error)
        (output/'result.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    return report


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model',type=Path);parser.add_argument('inputs',type=Path)
    parser.add_argument('manifest_sha');parser.add_argument('new_output',type=Path)
    args=parser.parse_args()
    result=produce(args.model,args.inputs,args.manifest_sha,args.new_output)
    print(json.dumps(result,indent=2,allow_nan=False));raise SystemExit(0 if result['completed'] else 1)

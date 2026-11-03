"""Frozen short original-FP32 alignment; descriptive native comparison is separate."""
import argparse
import gc
import hashlib
import importlib.metadata
import json
import math
import os
import platform
from pathlib import Path
import resource
import struct
import sys
import time

for name in ['VECLIB_MAXIMUM_THREADS', 'OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS']:
    os.environ[name] = '1'

import numpy as np
import llama3_original_fp32 as graph
from llama3_loss1024 import row_statistics


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def inputs(folder, expected_sha):
    folder = Path(folder)
    if sha(folder / 'manifest.json') != expected_sha:
        raise ValueError('Frozen short manifest differs')
    manifest = json.loads((folder / 'manifest.json').read_bytes())
    policy = manifest['consistency']
    if (manifest['format'] != 'toklet-llama3-original-short-v1' or manifest['add_special_tokens']
            or manifest['chat_rendering'] or manifest['position_offset'] != 0
            or policy['policy_pending_astra'] or policy['full_row_atol'] != 1e-3
            or policy['full_row_rtol'] != 1e-4 or policy['target_nll_absolute'] != 1e-4):
        raise ValueError('Frozen source policy differs')
    raw = (folder / 'tokens.u32le').read_bytes()
    if len(raw) != 72 or hashlib.sha256(raw).hexdigest() != manifest['tokens_sha256']:
        raise ValueError('Frozen input bytes differ')
    ids = list(struct.unpack('<18I', raw))
    if any(not 0 <= i < graph.Shape().vocab for i in ids):
        raise ValueError('Input vocabulary differs')
    cases = [{'prefix_tokens': n, 'logit_row': n-1, 'target_position': n, 'target_id': ids[n]} for n in [1, 4, 8, 17]]
    if manifest['cases'] != cases:
        raise ValueError('Frozen target positions differ')
    return manifest, ids


def comparison(batch, prefix, target):
    if batch.shape != prefix.shape or batch.dtype != np.float32 or prefix.dtype != np.float32:
        raise ValueError('Source row shape/dtype differs')
    if not np.isfinite(batch).all() or not np.isfinite(prefix).all():
        raise ValueError('Nonfinite source row')
    error = np.abs(batch.astype(np.float64) - prefix.astype(np.float64))
    gate = 1e-3 + 1e-4 * np.abs(prefix.astype(np.float64))
    ratio = error / gate
    worst = int(np.argmax(ratio))
    a, b = row_statistics(batch, target), row_statistics(prefix, target)
    top = np.partition(prefix, -2)[-2:]
    return {'passed': bool(np.all(error <= gate) and abs(a['nll'] - b['nll']) <= 1e-4),
            'maximum_logit_error': float(np.max(error)), 'worst_vocab_id': worst,
            'worst_error_over_gate': float(ratio[worst]), 'nll_absolute_difference': abs(a['nll'] - b['nll']),
            'prefix_greedy_margin': float(max(top) - min(top)), 'batch': a, 'prefix': b}


def produce(model, input_folder, manifest_sha, output):
    output = Path(output)
    output.mkdir()
    report = {'format': 'toklet-llama3-original-short-source-v1', 'passed': False, 'cases': [],
              'input_manifest_sha256': manifest_sha, 'full_quality_qualified': False,
              'precision': 'original BF16 widened to FP32; FP32 graph; stable FP64 NLL'}
    reader = None
    try:
        if importlib.metadata.version('numpy') != '2.3.2' or sys.byteorder != 'little':
            raise ValueError('Reference runtime differs')
        manifest, ids = inputs(input_folder, manifest_sha)
        paths = [Path(__file__), Path(graph.__file__), graph.ROOT/'tools/bench/acquire_llama3_original.py',
                 graph.ROOT/'tools/reference/llama3_loss1024.py']
        report['source_sha256'] = {str(p): sha(p) for p in paths}
        report['runtime'] = {'python': sys.version, 'numpy': np.__version__,
                             'platform': platform.platform(), 'threads': {n: os.environ[n] for n in ['VECLIB_MAXIMUM_THREADS', 'OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS']}}
        numpy_root = Path(np.__file__).resolve().parent
        provider = {str(p): sha(p) for p in sorted(numpy_root.rglob('*'))
                    if p.is_file() and p.suffix in ['.py','.so','.dylib'] and '__pycache__' not in p.parts}
        provider_file = output/'runtime-pins.json'
        provider_file.write_text(json.dumps(provider,indent=2)+'\n')
        report['runtime_pins'] = {'file':provider_file.name,'sha256':sha(provider_file),'entries':len(provider)}
        report['original_acquisition_sha256'] = sha(Path(model)/'acquisition.json')
        started = time.monotonic()
        reader = graph.Reader(model)
        batch = graph.forward(reader, ids[:17])
        for case in manifest['cases']:
            n, target = case['prefix_tokens'], case['target_id']
            prefix = graph.forward(reader, ids[:n])[-1]
            selected = batch[n-1]
            row = dict(case, consistency=comparison(selected, prefix, target), arrays={})
            for label, values in [('batch', selected), ('prefix', prefix)]:
                raw = values.astype('<f4', copy=False).tobytes()
                name = f'prefix-{n}-{label}.f32le'
                (output/name).write_bytes(raw)
                row['arrays'][label] = {'file': name, 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}
            report['cases'].append(row)
            print(json.dumps({'prefix': n, 'self_consistency': row['consistency']['passed']}), flush=True)
            if not row['consistency']['passed']:
                raise ValueError('Original FP32 batch/prefix consistency failed')
        closing = reader; reader = None; closing.close()
        del batch, prefix, selected
        gc.collect()
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == 'darwin' else 1024)
        report.update(peak_resident_bytes=peak, seconds=time.monotonic()-started)
        if peak > 6*1024**3:
            raise ValueError('Reference exceeds 6 GiB peak RSS')
        if any(sha(p) != digest for p, digest in report['source_sha256'].items()):
            raise ValueError('Reference source changed')
        if any(sha(p) != digest for p, digest in provider.items()):
            raise ValueError('Numerical runtime changed')
        report['passed'] = True
    except BaseException as error:
        report['error'] = type(error).__name__ + ': ' + str(error)
        raise
    finally:
        cleanup = None
        if reader is not None:
            try:
                reader.close()
            except BaseException as error:
                cleanup = error
                report['passed'] = False
                report['cleanup_error'] = type(error).__name__ + ': ' + str(error)
        (output/'source.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
        if cleanup is not None and sys.exc_info()[0] is None:
            raise cleanup
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model', type=Path); parser.add_argument('inputs', type=Path)
    parser.add_argument('manifest_sha'); parser.add_argument('output', type=Path)
    args = parser.parse_args()
    print(json.dumps(produce(args.model, args.inputs, args.manifest_sha, args.output), indent=2))

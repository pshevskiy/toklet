"""Explicit admitted literal/wide-row checks; never opens a model or parent."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import granite_quantize as quant
import granite_quant_runtime as runtime


def sources():
    folder = Path(__file__).resolve().parent
    return [folder / name for name in (
        'granite_quant_probe.py', 'granite_quantize.py', 'test_granite_quantize.py',
        'granite_quant_runtime.py', 'test_granite_quant_runtime.py',
        'minicpm5_quantize.py', 'test_minicpm5_quantize.py', 'minicpm5_quant_runtime.py')]


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def upstream_f32(buffers, rows, columns, bits, device, np, mx):
    target = mx.cpu if device == 'cpu' else mx.gpu
    packed = mx.array(np.frombuffer(buffers['weight'], dtype='<u4')
                      .reshape(rows, columns * bits // 32))
    metadata = {}
    for name in ('scales', 'biases'):
        raw = np.frombuffer(buffers[name], dtype='<u2').reshape(rows, columns // 64)
        metadata[name] = mx.array(raw).view(mx.bfloat16, stream=target).astype(mx.float32, stream=target)
    out = mx.dequantize(packed, metadata['scales'], metadata['biases'],
                        group_size=64, bits=bits, mode='affine', stream=target)
    mx.eval(out)
    mx.synchronize(target)
    quant.need(out.dtype == mx.float32 and tuple(out.shape) == (rows, columns),
               'Explicit FP32 upstream reconstruction required')
    return np.asarray(out).astype('<f4', copy=False).tobytes()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('environment-root', 'runtime-manifest', 'runtime-sha256',
                 'controls-file', 'controls-sha256', 'fresh-output'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    output = Path(args.fresh_output)
    output.mkdir()
    runtime.bind_environment(args.environment_root)
    raw = runtime.common.captured(Path(args.controls_file), 2 * 1024**2,
                                  runtime.common.sha(args.controls_sha256))
    controls = json.loads(raw, object_pairs_hook=runtime.common.json_unique,
                          parse_constant=runtime.common.bad_json_number)
    frozen = dict(format='toklet-granite-affine-literals-v1', cases=quant.controls())
    # JSON encoding also distinguishes booleans/integers that Python == merges.
    quant.need(json.dumps(controls, sort_keys=True) == json.dumps(frozen, sort_keys=True),
               'Prospective Granite literal controls differ')
    admitted, endpoint = runtime.admit(Path(args.runtime_manifest), args.runtime_sha256, sources())
    import numpy as np
    import mlx.core as mx
    mx.set_cache_limit(0)
    records = []
    for case in controls['cases']:
        for rows, columns in ((1, 64), (256, 8192)):
            repeats = rows * columns // 64
            for device in ('cpu', 'gpu'):
                raw = bytes.fromhex(case['input_bf16']) * repeats
                started = time.monotonic_ns()
                actual = quant.quantize(raw, rows, columns, case['bits'], device, np, mx)
                completed = time.monotonic_ns()
                expected = {key: bytes.fromhex(value) * repeats
                            for key, value in case[device].items()}
                buffers = quant.compare(actual, expected)
                reconstructed = quant.reconstruct_f32(actual, rows, columns, case['bits'], np)
                literal = quant.literal_reconstruction(case, device) * repeats
                upstream = upstream_f32(actual, rows, columns, case['bits'], device, np, mx)
                record = dict(name=case['name'], bits=case['bits'], device=device,
                              rows=rows, columns=columns, groups=repeats,
                              input_bytes=len(raw), input_sha256=sha(raw),
                              completed_quantize_and_readback_ns=completed - started,
                              buffers=buffers,
                              reconstruction_bytes=len(reconstructed),
                              independent_F32_sha256=sha(reconstructed),
                              literal_F32_sha256=sha(literal),
                              upstream_F32_sha256=sha(upstream),
                              literal_reconstruction_equal=reconstructed == literal,
                              upstream_reconstruction_equal=reconstructed == upstream,
                              passed=all(v['mismatching_bytes'] == 0 for v in buffers.values())
                              and reconstructed == literal == upstream)
                with (output / 'records.jsonl').open('a') as stream:
                    stream.write(json.dumps(record, separators=(',', ':')) + '\n')
                records.append(record)
                # Keep only scalar receipts across cases and before endpoint
                # inventory hashing. The maximum individual raw row tile is 4 MiB.
                del raw, actual, expected, reconstructed, literal, upstream
    final = endpoint()
    result = dict(format='toklet-granite-affine-controls-v1',
                  passed=len(records) == 48 and all(row['passed'] for row in records),
                  cases=len(controls['cases']), executions=len(records),
                  admitted_runtime=admitted, endpoint_origins=final,
                  controls_sha256=args.controls_sha256, device=mx.metal.device_info(),
                  python_optimization=sys.flags.optimize,
                  model_or_parent_weights_consumed=False, conversion_lineage_qualified=False,
                  model_performance_quality_or_adoption_qualified=False)
    (output / 'observation.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({key: value for key, value in result.items()
                      if key not in ('admitted_runtime', 'endpoint_origins', 'device')}))
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    sys.exit(main())

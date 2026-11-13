"""Admitted Ministral literal/wide-row controls; no model or parent is opened."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import granite_quant_runtime as runtime
from granite_quant_probe import upstream_f32
import ministral3_quantize as quant

SHAPES = ((1, 64), (224, 9216), (32, 9216), (64, 9216), (128, 9216), (160, 9216))
CONTROL_FORMAT = 'toklet-ministral3-affine-literals-v1'


def sources():
    """Explicit new and unchanged transitive helpers/tests, before admission."""
    folder = Path(__file__).resolve().parent
    return [folder / name for name in (
        'ministral3_quant_probe.py', 'ministral3_quantize.py', 'test_ministral3_quantize.py',
        'granite_quant_probe.py', 'granite_quantize.py', 'test_granite_quantize.py',
        'granite_quant_runtime.py', 'test_granite_quant_runtime.py',
        'minicpm5_quantize.py', 'test_minicpm5_quantize.py',
        'minicpm5_quant_runtime.py',
        'reference_runtime.py', 'text_reference_runtime.py')]


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def load_controls(path, expected_sha256):
    raw = runtime.common.captured(Path(path), 2 * 1024**2,
                                  runtime.common.sha(expected_sha256))
    controls = json.loads(raw, object_pairs_hook=runtime.common.json_unique,
                          parse_constant=runtime.common.bad_json_number)
    frozen = dict(format=CONTROL_FORMAT, cases=quant.controls(),
                  shapes=[list(shape) for shape in SHAPES])
    # Encoding distinguishes Boolean/number types that Python equality merges.
    quant.need(json.dumps(controls, sort_keys=True, allow_nan=False) ==
               json.dumps(frozen, sort_keys=True, allow_nan=False),
               'Prospective Ministral literal controls differ')
    return controls


def check_output(path):
    path = Path(path)
    quant.need(path.is_absolute() and '..' not in path.parts,
               'Absolute fresh Ministral output directory required')
    quant.need(not path.exists() and not path.is_symlink(), 'Never replace Ministral control output')
    quant.need(path.parent.is_dir() and not any(p.is_symlink() for p in path.parents),
               'Existing nonsymlink output parent required')
    return path


def execute_case(case, rows, columns, device, np, mx):
    quant.shape(rows, columns, case['bits'])
    quant.need(device in ('cpu', 'gpu'), 'Explicit control device required')
    repeats = rows * columns // 64
    raw = bytes.fromhex(case['input_bf16']) * repeats
    started = time.monotonic_ns()
    actual = quant.quantize(raw, rows, columns, case['bits'], device, np, mx)
    completed = time.monotonic_ns()
    expected = {key: bytes.fromhex(value) * repeats for key, value in case[device].items()}
    buffers = quant.compare(actual, expected)
    reconstructed = quant.reconstruct_f32(actual, rows, columns, case['bits'], np)
    literal = quant.literal_reconstruction(case, device) * repeats
    upstream = upstream_f32(actual, rows, columns, case['bits'], device, np, mx)
    return dict(name=case['name'], bits=case['bits'], device=device,
                rows=rows, columns=columns, groups=repeats,
                input_bytes=len(raw), input_sha256=sha(raw),
                completed_quantize_and_readback_ns=completed - started,
                buffers=buffers, reconstruction_bytes=len(reconstructed),
                independent_F32_sha256=sha(reconstructed), literal_F32_sha256=sha(literal),
                upstream_F32_sha256=sha(upstream),
                literal_reconstruction_equal=reconstructed == literal,
                upstream_reconstruction_equal=reconstructed == upstream,
                passed=all(v['mismatching_bytes'] == 0 for v in buffers.values())
                and reconstructed == literal == upstream)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('environment-root', 'runtime-manifest', 'runtime-sha256',
                 'controls-file', 'controls-sha256', 'fresh-output'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args(argv)
    output = check_output(args.fresh_output)
    controls = load_controls(args.controls_file, args.controls_sha256)
    runtime.bind_environment(args.environment_root)
    output.mkdir()
    admitted, endpoint = runtime.admit(Path(args.runtime_manifest), args.runtime_sha256, sources())
    import numpy as np
    import mlx.core as mx
    mx.set_cache_limit(0)
    records = []
    with (output / 'records.jsonl').open('x') as journal:
        for case in controls['cases']:
            for rows, columns in controls['shapes']:
                for device in ('cpu', 'gpu'):
                    record = execute_case(case, rows, columns, device, np, mx)
                    journal.write(json.dumps(record, separators=(',', ':'), allow_nan=False) + '\n')
                    journal.flush()
                    records.append(record)
    # execute_case returns scalar-only receipts; all tensor owners die before
    # endpoint runtime inventory/hashing, including on the maximum-width tile.
    final = endpoint()
    observed = {(r['bits'], r['name'], r['rows'], r['columns'], r['device']) for r in records}
    expected = {(c['bits'], c['name'], r, n, d) for c in controls['cases']
                for r, n in SHAPES for d in ('cpu', 'gpu')}
    result = dict(format='toklet-ministral3-affine-controls-v1',
                  passed=len(records) == 144 and len(observed) == 144 and observed == expected
                  and all(row['passed'] for row in records),
                  cases=len(controls['cases']), executions=len(records), shapes=controls['shapes'],
                  admitted_runtime=admitted, endpoint_origins=final,
                  controls_sha256=args.controls_sha256, device=mx.metal.device_info(),
                  python_optimization=sys.flags.optimize,
                  model_or_parent_weights_consumed=False, conversion_lineage_qualified=False,
                  model_performance_quality_or_adoption_qualified=False)
    with (output / 'observation.json').open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({key: value for key, value in result.items()
                      if key not in ('admitted_runtime', 'endpoint_origins', 'device')}, allow_nan=False))
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    sys.exit(main())

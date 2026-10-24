"""Independent FP64 gated RMS normalization and explicit pinned MLX comparison.

This is a tiny operator witness, not a hybrid model. Norm precedes SiLU gating;
learned scale is used directly, without the dense Qwen norm's +1 conversion.
"""
import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import signal
import struct
import sys

import reference_runtime as runtime

FORMAT = 'toklet-gated-output-norm-math-v1'
MAX_JSON = 256 * 1024
MAX_RAW = 1024 * 1024
REVISION = '0d4d3d04f8ab7ce4a51e7209e8615755a48ba94d'
SOURCES = {
    'qwen3_5.py': '851449fd9989d6bf4b3c4fed92ddc3be5f1e781eb2246f89c54d7eff2518d9c6',
    'qwen3_next.py': '67ca08a4d08cb7be124965e95399735e556f1eaeb21a16bc23cb88d1ac035e11',
    'activations.py': '6f7d756897deb322af4fe98e040df76e9358a91d76521e51f14d734fdd0fb05f',
}
SPECS = (('empty', 0, 2, 3, 1e-6), ('scalar', 1, 1, 1, 1.0),
         ('asymmetric', 2, 3, 5, 1e-6), ('epsilon', 1, 2, 3, 1e-6),
         ('wide', 2, 2, 128, 1e-6))
ATOL, RTOL = 1e-5, 1e-4


def encoded(value):
    return (json.dumps(value, indent=2, allow_nan=False) + '\n').encode()


def math_values(hidden, gate, weight, epsilon):
    import numpy as np
    hidden, gate, weight = [np.asarray(value) for value in (hidden, gate, weight)]
    if (hidden.ndim != 3 or gate.shape != hidden.shape or weight.ndim != 1
            or hidden.shape[-1] != weight.size or not 1 <= weight.size <= 128
            or not 0 <= hidden.shape[0] <= 2 or not 1 <= hidden.shape[1] <= 3
            or type(epsilon) not in (int, float) or not math.isfinite(epsilon) or epsilon <= 0):
        raise ValueError('Gated norm geometry/epsilon differs')
    if any(value.dtype != np.dtype('float32') or not np.isfinite(value).all() for value in (hidden, gate, weight)):
        raise ValueError('Finite FP32 gated norm inputs required')
    hidden, gate, weight = [value.astype(np.float64) for value in (hidden, gate, weight)]
    # FP32 squares and their bounded width fit FP64. Epsilon is added to MEAN.
    with np.errstate(over='raise', invalid='raise', under='ignore'):
        normalized = hidden / np.sqrt(np.mean(hidden * hidden, axis=-1, keepdims=True) + epsilon) * weight
        exp_negative = np.exp(-np.abs(gate))
        sigmoid = np.where(gate >= 0, 1 / (1 + exp_negative), exp_negative / (1 + exp_negative))
        activated = gate * sigmoid
        output = normalized * activated
    if any(not np.isfinite(value).all() for value in (normalized, activated, output)):
        raise ValueError('Nonfinite gated norm mathematical output')
    return normalized, activated, output


def export():
    import numpy as np
    cases = []
    for name, tokens, heads, width, epsilon in SPECS:
        shape = (tokens, heads, width)
        def values(shape, multiplier, offset, scale):
            return (((np.arange(math.prod(shape)) * multiplier + offset) % 31 - 15) * scale).astype(np.float32).reshape(shape)
        hidden = values(shape, 7, 4, 0.125 if name != 'epsilon' else 1e-5)
        gate = values(shape, 11, 3, 0.25)
        weight = values((width,), 13, 8, 0.125)
        if name == 'scalar':
            hidden.fill(-2)
            gate.fill(-24)
            weight.fill(-0.5)
        elif tokens:
            gate.flat[0] = -1000
            gate.flat[-1] = 1000
        normalized, activated, output = math_values(hidden, gate, weight, epsilon)
        arrays = dict(hidden=hidden, gate=gate, weight=weight, normalized=normalized, activated=activated, output=output)
        cases.append(dict(name=name, tokens=tokens, heads=heads, width=width, epsilon=epsilon,
            arrays={key: dict(dtype='<f4' if value.dtype == np.dtype('float32') else '<f8',
                shape=list(value.shape), values=value.ravel().tolist()) for key, value in arrays.items()}))
    return encoded(dict(format=FORMAT, source_revision=REVISION, source_sha256=SOURCES,
        producer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        precision='FP32 inputs; independent FP64 mean RMS normalization, SiLU and product',
        runtime=dict(numpy=np.__version__), native_graph_implemented=False, device_qualified=False, cases=cases))


def verify(path, expected_sha):
    raw = runtime.captured(path, MAX_JSON, runtime.sha(expected_sha))
    data = json.loads(raw, object_pairs_hook=runtime.json_unique,
        parse_constant=runtime.bad_json_number, parse_float=runtime.finite_json_number)
    if (not isinstance(data, dict) or set(data) != {'format', 'source_revision', 'source_sha256',
            'producer_sha256', 'precision', 'runtime', 'native_graph_implemented', 'device_qualified', 'cases'}
            or data['format'] != FORMAT or data['source_revision'] != REVISION or data['source_sha256'] != SOURCES
            or data['precision'] != 'FP32 inputs; independent FP64 mean RMS normalization, SiLU and product'
            or data['runtime'] != dict(numpy=runtime.VERSIONS['numpy'])
            or data['native_graph_implemented'] is not False or data['device_qualified'] is not False
            or not isinstance(data['cases'], list) or len(data['cases']) != len(SPECS)):
        raise ValueError('Gated norm reference identity/scope differs')
    runtime.sha(data['producer_sha256'])
    for case, spec in zip(data['cases'], SPECS):
        if not isinstance(case, dict) or set(case) != {'name', 'tokens', 'heads', 'width', 'epsilon', 'arrays'}:
            raise ValueError('Gated norm case inventory differs')
        if (tuple(case[key] for key in ('name', 'tokens', 'heads', 'width', 'epsilon')) != spec
                or any(type(case[key]) is not int for key in ('tokens', 'heads', 'width'))
                or type(case['epsilon']) not in (int, float)):
            raise ValueError('Gated norm case geometry differs')
        if not isinstance(case['arrays'], dict) or set(case['arrays']) != {'hidden', 'gate', 'weight', 'normalized', 'activated', 'output'}:
            raise ValueError('Gated norm array inventory differs')
        for name, array in case['arrays'].items():
            shape = [case['width']] if name == 'weight' else [case['tokens'], case['heads'], case['width']]
            dtype = '<f4' if name in ('hidden', 'gate', 'weight') else '<f8'
            if (not isinstance(array, dict) or set(array) != {'dtype', 'shape', 'values'}
                    or array['dtype'] != dtype or array['shape'] != shape
                    or not isinstance(array['shape'], list) or any(type(n) is not int for n in array['shape'])
                    or not isinstance(array['values'], list) or len(array['values']) != math.prod(shape)):
                raise ValueError('Gated norm array descriptor differs')
            for number in array['values']:
                if type(number) not in (float, int) or not -sys.float_info.max <= number <= sys.float_info.max or not math.isfinite(number):
                    raise ValueError('Finite gated norm numeric array required')
                if dtype == '<f4':
                    try:
                        if struct.unpack('<f', struct.pack('<f', number))[0] != number:
                            raise ValueError('Input not exactly representable as FP32')
                    except OverflowError as error:
                        raise ValueError('Input exceeds FP32') from error
    return data


def source_nodes(sources):
    result = {}
    for filename, name, kind in [('qwen3_next.py', 'Qwen3NextRMSNormGated', ast.ClassDef),
            ('activations.py', 'precise_swiglu', ast.FunctionDef)]:
        nodes = [node for node in ast.parse(sources[filename]).body if isinstance(node, kind) and node.name == name]
        if len(nodes) != 1:
            raise ValueError('Expected unique pinned gated norm source definition')
        result[filename] = nodes[0]
    return result


def compare(reference, reference_sha, sources, device, destination, runtime_manifest, runtime_sha):
    if device not in ('cpu', 'gpu'):
        raise ValueError('Explicit gated norm reference device required')
    data = verify(reference, reference_sha)
    if (set(sources) != set(SOURCES) or any(not isinstance(raw, bytes) or len(raw) > 128 * 1024
            or hashlib.sha256(raw).hexdigest() != SOURCES[name] for name, raw in sources.items())):
        raise ValueError('Authenticated gated norm source buffers differ')
    nodes = source_nodes(sources)
    admitted = runtime.admit_runtime(runtime_manifest, runtime_sha, [Path(__file__)])
    import numpy as np
    import mlx.core as mx
    import mlx.nn as nn
    from functools import partial
    if (np.__file__ != admitted['imports']['numpy'] or mx.__file__ != admitted['imports']['mlx.core']
            or nn.__file__ != admitted['imports']['mlx.nn']):
        raise ValueError('Actual gated norm imports differ from admitted files')
    if destination.is_symlink() or not destination.is_dir() or any(destination.iterdir()):
        raise ValueError('Empty regular gated norm output required')
    previous_device = mx.default_device()
    previous_signals = {}
    blob = bytearray()
    rows, comparisons, cleanup = [], [], []
    primary = None

    def interrupted(number, _frame):
        for value in (signal.SIGTERM, signal.SIGINT):
            signal.signal(value, signal.SIG_IGN)
        raise InterruptedError('Received signal ' + str(number))

    def array(case, name):
        item = case['arrays'][name]
        return np.array(item['values'], dtype=item['dtype']).reshape(item['shape'])

    def observe(row, name, value, expected):
        mx.eval(value)
        actual = np.asarray(value)
        if actual.dtype.hasobject or actual.dtype.kind not in 'biufc' or actual.nbytes > MAX_RAW:
            row['arrays'][name] = dict(raw_unavailable='Unsupported numeric observation', dtype=str(actual.dtype), shape=list(actual.shape))
            raise ValueError('Bounded numeric observation required')
        dtype = actual.dtype.newbyteorder('<')
        raw = np.ascontiguousarray(actual, dtype=dtype).tobytes()
        padding = (-len(blob)) % 8
        if len(blob) + padding + len(raw) > MAX_RAW:
            raise ValueError('Gated norm observation buffer exceeds bound')
        blob.extend(b'\0' * padding)
        row['arrays'][name] = dict(dtype=dtype.str, shape=list(actual.shape), offset=len(blob), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
        blob.extend(raw)
        if actual.dtype != np.dtype('float32') or actual.shape != expected.shape or not np.isfinite(actual).all():
            raise ValueError('Observed gated norm dtype/shape/finite values differ')
        difference = np.abs(actual.astype(np.float64) - expected)
        ratio = difference / (ATOL + RTOL * np.abs(expected))
        comparisons.append(dict(case=row['name'], name=name, passed=bool(np.all(ratio <= 1)),
            maximum_absolute_error=float(difference.max()) if difference.size else 0,
            maximum_error_to_policy_ratio=float(ratio.max()) if ratio.size else 0))

    try:
        for value in (signal.SIGTERM, signal.SIGINT):
            previous_signals[value] = signal.signal(value, interrupted)
        mx.set_default_device(mx.cpu if device == 'cpu' else mx.gpu)
        environment = dict(mx=mx, nn=nn, partial=partial)
        for filename in ('activations.py', 'qwen3_next.py'):
            exec(compile(ast.Module(body=[nodes[filename]], type_ignores=[]), filename, 'exec'), environment)
        for case in data['cases']:
            row = dict(name=case['name'], completed=False, upstream_evaluated=False, arrays={})
            rows.append(row)
            if case['tokens'] == 0:
                row.update(completed=True, reason='Empty sequence is mathematical no-call policy')
                continue
            model = environment['Qwen3NextRMSNormGated'](case['width'], eps=case['epsilon'])
            model.weight = mx.array(array(case, 'weight'))
            hidden, gate = [mx.array(array(case, name)) for name in ('hidden', 'gate')]
            # Retain each evaluated result before constructing a later call.
            observe(row, 'normalized', model(hidden), array(case, 'normalized'))
            observe(row, 'activated', nn.silu(gate), array(case, 'activated'))
            observe(row, 'output', model(hidden, gate), array(case, 'output'))
            if not np.array_equal(np.asarray(hidden), array(case, 'hidden')) or not np.array_equal(np.asarray(gate), array(case, 'gate')):
                raise ValueError('Gated norm mutated immutable inputs')
            row.update(completed=True, upstream_evaluated=True)
    except (Exception, KeyboardInterrupt) as error:
        primary = type(error).__name__ + ': ' + str(error)
    finally:
        for value in previous_signals:
            signal.signal(value, signal.SIG_IGN)
        try:
            mx.set_default_device(previous_device)
        except Exception as error:
            cleanup.append('Restore device: ' + type(error).__name__ + ': ' + str(error))
    passed = primary is None and not cleanup and len(rows) == len(SPECS) and all(row['completed'] for row in rows) and len(comparisons) == 12 and all(row['passed'] for row in comparisons)
    report = dict(format='toklet-gated-output-norm-observation-v1', passed=passed,
        mathematical_reference_sha256=reference_sha, source_revision=REVISION, source_sha256=SOURCES,
        admitted_runtime=admitted, requested_device=device, source_policy=dict(atol=ATOL, rtol=RTOL),
        selected_source_definitions={name: dict(line=node.lineno, source=ast.unparse(node)) for name, node in nodes.items()},
        cases=rows, comparisons=comparisons, binary=dict(path='observed.bin', bytes=len(blob), alignment=8, sha256=hashlib.sha256(blob).hexdigest()),
        primary_error=primary, cleanup_errors=cleanup, native_graph_implemented=False, device_qualified=False,
        scope='Tiny gated output normalization source/operator comparison; no native graph or performance qualification')
    try:
        for name, raw in [('observed.bin', bytes(blob)), ('observation.json', encoded(report))]:
            with (destination / name).open('xb') as stream:
                stream.write(raw)
    finally:
        for value, handler in previous_signals.items():
            signal.signal(value, handler)
    return passed


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['export', 'compare'])
    parser.add_argument('--sources', required=True, type=Path)
    parser.add_argument('--runtime-manifest', required=True, type=Path)
    parser.add_argument('--expected-runtime-sha256', required=True)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--expected-reference-sha256')
    parser.add_argument('--device', choices=['cpu', 'gpu'])
    options = parser.parse_args()
    sources = {name: runtime.captured(options.sources / name, 128 * 1024, expected) for name, expected in SOURCES.items()}
    source_nodes(sources)
    if options.mode == 'compare':
        if options.reference is None or options.expected_reference_sha256 is None or options.device is None:
            parser.error('Comparison requires explicit reference identity and device')
        verify(options.reference, options.expected_reference_sha256)
    if options.mode == 'export':
        admitted = runtime.admit_runtime(options.runtime_manifest, options.expected_runtime_sha256, [Path(__file__)])
        import numpy as np
        if np.__file__ != admitted['imports']['numpy']:
            raise ValueError('Imported NumPy differs from admitted producer')
        with options.output.open('xb') as stream:
            stream.write(export())
    else:
        passed = compare(options.reference, options.expected_reference_sha256, sources, options.device, options.output, options.runtime_manifest, options.expected_runtime_sha256)
        print(json.dumps(dict(passed=passed, mathematical_reference_only=True)))
        raise SystemExit(0 if passed else 1)

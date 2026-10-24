"""Independent FP64 tanh-GELU/GeGLU and explicit pinned Gemma4 comparison.

The actual inspected variant uses tanh GELU, not erf GELU or SiLU. This tiny
operator witness precedes native implementation and does not qualify a model.
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

FORMAT = 'toklet-gemma-gelu-math-v1'
MAX_JSON = 256 * 1024
MAX_RAW = 1024 * 1024
REVISION = '0d4d3d04f8ab7ce4a51e7209e8615755a48ba94d'
SOURCES = {'gemma4_text.py': '73ffa75e9069ce7b2f864ccd8f0267bb80c5b5d4e2b73102b8a2a93fe03f936d'}
SPECS = (('empty', 0), ('scalar', 1), ('grid', 129), ('tails', 17), ('odd', 33))
ATOL, RTOL = 1e-5, 1e-4


def encoded(value):
    return (json.dumps(value, indent=2, allow_nan=False) + '\n').encode()


def math_values(gate, up):
    import numpy as np
    gate, up = [np.asarray(value) for value in (gate, up)]
    if gate.ndim != 1 or gate.shape != up.shape or not 0 <= gate.size <= 129:
        raise ValueError('Bounded equal GELU vectors required')
    if any(value.dtype != np.dtype('float32') or not np.isfinite(value).all() for value in (gate, up)):
        raise ValueError('Finite FP32 GELU inputs required')
    gate, up = gate.astype(np.float64), up.astype(np.float64)
    with np.errstate(over='raise', invalid='raise', under='ignore'):
        activated = 0.5 * gate * (1 + np.tanh(math.sqrt(2 / math.pi) * (gate + 0.044715 * gate**3)))
        output = activated * up
    if not np.isfinite(activated).all() or not np.isfinite(output).all():
        raise ValueError('Nonfinite GELU mathematical output')
    return activated, output


def export():
    import numpy as np
    cases = []
    for name, width in SPECS:
        gate = ((np.arange(width) * 7 + 4) % 31 - 15).astype(np.float32) * np.float32(0.25)
        up = ((np.arange(width) * 11 + 3) % 19 - 9).astype(np.float32) * np.float32(0.5)
        if name == 'scalar':
            gate.fill(0.5)
            up.fill(-3)
        elif name == 'grid':
            gate = np.arange(-64, 65, dtype=np.float32) * np.float32(0.125)
        elif name == 'tails':
            gate = np.array([-1000,-32,-10,-6,-3,-1,-0.1,-1e-8,0,1e-8,0.1,1,3,6,10,32,1000],np.float32)
            up.fill(1.5)
        activated, output = math_values(gate, up)
        arrays = dict(gate=gate, up=up, activated=activated, output=output)
        cases.append(dict(name=name, width=width,
            arrays={key: dict(dtype='<f4' if value.dtype == np.dtype('float32') else '<f8',
                shape=list(value.shape), values=value.tolist()) for key, value in arrays.items()}))
    return encoded(dict(format=FORMAT, source_revision=REVISION, source_sha256=SOURCES,
        producer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        precision='FP32 inputs; independent FP64 tanh GELU and gated product',
        runtime=dict(numpy=np.__version__), native_graph_implemented=False, device_qualified=False, cases=cases))


def verify(path, expected_sha):
    raw = runtime.captured(path, MAX_JSON, runtime.sha(expected_sha))
    data = json.loads(raw, object_pairs_hook=runtime.json_unique,
        parse_constant=runtime.bad_json_number, parse_float=runtime.finite_json_number)
    if (not isinstance(data, dict) or set(data) != {'format', 'source_revision', 'source_sha256',
            'producer_sha256', 'precision', 'runtime', 'native_graph_implemented', 'device_qualified', 'cases'}
            or data['format'] != FORMAT or data['source_revision'] != REVISION or data['source_sha256'] != SOURCES
            or data['precision'] != 'FP32 inputs; independent FP64 tanh GELU and gated product'
            or data['runtime'] != dict(numpy=runtime.VERSIONS['numpy'])
            or data['native_graph_implemented'] is not False or data['device_qualified'] is not False
            or not isinstance(data['cases'], list) or len(data['cases']) != len(SPECS)):
        raise ValueError('GELU reference identity/scope differs')
    runtime.sha(data['producer_sha256'])
    for case, spec in zip(data['cases'], SPECS):
        if not isinstance(case, dict) or set(case) != {'name', 'width', 'arrays'}:
            raise ValueError('GELU case inventory differs')
        if (case['name'], case['width']) != spec or type(case['width']) is not int:
            raise ValueError('GELU case geometry differs')
        if not isinstance(case['arrays'], dict) or set(case['arrays']) != {'gate', 'up', 'activated', 'output'}:
            raise ValueError('GELU array inventory differs')
        for name, array in case['arrays'].items():
            dtype = '<f4' if name in ('gate', 'up') else '<f8'
            if (not isinstance(array, dict) or set(array) != {'dtype', 'shape', 'values'}
                    or array['dtype'] != dtype or array['shape'] != [case['width']]
                    or not isinstance(array['shape'], list) or any(type(n) is not int for n in array['shape'])
                    or not isinstance(array['values'], list) or len(array['values']) != case['width']):
                raise ValueError('GELU array descriptor differs')
            for number in array['values']:
                if type(number) not in (float, int) or not -sys.float_info.max <= number <= sys.float_info.max or not math.isfinite(number):
                    raise ValueError('Finite GELU numeric array required')
                if dtype == '<f4':
                    try:
                        if struct.unpack('<f', struct.pack('<f', number))[0] != number:
                            raise ValueError('Input not exactly representable as FP32')
                    except OverflowError as error:
                        raise ValueError('Input exceeds FP32') from error
    return data


def source_nodes(sources):
    nodes = [node for node in ast.parse(sources['gemma4_text.py']).body
        if isinstance(node, ast.FunctionDef) and node.name == 'geglu']
    if len(nodes) != 1:
        raise ValueError('Expected unique pinned GeGLU definition')
    return {'gemma4_text.py': nodes[0]}


def compare(reference, reference_sha, sources, device, destination, runtime_manifest, runtime_sha):
    if device not in ('cpu', 'gpu'):
        raise ValueError('Explicit GELU reference device required')
    data = verify(reference, reference_sha)
    if (set(sources) != set(SOURCES) or any(not isinstance(raw, bytes) or len(raw) > 128 * 1024
            or hashlib.sha256(raw).hexdigest() != SOURCES[name] for name, raw in sources.items())):
        raise ValueError('Authenticated GELU source buffers differ')
    nodes = source_nodes(sources)
    admitted = runtime.admit_runtime(runtime_manifest, runtime_sha, [Path(__file__)])
    import numpy as np
    import mlx.core as mx
    import mlx.nn as nn
    from functools import partial
    if (np.__file__ != admitted['imports']['numpy'] or mx.__file__ != admitted['imports']['mlx.core']
            or nn.__file__ != admitted['imports']['mlx.nn']):
        raise ValueError('Actual GELU imports differ from admitted files')
    if destination.is_symlink() or not destination.is_dir() or any(destination.iterdir()):
        raise ValueError('Empty regular GELU output required')
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
            raise ValueError('GELU observation buffer exceeds bound')
        blob.extend(b'\0' * padding)
        row['arrays'][name] = dict(dtype=dtype.str, shape=list(actual.shape), offset=len(blob), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
        blob.extend(raw)
        if actual.dtype != np.dtype('float32') or actual.shape != expected.shape or not np.isfinite(actual).all():
            raise ValueError('Observed GELU dtype/shape/finite values differ')
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
        for filename, node in nodes.items():
            exec(compile(ast.Module(body=[node], type_ignores=[]), filename, 'exec'), environment)
        for case in data['cases']:
            row = dict(name=case['name'], completed=False, upstream_evaluated=False, arrays={})
            rows.append(row)
            if case['width'] == 0:
                row.update(completed=True, reason='Empty vector is mathematical no-call policy')
                continue
            gate, up = [mx.array(array(case, name)) for name in ('gate', 'up')]
            observe(row, 'activated', nn.gelu_approx(gate), array(case, 'activated'))
            observe(row, 'output', environment['geglu'](gate, up), array(case, 'output'))
            if not np.array_equal(np.asarray(gate), array(case, 'gate')) or not np.array_equal(np.asarray(up), array(case, 'up')):
                raise ValueError('GELU mutated immutable inputs')
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
    passed = primary is None and not cleanup and len(rows) == len(SPECS) and all(row['completed'] for row in rows) and len(comparisons) == 8 and all(row['passed'] for row in comparisons)
    report = dict(format='toklet-gemma-gelu-observation-v1', passed=passed,
        mathematical_reference_sha256=reference_sha, source_revision=REVISION, source_sha256=SOURCES,
        admitted_runtime=admitted, requested_device=device, source_policy=dict(atol=ATOL, rtol=RTOL),
        selected_source_definitions={name: dict(line=node.lineno, source=ast.unparse(node)) for name, node in nodes.items()},
        cases=rows, comparisons=comparisons, binary=dict(path='observed.bin', bytes=len(blob), alignment=8, sha256=hashlib.sha256(blob).hexdigest()),
        primary_error=primary, cleanup_errors=cleanup, native_graph_implemented=False, device_qualified=False,
        scope='Tiny tanh GELU/GeGLU source/operator comparison; no native graph or performance qualification')
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

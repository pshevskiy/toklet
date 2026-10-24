"""Explicit pinned Conv1d/SiLU snippets against independent mathematical arrays.

Only authenticated selected assignments from Qwen3.5 are executed. This is an
operator/layout comparison, not execution or qualification of its model graph.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import signal
import sys
import types

import causal_conv_fixture as fixture
import reference_runtime as runtime

# Existing source-reference policy, selected before observing this operator.
SOURCE_ATOL = 1e-5
SOURCE_RTOL = 1e-4


def selected_assignments(source):
    tree = ast.parse(source)
    classes = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'GatedDeltaNet']
    if len(classes) != 1:
        raise ValueError('Expected unique pinned GatedDeltaNet source')
    selected = {}
    contracts = {'constructor': ('__init__', 'self.conv1d'),
        'concatenate': ('__call__', 'conv_input'), 'history': ('__call__', 'cache[0]'),
        'activation': ('__call__', 'conv_out')}
    for name, (method, target) in contracts.items():
        methods = [node for node in classes[0].body if isinstance(node, ast.FunctionDef) and node.name == method]
        if len(methods) != 1:
            raise ValueError('Expected unique pinned convolution method')
        nodes = [node for node in ast.walk(methods[0]) if isinstance(node, ast.Assign)
            and len(node.targets) == 1 and ast.unparse(node.targets[0]) == target]
        # cache[0] has two branches. Select only the unmasked no-lengths branch;
        # masked/length-dependent history is outside this operator reference.
        if name == 'history':
            nodes = [node for node in nodes if isinstance(node.value, ast.Call)
                and ast.unparse(node.value.func) == 'mx.contiguous']
        if len(nodes) != 1:
            raise ValueError('Expected unique pinned convolution assignment: ' + name)
        selected[name] = nodes[0]
    return selected


def capture_evaluations(steps, evaluate, retain):
    """Retain each completed evaluation before constructing/evaluating the next."""
    results = []
    for name, value in steps:
        results.append(retain(name, evaluate(value)))
    return results


def retain_array(blob, row, name, actual, expected):
    """Save bounded numeric raw bytes before dtype/finite/numerical checks."""
    import numpy as np
    actual = np.asarray(actual)
    if actual.dtype.hasobject or actual.dtype.kind not in 'biufc':
        row['arrays'][name] = dict(raw_unavailable='Unsupported nonnumeric dtype', dtype=str(actual.dtype), shape=list(actual.shape))
        raise ValueError('Numeric convolution observation required')
    if actual.nbytes > fixture.MAX_BYTES:
        raise ValueError('Observed convolution array exceeds bound')
    dtype = actual.dtype.newbyteorder('<')
    raw = np.ascontiguousarray(actual, dtype=dtype).tobytes()
    padding = (-len(blob)) % 8
    if len(blob) + padding + len(raw) > fixture.MAX_BYTES:
        raise ValueError('Observed convolution buffer exceeds bound')
    blob.extend(b'\0' * padding)
    row['arrays'][name] = dict(dtype=dtype.str, shape=list(actual.shape), offset=len(blob), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
    blob.extend(raw)
    if actual.dtype != np.dtype('float32') or actual.shape != expected.shape or not np.isfinite(actual).all():
        raise ValueError('Observed dtype/shape/finite values differ: ' + name)
    difference = np.abs(actual.astype(np.float64) - expected)
    limit = SOURCE_ATOL + SOURCE_RTOL * np.abs(expected)
    comparison = dict(case=row['name'], name=name, passed=bool(np.all(difference <= limit)),
        maximum_absolute_error=float(difference.max()) if difference.size else 0.0,
        maximum_error_to_policy_ratio=float((difference / limit).max()) if difference.size else 0.0)
    return actual.copy(), comparison


def compare(directory, manifest_sha, binary_sha, model_source, device, destination, runtime_manifest, runtime_sha):
    if destination.is_symlink() or not destination.is_dir() or any(destination.iterdir()):
        raise ValueError('Empty regular comparison output required')
    if device not in ('cpu', 'gpu'):
        raise ValueError('Explicit comparison device required')
    manifest = fixture.verify(directory, manifest_sha, binary_sha)
    binary = runtime.captured(directory / 'arrays.bin', fixture.MAX_BYTES, binary_sha)
    source = runtime.captured(model_source, 128 * 1024, fixture.MODEL_SOURCE_SHA)
    selected = selected_assignments(source)
    programs = {name: compile(ast.Module(body=[node], type_ignores=[]), str(model_source), 'exec')
        for name, node in selected.items()}
    admitted = runtime.admit_runtime(runtime_manifest, runtime_sha, [Path(__file__), Path(fixture.__file__)])
    import numpy as np
    import mlx.core as mx
    import mlx.nn as nn
    if (np.__file__ != admitted['imports']['numpy'] or mx.__file__ != admitted['imports']['mlx.core']
            or nn.__file__ != admitted['imports']['mlx.nn']):
        raise ValueError('Actual numerical imports differ from admitted sources')
    previous_device = mx.default_device()
    previous_signals = {}
    rows, comparisons, cleanup_errors = [], [], []
    primary = None
    blob = bytearray()

    def interrupted(number, _frame):
        for value in (signal.SIGTERM, signal.SIGINT):
            signal.signal(value, signal.SIG_IGN)
        raise InterruptedError('Received signal ' + str(number))

    def array(row, name):
        item = row['arrays'][name]
        return np.frombuffer(binary, dtype=item['dtype'], count=item['bytes'] // np.dtype(item['dtype']).itemsize,
            offset=item['offset']).reshape(item['shape'])

    def retain(row, name, actual, expected):
        actual, comparison = retain_array(blob, row, name, actual, expected)
        comparisons.append(comparison)
        return actual

    def evaluated(value):
        mx.eval(value)
        return np.asarray(value)

    try:
        for value in (signal.SIGTERM, signal.SIGINT):
            previous_signals[value] = signal.signal(value, interrupted)
        mx.set_default_device(mx.cpu if device == 'cpu' else mx.gpu)
        for case in manifest['cases']:
            tokens, channels, kernel = [case[key] for key in ('tokens', 'channels', 'kernel')]
            if not tokens:
                rows.append(dict(name=case['name'], completed=True, upstream_evaluated=False,
                    reason='Empty-prefix no-op is a mathematical policy; upstream empty call not evaluated'))
                continue
            row = dict(name=case['name'], completed=False, upstream_evaluated=False, arrays={},
                upstream_history_evaluated=kernel > 1,
                history_scope='Unmasked no-lengths cache branch' if kernel > 1 else 'K1 cache branch excluded; mathematical empty history')
            rows.append(row)
            state = types.SimpleNamespace(conv_dim=channels, conv_kernel_size=kernel)
            environment = dict(nn=nn, mx=mx, self=state)
            exec(programs['constructor'], environment)
            state.conv1d.weight = mx.array(array(case, 'weight')[:, :, None])
            qkv = mx.array(array(case, 'input')[None])
            initial = mx.array(array(case, 'initial_history')[None])

            def call(first, last, history, label):
                environment.update(qkv=qkv[:, first:last], conv_state=history, n_keep=kernel - 1, cache=[None])
                exec(programs['concatenate'], environment)
                names = ['output', 'activated', 'final_history'] if not label else [label + '_' + suffix for suffix in ('output', 'activated', 'history')]
                expected = [array(case, 'output')[first:last], array(case, 'activated')[first:last]]
                if kernel > 1:
                    expected.append(array(case, 'history')[last - 1])
                next_state = [initial]

                def steps():
                    yield names[0], state.conv1d(environment['conv_input'])
                    exec(programs['activation'], environment)
                    yield names[1], environment['conv_out']
                    if kernel > 1:
                        exec(programs['history'], environment)
                        next_state[0] = environment['cache'][0]
                        yield names[2], next_state[0]

                references = dict(zip(names, expected))
                values = capture_evaluations(steps(), lambda value: evaluated(value)[0],
                    lambda name, value: retain(row, name, value, references[name]))
                return values, next_state[0]

            call(0, tokens, initial, '')
            if kernel > 1:
                histories = []
                for end in range(1, tokens + 1):
                    values, _ = call(0, end, initial, 'prefix_' + str(end))
                    histories.append(values[2])
                retain(row, 'history', np.stack(histories), array(case, 'history'))
            for chunk in (1, 3):
                history = initial
                outputs, activations = [], []
                for first in range(0, tokens, chunk):
                    values, history = call(first, min(tokens, first + chunk), history, 'chunk_' + str(chunk) + '_' + str(first))
                    outputs.append(values[0])
                    activations.append(values[1])
                retain(row, 'chunk_' + str(chunk) + '_output', np.concatenate(outputs), array(case, 'output'))
                retain(row, 'chunk_' + str(chunk) + '_activated', np.concatenate(activations), array(case, 'activated'))
                if kernel > 1:
                    retain(row, 'chunk_' + str(chunk) + '_final_history', evaluated(history)[0], array(case, 'final_history'))
            if not np.array_equal(evaluated(initial)[0], array(case, 'initial_history')) or not np.array_equal(evaluated(qkv)[0], array(case, 'input')):
                raise ValueError('Upstream mutated immutable input/history')
            row.update(completed=True, upstream_evaluated=True)
    except (Exception, KeyboardInterrupt) as error:
        primary = type(error).__name__ + ': ' + str(error)
    finally:
        for value in previous_signals:
            signal.signal(value, signal.SIG_IGN)
        try:
            mx.set_default_device(previous_device)
        except Exception as error:
            cleanup_errors.append('Restore device: ' + type(error).__name__ + ': ' + str(error))
    passed = (primary is None and not cleanup_errors and len(rows) == len(fixture.SPECS)
        and all(row.get('completed') for row in rows) and len(comparisons) == 154
        and all(row['passed'] for row in comparisons))
    report = dict(format='toklet-causal-conv-upstream-observation-v1', passed=passed,
        mathematical_manifest_sha256=manifest_sha, mathematical_binary_sha256=binary_sha,
        model_revision=fixture.MODEL_REVISION, model_source_sha256=fixture.MODEL_SOURCE_SHA,
        selected_source_assignments={name: dict(line=node.lineno, source=ast.unparse(node)) for name, node in selected.items()},
        requested_device=device, admitted_runtime=admitted, source_policy=dict(atol=SOURCE_ATOL, rtol=SOURCE_RTOL),
        cases=rows, comparisons=comparisons,
        binary=dict(path='observed.bin', bytes=len(blob), sha256=hashlib.sha256(blob).hexdigest(), alignment=8),
        primary_error=primary, cleanup_errors=cleanup_errors, native_graph_implemented=False,
        numerical_device_qualified=False, scope='Tiny unmasked Conv1d/SiLU selected-source operator/layout comparison only')
    try:
        for name, raw in [('observed.bin', bytes(blob)), ('observation.json', (json.dumps(report, indent=2, allow_nan=False) + '\n').encode())]:
            with (destination / name).open('xb') as stream:
                stream.write(raw)
    finally:
        for value, handler in previous_signals.items():
            signal.signal(value, handler)
    return passed


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--expected-manifest-sha256', required=True)
    parser.add_argument('--expected-binary-sha256', required=True)
    parser.add_argument('--model-source', required=True, type=Path)
    parser.add_argument('--device', required=True, choices=['cpu', 'gpu'])
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--runtime-manifest', required=True, type=Path)
    parser.add_argument('--expected-runtime-sha256', required=True)
    options = parser.parse_args()
    passed = compare(options.directory, options.expected_manifest_sha256, options.expected_binary_sha256,
        options.model_source, options.device, options.output, options.runtime_manifest, options.expected_runtime_sha256)
    print(json.dumps(dict(passed=passed, operator_reference_only=True)))
    raise SystemExit(0 if passed else 1)

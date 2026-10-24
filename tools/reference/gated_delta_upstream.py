"""Explicit pinned upstream comparison against independent mathematical arrays.

Admit the external math-fixture identities and exact source buffer before any
MLX import. Execute only that captured pinned source. Raw observed FP32 arrays
are retained alongside errors; CPU/GPU requests are separate owned invocations.
"""
import argparse
import hashlib
from importlib.metadata import version
from importlib.machinery import ExtensionFileLoader, PathFinder, SourceFileLoader
from importlib.util import cache_from_source, find_spec
import json
import os
from pathlib import Path
import signal
import sys
import sysconfig
import types

import gated_delta_fixture as fixture

VERSIONS = {'numpy': '2.3.2', 'mlx': '0.32.3'}
# Existing Qwen3 source-reference policy, fixed before any observations here.
SOURCE_ATOL = 1e-5
SOURCE_RTOL = 1e-4


def captured(path, maximum, expected):
    if path.is_symlink() or not path.is_file():
        raise ValueError('Expected a regular captured reference file')
    with path.open('rb') as stream:
        raw = stream.read(maximum + 1)
    if len(raw) > maximum or hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError('Captured reference buffer identity differs')
    return raw


def runtime_paths():
    """Enumerate the installed selected producer closure without importing it."""
    site = Path(sysconfig.get_path('purelib')).resolve()
    root = Path(__file__).resolve().parents[2]
    if not site.is_relative_to(root / '.venv'):
        raise ValueError('Repository virtual environment required')
    paths = {Path(__file__).resolve(), Path(fixture.__file__).resolve(), Path(sys.executable).resolve()}
    for package in ('numpy', 'mlx'):
        folder = site / package
        if not folder.is_dir():
            raise ValueError('Required reference runtime package absent')
        paths.update(path for path in folder.rglob('*')
            if path.is_file() and '__pycache__' not in path.parts)
    for pattern in ('numpy*.libs', 'numpy*.dist-info', 'mlx*.dist-info'):
        for folder in site.glob(pattern):
            paths.update(path for path in folder.rglob('*') if path.is_file() and '__pycache__' not in path.parts)
    if not any(path.name == 'mlx.metallib' for path in paths) or not any(path.name == 'libmlx.dylib' for path in paths):
        raise ValueError('Full MLX Metal/native runtime required')
    return sorted(paths)


def selected_imports(site, paths):
    """Resolve package roots without importing or running numerical code."""
    admitted = {str(path) for path in paths}
    selected = {}
    for name in ('numpy', 'mlx'):
        spec = find_spec(name)
        folder = site / name
        if spec is None or spec.submodule_search_locations is None:
            raise ValueError('Reference numerical package resolution differs')
        locations = list(spec.submodule_search_locations)
        if locations != [str(folder)]:
            raise ValueError('Reference numerical package search path differs')
        if name == 'numpy':
            if spec.origin != str(folder / '__init__.py') or not isinstance(spec.loader, SourceFileLoader) or spec.origin not in admitted:
                raise ValueError('Reference NumPy import source differs')
            selected[name] = spec.origin
        elif spec.origin is not None:
            raise ValueError('Expected the pinned MLX namespace package')
        for child, loader in ([('core', ExtensionFileLoader), ('nn', SourceFileLoader)] if name == 'mlx' else []):
            child_spec = PathFinder.find_spec(name + '.' + child, locations)
            if child_spec is None or child_spec.origin not in admitted or not isinstance(child_spec.loader, loader):
                raise ValueError('Reference MLX import source differs')
            selected[name + '.' + child] = child_spec.origin
    return selected


def admit_runtime(path, expected_sha):
    """Admit externally pinned exact runtime files before numerical imports.

    The owned caller creates the manifest and fresh pycache prefix before launch.
    Version strings alone, self-reported hashes and an already imported runtime
    cannot admit a reference execution.
    """
    raw = captured(path, 2 * 1024 * 1024, fixture._sha(expected_sha))
    manifest = json.loads(raw, object_pairs_hook=fixture._unique,
        parse_constant=fixture._bad_number, parse_float=fixture._finite_number)
    paths = runtime_paths()
    if (not isinstance(manifest, dict) or set(manifest) != {'format', 'versions', 'files'}
            or manifest['format'] != 'toklet-gated-delta-runtime-v1'
            or manifest['versions'] != VERSIONS or not isinstance(manifest['files'], dict)
            or set(manifest['files']) != {str(path) for path in paths}):
        raise ValueError('Exact externally pinned reference runtime closure required')
    if any(name == 'numpy' or name.startswith('numpy.') or name == 'mlx' or name.startswith('mlx.') for name in sys.modules):
        raise ValueError('Reference numerical runtime was already imported')
    if not sys.pycache_prefix or not Path(sys.pycache_prefix).is_absolute():
        raise ValueError('Fresh owned Python cache prefix required')
    site = Path(sysconfig.get_path('purelib')).resolve()
    imports = selected_imports(site, paths)
    for item in paths:
        if item.is_relative_to(site) and item.suffix == '.py' and Path(cache_from_source(str(item))).exists():
            raise ValueError('Reference producer bytecode cache already exists')
        pin = manifest['files'][str(item)]
        if (not isinstance(pin, dict) or set(pin) != {'bytes', 'sha256'}
                or type(pin['bytes']) is not int or not 0 <= pin['bytes'] <= 512 * 1024 * 1024
                or item.is_symlink()):
            raise ValueError('Bounded regular runtime file identity required')
        expected = fixture._sha(pin['sha256'])
        digest = hashlib.sha256()
        count = 0
        with item.open('rb') as stream:
            while chunk := stream.read(128 * 1024):
                count += len(chunk)
                if count > pin['bytes']:
                    raise ValueError('Reference runtime file exceeds its admitted length')
                digest.update(chunk)
        if count != pin['bytes'] or digest.hexdigest() != expected:
            raise ValueError('Reference runtime bytes differ')
    return dict(manifest_sha256=expected_sha, files=len(paths), pycache_prefix=sys.pycache_prefix, imports=imports)


def compare(directory, manifest_sha, binary_sha, upstream_source, device, destination, runtime_manifest, runtime_sha):
    if destination.is_symlink() or not destination.is_dir() or any(destination.iterdir()):
        raise ValueError('Empty existing comparison output directory required')
    if device not in ('cpu', 'gpu'):
        raise ValueError('Expected an explicit CPU or GPU reference device')
    manifest = fixture.verify(directory, manifest_sha, binary_sha)
    # Re-admit the consumed binary buffer; no unchecked reread after verification.
    binary = captured(directory / 'arrays.bin', fixture.MAX_ARRAY_BYTES, binary_sha)
    source = captured(upstream_source, 256 * 1024, fixture.UPSTREAM_SHA256)
    admitted_runtime = admit_runtime(runtime_manifest, runtime_sha)
    observed_versions = {name: version(name) for name in VERSIONS}
    if observed_versions != VERSIONS:
        raise ValueError('Reference runtime versions differ')
    import numpy as np
    import mlx.core as mx
    if (np.__file__ != admitted_runtime['imports']['numpy']
            or mx.__file__ != admitted_runtime['imports']['mlx.core']):
        raise ValueError('Imported numerical runtime differs from admitted files')

    if device == 'gpu' and not mx.metal.is_available():
        raise ValueError('Requested upstream Metal device is unavailable')
    namespace = '__toklet_pinned_gated_delta_reference__'
    if namespace in sys.modules:
        raise ValueError('Reference module namespace is already in use')
    module = types.ModuleType(namespace)
    module.__file__ = str(upstream_source)
    previous_device = mx.default_device()
    blob = bytearray()
    rows = []
    primary = None
    errors = []
    comparisons = []
    source_execution_attempted = False
    source_execution_completed = False
    captured_packed_policy = None
    previous_signals = {}
    sys.modules[namespace] = module

    def interrupted(number, _frame):
        for value in (signal.SIGTERM, signal.SIGINT):
            signal.signal(value, signal.SIG_IGN)
        raise InterruptedError('Received signal ' + str(number))

    def array(row, name):
        descriptor = row['arrays'][name]
        return np.frombuffer(binary, dtype=descriptor['dtype'],
            count=descriptor['bytes'] // np.dtype(descriptor['dtype']).itemsize,
            offset=descriptor['offset']).reshape(descriptor['shape'])

    def retain(observation, name, actual):
        actual = np.asarray(actual)
        if actual.dtype.hasobject or actual.dtype.kind not in 'biufc':
            observation['arrays'][name] = dict(raw_unavailable='Unsupported nonnumeric dtype', dtype=str(actual.dtype), shape=list(actual.shape))
            raise ValueError('Observed upstream dtype cannot be serialized')
        if actual.nbytes > fixture.MAX_ARRAY_BYTES:
            raise ValueError('Observed upstream array exceeds its bound')
        dtype = actual.dtype.newbyteorder('<')
        raw = np.ascontiguousarray(actual, dtype=dtype).tobytes()
        padding = (-len(blob)) % 8
        if len(blob) + padding + len(raw) > fixture.MAX_ARRAY_BYTES:
            raise ValueError('Observed upstream binary exceeds its bound')
        blob.extend(b'\0' * padding)
        offset = len(blob)
        blob.extend(raw)
        observation['arrays'][name] = dict(dtype=dtype.str, shape=list(actual.shape),
            offset=offset, bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
        return actual

    def observe(row, observation, name, actual, expected):
        actual = retain(observation, name, actual)
        # Retain the raw observation even when a shape/value check fails. The
        # case row is published before any call, so partial arrays stay indexed.
        if actual.dtype != np.dtype('float32') or actual.shape != expected.shape or not np.isfinite(actual).all():
            raise ValueError('Observed upstream dtype/shape/values differ: ' + row['name'] + '/' + name)
        difference = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
        limit = SOURCE_ATOL + SOURCE_RTOL * np.abs(expected.astype(np.float64))
        passed = bool(np.all(difference <= limit))
        comparisons.append(dict(case=row['name'], name=name, passed=passed,
            maximum_absolute_error=float(difference.max()) if difference.size else 0.0,
            maximum_error_to_policy_ratio=float((difference / limit).max()) if difference.size else 0.0))

    def numpy_value(value):
        mx.eval(value)
        return np.asarray(value)

    try:
        for value in (signal.SIGTERM, signal.SIGINT):
            previous_signals[value] = signal.signal(value, interrupted)
        mx.set_default_device(mx.cpu if device == 'cpu' else mx.gpu)
        source_execution_attempted = True
        exec(compile(source, str(upstream_source), 'exec'), module.__dict__)
        source_execution_completed = True
        if sys.modules['mlx.nn'].__file__ != admitted_runtime['imports']['mlx.nn']:
            raise ValueError('Imported MLX nn differs from admitted files')
        captured_packed_policy = bool(module._ENABLE_GDN_PACKED)
        for row in manifest['cases']:
            tokens = row['geometry']['tokens']
            if tokens == 0:
                rows.append(dict(name=row['name'], upstream_evaluated=False,
                    reason='Empty-prefix no-op is mathematical/target admission policy; no upstream empty-call claim'))
                continue
            observation = dict(name=row['name'], upstream_evaluated=False,
                completed=False, requested_use_kernel=device == 'gpu', arrays={})
            rows.append(observation)
            dk, dv = row['geometry']['key_dim'], row['geometry']['value_dim']
            kernel_eligible = device == 'gpu' and dk >= 32 and dk % 32 == 0
            observation['resolved_path_from_pinned_dispatch'] = (
                'ops' if not kernel_eligible else
                'packed' if captured_packed_policy and dk == 128 and dv % 8 == 0 else 'generic')
            values = {name: mx.array(array(row, name)[None, ...])
                      for name in ('raw_q', 'raw_k', 'v', 'a', 'b', 'initial_state')}
            a_log, dt_bias = [mx.array(array(row, name)) for name in ('a_log', 'dt_bias')]
            q, k = module.normalize_qk(values['raw_q'], values['raw_k'],
                inv_scale=row['geometry']['key_dim'] ** -0.5, eps=row['geometry']['epsilon'])
            decay = module.compute_g(a_log, values['a'], dt_bias)
            beta = mx.sigmoid(values['b'])
            for name, value in [('normalized_q', q), ('normalized_k', k), ('decay', decay), ('beta', beta)]:
                observe(row, observation, name, numpy_value(value)[0], array(row, name))
            states = []
            full_output = None
            for end in range(1, tokens + 1):
                output, state = module.gated_delta_update(q[:, :end], k[:, :end], values['v'][:, :end],
                    values['a'][:, :end], values['b'][:, :end], a_log, dt_bias,
                    state=values['initial_state'], use_kernel=device == 'gpu')
                actual_state = numpy_value(state)[0]
                retain(observation, 'prefix_' + str(end) + '_state', actual_state)
                actual_output = numpy_value(output)[0]
                retain(observation, 'prefix_' + str(end) + '_output', actual_output)
                states.append(np.array(actual_state, copy=True))
                if end == tokens:
                    full_output = np.array(actual_output, copy=True)
            observe(row, observation, 'output', full_output, array(row, 'output'))
            observe(row, observation, 'states', np.stack(states), array(row, 'states'))
            observe(row, observation, 'final_state', states[-1], array(row, 'final_state'))
            # Separate chunked calls carry their returned FP32 state forward.
            # Repeated prefixes above deliberately reset to the same initial state.
            current = values['initial_state']
            chunks = []
            for first in range(0, tokens, 3):
                last = min(tokens, first + 3)
                output, current = module.gated_delta_update(q[:, first:last], k[:, first:last], values['v'][:, first:last],
                    values['a'][:, first:last], values['b'][:, first:last], a_log, dt_bias,
                    state=current, use_kernel=device == 'gpu')
                actual_output = numpy_value(output)[0]
                retain(observation, 'chunk_' + str(first) + '_output', actual_output)
                retain(observation, 'chunk_' + str(first) + '_state', numpy_value(current)[0])
                chunks.append(np.array(actual_output, copy=True))
            observe(row, observation, 'chunked_output', np.concatenate(chunks), array(row, 'output'))
            observe(row, observation, 'chunked_final_state', numpy_value(current)[0], array(row, 'final_state'))
            if not np.array_equal(numpy_value(values['initial_state'])[0], array(row, 'initial_state')):
                raise ValueError('Upstream mutated the immutable initial state')
            observation.update(upstream_evaluated=True, completed=True)
    except (Exception, KeyboardInterrupt) as error:
        primary = type(error).__name__ + ': ' + str(error)
    finally:
        # Suppress repeat TERM/INT while publishing bounded data. The parent has
        # its independent SIGKILL deadline if an extension or filesystem stalls.
        for value in previous_signals:
            signal.signal(value, signal.SIG_IGN)
        try:
            mx.set_default_device(previous_device)
        except Exception as error:
            errors.append('Restore reference device: ' + type(error).__name__ + ': ' + str(error))
        if sys.modules.get(namespace) is module:
            del sys.modules[namespace]
        else:
            errors.append('Reference namespace ownership changed')
    passed = (primary is None and not errors and source_execution_completed
        and len(rows) == len(fixture.CASE_SPECS)
        and all(row.get('completed', False) for row in rows if row.get('arrays') is not None)
        and len(comparisons) == 9 * (len(fixture.CASE_SPECS) - 1)
        and all(row['passed'] for row in comparisons))
    report = dict(format='toklet-gated-delta-upstream-observation-v1', passed=passed,
        mathematical_manifest_sha256=manifest_sha, mathematical_binary_sha256=binary_sha,
        upstream_revision=fixture.UPSTREAM_REVISION, upstream_source_sha256=fixture.UPSTREAM_SHA256,
        source_execution_attempted=source_execution_attempted,
        source_execution_completed=source_execution_completed,
        source_executed_from_authenticated_buffer=source_execution_completed,
        requested_device=device,
        runtime=observed_versions, admitted_runtime=admitted_runtime,
        environment_gdn_packed=os.environ.get('MLX_GDN_PACKED', '1'),
        captured_upstream_packed_policy=captured_packed_policy,
        source_policy=dict(atol=SOURCE_ATOL, rtol=SOURCE_RTOL), comparisons=comparisons,
        cases=rows, binary=dict(path='observed.bin', bytes=len(blob), sha256=hashlib.sha256(blob).hexdigest(), alignment=8),
        primary_error=primary, cleanup_errors=errors, numerical_device_qualified=False,
        scope='Tiny unmasked FP32 upstream observations against ideal FP64 math, not a native graph or performance/quality result')
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
    parser.add_argument('--upstream-source', required=True, type=Path)
    parser.add_argument('--device', required=True, choices=['cpu', 'gpu'])
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--runtime-manifest', required=True, type=Path)
    parser.add_argument('--expected-runtime-sha256', required=True)
    options = parser.parse_args()
    passed = compare(options.directory, options.expected_manifest_sha256, options.expected_binary_sha256,
                     options.upstream_source, options.device, options.output,
                     options.runtime_manifest, options.expected_runtime_sha256)
    print(json.dumps(dict(passed=passed, mathematical_reference_only=True)))
    raise SystemExit(0 if passed else 1)

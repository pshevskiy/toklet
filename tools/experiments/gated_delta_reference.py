"""Owned CPU/GPU reference capture with explicit source and runtime identities.

Consume a separately produced mathematical fixture, never freeze it in place.
Each comparison owns its child process group, logs, output and fresh bytecode
cache. A final provisional record keeps primary, endpoint and capture failures.
"""
import argparse
import hashlib
import json
from pathlib import Path
import signal
import sys

import m7_apple_package as owner

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools/reference'))
import gated_delta_upstream as upstream


def encoded(value):
    return (json.dumps(value, indent=2, allow_nan=False) + '\n').encode()


def pin(path):
    if path.is_symlink() or not path.is_file():
        raise ValueError('Regular reference input required: ' + str(path))
    digest = hashlib.sha256()
    count = 0
    with path.open('rb') as stream:
        while chunk := stream.read(128 * 1024):
            count += len(chunk)
            if count > 512 * 1024 * 1024:
                raise ValueError('Reference input exceeds its bound')
            digest.update(chunk)
    return dict(bytes=count, sha256=digest.hexdigest())


def capture(path, maximum):
    if path.is_symlink() or not path.is_file():
        raise ValueError('Regular reference output required')
    with path.open('rb') as stream:
        raw = stream.read(maximum + 1)
    if len(raw) > maximum:
        raise ValueError('Reference output exceeds its capture bound')
    return raw, dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def collect(fixture, manifest_sha, binary_sha, source, destination):
    fixture, source = fixture.absolute(), source.absolute()
    # mkdir is exclusive, including every retry. No prior artifact is replaced.
    if (destination.is_symlink() or not destination.is_absolute()
            or not destination.parent.is_dir()
            or not destination.resolve().is_relative_to(ROOT / 'artifacts')):
        raise ValueError('New absolute output below repository artifacts required')
    destination.mkdir()
    commands, endpoint_errors, capture_errors = [], [], []
    inputs, outputs = {}, {}
    primary = None
    runtime_files = None
    observations = {}
    runtime_manifest = None
    source_copy = destination / 'gated_delta.py'
    controlled_environment = {}
    previous = {}

    def interrupted(number, _frame):
        for value in (signal.SIGTERM, signal.SIGINT):
            signal.signal(value, signal.SIG_IGN)
        raise InterruptedError('Received signal ' + str(number))

    def guard():
        if runtime_files is not None and upstream.runtime_paths() != runtime_files:
            raise ValueError('Reference runtime inventory changed')
        for name, expected in inputs.items():
            if pin(Path(name)) != expected:
                raise ValueError('Reference input changed: ' + name)
        upstream.fixture.verify(fixture, manifest_sha, binary_sha)

    def retained(path, maximum):
        raw, descriptor = capture(path, maximum)
        name = str(path.relative_to(destination))
        if name in outputs and outputs[name] != descriptor:
            raise ValueError('Previously captured output changed: ' + name)
        outputs[name] = descriptor
        return raw

    try:
        for value in (signal.SIGTERM, signal.SIGINT):
            previous[value] = signal.signal(value, interrupted)
        # Authenticate the actual mathematical buffers and inert upstream source
        # before creating any runtime manifest or starting a numerical process.
        upstream.fixture.verify(fixture, manifest_sha, binary_sha)
        source_raw = upstream.captured(source, 256 * 1024, upstream.fixture.UPSTREAM_SHA256)
        source_pin = dict(bytes=len(source_raw), sha256=hashlib.sha256(source_raw).hexdigest())
        inputs[str(source)] = source_pin
        with source_copy.open('xb') as stream:
            stream.write(source_raw)
        inputs[str(source_copy)] = source_pin
        runtime_files = upstream.runtime_paths()
        selected = [Path(__file__).resolve(), Path(owner.__file__).resolve(),
            *sorted((ROOT / 'tools/reference').glob('*gated_delta*.py')),
            fixture / 'manifest.json', fixture / 'arrays.bin', source, source_copy]
        for path in sorted(set([*selected, *runtime_files])):
            actual_pin = pin(path)
            if str(path) in inputs and inputs[str(path)] != actual_pin:
                raise ValueError('Captured upstream source changed: ' + str(path))
            inputs[str(path)] = actual_pin
        # Versions are expected constants; child admission checks actual metadata.
        runtime_manifest = dict(format='toklet-gated-delta-runtime-v1',
            versions=upstream.VERSIONS, files={str(path): inputs[str(path)] for path in runtime_files})
        runtime_path = destination / 'runtime.json'
        runtime_raw = encoded(runtime_manifest)
        with runtime_path.open('xb') as stream:
            stream.write(runtime_raw)
        runtime_sha = hashlib.sha256(runtime_raw).hexdigest()
        inputs[str(runtime_path)] = dict(bytes=len(runtime_raw), sha256=runtime_sha)
        logs = destination / 'logs'
        logs.mkdir()
        temporary = destination / 'tmp'
        temporary.mkdir()
        environment = dict(HOME=str(Path.home()), PATH='/usr/bin:/bin:/usr/sbin:/sbin',
            LANG='en_US.UTF-8', TMPDIR=str(temporary), PYTHONNOUSERSITE='1', MLX_GDN_PACKED='1')
        controlled_environment = dict(environment)
        for device in ('cpu', 'gpu'):
            guard()
            cache = destination / (device + '-pycache')
            cache.mkdir()
            output = destination / device
            output.mkdir()
            current_environment = dict(environment, PYTHONPYCACHEPREFIX=str(cache))
            command = [sys.executable, '-s', str(ROOT / 'tools/reference/gated_delta_upstream.py'),
                str(fixture), '--expected-manifest-sha256', manifest_sha,
                '--expected-binary-sha256', binary_sha, '--upstream-source', str(source_copy),
                '--device', device, '--output', str(output), '--runtime-manifest', str(runtime_path),
                '--expected-runtime-sha256', runtime_sha]
            attempt = dict(command=command, device=device, pycache_prefix=str(cache),
                owned_result=None, invocation_error=None)
            commands.append(attempt)
            try:
                attempt['owned_result'] = owner.owned(command, logs / (device + '.stdout'), logs / (device + '.stderr'),
                    current_environment, timeout=1200)
            except (Exception, KeyboardInterrupt) as error:
                attempt['invocation_error'] = type(error).__name__ + ': ' + str(error)
            finally:
                # Capture known owned paths even when the helper could not return
                # a row. In that case PID/exit/cleanup remain unknown, never green.
                for suffix in ('stdout', 'stderr'):
                    try:
                        retained(logs / (device + '.' + suffix), 8 * 1024 * 1024)
                    except (Exception, KeyboardInterrupt) as error:
                        capture_errors.append(device + '/' + suffix + ': ' + type(error).__name__ + ': ' + str(error))
                for name, maximum in [('observed.bin', upstream.fixture.MAX_ARRAY_BYTES), ('observation.json', 2 * 1024 * 1024)]:
                    try:
                        raw = retained(output / name, maximum)
                        if name == 'observation.json':
                            observations[device] = json.loads(raw, object_pairs_hook=upstream.fixture._unique,
                                parse_constant=upstream.fixture._bad_number, parse_float=upstream.fixture._finite_number)
                    except (Exception, KeyboardInterrupt) as error:
                        capture_errors.append(device + '/' + name + ': ' + type(error).__name__ + ': ' + str(error))
            if attempt['invocation_error']:
                raise RuntimeError('Owned comparison invocation failed: ' + attempt['invocation_error'])
            result = attempt['owned_result']
            if result['primary_error'] or result['cleanup_errors']:
                raise RuntimeError('Owned ' + device + ' comparison failed: ' + str(result['primary_error'] or result['cleanup_errors']))
            if capture_errors:
                raise ValueError('Reference capture failed')
            guard()
            observation = observations.get(device)
            if not isinstance(observation, dict) or observation.get('passed') is not True:
                raise ValueError('Upstream comparison did not pass')
            binary_pin = outputs[device + '/observed.bin']
            if (observation.get('format') != 'toklet-gated-delta-upstream-observation-v1'
                    or observation.get('requested_device') != device
                    or observation.get('mathematical_manifest_sha256') != manifest_sha
                    or observation.get('mathematical_binary_sha256') != binary_sha
                    or observation.get('upstream_source_sha256') != upstream.fixture.UPSTREAM_SHA256
                    or observation.get('admitted_runtime', {}).get('manifest_sha256') != runtime_sha
                    or observation.get('source_execution_completed') is not True
                    or observation.get('primary_error') is not None or observation.get('cleanup_errors') != []
                    or observation.get('numerical_device_qualified') is not False
                    or observation.get('binary') != dict(path='observed.bin', alignment=8, **binary_pin)):
                raise ValueError('Consumed upstream observation identity/scope differs')
    except (Exception, KeyboardInterrupt) as error:
        primary = type(error).__name__ + ': ' + str(error)
    finally:
        for value in previous:
            signal.signal(value, signal.SIG_IGN)
        try:
            guard()
        except (Exception, KeyboardInterrupt) as error:
            endpoint_errors.append(type(error).__name__ + ': ' + str(error))
        for name, expected in outputs.items():
            try:
                if pin(destination / name) != expected:
                    endpoint_errors.append('Captured output changed: ' + name)
            except (Exception, KeyboardInterrupt) as error:
                endpoint_errors.append(name + ': ' + type(error).__name__ + ': ' + str(error))
    passed = (primary is None and not endpoint_errors and not capture_errors
        and set(observations) == {'cpu', 'gpu'} and len(commands) == 2
        and all(row.get('passed') is True for row in observations.values()))
    report = dict(format='toklet-gated-delta-owned-reference-v1', passed=passed,
        provisional=True, fixture_accepted=False, native_graph_implemented=False,
        numerical_device_qualified=False, inputs=inputs, outputs=outputs,
        fixture_manifest_sha256=manifest_sha, fixture_binary_sha256=binary_sha,
        commands=commands, controlled_environment=controlled_environment,
        primary_error=primary, endpoint_errors=endpoint_errors, capture_errors=capture_errors,
        scope='Owned tiny unmasked upstream source comparison only; final source/evidence review and acceptance required')
    try:
        with (destination / 'capture.json').open('xb') as stream:
            stream.write(encoded(report))
    finally:
        for value, handler in previous.items():
            signal.signal(value, handler)
    print(json.dumps(dict(passed=passed, provisional=True, output=str(destination), primary_error=primary)))
    return 0 if passed else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('fixture', type=Path)
    parser.add_argument('--expected-manifest-sha256', required=True)
    parser.add_argument('--expected-binary-sha256', required=True)
    parser.add_argument('--upstream-source', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    options = parser.parse_args()
    raise SystemExit(collect(options.fixture, options.expected_manifest_sha256,
        options.expected_binary_sha256, options.upstream_source, options.output))

"""Owned independent convolution export and explicit CPU/GPU source comparison.

Each child owns fresh output/cache/log paths and a process group. The provisional
capture retains failures and endpoint guards; acceptance requires source/evidence
review. No native model, device capacity or performance claim is made here.
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
import causal_conv_fixture as fixture
import reference_runtime as runtime


def encoded(value):
    return (json.dumps(value, indent=2, allow_nan=False) + '\n').encode()


def pin(path):
    if path.is_symlink() or not path.is_file():
        raise ValueError('Regular reference input required: ' + str(path))
    digest, count = hashlib.sha256(), 0
    with path.open('rb') as stream:
        while chunk := stream.read(128 * 1024):
            count += len(chunk)
            if count > 512 * 1024 * 1024:
                raise ValueError('Reference input exceeds bound')
            digest.update(chunk)
    return dict(bytes=count, sha256=digest.hexdigest())


def collect(source, destination):
    source = source.absolute()
    if (destination.is_symlink() or not destination.is_absolute() or not destination.parent.is_dir()
            or not destination.resolve().is_relative_to(ROOT / 'artifacts')):
        raise ValueError('New absolute output below repository artifacts required')
    destination.mkdir()
    inputs, outputs, commands, observations = {}, {}, [], {}
    endpoint_errors, capture_errors, primary = [], [], None
    runtime_files = None
    selected_sources = None
    previous = {}
    math_manifest_sha = math_binary_sha = None
    controlled_environment = {}

    def interrupted(number, _frame):
        for value in (signal.SIGTERM, signal.SIGINT):
            signal.signal(value, signal.SIG_IGN)
        raise InterruptedError('Received signal ' + str(number))

    def guard():
        if selected_sources is not None and runtime.runtime_paths(selected_sources) != runtime_files:
            raise ValueError('Reference runtime inventory changed')
        for name, expected in inputs.items():
            if pin(Path(name)) != expected:
                raise ValueError('Reference input changed: ' + name)

    def retained(path, maximum):
        if path.is_symlink() or not path.is_file():
            raise ValueError('Regular reference output required')
        with path.open('rb') as stream:
            raw = stream.read(maximum + 1)
        if len(raw) > maximum:
            raise ValueError('Reference output exceeds capture bound')
        descriptor = dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
        name = str(path.relative_to(destination))
        if name in outputs and outputs[name] != descriptor:
            raise ValueError('Previously captured output changed')
        outputs[name] = descriptor
        return raw

    def invoke(label, command, expected_outputs=()):
        guard()
        cache = destination / (label + '-pycache')
        cache.mkdir()
        attempt = dict(label=label, command=command, pycache_prefix=str(cache), owned_result=None, invocation_error=None)
        commands.append(attempt)
        try:
            attempt['owned_result'] = owner.owned(command, destination / 'logs' / (label + '.stdout'),
                destination / 'logs' / (label + '.stderr'), dict(controlled_environment, PYTHONPYCACHEPREFIX=str(cache)), timeout=1200)
        except (Exception, KeyboardInterrupt) as error:
            attempt['invocation_error'] = type(error).__name__ + ': ' + str(error)
        finally:
            for path, maximum in [(destination / 'logs' / (label + '.' + suffix), 8 * 1024 * 1024) for suffix in ('stdout', 'stderr')] + list(expected_outputs):
                try:
                    retained(path, maximum)
                except (Exception, KeyboardInterrupt) as error:
                    capture_errors.append(label + '/' + str(path) + ': ' + type(error).__name__ + ': ' + str(error))
        if attempt['invocation_error']:
            raise RuntimeError('Owned invocation failed: ' + attempt['invocation_error'])
        result = attempt['owned_result']
        if result['primary_error'] or result['cleanup_errors']:
            raise RuntimeError('Owned command failed: ' + str(result['primary_error'] or result['cleanup_errors']))
        if capture_errors:
            raise ValueError('Reference output capture failed')
        guard()

    try:
        for value in (signal.SIGTERM, signal.SIGINT):
            previous[value] = signal.signal(value, interrupted)
        raw = runtime.captured(source, 128 * 1024, fixture.MODEL_SOURCE_SHA)
        source_copy = destination / 'qwen3_5.py'
        with source_copy.open('xb') as stream:
            stream.write(raw)
        sources = [ROOT / 'tools/reference' / name for name in (
            'reference_runtime.py', 'causal_conv_fixture.py', 'causal_conv_oracle.py',
            'causal_conv_upstream.py', 'test_causal_conv.py', 'test_reference_runtime.py')]
        sources += [Path(__file__).resolve(), Path(owner.__file__).resolve(),
            ROOT / 'tools/experiments/test_causal_conv_reference.py']
        selected_sources = [path.resolve() for path in sources]
        runtime_files = runtime.runtime_paths(selected_sources)
        for path in sorted(set([*runtime_files, source, source_copy,
                ROOT / 'target/release/toklet', ROOT / 'target/release/examples/cpu_bench'])):
            inputs[str(path)] = pin(path)
        captured_pin = dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
        if inputs[str(source)] != captured_pin or inputs[str(source_copy)] != captured_pin:
            raise ValueError('Authenticated model source buffer changed before pinning')
        snapshot = destination / 'source-snapshot'
        snapshot.mkdir()
        for path in sources:
            target = snapshot / path.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open('xb') as stream:
                stream.write(runtime.captured(path, 512 * 1024, inputs[str(path)]['sha256']))
            inputs[str(target)] = pin(target)
        (destination / 'logs').mkdir()
        (destination / 'tmp').mkdir()
        controlled_environment = dict(HOME=str(Path.home()), PATH='/usr/bin:/bin:/usr/sbin:/sbin',
            LANG='en_US.UTF-8', TMPDIR=str(destination / 'tmp'), PYTHONNOUSERSITE='1')
        invoke('ownership', [sys.executable, '-s', '-m', 'unittest', 'discover', '-s', str(ROOT / 'tools/experiments'), '-p', 'test_causal_conv_reference.py', '-v'])
        invoke('portable', [sys.executable, '-s', '-m', 'unittest', 'discover', '-s', str(ROOT / 'tools/reference'), '-p', 'test_causal_conv.py', '-v'])
        invoke('admission', [sys.executable, '-s', '-m', 'unittest', 'discover', '-s', str(ROOT / 'tools/reference'), '-p', 'test_reference_runtime.py', '-v'])

        def manifest_for(label, selected):
            files = runtime.runtime_paths(selected)
            content = encoded(dict(format='toklet-mlx-reference-runtime-v1', versions=runtime.VERSIONS,
                files={str(path): inputs[str(path)] for path in files}))
            path = destination / (label + '-runtime.json')
            with path.open('xb') as stream:
                stream.write(content)
            inputs[str(path)] = pin(path)
            return path, inputs[str(path)]['sha256']

        producer = ROOT / 'tools/reference/causal_conv_oracle.py'
        reader = ROOT / 'tools/reference/causal_conv_fixture.py'
        producer_manifest, producer_sha = manifest_for('producer', [producer, reader])
        math_output = destination / 'math'
        math_output.mkdir()
        invoke('export', [sys.executable, '-s', str(producer), '--output', str(math_output),
            '--runtime-manifest', str(producer_manifest), '--expected-runtime-sha256', producer_sha,
            '--model-source', str(source_copy)], [(math_output / 'manifest.json', 128 * 1024), (math_output / 'arrays.bin', fixture.MAX_BYTES)])
        math_manifest_sha = outputs['math/manifest.json']['sha256']
        math_binary_sha = outputs['math/arrays.bin']['sha256']
        fixture.verify(math_output, math_manifest_sha, math_binary_sha)
        for name in ('manifest.json', 'arrays.bin'):
            inputs[str(math_output / name)] = pin(math_output / name)
        driver = ROOT / 'tools/reference/causal_conv_upstream.py'
        driver_manifest, driver_sha = manifest_for('upstream', [driver, reader])
        for device in ('cpu', 'gpu'):
            output = destination / device
            output.mkdir()
            invoke(device, [sys.executable, '-s', str(driver), str(math_output),
                '--expected-manifest-sha256', math_manifest_sha, '--expected-binary-sha256', math_binary_sha,
                '--model-source', str(source_copy), '--device', device, '--output', str(output),
                '--runtime-manifest', str(driver_manifest), '--expected-runtime-sha256', driver_sha],
                [(output / 'observed.bin', fixture.MAX_BYTES), (output / 'observation.json', 2 * 1024 * 1024)])
            observation = json.loads(retained(output / 'observation.json', 2 * 1024 * 1024),
                object_pairs_hook=runtime.json_unique, parse_constant=runtime.bad_json_number, parse_float=runtime.finite_json_number)
            observations[device] = observation
            if (observation.get('passed') is not True or observation.get('format') != 'toklet-causal-conv-upstream-observation-v1'
                    or observation.get('requested_device') != device or observation.get('primary_error') is not None
                    or observation.get('cleanup_errors') != [] or observation.get('mathematical_manifest_sha256') != math_manifest_sha
                    or observation.get('mathematical_binary_sha256') != math_binary_sha
                    or observation.get('model_source_sha256') != fixture.MODEL_SOURCE_SHA
                    or observation.get('admitted_runtime', {}).get('manifest_sha256') != driver_sha
                    or observation.get('native_graph_implemented') is not False or observation.get('numerical_device_qualified') is not False
                    or observation.get('binary') != dict(path='observed.bin', alignment=8, **outputs[device + '/observed.bin'])):
                raise ValueError('Upstream observation identity/scope differs')
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
    passed = (primary is None and not endpoint_errors and not capture_errors and len(commands) == 6
        and set(observations) == {'cpu', 'gpu'} and all(row.get('passed') is True for row in observations.values()))
    report = dict(format='toklet-causal-conv-owned-reference-v1', passed=passed, provisional=True,
        fixture_accepted=False, native_graph_implemented=False, numerical_device_qualified=False,
        inputs=inputs, outputs=outputs, commands=commands, controlled_environment=controlled_environment,
        mathematical_manifest_sha256=math_manifest_sha, mathematical_binary_sha256=math_binary_sha,
        primary_error=primary, endpoint_errors=endpoint_errors, capture_errors=capture_errors,
        scope='Tiny independent math and unmasked selected-source operator comparison; final review required')
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
    parser.add_argument('--model-source', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    options = parser.parse_args()
    raise SystemExit(collect(options.model_source, options.output))

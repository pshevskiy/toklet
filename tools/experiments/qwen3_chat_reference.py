"""Owned, externally admitted capture/replay of the isolated Qwen3 text witness."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import signal
import sys

ROOT = Path.cwd()
SOURCE = Path(__file__).resolve().parents[1] / 'reference'
sys.path.insert(0, str(SOURCE))
sys.path.insert(0, str(ROOT / 'tools/experiments'))
import chat_reference_runtime as runtime
import m7_apple_package as owner


def encoded(value):
    return (json.dumps(value, indent=2, ensure_ascii=False) + '\n').encode()


def capture_stream(path):
    if path.is_symlink() or not path.is_file():
        raise ValueError('Regular command stream required')
    with path.open('rb') as stream:
        raw = stream.read(8 * 1024 * 1024 + 1)
    if len(raw) > 8 * 1024 * 1024:
        raise ValueError('Command stream exceeds capture bound')
    return dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def invoke_owned(label, command, output, environment, guard, commands, captures,
                 capture_errors, endpoint_errors, expected_exit=0, runner=None):
    cache = output / (label + '-pycache')
    attempt = dict(label=label, command=list(command), expected_exit=expected_exit,
                   pycache_prefix=str(cache), result=None, invocation_error=None)
    commands.append(attempt)
    try:
        guard()
        cache.mkdir()
        attempt['result'] = (runner or owner.owned)(command,
            output / 'logs' / (label + '.stdout'), output / 'logs' / (label + '.stderr'),
            dict(environment, PYTHONPYCACHEPREFIX=str(cache)), timeout=120)
    except BaseException as error:
        attempt['invocation_error'] = type(error).__name__ + ': ' + str(error)
    finally:
        for suffix in ['stdout', 'stderr']:
            path = output / 'logs' / (label + '.' + suffix)
            try:
                captures[str(path)] = capture_stream(path)
            except BaseException as error:
                capture_errors.append(label + '/' + suffix + ': ' + type(error).__name__ + ': ' + str(error))
        try:
            guard()
        except BaseException as error:
            endpoint_errors.append(label + ': ' + type(error).__name__ + ': ' + str(error))
    # Preserve the returned owner row unchanged, and prioritize its original
    # failure over later stream/endpoint capture failures. All causes are retained.
    if attempt['invocation_error']:
        raise RuntimeError('Owned invocation failed: ' + attempt['invocation_error'])
    result = attempt['result']
    if result is None:
        raise RuntimeError('Owned helper returned no command record')
    if result['exit_code'] != expected_exit or result['cleanup_errors'] or (expected_exit == 0 and result['primary_error']):
        raise RuntimeError('Owned command failed: ' + str(result['primary_error'] or result['cleanup_errors'] or result['exit_code']))
    if capture_errors:
        raise ValueError('Command stream capture failed; separate causes retained')
    if endpoint_errors:
        raise ValueError('Command endpoint check failed; separate causes retained')
    print(label, result['exit_code'], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--preliminary', type=Path)
    args = parser.parse_args()
    output = args.output.absolute()
    if not output.resolve().is_relative_to(ROOT / 'artifacts'):
        raise ValueError('Fresh repository artifact output required')
    output.mkdir()
    for name in ['logs', 'tmp', 'assets']:
        (output / name).mkdir()
    producer = SOURCE / 'app_chat_qwen3.py'
    selected = [producer, Path(__file__).resolve()]
    files = runtime.paths(selected)
    inputs = {str(p): runtime.pin(p) for p in files}
    inputs[str(Path(owner.__file__).resolve())] = runtime.pin(Path(owner.__file__).resolve())
    portable = SOURCE / 'test_chat_reference_runtime.py'
    inputs[str(portable)] = runtime.pin(portable)
    ownership = Path(__file__).with_name('test_qwen3_chat_reference.py')
    inputs[str(ownership)] = runtime.pin(ownership)
    for name in ['src/app_chat.rs', 'src/native/text.rs', 'src/app_chat/stream.rs']:
        inputs[str(ROOT / name)] = runtime.pin(ROOT / name)
    model = ROOT / 'models/MLX-Qwen3-0.6B-8bit-11de968'
    pins = {'config.json': '9f387dcb1bf045cadb34a36e67569abd73c00e5322603df2b19d0c0f8cc9d240',
            'tokenizer.json': 'aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4',
            'tokenizer_config.json': '8e7dd4250db58b2a29792562d727b2715dc0fe79a94b446caa4c07db39eceba6'}
    for name, sha in pins.items():
        raw = runtime.captured(model / name, 16 * 1024 * 1024, sha)
        inputs[str(model / name)] = runtime.pin(model / name)
        copied = output / 'assets' / name
        with copied.open('xb') as stream:
            stream.write(raw)
        inputs[str(copied)] = runtime.pin(copied)
    manifest = dict(format='toklet-chat-reference-runtime-v1', versions=runtime.VERSIONS,
                    files={str(p): inputs[str(p)] for p in files})
    manifest_path = output / 'runtime.json'
    manifest_path.write_bytes(encoded(manifest))
    manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    inputs[str(manifest_path)] = runtime.pin(manifest_path)
    commands, captures, endpoints, capture_errors, primary = [], {}, [], [], None
    env = dict(HOME=str(Path.home()), PATH='/usr/bin:/bin:/usr/sbin:/sbin', LANG='en_US.UTF-8',
               TMPDIR=str(output / 'tmp'), PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1')

    def guard():
        if runtime.paths(selected) != files:
            raise ValueError('Text reference runtime inventory changed')
        for name, expected in inputs.items():
            if runtime.pin(Path(name)) != expected:
                raise ValueError('Text reference input changed: ' + name)

    def invoke(label, command, expected_exit=0):
        invoke_owned(label, command, output, env, guard, commands, captures, capture_errors, endpoints, expected_exit)

    reproduced = False
    previous = signal.signal(signal.SIGTERM, owner.interrupted)
    try:
        if args.preliminary:
            old = args.preliminary.resolve()
            inputs[str(old)] = runtime.pin(old)
            script = 'import json,sys;f=json.load(open(sys.argv[1]));assert all(c["decoded"]==c["text"] for c in f["decode_cases"]),"Input/decode identity was too strong: upstream NFC normalization is preserved"'
            invoke('preliminary-identity-assertion-reproduction', [sys.executable, '-s', '-c', script, str(old)], expected_exit=1)
            proof = output / 'logs/preliminary-identity-assertion-reproduction.stderr'
            raw = runtime.captured(proof, 8 * 1024 * 1024, captures[str(proof)]['sha256'])
            if b'AssertionError' not in raw or b'Input/decode identity was too strong' not in raw:
                raise ValueError('Expected preliminary assertion failure was not reproduced')
            reproduced = True
        env['TOKLET_CHAT_REFERENCE_OUTPUT'] = str(output)
        invoke('ownership', [sys.executable, '-s', '-B', '-m', 'unittest', 'discover', '-s', str(ownership.parent), '-p', ownership.name, '-v'])
        invoke('portable', [sys.executable, '-s', '-B', '-m', 'unittest', 'discover', '-s', str(SOURCE), '-p', portable.name, '-v'])
        common = [sys.executable, '-s', '-B', str(producer), str(output / 'assets'), '--runtime', str(manifest_path),
                  '--runtime-sha256', manifest_sha, '--owner-source', str(Path(__file__).resolve())]
        fixture = output / 'reference.json'
        invoke('capture', [*common, '--output', str(fixture)])
        inputs[str(fixture)] = runtime.pin(fixture)
        invoke('replay', [*common, '--check', str(fixture), '--check-sha256', inputs[str(fixture)]['sha256']])
        f = json.loads(fixture.read_bytes())
        assert len(f['cases']) == 5 and len(f['decode_cases']) == 4 and len(f['special_literals']) == 26
        assert sum(not x['special'] for x in f['special_literals']) == 12
        assert f['stop_ids'] == [151645] and f['thinking_token_ids'] == [151667, 151668]
        assert f['decode_cases'][1]['decoded'] == '你好，🌍! é'
        assert f['decode_cases'][3]['decoded'] == '<think>work</think> <tool_call>call</tool_call>'
        for case in f['decode_cases']:
            assert case['prefixes'][0] == '' and case['prefixes'][-1] == case['decoded']
        guard()
        captures[str(fixture.relative_to(ROOT))] = runtime.pin(fixture)
    except BaseException as error:
        primary = type(error).__name__ + ': ' + str(error)
    finally:
        signal.signal(signal.SIGTERM, previous)
        for name, expected in inputs.items():
            try:
                if runtime.pin(Path(name)) != expected:
                    endpoints.append('Changed input: ' + name)
            except BaseException as error:
                endpoints.append(str(error))
        result = dict(format='toklet-qwen3-chat-reference-capture-v1', passed=primary is None and not endpoints and not capture_errors,
                      commands=commands, input_pins=inputs, capture_pins=captures, runtime_manifest_sha256=manifest_sha,
                      runtime_versions=runtime.VERSIONS, primary_error=primary, endpoint_errors=endpoints, capture_errors=capture_errors,
                      preliminary_checks_unaccepted=bool(args.preliminary), failed_assertion_reproduced=reproduced,
                      expected_nonzero_command='preliminary-identity-assertion-reproduction' if args.preliminary else None,
                      native_profile_changed=False, numerical_or_phone_qualified=False)
        (output / 'capture.json').write_bytes(encoded(result))
    print(json.dumps(dict(passed=result['passed'], primary_error=primary, endpoint_errors=endpoints, capture_errors=capture_errors)))
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())

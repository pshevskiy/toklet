"""Acquire one pinned official checkpoint into one exclusive staging file.

No second full assembly copy, remote code, numerical runtime or model inference.
Diagnostic pipes have individual disk caps; the body has a child file-size cap.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import resource
import re
import selectors
import shutil
import signal
import stat
import struct
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools/reference'))
import gemma4_unquantized_layout as layout
import m8_gemma4_checkpoint as transport

SIZE = layout.FILE_BYTES
SHA = layout.FILE_SHA
DIAGNOSTIC_CAP = 2 * 1024**2
COPY_BLOCK = 1024**2
DESTINATION = ROOT / 'models/Gemma4-E2B-it-BF16-70af34e'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def identity(s):
    return s.st_dev, s.st_ino, s.st_mode, s.st_size, s.st_mtime_ns, s.st_ctime_ns


def deadline_check(deadline, stop):
    if stop.is_set():
        raise InterruptedError('acquisition cancelled')
    if time.monotonic() >= deadline:
        raise TimeoutError('acquisition deadline')


def regular_endpoint(file, path, maximum):
    value = os.fstat(file.fileno())
    require(stat.S_ISREG(value.st_mode) and value.st_size <= maximum,
            'regular bounded body required')
    require(identity(value) == identity(path.lstat()), 'body descriptor/path changed')
    return value


def hash_owned(file, path, size, prefix_bytes, deadline, stop):
    layout.integer(size, 1, 12 * 1024**3)
    layout.integer(prefix_bytes, 0, layout.MAX_HEADER + 8)
    before = regular_endpoint(file, path, size)
    require(before.st_size == size, 'complete body size required')
    file.seek(0)
    digest = hashlib.sha256()
    prefix = bytearray()
    count = 0
    while True:
        deadline_check(deadline, stop)
        raw = file.read(min(COPY_BLOCK, size - count + 1))
        if not raw:
            break
        count += len(raw)
        require(count <= size, 'body grew during hashing')
        digest.update(raw)
        prefix.extend(raw[:max(0, prefix_bytes - len(prefix))])
    after = regular_endpoint(file, path, size)
    require(count == size and identity(before) == identity(after), 'body changed during hashing')
    return dict(bytes=count, sha256=digest.hexdigest()), bytes(prefix), identity(after)


def run_owned(factory, file, directory, size, deadline, stop, env):
    """Drain all three child diagnostic pipes, retaining at most2MiB each.

    The factory receives inherited header/body FDs and remaining seconds. It is
    local code, used by the fixed curl invocation and synthetic offline controls.
    """
    layout.integer(size, 1, 12 * 1024**3)
    result = dict(command=None, pid=None, exit_code=None, primary_error=None,
                  cleanup_errors=[], diagnostic_bytes={}, diagnostic_overflow=[])
    process = None
    selector = selectors.DefaultSelector()
    logs = {}
    pipe_read, pipe_write = os.pipe()
    header_stream = os.fdopen(pipe_read, 'rb', buffering=0)
    streams = [header_stream]

    def consume(key, raw):
        old = result['diagnostic_bytes'].get(key, 0)
        available = max(0, DIAGNOSTIC_CAP - old)
        if available:
            logs[key].write(raw[:available])
        result['diagnostic_bytes'][key] = old + len(raw)
        if old + len(raw) > DIAGNOSTIC_CAP and key not in result['diagnostic_overflow']:
            result['diagnostic_overflow'].append(key)

    try:
        for key in ('headers', 'stdout', 'stderr'):
            logs[key] = (directory / ('download.' + key)).open('xb')
        deadline_check(deadline, stop)
        command = factory(pipe_write, file.fileno(), max(1, int(deadline - time.monotonic())))
        result['command'] = command
        # Set only this isolated Python process's soft limit while spawning.
        # The child inherits it; no unsafe pre-exec callback is needed.
        original = resource.getrlimit(resource.RLIMIT_FSIZE)
        require(original[0] == resource.RLIM_INFINITY or original[0] >= size,
                'existing file-size limit too small')
        try:
            resource.setrlimit(resource.RLIMIT_FSIZE, (size, original[1]))
            process = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL,
                                       stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                       start_new_session=True, pass_fds=(pipe_write, file.fileno()), env=env)
            result['pid'] = process.pid
        finally:
            resource.setrlimit(resource.RLIMIT_FSIZE, original)
        os.close(pipe_write)
        pipe_write = None
        streams.extend([process.stdout, process.stderr])
        for stream, key in zip(streams, ('headers', 'stdout', 'stderr'), strict=True):
            os.set_blocking(stream.fileno(), False)
            selector.register(stream, selectors.EVENT_READ, key)
        journal = layout.encoded(dict(pid=process.pid, command=command))
        require(len(journal) <= DIAGNOSTIC_CAP, 'bounded child journal required')
        with (directory / 'download.child.json').open('xb') as journal_sink:
            require(journal_sink.write(journal) == len(journal), 'short child journal write')
        while process.poll() is None or selector.get_map():
            deadline_check(deadline, stop)
            for event, _ in selector.select(.1):
                raw = os.read(event.fd, 65536)
                if raw:
                    consume(event.data, raw)
                else:
                    selector.unregister(event.fileobj)
            require(not result['diagnostic_overflow'], 'diagnostic cap exceeded')
        deadline_check(deadline, stop)
        result['exit_code'] = process.wait(timeout=15)
        require(result['exit_code'] == 0, 'download child exit ' + str(result['exit_code']))
    except BaseException as error:
        result['primary_error'] = type(error).__name__ + ': ' + str(error)
    finally:
        if process is not None:
            try:
                transport.package.drain_group(process, result['cleanup_errors'], 15)
                result['exit_code'] = process.wait(timeout=15)
            except BaseException as error:
                result['cleanup_errors'].append('owned cleanup: ' + str(error))
        if pipe_write is not None:
            os.close(pipe_write)
        # Preserve available residual bytes after termination, within each cap.
        for event in list(selector.get_map().values()):
            try:
                while True:
                    raw = os.read(event.fd, 65536)
                    if not raw:
                        break
                    consume(event.data, raw)
            except BlockingIOError:
                pass
            except BaseException as error:
                result['cleanup_errors'].append('diagnostic drain: ' + str(error))
        selector.close()
        for stream in streams:
            try:
                stream.close()
            except BaseException as error:
                result['cleanup_errors'].append('pipe close: ' + str(error))
        for key, log in logs.items():
            try:
                log.close()
            except BaseException as error:
                result['cleanup_errors'].append(key + ' close: ' + str(error))
    return result


def full_response(headers, status, size):
    transport.response(headers, status)
    blocks = [part for part in re.split(rb'\r?\n\r?\n', headers) if part.startswith(b'HTTP/')]
    # response() already rejects duplicate lengths. Check LF-only servers too.
    lines = blocks[-1].splitlines()
    lengths = [line.split(b':', 1)[1].strip() for line in lines[1:]
               if line.lower().startswith(b'content-length:')]
    require(not lengths or lengths == [str(size).encode()], 'full Content-Length differs')


def publish_authenticated(staging, destination, authenticated, staged_directory,
                          config, pins, deadline, stop, result):
    # Only the caller's already-authenticated file may cross this boundary.
    for name, pin in pins.items():
        require(layout.sha(layout.read(ROOT / name, 4 * 1024**2)) == pin,
                'source endpoint changed: ' + name)
    body = staging / 'model.safetensors'
    require(identity(body.lstat()) == authenticated, 'staged body changed before publish')
    with (staging / 'config.json').open('xb') as file:
        require(file.write(config) == len(config), 'short config write')
    directory = staging.lstat()
    require(stat.S_ISDIR(directory.st_mode)
            and (directory.st_dev, directory.st_ino) == staged_directory, 'staging directory replaced')
    require(identity(body.lstat()) == authenticated, 'body changed before publication')
    deadline_check(deadline, stop)
    transport.publish(staging, destination)
    result['published'] = True
    require(identity((destination / 'model.safetensors').lstat()) == authenticated,
            'published file identity differs')


def write_receipt(output, result):
    try:
        with (output / 'download.json').open('x') as file:
            json.dump(result, file, indent=2); file.write('\n')
    except BaseException as error:
        result['published_receipt_error'] = str(error)
        print(json.dumps(dict(published=result['published'], receipt_error=str(error))), file=sys.stderr)
        result['passed'] = False


def main(output):
    output = output.resolve()
    staging = output / 'staging'
    logs = output / 'logs'
    require(not any(os.path.lexists(path) for path in [DESTINATION, staging, logs, output / 'download.json']), 'existing output refused')
    require(output.is_relative_to(ROOT), 'output must be inside repository')
    fixture = ROOT / 'tests/fixtures/gemma-unquantized-header-v1'
    api = ROOT / 'artifacts/m8-gemma-quality-source-metadata-v2/inventory.json'
    files = [Path(__file__).resolve(), Path(transport.__file__).resolve(),
             Path(transport.package.__file__).resolve(), Path(layout.__file__).resolve(),
             api, fixture / 'header.json', fixture / 'config.json', fixture / 'layout.json']
    pins = {str(path.relative_to(ROOT)): layout.sha(layout.read(path, 4 * 1024**2)) for path in files}
    require(pins[str(api.relative_to(ROOT))] == '87aae76fab388e3e6d82e97f9210fb2e85897037fdb81fd1eb54b1389920ea53', 'pinned source API')
    output.mkdir(parents=True, exist_ok=True)
    require(shutil.disk_usage(output).free >= SIZE + 256 * 1024**2, 'insufficient volume storage')
    staging.mkdir(mode=0o700)
    logs.mkdir(mode=0o700)
    require(staging.stat().st_dev == DESTINATION.parent.stat().st_dev, 'same-volume publication required')
    staged_directory = (staging.stat().st_dev, staging.stat().st_ino)
    deadline = time.monotonic() + 3600
    stop = transport.STOP
    stop.clear()
    transport.DEADLINE = deadline
    previous = {sig: signal.signal(sig, lambda *_: stop.set()) for sig in (signal.SIGINT, signal.SIGTERM)}
    result = dict(passed=False, primary_error=None, endpoint_errors=[], command=None, captures=[],
                  published=False, published_receipt_error=None, source_pins=pins,
                  destination=str(DESTINATION.relative_to(ROOT)), expected_bytes=SIZE, expected_sha256=SHA,
                  full_payload_verified=False, inference_executed=False, quality_scored=False,
                  model_fit_qualified=False, app_eligibility_changed=False)
    body = staging / 'model.safetensors'
    authenticated = None
    try:
        header = layout.read(fixture / 'header.json', layout.MAX_HEADER)
        config = layout.read(fixture / 'config.json', 8192)
        layout.inspect(header, config)
        inventory = layout.document(layout.read(api, 2 * 1024**2))
        require(inventory['id'] == 'google/gemma-4-E2B-it' and inventory['sha'] == '70af34e20bd4b7a91f0de6b22675850c43922a03', 'official repository/revision')
        rows = [row for row in inventory['siblings'] if row['rfilename'] == 'model.safetensors']
        require(len(rows) == 1 and rows[0]['size'] == SIZE and rows[0]['lfs']['sha256'] == SHA, 'advertised full identity')
        env, _ = transport.package.build_environment()
        def command(header_fd, body_fd, seconds):
            return ['curl', '--disable', '--http1.1', '--fail', '--location', '--max-redirs', '8',
                    '--silent', '--show-error', '--connect-timeout', '30', '--speed-time', '60',
                    '--speed-limit', '65536', '--max-time', str(seconds), '--max-filesize', str(SIZE),
                    '--proto', '=https', '--proto-redir', '=https', '--header', 'Accept-Encoding: identity',
                    '--dump-header', f'/dev/fd/{header_fd}', '--write-out', '%{http_code}',
                    '--output', f'/dev/fd/{body_fd}',
                    'https://huggingface.co/google/gemma-4-E2B-it/resolve/70af34e20bd4b7a91f0de6b22675850c43922a03/model.safetensors']
        with body.open('x+b') as file:
            try:
                result['command'] = run_owned(command, file, logs, SIZE, deadline, stop, env)
                owned = result['command']
                require(owned['primary_error'] is None and not owned['cleanup_errors'], 'owned transfer failed')
                full_response(layout.read(logs / 'download.headers', DIAGNOSTIC_CAP), layout.read(logs / 'download.stdout', DIAGNOSTIC_CAP), SIZE)
                info, prefix, authenticated = hash_owned(file, body, SIZE, 8 + len(header), deadline, stop)
                require(info == dict(bytes=SIZE, sha256=SHA), 'full checkpoint SHA-256 differs')
                require(prefix == struct.pack('<Q', len(header)) + header, 'full checkpoint header differs')
                result['identity'] = info
                regular_endpoint(file, body, SIZE)
            finally:
                try:
                    value = os.fstat(file.fileno())
                    result['retained_descriptor'] = dict(bytes=value.st_size, device=value.st_dev,
                                                        inode=value.st_ino, regular=stat.S_ISREG(value.st_mode))
                except BaseException as error:
                    result['endpoint_errors'].append('retained descriptor: ' + str(error))
        publish_authenticated(staging, DESTINATION, authenticated, staged_directory, config, pins, deadline, stop, result)
        result.update(passed=True, full_payload_verified=True)
    except BaseException as error:
        result['primary_error'] = type(error).__name__ + ': ' + str(error)
    finally:
        for sig in previous:
            signal.signal(sig, signal.SIG_IGN)
        for key in ('headers', 'stdout', 'stderr', 'child.json'):
            path = logs / ('download.' + key)
            if os.path.lexists(path):
                try:
                    raw = layout.read(path, DIAGNOSTIC_CAP)
                    result['captures'].append(dict(path=str(path.relative_to(ROOT)), bytes=len(raw), sha256=layout.sha(raw)))
                except BaseException as error:
                    result['endpoint_errors'].append(key + ': ' + str(error))
        path = DESTINATION / 'model.safetensors' if result['published'] else body
        try:
            value = path.lstat()
            result['retained_body'] = dict(path=str(path.relative_to(ROOT)), bytes=value.st_size,
                                           descriptor_identity=[value.st_dev, value.st_ino], regular=stat.S_ISREG(value.st_mode))
        except OSError as error:
            result['endpoint_errors'].append('partial body: ' + str(error))
        for name, pin in pins.items():
            try:
                require(layout.sha(layout.read(ROOT / name, 4 * 1024**2)) == pin, 'source endpoint: ' + name)
            except BaseException as error:
                result['endpoint_errors'].append(str(error))
        if result['endpoint_errors']:
            result['passed'] = False
        try:
            write_receipt(output, result)
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)
    print(json.dumps({key: result.get(key) for key in ['passed', 'primary_error', 'endpoint_errors', 'published', 'full_payload_verified', 'identity', 'published_receipt_error']}))
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT / 'artifacts/m8-gemma4-unquantized-checkpoint-v1')
    sys.exit(main(parser.parse_args().output.resolve()))

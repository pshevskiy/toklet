"""Coarse observer adapter around the unchanged qualified literal controller.

The root guest observer starts before the container. The attach Popen returns
to the original stdin transport only after an observer acknowledgement, keeping
candidate payload delivery after identity binding. Missing observer evidence
never becomes a CPU-budget verdict. Original isolation and cleanup stay owned
by linux_task_controller. No generated code executes in this host process.
"""
from pathlib import Path
import base64
import hashlib
import json
import os
import selectors
import shlex
import signal
import subprocess
import time
from unittest.mock import patch
from urllib.parse import urlsplit
import linux_task_controller as controller
import linux_code_isolation_probe as boundary
import podman_machine_lifecycle as lifecycle

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT/'tools/bench/linux_process_cpu_guest.py'


class Observer:
    def __init__(self, podman, identifier, image, program_sha):
        prefix = boundary.podman_prefix(podman)
        uri = prefix[prefix.index('--url')+1]
        api_socket = urlsplit(uri).path
        source = SOURCE.read_text()
        self.source_sha = hashlib.sha256(source.encode()).hexdigest()
        command = [prefix[0], 'machine', 'ssh', lifecycle.MACHINE,
                   shlex.join(['sudo', '-n', '/usr/bin/python3', '-I', '-S', '-u', '-c',
                               source, api_socket, identifier, image.removeprefix('sha256:'), program_sha])]
        self.command = command
        self.process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE, start_new_session=True, close_fds=True)
        self.selector = selectors.DefaultSelector()
        self.channels = dict(stdout=bytearray(), stderr=bytearray())
        self.pending = bytearray()
        self.events = []
        self.error = None
        for stream, name in [(self.process.stdout, 'stdout'), (self.process.stderr, 'stderr')]:
            os.set_blocking(stream.fileno(), False)
            self.selector.register(stream, selectors.EVENT_READ, name)

    def pump(self, seconds):
        for key, _ in self.selector.select(seconds):
            block = os.read(key.fileobj.fileno(), 4096)
            if not block:
                self.selector.unregister(key.fileobj)
                key.fileobj.close()
                continue
            self.channels[key.data].extend(block)
            if sum(map(len, self.channels.values())) > 65536:
                raise ValueError('Observer output exceeded bound')
            if key.data == 'stdout':
                self.pending.extend(block)
                while b'\n' in self.pending:
                    line, _, rest = self.pending.partition(b'\n')
                    self.pending = bytearray(rest)
                    event = json.loads(line)
                    if type(event) is not dict or event.get('event') not in ['watching', 'attached', 'result', 'startup_failure']:
                        raise ValueError('Unexpected observer frame')
                    self.events.append(event)

    def wait_event(self, event, seconds):
        deadline = time.monotonic()+seconds
        while True:
            found = [row for row in self.events if row['event'] == event]
            if found:
                if len(found) != 1:
                    raise ValueError('Repeated observer frame')
                return found[0]
            if time.monotonic() >= deadline:
                raise TimeoutError('Observer acknowledgement deadline: '+event)
            if self.process.poll() is not None and not self.selector.get_map():
                raise ValueError('Observer exited before '+event)
            self.pump(.01)

    def finish(self):
        errors = []
        deadline = time.monotonic()+18
        try:
            while self.process.poll() is None or self.selector.get_map():
                if time.monotonic() >= deadline:
                    raise TimeoutError('Observer completion deadline')
                self.pump(.01)
            if self.pending:
                raise ValueError('Incomplete observer frame')
        except BaseException as error:
            self.error = repr(error)
        finally:
            if self.process.poll() is None:
                try:
                    os.killpg(self.process.pid, signal.SIGTERM)
                    self.process.wait(timeout=5)
                except BaseException as error:
                    errors.append(repr(error))
                    if self.process.poll() is None:
                        os.killpg(self.process.pid, signal.SIGKILL)
                        self.process.wait(timeout=5)
            self.selector.close()
            for stream in [self.process.stdout, self.process.stderr]:
                if not stream.closed:
                    stream.close()
        results = [event['record'] for event in self.events if event['event'] == 'result']
        members = boundary.live_group_members(self.process.pid)
        record = dict(format='toklet-linux-process-cpu-capture-v1', command=self.command, host_pid=self.process.pid,
                    observer_source_sha256=self.source_sha, exit_code=self.process.returncode,
                    failure=self.error, cleanup_errors=errors, surviving_group_members=members,
                    events=self.events, result=results[0] if len(results) == 1 else None,
                    stdout_base64=base64.b64encode(self.channels['stdout']).decode(),
                    stderr_base64=base64.b64encode(self.channels['stderr']).decode())
        for name, body in self.channels.items():
            record[name+'_captured_bytes'] = len(body)
            record[name+'_captured_sha256'] = hashlib.sha256(body).hexdigest()
        return record


def execute(podman, image, source, payload):
    """Collect an authored literal evaluation plus independently owned witness.

    Qualification must approve this wrapper before it is used for model tasks.
    The unchanged controller still validates image/configuration, sends the
    authenticated payload and owns all container termination/removal.
    """
    original_capture = controller.transport.capture_input
    captures = []

    def observed_capture(command, data):
        identifier = command[-1]
        prefix = boundary.podman_prefix(podman)
        if command != [*prefix, 'start', '--interactive', '--attach', identifier]:
            raise ValueError('Unexpected observed container attach command')
        observer = Observer(podman, identifier, image, payload['program_sha256'])
        before_input = False
        gate_failure = None
        try:
            observer.wait_event('watching', 5)
            original_spawn = subprocess.Popen

            def spawn(*args, **kwargs):
                nonlocal before_input, gate_failure
                process = original_spawn(*args, **kwargs)
                if list(args[0]) == command:
                    try:
                        observer.wait_event('attached', 2)
                        before_input = True
                    except BaseException as error:
                        # Returning the owned process keeps the original
                        # transport's cleanup intact. Such a missing gate is
                        # ineligible for a CPU witness, regardless of outcome.
                        gate_failure = repr(error)
                return process

            with patch.object(controller.transport.subprocess, 'Popen', spawn):
                return original_capture(command, data)
        finally:
            record = observer.finish()
            record.update(attached_before_input=before_input, gate_failure=gate_failure)
            captures.append(record)

    with patch.object(controller.transport, 'capture_input', observed_capture):
        result = controller.execute(podman, image, source, payload)
    result['process_cpu_captures'] = captures
    return result

"""Trusted guest-side process CPU witness; generated code never enters here.

Run outside the candidate container as guest root. A full container lease is
observed through the existing rootless API, then bound to a seized evaluator
before its stdin payload is released. TRACEEXIT samples Linux process PROF
and SCHED clocks; only PROF measures the frozen user+system budget. No EXITKILL,
memory/register writes, syscall interception or policy changes are used.
"""
import ctypes
import hashlib
import http.client
import json
import os
from pathlib import Path
import platform
import re
import resource
import signal
import socket
import sys
import time

SEIZE, CONT, LISTEN, GETEVENTMSG = 0x4206, 7, 0x4208, 0x4201
TRACEEXIT, EVENT_EXIT, EVENT_STOP = 0x40, 6, 128
WALL = 0x40000000


def require(value, message):
    if not value:
        raise ValueError(message)


def stat_identity(raw):
    end = raw.rfind(')')
    require(end > 0, 'Process stat comm missing')
    words = raw[end+2:].split()
    require(len(words) >= 20, 'Process stat truncated')
    return dict(start_ticks=int(words[19]), ppid=int(words[1]))


def cpu_nanoseconds(seconds, nanoseconds):
    require(type(seconds) is int and 0 <= seconds <= 60
            and type(nanoseconds) is int and 0 <= nanoseconds < 10**9, 'Invalid process CPU clock')
    return seconds*10**9+nanoseconds


def process_clock_id(pid, kind):
    require(type(pid) is int and 1 < pid < 2**28 and type(kind) is int and kind in [0, 2],
            'Bounded Linux process PROF/SCHED clock required')
    return ((~pid) << 3) | kind


class UnixHTTP(http.client.HTTPConnection):
    def __init__(self, path):
        super().__init__('localhost', timeout=.5)
        self.path = path

    def connect(self):
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.settimeout(self.timeout)
        self.sock.connect(self.path)


def inspect_container(api_socket, identifier, image):
    connection = UnixHTTP(api_socket)
    try:
        connection.request('GET', '/v5.7.1/libpod/containers/'+identifier+'/json')
        response = connection.getresponse()
        raw = response.read(262145)
        require(response.status == 200 and len(raw) <= 262144, 'Bounded owned container inspection required')
        value = json.loads(raw)
    finally:
        connection.close()
    require(value['Id'].removeprefix('sha256:') == identifier
            and value['Image'].removeprefix('sha256:') == image
            and value['HostConfig']['PidsLimit'] == 1, 'Container lease/image/one-PID policy differs')
    return value


def process_identity(pid):
    base = Path('/proc')/str(pid)
    fields = dict(line.split(':', 1) for line in (base/'status').read_text().splitlines() if ':' in line)
    result = stat_identity((base/'stat').read_text())
    result.update(pid=pid, tgid=int(fields['Tgid']), threads=int(fields['Threads']),
                  tracer_pid=int(fields['TracerPid']), nspid=[int(v) for v in fields['NSpid'].split()],
                  namespace=os.readlink(base/'ns/pid'), cgroup=(base/'cgroup').read_text().strip())
    require(result['tgid'] == pid and result['threads'] == 1 and result['nspid'][0] == pid
            and result['nspid'][-1] == 1 and len(result['nspid']) >= 2, 'Single evaluator thread-group identity differs')
    require(result['namespace'] != os.readlink('/proc/self/ns/pid'), 'Observer must remain outside candidate PID namespace')
    require(re.search(r'^Max cpu time\s+4\s+4\s+seconds\s*$', (base/'limits').read_text(), re.MULTILINE),
            'Actual evaluator CPU limits differ')
    return result


def stable_identity(before, after):
    keys = ['pid', 'tgid', 'threads', 'start_ticks', 'namespace', 'cgroup', 'nspid']
    require(all(before[k] == after[k] for k in keys), 'Evaluator identity changed')


def cgroup_policy(identity, identifier):
    group = identity['cgroup']
    require(group.startswith('0::/') and '\n' not in group and identifier in group, 'Evaluator cgroup lease differs')
    path = group.removeprefix('0::')
    require('..' not in Path(path).parts, 'Invalid cgroup path')
    base = Path('/sys/fs/cgroup')/path.lstrip('/')
    require((base/'pids.max').read_text().strip() == '1'
            and (base/'pids.current').read_text().strip() == '1', 'Actual one-PID cgroup policy differs')


class Native:
    class Timespec(ctypes.Structure):
        _fields_ = [('seconds', ctypes.c_long), ('nanoseconds', ctypes.c_long)]

    def __init__(self):
        require(sys.platform == 'linux' and platform.machine() == 'aarch64'
                and ctypes.sizeof(ctypes.c_long) == 8 and ctypes.sizeof(ctypes.c_void_p) == 8,
                'Qualified Linux AArch64 LP64 observer ABI required')
        self.library = ctypes.CDLL(None, use_errno=True)
        self.library.ptrace.argtypes = [ctypes.c_uint, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p]
        self.library.ptrace.restype = ctypes.c_long
        self.library.clock_getcpuclockid.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
        self.library.clock_getcpuclockid.restype = ctypes.c_int
        self.library.clock_gettime.argtypes = [ctypes.c_int, ctypes.POINTER(self.Timespec)]
        self.library.clock_gettime.restype = ctypes.c_int

    def trace(self, request, pid, data=0):
        result = self.library.ptrace(request, pid, None, ctypes.c_void_p(data))
        if result == -1:
            error = ctypes.get_errno()
            raise OSError(error, os.strerror(error))
        return result

    def clock(self, pid):
        clock = ctypes.c_int()
        error = self.library.clock_getcpuclockid(pid, ctypes.byref(clock))
        if error:
            raise OSError(error, os.strerror(error))
        require(clock.value == process_clock_id(pid, 2), 'glibc process-clock encoding differs')
        result = dict(decision_clock='linux_process_prof')
        for name, kind in [('prof', 0), ('sched', 2)]:
            identifier = process_clock_id(pid, kind)
            value = self.Timespec()
            if self.library.clock_gettime(identifier, ctypes.byref(value)):
                error = ctypes.get_errno()
                raise OSError(error, os.strerror(error))
            result[name] = dict(clock_id=identifier, seconds=value.seconds, nanoseconds=value.nanoseconds,
                process_cpu_ns=cpu_nanoseconds(value.seconds, value.nanoseconds))
        return result


def emit(value):
    print(json.dumps(value, allow_nan=False), flush=True)


def run(api_socket, identifier, image, program_sha):
    require(re.fullmatch(r'/run/user/[0-9]+/podman/podman.sock', api_socket)
            and all(re.fullmatch('[0-9a-f]{64}', v) for v in [identifier, image, program_sha]),
            'Bounded fixed observer lease required')
    require(os.geteuid() == 0, 'Trusted observer must run as guest root')
    native = Native()
    executable = Path(sys.executable).resolve()
    record = dict(format='toklet-linux-process-cpu-witness-v2', completed=False, container_id=identifier,
        image=image, program_sha256=program_sha, observer_pid=os.getpid(),
        observer_namespace=os.readlink('/proc/self/ns/pid'),
        boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(), kernel=platform.release(),
        python_sha256=hashlib.sha256(executable.read_bytes()).hexdigest(), stops=[])
    cmdline = Path('/proc/self/cmdline').read_bytes().split(b'\0')
    position = cmdline.index(b'-c')
    record['observer_source_sha256'] = hashlib.sha256(cmdline[position+1]).hexdigest()
    deadline = time.monotonic()+15
    try:
        inspect_container(api_socket, identifier, image)
        emit(dict(event='watching', container_id=identifier, observer_pid=os.getpid()))
        while True:
            require(time.monotonic() < deadline, 'Observer start deadline exceeded')
            container = inspect_container(api_socket, identifier, image)
            pid = container['State']['Pid']
            if container['State']['Running'] and type(pid) is int and pid > 1:
                break
            time.sleep(.01)
        before = process_identity(pid)
        cgroup_policy(before, identifier)
        require(before['tracer_pid'] == 0, 'Evaluator already traced')
        native.trace(SEIZE, pid, TRACEEXIT)
        after = process_identity(pid)
        stable_identity(before, after)
        cgroup_policy(after, identifier)
        require(after['tracer_pid'] == os.getpid(), 'Observer tracer identity differs')
        record.update(identity_before=before, identity_attached=after)
        emit(dict(event='attached', container_id=identifier, identity=after))
        while True:
            require(time.monotonic() < deadline, 'Observer exit deadline exceeded')
            found, status = os.waitpid(pid, os.WNOHANG | WALL)
            if found == 0:
                time.sleep(.005)
                continue
            require(found == pid, 'Unexpected waited process')
            if os.WIFSTOPPED(status):
                event, stop_signal = status >> 16, os.WSTOPSIG(status)
                require(len(record['stops']) < 16, 'Observer stop inventory exceeded')
                record['stops'].append(dict(event=event, signal=stop_signal))
                if event == EVENT_EXIT:
                    require('exit_clock' not in record, 'Repeated process exit stop')
                    exiting = process_identity(pid)
                    stable_identity(before, exiting)
                    require(exiting['tracer_pid'] == os.getpid(), 'Exit-stop tracer differs')
                    record['identity_exit_stop'] = exiting
                    record['exit_clock'] = native.clock(pid)
                    message = ctypes.c_ulong()
                    native.trace(GETEVENTMSG, pid, ctypes.addressof(message))
                    record['exit_event_status'] = message.value
                    native.trace(CONT, pid)
                elif event == EVENT_STOP and stop_signal in [signal.SIGSTOP, signal.SIGTSTP, signal.SIGTTIN, signal.SIGTTOU]:
                    native.trace(LISTEN, pid)
                elif event == 0:
                    native.trace(CONT, pid, stop_signal)
                else:
                    raise ValueError('Unsupported ptrace stop')
            elif os.WIFEXITED(status) or os.WIFSIGNALED(status):
                require('exit_clock' in record, 'Final status without process CPU exit stop')
                record.update(completed=True, final_status=status,
                    final_signal=os.WTERMSIG(status) if os.WIFSIGNALED(status) else None,
                    final_exit_code=os.WEXITSTATUS(status) if os.WIFEXITED(status) else None)
                break
            else:
                raise ValueError('Unsupported wait status')
    except BaseException as error:
        record['failure'] = dict(type=type(error).__name__, message=str(error)[:2048])
    record['observer_peak_rss_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    emit(dict(event='result', record=record))
    # Exiting the tracer releases any remaining attachment. EXITKILL is absent;
    # the separate original container controller retains cleanup responsibility.
    return record['completed']


if __name__ == '__main__':
    try:
        success = run(*sys.argv[1:])
    except BaseException as error:
        emit(dict(event='startup_failure', type=type(error).__name__, message=str(error)[:2048]))
        success = False
    raise SystemExit(0 if success else 1)

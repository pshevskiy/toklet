"""Validate independent process-CPU evidence; legacy evaluator stays unchanged."""
import base64
import hashlib
import json
import os
import re
import linux_process_cpu_guest as guest
import linux_task_evidence as legacy


def validate(capture, *, container_id, image, program_sha, observer_source_sha,
             observer_python_sha, boot_id, kernel, observer_namespace):
    require = guest.require
    require(capture['format'] == 'toklet-linux-process-cpu-capture-v1'
            and capture['exit_code'] == 0 and capture['failure'] is None
            and capture['cleanup_errors'] == [] and capture['surviving_group_members'] == []
            and capture['attached_before_input'] is True and capture['gate_failure'] is None,
            'Observer admission/exit/ownership is unresolved')
    channels = {}
    for name in ['stdout', 'stderr']:
        raw = base64.b64decode(capture[name+'_base64'], validate=True)
        require(len(raw) == capture[name+'_captured_bytes']
                and hashlib.sha256(raw).hexdigest() == capture[name+'_captured_sha256'], 'Observer raw capture differs')
        channels[name] = raw
    require(sum(map(len, channels.values())) <= 65536 and not channels['stderr']
            and channels['stdout'].endswith(b'\n'), 'Observer output framing differs')
    events = [json.loads(line, object_pairs_hook=legacy.unique_object) for line in channels['stdout'].splitlines()]
    require(events == capture['events'] and [event['event'] for event in events] == ['watching', 'attached', 'result'],
            'Complete unique observer frames required')
    value = events[-1]['record']
    require(value == capture['result'] and value['format'] == 'toklet-linux-process-cpu-witness-v2'
            and value['completed'] is True and not value.get('failure')
            and value['container_id'] == container_id and value['image'] == image.removeprefix('sha256:')
            and value['program_sha256'] == program_sha and value['boot_id'] == boot_id
            and value['kernel'] == kernel and value['observer_namespace'] == observer_namespace
            and value['observer_source_sha256'] == capture['observer_source_sha256'] == observer_source_sha
            and value['python_sha256'] == observer_python_sha, 'Observer runtime/lease identity differs')
    require(all(event['container_id'] == container_id for event in events[:2]), 'Observer handshake lease differs')
    before = value['identity_before']
    attached = value['identity_attached']
    exiting = value['identity_exit_stop']
    guest.stable_identity(before, attached)
    guest.stable_identity(before, exiting)
    require(events[1]['identity'] == attached and events[0]['observer_pid'] == value['observer_pid']
            and type(value['observer_pid']) is int and value['observer_pid'] > 1,
            'Observer handshake identity differs')
    pid = before['pid']
    require(type(pid) is int and 1 < pid < 2**28 and before['tgid'] == pid and before['threads'] == 1
            and before['nspid'][0] == pid and before['nspid'][-1] == 1 and len(before['nspid']) >= 2
            and before['namespace'] != observer_namespace and before['tracer_pid'] == 0
            and attached['tracer_pid'] == exiting['tracer_pid'] == value['observer_pid']
            and container_id in before['cgroup'] and before['cgroup'].startswith('0::/'),
            'Independent evaluator process binding differs')
    require(re.fullmatch(r'pid:\[[0-9]+\]', before['namespace']) and type(before['start_ticks']) is int
            and before['start_ticks'] > 0, 'Evaluator start/namespace identity missing')
    clock = value['exit_clock']
    require(clock['decision_clock'] == 'linux_process_prof', 'User+system enforcement clock required')
    for name, kind in [('prof', 0), ('sched', 2)]:
        part = clock[name]
        require(type(part['clock_id']) is int and part['clock_id'] == guest.process_clock_id(pid, kind),
                'Linux process clock belongs to another process or thread')
        require(guest.cpu_nanoseconds(part['seconds'], part['nanoseconds']) == part['process_cpu_ns'],
                'Process CPU units differ')
    nanos = clock['prof']['process_cpu_ns']
    status = value['final_status']
    require(type(status) is int and 0 <= status <= 65535 and status == value['exit_event_status'],
            'Final wait status contradicts exit-stop evidence')
    require(os.WIFEXITED(status) or os.WIFSIGNALED(status), 'Final process wait outcome missing')
    require(value['final_signal'] == (os.WTERMSIG(status) if os.WIFSIGNALED(status) else None)
            and value['final_exit_code'] == (os.WEXITSTATUS(status) if os.WIFEXITED(status) else None),
            'Final wait status fields differ')
    require(type(value['stops']) is list and 1 <= len(value['stops']) <= 16, 'Bounded observer stops required')
    for stop in value['stops']:
        require(set(stop) == {'event', 'signal'} and type(stop['event']) is int and type(stop['signal']) is int
                and ((stop['event'] == 0 and 1 <= stop['signal'] <= 64)
                     or (stop['event'] == guest.EVENT_STOP and stop['signal'] in [19, 20, 21, 22])
                     or (stop['event'] == guest.EVENT_EXIT and stop['signal'] == 5)),
                'Unsupported observer stop inventory')
    exit_stops = [stop for stop in value['stops'] if stop['event'] == guest.EVENT_EXIT]
    require(exit_stops == [dict(event=guest.EVENT_EXIT, signal=5)], 'Unique kernel exit stop required')
    return dict(process_cpu_ns=nanos, budget_exhausted=nanos >= 4_000_000_000 and value['final_signal'] == 9,
                sched_cpu_ns=clock['sched']['process_cpu_ns'], container_id=container_id, pid=pid,
                scope='Linux process user+system PROF at exit stop; no signal-cause attribution')

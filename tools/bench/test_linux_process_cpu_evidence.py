import base64
import copy
import hashlib
import json
import unittest
import linux_process_cpu_evidence as subject


EXPECTED = dict(container_id='a'*64, image='b'*64, program_sha='c'*64, observer_source_sha='d'*64,
                observer_python_sha='e'*64, boot_id='boot-identity', kernel='qualified-kernel', observer_namespace='pid:[1]')


def witness(prof=4_000_000_000, sched=3_900_000_000, status=9):
    before = dict(pid=42, tgid=42, threads=1, start_ticks=100, namespace='pid:[2]',
                  cgroup='0::/libpod-'+'a'*64+'.scope', nspid=[42, 1], tracer_pid=0)
    attached = dict(before, tracer_pid=77)
    def clock(value, kind):
        return dict(clock_id=subject.guest.process_clock_id(42, kind), seconds=value//10**9,
                    nanoseconds=value % 10**9, process_cpu_ns=value)
    return dict(format='toklet-linux-process-cpu-witness-v2', completed=True,
        container_id=EXPECTED['container_id'], image=EXPECTED['image'], program_sha256=EXPECTED['program_sha'],
        observer_source_sha256=EXPECTED['observer_source_sha'], python_sha256=EXPECTED['observer_python_sha'],
        boot_id=EXPECTED['boot_id'], kernel=EXPECTED['kernel'], observer_namespace=EXPECTED['observer_namespace'],
        observer_pid=77, identity_before=before, identity_attached=attached, identity_exit_stop=copy.deepcopy(attached),
        exit_clock=dict(decision_clock='linux_process_prof', prof=clock(prof, 0), sched=clock(sched, 2)),
        stops=[dict(event=6, signal=5)], final_status=status, exit_event_status=status,
        final_signal=9 if status == 9 else None, final_exit_code=None if status == 9 else status >> 8)


def capture(value):
    events = [dict(event='watching', container_id=value['container_id'], observer_pid=value['observer_pid']),
              dict(event='attached', container_id=value['container_id'], identity=value['identity_attached']),
              dict(event='result', record=value)]
    channels = dict(stdout=b''.join(json.dumps(event).encode()+b'\n' for event in events), stderr=b'')
    result = dict(format='toklet-linux-process-cpu-capture-v1', exit_code=0, failure=None, cleanup_errors=[],
        surviving_group_members=[], attached_before_input=True, gate_failure=None, events=events, result=value,
        observer_source_sha256=value['observer_source_sha256'])
    for name, raw in channels.items():
        result[name+'_base64'] = base64.b64encode(raw).decode()
        result[name+'_captured_bytes'] = len(raw)
        result[name+'_captured_sha256'] = hashlib.sha256(raw).hexdigest()
    return result


class EvidenceControls(unittest.TestCase):
    def test_only_prof_counter_decides_unchanged_boundary(self):
        self.assertTrue(subject.validate(capture(witness()), **EXPECTED)['budget_exhausted'])
        self.assertFalse(subject.validate(capture(witness(prof=3_999_999_999, sched=4_100_000_000)), **EXPECTED)['budget_exhausted'])
        self.assertFalse(subject.validate(capture(witness(status=137 << 8)), **EXPECTED)['budget_exhausted'])

    def test_coherent_wrong_identity_clock_or_exit_stop_rejects(self):
        for change in ['pid', 'boot', 'program', 'source', 'namespace', 'clock', 'counter', 'units', 'status', 'missing', 'unknown']:
            value = witness()
            if change == 'pid':
                value['identity_exit_stop']['pid'] = 43
            elif change == 'boot':
                value['boot_id'] = 'other'
            elif change == 'program':
                value['program_sha256'] = 'f'*64
            elif change == 'source':
                value['observer_source_sha256'] = 'f'*64
            elif change == 'namespace':
                value['identity_exit_stop']['namespace'] = 'pid:[3]'
            elif change == 'clock':
                value['exit_clock']['prof']['clock_id'] |= 4
            elif change == 'counter':
                value['exit_clock']['decision_clock'] = 'linux_process_sched'
            elif change == 'units':
                value['exit_clock']['prof']['nanoseconds'] = 10**9
            elif change == 'status':
                value['exit_event_status'] = 0
            elif change == 'missing':
                value['stops'] = []
            else:
                value['stops'].insert(0, dict(event=99, signal=5))
            with self.subTest(change=change), self.assertRaises(ValueError):
                subject.validate(capture(value), **EXPECTED)

    def test_capture_mutation_missing_gate_and_owner_failure_rejects(self):
        for field, bad in [('stdout_base64', ''), ('attached_before_input', False), ('gate_failure', 'timeout'),
                           ('cleanup_errors', ['live observer']), ('exit_code', 1), ('surviving_group_members', [123])]:
            value = capture(witness())
            value[field] = bad
            with self.subTest(field=field), self.assertRaises(ValueError):
                subject.validate(value, **EXPECTED)

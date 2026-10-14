"""Window checks. All process and VM interactions are mocked."""
import json
from contextlib import ExitStack
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import task_grading_window as window


def capture(value, exit_code=0):
    return {'exit_code': exit_code, 'wall_timed_out': False,
            'output_limit_exceeded': False, 'cleanup_errors': [],
            'surviving_group_members': [], 'authored_output': json.dumps(value)}


class ContainerInventory(unittest.TestCase):
    def observe(self, items):
        record = {'commands': []}
        with patch.object(window.boundary, 'capture', side_effect=items) as calls, \
                patch.object(window.boundary, 'output', side_effect=lambda item: item['authored_output']):
            result = window.running_containers(('fixed-api',), record)
        return result, record, calls

    def test_api_inventory_is_used_when_healthy(self):
        result, record, calls = self.observe([capture([])])
        self.assertEqual(result, [])
        self.assertEqual(len(record['commands']), 1)
        calls.assert_called_once_with(['fixed-api', 'ps', '--format=json'], 20, 262144)

    def test_failed_api_uses_fixed_machine_ssh_fallback(self):
        result, record, calls = self.observe([capture(None, 1), capture([{'Id': 'foreign'}])])
        self.assertEqual(result, [{'Id': 'foreign'}])
        self.assertEqual(len(record['commands']), 2)
        self.assertEqual(calls.call_args_list[1].args[0],
                         [window.PODMAN, 'machine', 'ssh', window.lifecycle.MACHINE,
                          'podman', 'ps', '--format=json'])

    def test_unknown_inventory_is_never_assumed_empty(self):
        with self.assertRaisesRegex(ValueError, 'machine retained'):
            self.observe([capture(None, 1), capture(None, 1)])

    def test_healthy_malformed_inventory_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'not a list'):
            self.observe([capture({'unexpected': 'object'})])

    def test_surviving_control_group_cannot_admit_an_inventory(self):
        bad = capture([])
        bad['surviving_group_members'] = [{'pid': 123}]
        _, _, calls = self.observe([bad, capture([])])
        self.assertEqual(calls.call_count, 2)

    def test_unhealthy_fallback_cannot_admit_an_empty_inventory(self):
        mutations = ({'failure': {'type': 'CaptureError'}}, {'wall_timed_out': True},
                     {'output_limit_exceeded': True}, {'cleanup_errors': [{'type': 'ReapError'}]})
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                bad = capture([])
                bad.update(mutation)
                with self.assertRaisesRegex(ValueError, 'machine retained'):
                    self.observe([capture(None, 1), bad])


class WindowRestoration(unittest.TestCase):
    def invoke(self, foreign=False, initially_running=False, quality_completed=True, replaced=False, late_live=False):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            (root / 'benchmarks/raw').mkdir(parents=True)
            results = root / 'benchmarks/results'
            results.mkdir()
            producer = root / 'tools/bench/held_out_task_window.py'
            producer.parent.mkdir(parents=True)
            producer.write_text('# authored source\n')
            idle = root / 'idle.json'
            idle.write_text(json.dumps({'format': 'toklet-held-out-task-natural-exit-idle-v1',
                'all_owned_processes_absent': True, 'exited_owned_pids': [101, 102, 103],
                'collection_sha256': 'authored-sha', 'end_verified': {}}))
            (results / '2026-10-11-interrupted-workload-idle-v1.json').write_text('{}')
            (results / '2026-10-11-literal-native-v3-probe.json').write_text(json.dumps({
                'completed': True, 'runs': [{'expectation_passed': True} for _ in range(28)],
                'source_sha256': {}}))
            image = {'Id': window.IMAGE, 'Digest': None, 'RepoDigests': None,
                     'Architecture': 'arm64', 'Os': 'linux', 'RootFS': None}
            binding = {'URI': 'fixed-api', 'Identity': 'fixed-identity'}
            prior = results / 'runtime.json'
            prior.write_text(json.dumps({'completed': True, 'runs': [{} for _ in range(14)],
                'admitted_runtime': {}, 'admitted_python_version': 'fixed-python',
                'connection_binding': binding, 'image': image}))
            args = SimpleNamespace(prefix='2026-10-14-backend-task-restoration-authored',
                                   idle_proof=idle, collection=root / 'collection.json')
            owner = Mock()
            commands = []
            observations = 0

            def control(argv, seconds, limit):
                nonlocal observations
                commands.append(argv)
                self.assertEqual(limit, 262144)
                if argv == [window.PODMAN, 'machine', 'list', '--format=json']:
                    observations += 1
                    running = initially_running if observations == 1 else observations == 2
                    return capture([{'Name': window.lifecycle.MACHINE,
                                     'Running': running, 'Starting': False}])
                if argv == [window.PODMAN, 'machine', 'inspect', window.lifecycle.MACHINE]:
                    return capture([{}])
                if argv == [window.PODMAN, 'system', 'connection', 'list', '--format=json']:
                    return capture([])
                if argv[-2:] == ['info', '--format=json']:
                    return capture({})
                if argv[-3:] == ['image', 'inspect', 'sha256:' + window.IMAGE]:
                    return capture([image])
                if argv[-2:] == ['ps', '--format=json']:
                    return capture([{'Id': 'unexplained-container'}] if foreign else [])
                if argv == [window.PODMAN, 'machine', 'stop', window.lifecycle.MACHINE]:
                    return capture({})
                self.fail('Unexpected control command: ' + repr(argv))

            with ExitStack() as stack:
                for obj, name, value in ((window, 'ROOT', root), (window, '__file__', str(producer)),
                                         (window, 'PRIOR', prior)):
                    stack.enter_context(patch.object(obj, name, value))
                stack.enter_context(patch.object(window.boundary, 'digest', return_value='authored-sha'))
                stack.enter_context(patch.object(window.boundary, 'idle', return_value={'held': True}))
                stack.enter_context(patch.object(window.boundary, 'live_group_members', return_value=[]))
                stack.enter_context(patch.object(window.subprocess, 'run',
                    return_value=SimpleNamespace(returncode=1, stdout='', stderr='')))
                stack.enter_context(patch.object(window.boundary, 'check_record'))
                stack.enter_context(patch.object(window.boundary, 'bound_machine_connection', return_value=binding))
                stack.enter_context(patch.object(window.boundary, 'capture', side_effect=control))
                stack.enter_context(patch.object(window.boundary, 'output',
                    side_effect=lambda item: item['authored_output']))
                stack.enter_context(patch.object(window.admission, 'validate_collection',
                    return_value={'identity':{'backend':'metal','model':'larger'},'collection_sha256':'authored-sha','raw_sha256':'authored-sha'}))
                stack.enter_context(patch.object(window, 'admit_idle', return_value={'proof_sha256':'authored-sha','collections':[]}))
                idle_checks = iter((None, ValueError('Task group became live')) if late_live else (None,)*8)
                def idle_check(_):
                    value=next(idle_checks)
                    if value:raise value
                stack.enter_context(patch.object(window, 'check_idle',side_effect=idle_check))
                stack.enter_context(patch.object(window.probe, 'check_collection',side_effect=ValueError('Admitted task collection changed') if replaced else None))
                stack.enter_context(patch.object(window, 'pinned_record',side_effect=lambda p,sha:json.loads(Path(p).read_text())))
                grade = stack.enter_context(patch.object(window.probe, 'run', return_value={
                    'completed': True, 'quality_completed': quality_completed,
                    'unresolved_case_ids': [] if quality_completed else ['unresolved-case']}))
                start = stack.enter_context(patch.object(window.lifecycle, 'start_machine',
                    return_value=({'client_succeeded': True}, owner)))
                await_owner = stack.enter_context(patch.object(window.lifecycle, 'await_stopped_owner'))
                result = window.run(args)
                record = json.loads((root / 'benchmarks/raw' / (args.prefix + '-window.json')).read_text())
            return result, record, commands, owner, start, await_owner, grade

    def test_initially_running_machine_is_never_started_or_stopped(self):
        result, record, commands, owner, start, await_owner, grade = self.invoke(initially_running=True)
        self.assertEqual(result, 1)
        self.assertFalse(record['started_by_this_task'])
        self.assertIn('initially stopped', record['failure']['message'])
        start.assert_not_called()
        grade.assert_not_called()
        owner.finish.assert_not_called()
        self.assertIsNone(await_owner.call_args.args[0])
        self.assertNotIn([window.PODMAN, 'machine', 'stop', window.lifecycle.MACHINE], commands)

    def test_unexplained_live_container_retains_owned_machine(self):
        result, record, commands, owner, _, await_owner, _ = self.invoke(foreign=True)
        self.assertEqual(result, 1)
        self.assertFalse(record['restored_stopped'])
        self.assertIn('ownership needs resolution', record['restoration_failure']['message'])
        owner.finish.assert_not_called()
        self.assertIs(await_owner.call_args.args[0], owner)
        self.assertNotIn([window.PODMAN, 'machine', 'stop', window.lifecycle.MACHINE], commands)

    def test_empty_inventory_allows_stop_and_pipe_owner_finish(self):
        result, record, commands, owner, start, await_owner, grade = self.invoke()
        self.assertEqual(result, 0)
        self.assertTrue(record['restored_stopped'])
        self.assertTrue(record['pipe_lifetime_finished'])
        self.assertEqual(commands.count([window.PODMAN, 'machine', 'stop', window.lifecycle.MACHINE]), 1)
        start.assert_called_once_with(window.PODMAN)
        grade.assert_called_once()
        owner.finish.assert_called_once_with([{'Name': window.lifecycle.MACHINE,
                                               'Running': False, 'Starting': False}])
        self.assertIs(await_owner.call_args.args[0], owner)

    def test_completed_attempt_can_retain_unresolved_quality_without_a_score(self):
        result, record, _, _, _, _, _ = self.invoke(quality_completed=False)
        self.assertEqual(result, 0)
        self.assertTrue(record['completed'])
        self.assertFalse(record['quality_completed'])
        self.assertEqual(record['unresolved_case_ids'], ['unresolved-case'])
        self.assertNotIn('aggregate', record)


    def test_replacement_after_admission_prevents_vm_start(self):
        result,record,commands,owner,start,_,grade=self.invoke(replaced=True)
        self.assertEqual(result,1);self.assertIn('collection changed',record['failure']['message'])
        start.assert_not_called();grade.assert_not_called();owner.finish.assert_not_called()
        self.assertEqual(commands,[])

    def test_fresh_idle_check_immediately_precedes_vm_start(self):
        result,record,commands,owner,start,_,grade=self.invoke(late_live=True)
        self.assertEqual(result,1);self.assertIn('became live',record['failure']['message'])
        self.assertTrue(any('inspect' in c for c in commands))
        start.assert_not_called();grade.assert_not_called();owner.finish.assert_not_called()
        self.assertFalse(record['started_by_this_task'])

class MatchedIdle(unittest.TestCase):
    def fixture(self, directory):
        root=Path(directory);paths=[root/f'{i}.json' for i in range(3)]
        proof=root/'idle.json';proof.write_text(json.dumps({'format':'toklet-backend-task-idle-input-v1','collections':list(map(str,paths))}))
        accepted=[{'identity':{'backend':b,'model':m},'worker_pid':100+i,'collection_sha256':'saved','raw_sha256':'raw'} for i,(b,m) in enumerate((('metal','small'),('metal','larger'),('cpu','larger')))]
        return proof,paths,accepted

    def test_all_three_roles_and_distinct_workers_required(self):
        with tempfile.TemporaryDirectory() as directory:
            proof,paths,accepted=self.fixture(directory)
            with patch.object(window.admission,'validate_collection',side_effect=accepted),patch.object(window,'check_idle') as check:
                result=window.admit_idle(proof,paths[0]);self.assertEqual(len(result['collections']),3);check.assert_called_once_with(result)
            accepted[2]['identity']={'backend':'metal','model':'larger'}
            with patch.object(window.admission,'validate_collection',side_effect=accepted),patch.object(window,'check_idle') as check:
                with self.assertRaisesRegex(ValueError,'Matched full'):window.admit_idle(proof,paths[0])
                check.assert_not_called()
            accepted[2]['identity']={'backend':'cpu','model':'larger'};accepted[2]['worker_pid']=100
            with patch.object(window.admission,'validate_collection',side_effect=accepted),patch.object(window,'check_idle'):
                with self.assertRaisesRegex(ValueError,'distinct owned'):window.admit_idle(proof,paths[0])

    def test_incomplete_collection_rejects_before_process_or_vm_operation(self):
        with tempfile.TemporaryDirectory() as directory:
            proof,paths,_=self.fixture(directory)
            with patch.object(window.admission,'validate_collection',side_effect=ValueError('Incomplete collection')),patch.object(window,'check_idle') as check,patch.object(window.lifecycle,'start_machine') as start:
                with self.assertRaisesRegex(ValueError,'Incomplete'):window.admit_idle(proof,paths[0])
                check.assert_not_called();start.assert_not_called()
            proof.write_text(json.dumps({'format':'toklet-backend-task-idle-input-v1','collections':[str(paths[0])]*3}))
            with patch.object(window.admission,'validate_collection') as validate:
                with self.assertRaisesRegex(ValueError,'Distinct'):window.admit_idle(proof,paths[0])
                validate.assert_not_called()

    def binding(self):
        return {'proof_path':'proof','proof_sha256':'same','protected_binaries':{'cpu':'same'},'collections':[{'worker_pid':100+i,'path':str(i)+'.json','sha256':'same','raw_sha256':'same'} for i in range(3)]}

    def test_live_pid_or_descendant_rejects_and_checks_repeat(self):
        binding=self.binding()
        with patch.object(window.boundary,'digest',return_value='same'),patch.object(window.subprocess,'run',return_value=SimpleNamespace(returncode=1,stdout='',stderr='')) as query,patch.object(window.boundary,'live_group_members',return_value=[]):
            window.check_idle(binding);window.check_idle(binding);self.assertEqual(query.call_count,2)
        for process,group in ((SimpleNamespace(returncode=0,stdout='100',stderr=''),[]),(SimpleNamespace(returncode=1,stdout='',stderr=''),[101])):
            with patch.object(window.boundary,'digest',return_value='same'),patch.object(window.subprocess,'run',return_value=process),patch.object(window.boundary,'live_group_members',return_value=group):
                with self.assertRaisesRegex(ValueError,'remains live'):window.check_idle(binding)

    def test_proof_collection_raw_or_protected_identity_change_rejects(self):
        binding=self.binding()
        for changed in ('proof','0.json','0.raw.jsonl',str(window.ROOT/'cpu')):
            with patch.object(window.boundary,'digest',side_effect=lambda p:'changed' if str(p)==changed else 'same'),patch.object(window.subprocess,'run',return_value=SimpleNamespace(returncode=1,stdout='',stderr='')),patch.object(window.boundary,'live_group_members',return_value=[]):
                with self.subTest(changed=changed),self.assertRaises(ValueError):window.check_idle(binding)

    def test_receipts_are_pinned_before_json_parse(self):
        with patch.object(window.admission,'bounded',return_value=b'{}'),patch.object(window.admission.evidence,'decode_event',side_effect=AssertionError('must not parse')):
            with self.assertRaisesRegex(ValueError,'receipt differs'):window.pinned_record('receipt','bad')

if __name__ == '__main__':
    unittest.main()

"""Window checks. All process and VM interactions are mocked."""
import json
from contextlib import ExitStack
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import held_out_task_window as window


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


class WindowAdmission(unittest.TestCase):
    def invoke(self, proof, process_query=None, groups=None, digest='collection-sha', provenance=False):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            (root / 'benchmarks/raw').mkdir(parents=True)
            results = root / 'benchmarks/results'
            results.mkdir()
            (results / '2026-10-11-interrupted-workload-idle-v1.json').write_text('{}')
            (results / '2026-10-11-literal-native-v3-probe.json').write_text(json.dumps({
                'completed': True, 'runs': [{'expectation_passed': True} for _ in range(28)],
                'source_sha256': {}}))
            producer = root / 'tools/bench/held_out_task_window.py'
            producer.parent.mkdir(parents=True)
            producer.write_text('# authored source\n')
            idle = root / 'idle.json'
            idle.write_text(json.dumps(proof))
            args = SimpleNamespace(prefix='2026-10-11-held-out-task-authored',
                                   idle_proof=idle, collection=root / 'collection.json')
            with patch.object(window, 'ROOT', root), \
                    patch.object(window, '__file__', str(producer)), \
                    patch.object(window.boundary, 'digest', return_value=digest), \
                    patch.object(window.subprocess, 'run', return_value=process_query) as query, \
                    patch.object(window.boundary, 'live_group_members', return_value=groups or []) as group_query, \
                    patch.object(window.boundary, 'idle', return_value={'held': True}), \
                    patch.object(window.probe, 'validate_collection',
                                 side_effect=ValueError('collection provenance rejected')) as validate, \
                    patch.object(window.lifecycle, 'start_machine') as start, \
                    patch.object(window.lifecycle, 'await_stopped_owner') as restore, \
                    patch.object(window.boundary, 'capture') as control, \
                    patch.object(window.probe, 'run') as grade:
                result = window.run(args)
                record = json.loads((root / 'benchmarks/raw' / (args.prefix + '-window.json')).read_text())
                start.assert_not_called()
                control.assert_not_called()
                grade.assert_not_called()
                restore.assert_called_once()
                self.assertIsNone(restore.call_args.args[0])
                if provenance:
                    validate.assert_called_once_with(args.collection)
                else:
                    validate.assert_not_called()
            return result, record, query, group_query

    def proof(self):
        return {'format': 'toklet-held-out-task-natural-exit-idle-v1',
                'all_owned_processes_absent': True, 'exited_owned_pids': [101, 102, 103],
                'collection_sha256': 'collection-sha', 'end_verified': {}}

    def test_three_distinct_typed_owned_pids_required_before_start(self):
        for pids in ([101, 102], [101, 102, 102], [True, 102, 103], [1, 102, 103]):
            with self.subTest(pids=pids):
                proof = self.proof()
                proof['exited_owned_pids'] = pids
                result, record, query, _ = self.invoke(proof)
                self.assertEqual(result, 1)
                self.assertIn('idle identity differs', record['failure']['message'])
                query.assert_not_called()
                self.assertFalse(record['started_by_this_task'])

    def test_collection_hash_mismatch_rejected_before_process_checks(self):
        proof = self.proof()
        proof['collection_sha256'] = 'different'
        result, record, query, _ = self.invoke(proof)
        self.assertEqual(result, 1)
        self.assertIn('idle receipt differs', record['failure']['message'])
        query.assert_not_called()

    def test_live_collection_pid_rejected_before_any_vm_operation(self):
        result, record, query, groups = self.invoke(self.proof(),
            process_query=SimpleNamespace(returncode=0, stdout='101\n', stderr=''))
        self.assertEqual(result, 1)
        self.assertIn('processes are not idle', record['failure']['message'])
        query.assert_called_once_with(['ps', '-p', '101,102,103', '-o', 'pid='],
                                      capture_output=True, text=True)
        groups.assert_not_called()

    def test_live_collection_descendant_rejected_before_any_vm_operation(self):
        result, record, _, _ = self.invoke(self.proof(),
            process_query=SimpleNamespace(returncode=1, stdout='', stderr=''),
            groups=[{'pid': 104}])
        self.assertEqual(result, 1)
        self.assertIn('process group is not idle', record['failure']['message'])

    def test_collection_provenance_rejected_before_any_vm_operation(self):
        result, record, _, groups = self.invoke(self.proof(),
            process_query=SimpleNamespace(returncode=1, stdout='', stderr=''), provenance=True)
        self.assertEqual(result, 1)
        self.assertEqual(record['failure']['message'], 'collection provenance rejected')
        self.assertEqual(groups.call_count, 3)
        self.assertFalse(record['started_by_this_task'])


class WindowRestoration(unittest.TestCase):
    def invoke(self, foreign=False, initially_running=False, quality_completed=True):
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
            args = SimpleNamespace(prefix='2026-10-11-held-out-task-restoration-authored',
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
                stack.enter_context(patch.object(window.probe, 'validate_collection',
                    return_value=({'projection': 'scalar'}, [])))
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


if __name__ == '__main__':
    unittest.main()

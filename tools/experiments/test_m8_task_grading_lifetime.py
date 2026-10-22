"""Cooperative grading-owner lifetime regressions; all VM/process work mocked."""
from contextlib import ExitStack
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import task_grading_window as window
from test_task_grading_window import capture


class Lifetime(unittest.TestCase):
    def invoke(self, foreign=False, initially_running=False, quality_completed=True, replaced=False, late_live=False, custom_hooks=False, admission_failure=False, remote_checker=None, existing_output=None, slow_restoration=False):
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
            if existing_output is not None:
                (root/'benchmarks/raw'/(args.prefix+existing_output)).symlink_to(root/'missing')
            owner = Mock()
            if slow_restoration:
                owner.record = {'helper_pipe_lifetime_completed':False}
                owner.authored_pauses = []
                remaining = iter([False, False, True])
                def finish(_):
                    if not next(remaining):raise ValueError('Authored startup group still draining')
                    owner.record['helper_pipe_lifetime_completed'] = True
                owner.finish.side_effect = finish
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
                grade = stack.enter_context(patch.object(window.probe, 'run', side_effect=TimeoutError('Cooperative grading interruption') if slow_restoration else None, return_value={
                    'completed': True, 'quality_completed': quality_completed,
                    'unresolved_case_ids': [] if quality_completed else ['unresolved-case']}))
                start = stack.enter_context(patch.object(window.lifecycle, 'start_machine',
                    return_value=({'client_succeeded': True}, owner)))
                real_await = window.lifecycle.await_stopped_owner
                await_owner = stack.enter_context(patch.object(window.lifecycle, 'await_stopped_owner', side_effect=(lambda owner,observe:real_await(owner,observe,owner.authored_pauses.append)) if slow_restoration else None))
                if custom_hooks:
                    validator=Mock(side_effect=ValueError('V2 admission failed') if admission_failure else None,
                                   return_value={'identity':{'backend':'metal','model':'larger','packed_format_version':2},'collection_sha256':'authored-sha','raw_sha256':'authored-sha'})
                    injected=SimpleNamespace(validate_collection=validator)
                    result=window.run(args,collection_admission=injected,
                                      idle_admission=lambda *args: {'proof_sha256':'authored-sha','collections':[]},
                                      idle_check=remote_checker,
                                      extra_source_paths=(producer,),extra_grading_modules=(injected,))
                else:
                    result = window.run(args,idle_check=remote_checker)
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

    def test_cooperative_interrupt_retains_owner_through_slow_restoration(self):
        # The caller has no finite parent/group-kill deadline. A simulated
        # 30-second drain exceeds generic command cleanup's 15-second grace.
        with patch.object(window.lifecycle, 'signal_owned_group') as signal_group:
            result,record,commands,owner,start,await_owner,grade=self.invoke(slow_restoration=True)
        self.assertEqual(result,1)
        self.assertIn('Cooperative grading interruption',record['failure']['message'])
        self.assertIn('still draining',record['restoration_failure']['message'])
        self.assertEqual(owner.authored_pauses,[30]);self.assertEqual(owner.finish.call_count,3)
        self.assertTrue(owner.record['helper_pipe_lifetime_completed'])
        start.assert_called_once();grade.assert_called_once();await_owner.assert_called_once()
        signal_group.assert_not_called()
        self.assertIn([window.PODMAN,'machine','stop',window.lifecycle.MACHINE],commands)



if __name__=='__main__':unittest.main()

"""Portable admission/restoration controls; never start a VM."""
from contextlib import ExitStack
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import requalify_grading_host as subject


class RequalificationTests(unittest.TestCase):
    def fixture(self, *, probe_failure=False, literal_complete=True, running=False,
                memory='2147483648', stop_failure=False, start_failure=False):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name).resolve()
        (root / 'benchmarks/raw').mkdir(parents=True)
        (root / 'artifacts').mkdir()
        output = root / 'artifacts/qualification'
        sha = '0' * 64
        runtime = {'fixture': True}
        binding = {'URI': 'ssh://fixture', 'Identity': '/fixture/identity'}
        prior = dict(source_sha256={'tools/bench/code_isolation_probe.py': sha,
                                   'tools/bench/task_cases_probe.py': sha,
                                   'tools/bench/linux_code_isolation_child.py': sha},
                     podman_sha256=sha, connection_binding=binding, podman_version='{}',
                     image={'Id': 'fixture'}, admitted_runtime=runtime,
                     admitted_python_version='fixture')
        state = {'running': running}
        owner = Mock()
        owner.record = {'helper_pipe_lifetime_completed': True}

        def capture(argv, *_):
            if argv[1:3] == ['machine', 'list']:
                body = [dict(Name=subject.lifecycle.MACHINE, Running=state['running'],
                             Starting=False, Memory=memory)]
            elif argv[1:3] == ['machine', 'inspect']:
                body = [{}]
            elif argv[1:3] == ['system', 'connection']:
                body = []
            elif argv[1:3] == ['machine', 'stop']:
                if stop_failure:
                    raise ValueError('fixture stop failure')
                state['running'] = False
                body = {}
            elif 'version' in argv:
                body = {}
            elif 'image' in argv:
                body = [{'Id': 'fixture'}]
            else:
                body = {}
            return dict(exit_code=0, wall_timed_out=False, output_limit_exceeded=False,
                        cleanup_errors=[], surviving_group_members=[], text=json.dumps(body))

        def start(_):
            if start_failure:
                raise OSError('fixture startup failure')
            state['running'] = True
            return dict(client_succeeded=True), owner

        def execute(*args):
            if probe_failure:
                raise ValueError('fixture probe failure')
            return dict(container_cleanup_errors=[], container_removed=True)

        def literal(_api, _image, _runtime, _python, path):
            result = dict(completed=literal_complete,
                          runs=[dict(expectation_passed=True) for _ in range(28)])
            path.write_text(json.dumps(result))
            return result

        stack = ExitStack()
        self.addCleanup(stack.close)
        for target, name, value in (
            (subject, 'ROOT', root), (subject, 'prior_records', lambda: [prior, prior]),
            (subject, 'running_containers', lambda *_: []),
            (subject.boundary, 'digest', lambda _: sha),
            (subject.boundary, 'capture', capture),
            (subject.boundary, 'output', lambda item: item['text']),
            (subject.boundary, 'bound_machine_connection', lambda *_: binding),
            (subject.boundary, 'execute', execute),
            (subject.boundary, 'check_record', lambda _: dict(runtime=runtime, python='fixture')),
            (subject.lifecycle, 'start_machine', start),
            (subject.lifecycle, 'await_stopped_owner', Mock()),
            (subject.literal, 'run', literal),
            (subject.signal, 'signal', lambda *_: subject.signal.SIG_DFL),
        ):
            stack.enter_context(patch.object(target, name, value))
        return output, state, owner

    def test_complete_original_scope_restores_before_publication(self):
        output, state, owner = self.fixture()
        result = subject.run(output)
        self.assertTrue(result['completed'])
        self.assertEqual(len(result['runs']), 14)
        self.assertFalse(state['running'])
        self.assertTrue(result['restored_stopped'])
        self.assertTrue(result['pipe_lifetime_finished'])
        owner.finish.assert_called_once()
        self.assertEqual(json.loads((output / 'result.json').read_bytes()), result)
        self.assertIn('tools/bench/process_groups.py', result['source_sha256'])

    def test_probe_failure_stops_without_qualification(self):
        output, state, owner = self.fixture(probe_failure=True)
        result = subject.run(output)
        self.assertFalse(result['completed'])
        self.assertFalse(state['running'])
        self.assertTrue(result['restored_stopped'])
        owner.finish.assert_called_once()

    def test_incomplete_literal_scope_cannot_qualify(self):
        output, state, _ = self.fixture(literal_complete=False)
        result = subject.run(output)
        self.assertFalse(result['completed'])
        self.assertFalse(state['running'])

    def test_running_vm_is_never_owned_or_stopped(self):
        output, state, owner = self.fixture(running=True)
        result = subject.run(output)
        self.assertFalse(result['completed'])
        self.assertFalse(result['started_by_this_task'])
        self.assertTrue(state['running'])
        owner.finish.assert_not_called()

    def test_different_guest_reservation_rejected_before_start(self):
        output, state, owner = self.fixture(memory='4294967296')
        result = subject.run(output)
        self.assertFalse(result['completed'])
        self.assertFalse(result['started_by_this_task'])
        self.assertFalse(state['running'])
        owner.finish.assert_not_called()

    def test_failed_stop_never_publishes_success(self):
        output, state, owner = self.fixture(stop_failure=True)
        result = subject.run(output)
        self.assertFalse(result['completed'])
        self.assertIn('restoration_failure', result)
        self.assertTrue(state['running'])
        owner.finish.assert_not_called()

    def test_start_exception_still_checks_restoration(self):
        output, state, _ = self.fixture(start_failure=True)
        result = subject.run(output)
        self.assertFalse(result['completed'])
        self.assertTrue(result['restored_stopped'])
        self.assertFalse(state['running'])

    def test_original_proofs_and_guest_policy_are_still_bound(self):
        first, second = subject.prior_records()
        self.assertEqual(len(first['runs']), 14)
        self.assertEqual(len(second['runs']), 28)
        with patch.object(subject.child, 'POLICY', {'cpu_seconds': 5}):
            with self.assertRaisesRegex(ValueError, 'Guest program or original policy changed'):
                subject.prior_records()


if __name__ == '__main__':
    unittest.main()

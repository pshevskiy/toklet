"""Offline controller fixtures with all native process operations mocked."""
import copy
import hashlib
import json
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import linux_task_controller as controller
from test_linux_code_isolation import configuration


class ControllerTests(unittest.TestCase):
    def fixture(self):
        value = configuration()
        value['Config'].update(OpenStdin=True, Tty=False,
                               Cmd=['-I', '-S', '-B', '-c', 'trusted fixture'])
        return value

    def payload(self):
        program = 'assert True'
        return {'case_id': 'authored/pass', 'program': program,
                'program_sha256': hashlib.sha256(program.encode()).hexdigest()}

    def test_only_fixed_trusted_bootstrap_enters_arguments(self):
        with patch.object(controller, 'bundled_source', return_value='trusted fixture'):
            command = controller.create_command('/fixed/podman', '1' * 64,
                                                'toklet-authored-' + '3' * 32, 'trusted fixture')
        self.assertIn('--interactive', command)
        self.assertEqual(command[-5:], ['-I', '-S', '-B', '-c', 'trusted fixture'])
        self.assertIn('--network=none', command)
        self.assertIn('--pids-limit=1', command)
        self.assertIn('--memory=268435456', command)

    def test_interactive_config_retains_qualified_security_fields(self):
        value = self.fixture()
        original = copy.deepcopy(value)
        result = controller.validate_configuration(value, 'trusted fixture')
        self.assertTrue(result['open_stdin'])
        self.assertFalse(result['tty'])
        self.assertEqual(value, original)
        changes = [('OpenStdin', False), ('Tty', True), ('Cmd', ['candidate source'])]
        for key, data in changes:
            value = self.fixture()
            value['Config'][key] = data
            with self.subTest(key=key), self.assertRaises(ValueError):
                controller.validate_configuration(value, 'trusted fixture')
        value = self.fixture()
        value['HostConfig']['NetworkMode'] = 'host'
        with self.assertRaises(ValueError):
            controller.validate_configuration(value, 'trusted fixture')

    def mock_control(self, record):
        return {'stdout_base64': '', 'stderr_captured_bytes': 0, 'fixture_output': record}

    def test_wait_preserves_oom_and_literal_exit137(self):
        b = controller.boundary
        for code, oom in [(0, False), (137, True), (137, False)]:
            waited = self.mock_control(str(code))
            stopped = {'State': {'Running': False, 'ExitCode': code, 'OOMKilled': oom, 'Error': ''}}
            record = {}
            with patch.object(b, 'successful', return_value=waited), \
                    patch.object(b, 'output', return_value=str(code)), \
                    patch.object(b, 'inspect_owned', return_value=stopped):
                self.assertEqual(controller.wait_owned_exit('/fixed/podman', '2' * 64,
                                 'owned', '1' * 64, record), stopped)
                self.assertEqual(record['container_exit_wait'], waited)

    def test_wait_rejects_uncompleted_or_inconsistent_status(self):
        b = controller.boundary
        for text, state in [('-1', {}), ('256', {}), ('137', {'Running': True}),
                            ('0', {'Running': False, 'ExitCode': 1, 'OOMKilled': False, 'Error': ''}),
                            ('0', {'Running': False, 'ExitCode': 0, 'OOMKilled': 0, 'Error': ''})]:
            with patch.object(b, 'successful', return_value=self.mock_control(text)), \
                    patch.object(b, 'output', return_value=text), \
                    patch.object(b, 'inspect_owned', return_value={'State': state}):
                with self.subTest(text=text, state=state), self.assertRaises(ValueError):
                    controller.wait_owned_exit('/fixed/podman', '2' * 64, 'owned', '1' * 64, {})

    def test_unknown_wait_can_retry_only_to_agreed_literal_status(self):
        waits = [self.mock_control('-1'), self.mock_control('137')]
        states = [{'State': {'Running': False, 'ExitCode': code, 'OOMKilled': False, 'Error': ''}}
                  for code in (-1, 137)]
        record = {}
        with patch.object(controller.boundary, 'successful', side_effect=waits), \
                patch.object(controller.boundary, 'output', side_effect=lambda r: r['fixture_output']), \
                patch.object(controller.boundary, 'inspect_owned', side_effect=states), \
                patch.object(controller.time, 'sleep') as pause:
            self.assertEqual(controller.wait_owned_exit('/fixed/podman', '2' * 64,
                             'owned', '1' * 64, record), states[-1])
        self.assertEqual(record['container_exit_waits'], waits)
        self.assertEqual([item['State']['ExitCode'] for item in record['container_exit_inspections']], [-1, 137])
        pause.assert_called_once_with(0.1)

    def test_persistent_unknown_wait_is_bounded_and_unresolved(self):
        state = {'State': {'Running': False, 'ExitCode': -1, 'OOMKilled': False, 'Error': ''}}
        record = {}
        with patch.object(controller.boundary, 'successful', return_value=self.mock_control('-1')) as control, \
                patch.object(controller.boundary, 'output', return_value='-1'), \
                patch.object(controller.boundary, 'inspect_owned', return_value=state), \
                patch.object(controller.time, 'sleep') as pause:
            with self.assertRaisesRegex(ValueError, 'remains unknown'):
                controller.wait_owned_exit('/fixed/podman', '2' * 64, 'owned', '1' * 64, record)
        self.assertEqual(control.call_count, 3)
        self.assertEqual(pause.call_count, 2)
        self.assertEqual(len(record['container_exit_waits']), 3)
        self.assertEqual(len(record['container_exit_inspections']), 3)

    def test_malformed_or_nonnegative_mismatch_does_not_retry(self):
        state = {'State': {'Running': False, 'ExitCode': 137, 'OOMKilled': False, 'Error': ''}}
        for text, stderr in [('0', 0), ('-2', 0), ('0.0', 0), ('-1', 1)]:
            waited = self.mock_control(text)
            waited['stderr_captured_bytes'] = stderr
            with self.subTest(text=text, stderr=stderr), \
                    patch.object(controller.boundary, 'successful', return_value=waited) as control, \
                    patch.object(controller.boundary, 'output', return_value=text), \
                    patch.object(controller.boundary, 'inspect_owned', return_value=state), \
                    patch.object(controller.time, 'sleep') as pause:
                with self.assertRaises(ValueError):
                    controller.wait_owned_exit('/fixed/podman', '2' * 64, 'owned', '1' * 64, {})
                self.assertEqual(control.call_count, 1)
                pause.assert_not_called()

    def test_unknown_wait_does_not_hide_failed_inspection_or_control(self):
        state = {'State': {'Running': False, 'ExitCode': -1, 'OOMKilled': False, 'Error': ''}}
        for failed_inspection in (True, False):
            record = {}
            controls = [self.mock_control('-1'), ValueError('authored control')]
            inspected = ValueError('authored ownership') if failed_inspection else state
            with self.subTest(failed_inspection=failed_inspection), \
                    patch.object(controller.boundary, 'successful', side_effect=controls), \
                    patch.object(controller.boundary, 'output', side_effect=lambda r: r['fixture_output']), \
                    patch.object(controller.boundary, 'inspect_owned', side_effect=inspected if failed_inspection else None,
                                 return_value=state), \
                    patch.object(controller.time, 'sleep'):
                with self.assertRaises(ValueError):
                    controller.wait_owned_exit('/fixed/podman', '2' * 64, 'owned', '1' * 64, record)
            self.assertEqual(len(record['container_exit_waits']), 1)

    def run_fixture(self, *, config_failure=False, transport_failure=False, remove_failure=False):
        b = controller.boundary
        identifier = '2' * 64
        before = self.fixture()
        stopped = {'State': {'Running': False, 'ExitCode': 137, 'OOMKilled': False, 'Error': ''}}
        captured = {'transport_exit_code': 137, 'wall_timed_out': False}
        control = Mock()
        control.side_effect = [self.mock_control(identifier), self.mock_control('137')]
        if remove_failure:
            control.side_effect = [self.mock_control(identifier), self.mock_control('137'), ValueError('authored removal')]
        else:
            control.side_effect = [self.mock_control(identifier), self.mock_control('137'), self.mock_control('')]
        # output() reads only the authored fixtures, never real process output.
        with patch.object(controller, 'bundled_source', return_value='trusted fixture'), \
                patch.object(b, 'exists', return_value=False) as exists, \
                patch.object(b, 'successful', control), \
                patch.object(b, 'output', side_effect=lambda r: r['fixture_output']), \
                patch.object(b, 'inspect_owned', side_effect=[before, stopped, stopped, stopped]) as inspect, \
                patch.object(controller, 'validate_configuration', side_effect=ValueError('authored config') if config_failure else None), \
                patch.object(controller.transport, 'capture_input', side_effect=ValueError('authored pipe') if transport_failure else None,
                             return_value=captured) as pipe:
            record = controller.execute('/fixed/podman', '1' * 64, 'trusted fixture', self.payload())
        return record, pipe, inspect, exists, control

    def test_success_uses_input_pipe_and_retains_raw_exit(self):
        record, pipe, inspect, exists, control = self.run_fixture()
        self.assertEqual(record['container_exit_code'], 137)
        self.assertEqual(record['transport_exit_code'], 137)
        self.assertTrue(record['container_removed'])
        self.assertEqual(record['container_cleanup_errors'], [])
        self.assertEqual(pipe.call_args.args[0][-4:], ['start', '--interactive', '--attach', '2' * 64])
        self.assertEqual(json.loads(pipe.call_args.args[1]), self.payload())
        self.assertEqual(control.call_args.args[0][-3:], ['rm', '--force', '2' * 64])

    def test_invalid_payload_creates_nothing(self):
        with patch.object(controller.boundary, 'exists') as exists, \
                patch.object(controller.boundary, 'successful') as spawned:
            with self.assertRaises(ValueError):
                controller.execute('/fixed/podman', '1' * 64, 'trusted fixture', {})
            exists.assert_not_called()
            spawned.assert_not_called()

    def test_altered_or_oversized_bootstrap_creates_nothing(self):
        for source, expected in [('altered source', 'trusted fixture'),
                                 ('x' * (controller.BOOTSTRAP_ARGUMENT_BYTES + 1),
                                  'x' * (controller.BOOTSTRAP_ARGUMENT_BYTES + 1)),
                                 ('trusted\x00fixture', 'trusted\x00fixture')]:
            with patch.object(controller, 'bundled_source', return_value=expected), \
                    patch.object(controller.boundary, 'exists') as exists, \
                    patch.object(controller.boundary, 'successful') as spawned:
                with self.subTest(size=len(source)), self.assertRaises(ValueError):
                    controller.execute('/fixed/podman', '1' * 64, source, self.payload())
                exists.assert_not_called()
                spawned.assert_not_called()

    def test_configuration_failure_never_starts_and_preserves_cleanup(self):
        record, pipe, _, _, _ = self.run_fixture(config_failure=True)
        pipe.assert_not_called()
        self.assertEqual(record['failure']['message'], 'authored config')
        self.assertTrue(record['container_removed'])

    def test_pipe_failure_retains_error_and_removes_verified_owner(self):
        record, _, _, _, _ = self.run_fixture(transport_failure=True)
        self.assertEqual(record['failure']['message'], 'authored pipe')
        self.assertTrue(record['container_removed'])

    def test_remove_failure_prevents_cleanup_admission(self):
        record, _, _, _, _ = self.run_fixture(remove_failure=True)
        self.assertNotIn('container_removed', record)
        self.assertEqual(record['container_cleanup_errors'][0]['message'], 'authored removal')


if __name__ == '__main__':
    unittest.main()

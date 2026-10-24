"""Fault injection for the reference owner's separate attempt/stream causes."""
from pathlib import Path
import tempfile
import unittest

import qwen3_chat_reference as capture


class OwnerChecks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.output = Path(self.temp.name)
        (self.output / 'logs').mkdir()
        self.commands, self.streams, self.capture_errors, self.endpoints = [], {}, [], []
        self.guards = 0

    def tearDown(self):
        self.temp.cleanup()

    def guard(self):
        self.guards += 1

    def invoke(self, runner, guard=None):
        capture.invoke_owned('probe', ['test-only'], self.output, {}, guard or self.guard,
            self.commands, self.streams, self.capture_errors, self.endpoints, runner=runner)

    def row(self, code=0, primary=None, cleanup=None):
        return dict(command=['test-only'], exit_code=code, primary_error=primary, cleanup_errors=cleanup or [])

    def test_helper_raise_after_logs_keeps_attempt_and_both_streams(self):
        def runner(command, stdout, stderr, environment, timeout):
            self.assertEqual(len(self.commands), 1)
            stdout.write_bytes(b'out')
            stderr.write_bytes(b'original failure detail')
            raise OSError('injected helper exception after logs')
        with self.assertRaisesRegex(RuntimeError, 'injected helper exception'):
            self.invoke(runner)
        self.assertEqual(len(self.commands), 1)
        self.assertIsNone(self.commands[0]['result'])
        self.assertIn('OSError', self.commands[0]['invocation_error'])
        self.assertEqual(len(self.streams), 2)
        self.assertEqual(self.capture_errors, [])
        self.assertEqual(self.guards, 2)

    def test_missing_stdout_keeps_stderr_and_unchanged_owner_record(self):
        original = self.row(1, 'original owned failure', ['original cleanup cause'])
        def runner(command, stdout, stderr, environment, timeout):
            stderr.write_bytes(b'retained stderr')
            return original
        with self.assertRaisesRegex(RuntimeError, 'original owned failure'):
            self.invoke(runner)
        self.assertIs(self.commands[0]['result'], original)
        self.assertEqual(original, self.row(1, 'original owned failure', ['original cleanup cause']))
        self.assertIn(str(self.output / 'logs/probe.stderr'), self.streams)
        self.assertEqual(len(self.capture_errors), 1)
        self.assertIn('/stdout:', self.capture_errors[0])
        self.assertEqual(self.guards, 2)

    def test_endpoint_failure_does_not_replace_helper_exception(self):
        def guard():
            self.guards += 1
            if self.guards == 2:
                raise ValueError('separate changed endpoint')
        def runner(command, stdout, stderr, environment, timeout):
            stdout.write_bytes(b'out')
            stderr.write_bytes(b'err')
            raise OSError('original helper exception')
        with self.assertRaisesRegex(RuntimeError, 'original helper exception'):
            self.invoke(runner, guard)
        self.assertEqual(len(self.streams), 2)
        self.assertEqual(len(self.endpoints), 1)
        self.assertIn('separate changed endpoint', self.endpoints[0])
        self.assertIn('original helper exception', self.commands[0]['invocation_error'])

    def test_success_keeps_exact_returned_owner_row(self):
        original = self.row()
        def runner(command, stdout, stderr, environment, timeout):
            stdout.write_bytes(b'ok')
            stderr.write_bytes(b'')
            return original
        self.invoke(runner)
        self.assertIs(self.commands[0]['result'], original)
        self.assertEqual(len(self.streams), 2)
        self.assertEqual(self.capture_errors, [])
        self.assertEqual(self.endpoints, [])


if __name__ == '__main__':
    unittest.main()

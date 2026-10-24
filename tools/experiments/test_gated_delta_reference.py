"""Orchestration regressions; mock children never perform inference."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import gated_delta_reference as reference


class OwnedReference(unittest.TestCase):
    def test_regular_bounded_raw_capture_preserves_exact_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            path = root / 'bytes'
            raw = b'\0\xffowned'
            path.write_bytes(raw)
            captured, pin = reference.capture(path, len(raw))
            self.assertEqual(captured, raw)
            self.assertEqual(pin, dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest()))
            with self.assertRaisesRegex(ValueError, 'bound'):
                reference.capture(path, len(raw) - 1)
            alias = root / 'alias'
            alias.symlink_to(path)
            for operation in [reference.pin, lambda value: reference.capture(value, 64)]:
                with self.assertRaisesRegex(ValueError, 'Regular'):
                    operation(alias)

    def test_existing_or_escaping_destination_rejects_before_child(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            artifacts = root / 'artifacts'
            artifacts.mkdir()
            destination = artifacts / 'existing'
            destination.mkdir()
            (destination / 'owner').write_bytes(b'preserve')
            with mock.patch.object(reference, 'ROOT', root), mock.patch.object(reference.owner, 'owned') as child:
                with self.assertRaises(FileExistsError):
                    reference.collect(root, '0' * 64, '1' * 64, root / 'source', destination)
                with self.assertRaisesRegex(ValueError, 'below repository artifacts'):
                    reference.collect(root, '0' * 64, '1' * 64, root / 'source', root / 'escape')
                child.assert_not_called()
            self.assertEqual((destination / 'owner').read_bytes(), b'preserve')

    def test_bad_source_retains_provisional_failure_without_starting_child(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            (root / 'artifacts').mkdir()
            source = root / 'source'
            source.write_bytes(b'wrong upstream source')
            destination = root / 'artifacts' / 'failure'
            with mock.patch.object(reference, 'ROOT', root), mock.patch.object(reference.upstream.fixture, 'verify'), mock.patch.object(reference.owner, 'owned') as child:
                self.assertEqual(reference.collect(root, '0' * 64, '1' * 64, source, destination), 1)
                child.assert_not_called()
            report = json.loads((destination / 'capture.json').read_bytes())
            self.assertFalse(report['passed'])
            self.assertTrue(report['provisional'])
            self.assertFalse(report['fixture_accepted'])
            self.assertIn('identity', report['primary_error'])
            self.assertEqual(report['commands'], [])

    def test_child_or_helper_failure_retains_logs_and_checks_inputs_at_endpoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            (root / 'artifacts').mkdir()
            fixture = root / 'fixture'
            fixture.mkdir()
            for name in ('manifest.json', 'arrays.bin'):
                (fixture / name).write_bytes(b'synthetic fixture input')
            source = root / 'source'
            source.write_bytes(b'synthetic source')
            destination = root / 'artifacts' / 'failure'
            calls = []
            def failed_child(command, stdout, stderr, environment, timeout):
                calls.append(command)
                self.assertEqual(list(Path(environment['PYTHONPYCACHEPREFIX']).iterdir()), [])
                stdout.write_bytes(b'partial stdout\n')
                stderr.write_bytes(b'import failed\n')
                # Simulate a changed guarded input after child start. No real
                # child or numerical package is imported by this regression.
                source.write_bytes(b'changed source')
                return dict(command=command, exit_code=1, primary_error='tool exit 1',
                    cleanup_errors=[], pid=123, stdout=str(stdout), stderr=str(stderr))
            with mock.patch.object(reference, 'ROOT', root), mock.patch.object(reference.upstream.fixture, 'verify'), mock.patch.object(reference.upstream, 'captured', return_value=b'synthetic source'), mock.patch.object(reference.upstream, 'runtime_paths', return_value=[]), mock.patch.object(reference.owner, 'owned', side_effect=failed_child):
                self.assertEqual(reference.collect(fixture, '0' * 64, '1' * 64, source, destination), 1)
                source.write_bytes(b'synthetic source')
                raised_destination = root / 'artifacts' / 'helper-raised'
                def raised_child(*args, **kwargs):
                    failed_child(*args, **kwargs)
                    raise OSError('Synthetic helper close failure')
                with mock.patch.object(reference.owner, 'owned', side_effect=raised_child):
                    self.assertEqual(reference.collect(fixture, '0' * 64, '1' * 64, source, raised_destination), 1)
            report = json.loads((destination / 'capture.json').read_bytes())
            self.assertEqual(len(calls), 2)
            self.assertFalse(report['passed'])
            self.assertEqual(report['commands'][0]['owned_result']['primary_error'], 'tool exit 1')
            self.assertTrue(report['endpoint_errors'])
            self.assertTrue(report['capture_errors'])
            self.assertEqual((destination / 'logs/cpu.stdout').read_bytes(), b'partial stdout\n')
            self.assertEqual((destination / 'logs/cpu.stderr').read_bytes(), b'import failed\n')
            self.assertEqual(set(report['outputs']), {'logs/cpu.stdout', 'logs/cpu.stderr'})
            raised = json.loads((raised_destination / 'capture.json').read_bytes())
            self.assertFalse(raised['passed'])
            self.assertIsNone(raised['commands'][0]['owned_result'])
            self.assertIn('Synthetic helper close failure', raised['commands'][0]['invocation_error'])
            self.assertTrue(raised['endpoint_errors'])
            self.assertEqual(set(raised['outputs']), {'logs/cpu.stdout', 'logs/cpu.stderr'})


if __name__ == '__main__':
    unittest.main()

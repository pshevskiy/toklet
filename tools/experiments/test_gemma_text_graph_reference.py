"""Owner regressions with mocked children; no numerical runtime imported."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import gemma_text_graph_reference as reference


class OwnedReference(unittest.TestCase):
    def test_destination_and_bad_source_fail_before_any_child(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            artifacts = root / 'artifacts'
            artifacts.mkdir()
            source = root / 'source'
            source.mkdir()
            for name in reference.operator.SOURCES:
                (source/name).write_bytes(b'wrong source')
            destination = artifacts / 'capture'
            with mock.patch.object(reference, 'ROOT', root), mock.patch.object(reference.owner, 'owned') as child:
                self.assertEqual(reference.collect(source, destination), 1)
                with self.assertRaises(FileExistsError):
                    reference.collect(source, destination)
                with self.assertRaisesRegex(ValueError, 'below repository artifacts'):
                    reference.collect(source, root / 'escape')
                child.assert_not_called()
            report = json.loads((destination / 'capture.json').read_bytes())
            self.assertFalse(report['passed'])
            self.assertTrue(report['provisional'])
            self.assertFalse(report['fixture_accepted'])
            self.assertIn('identity', report['primary_error'])
            self.assertEqual(report['commands'], [])

    def test_source_substitution_after_buffer_capture_rejects_before_child(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            (root / 'artifacts').mkdir()
            source = root / 'source'
            source.mkdir()
            for name in reference.operator.SOURCES:
                (source/name).write_bytes(b'authenticated buffer')
            def capture_then_replace(path, maximum, expected):
                raw = path.read_bytes()
                path.write_bytes(b'replacement before independent pin')
                return raw
            destination = root / 'artifacts' / 'replaced'
            with mock.patch.object(reference, 'ROOT', root), mock.patch.object(reference.runtime, 'captured', side_effect=capture_then_replace), mock.patch.object(reference.runtime, 'runtime_paths', return_value=[]), mock.patch.object(reference, 'pin', side_effect=lambda path: dict(bytes=path.stat().st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest()) if path.exists() else dict(bytes=0, sha256='0' * 64)), mock.patch.object(reference.owner, 'owned') as child:
                self.assertEqual(reference.collect(source, destination), 1)
                child.assert_not_called()
            report = json.loads((destination / 'capture.json').read_bytes())
            self.assertIn('source buffer changed', report['primary_error'])
            self.assertEqual(report['commands'], [])
            self.assertEqual((destination / 'sources/gemma4_text.py').read_bytes(), b'authenticated buffer')

    def test_failed_child_and_raised_helper_preserve_logs_and_endpoint_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            (root / 'artifacts').mkdir()
            for name in ('gemma_text_graph.py', 'test_gemma_text_graph.py', 'test_reference_runtime.py'):
                path = root / 'tools/reference' / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b'synthetic source')
            helper = root / 'tools/experiments/m7_apple_package.py'
            runner = root / 'tools/experiments/gemma_text_graph_reference.py'
            for path in (helper, runner, root / 'tools/experiments/test_gemma_text_graph_reference.py', root / 'target/release/toklet', root / 'target/release/examples/cpu_bench'):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b'synthetic source')
            source = root / 'source'
            source.mkdir()
            for name in reference.operator.SOURCES:
                (source/name).write_bytes(b'synthetic source')
            calls = []
            def failed(command, stdout, stderr, environment, timeout):
                calls.append(command)
                self.assertEqual(list(Path(environment['PYTHONPYCACHEPREFIX']).iterdir()), [])
                stdout.write_bytes(b'partial stdout\n')
                stderr.write_bytes(b'partial stderr\n')
                (source/'gemma4_text.py').write_bytes(b'changed input')
                return dict(exit_code=1, primary_error='synthetic child failure', cleanup_errors=[])
            def raised(*args, **kwargs):
                failed(*args, **kwargs)
                raise OSError('synthetic helper failure')
            with mock.patch.object(reference, 'ROOT', root), mock.patch.object(reference, '__file__', str(runner)), mock.patch.object(reference.owner, '__file__', str(helper)), mock.patch.object(reference.runtime, 'runtime_paths', side_effect=lambda selected: sorted(set(selected))), mock.patch.object(reference.operator, 'SOURCES', {name:hashlib.sha256(b'synthetic source').hexdigest() for name in reference.operator.SOURCES}):
                for name, operation in [('failed', failed), ('raised', raised)]:
                    for filename in reference.operator.SOURCES:
                        (source/filename).write_bytes(b'synthetic source')
                    destination = root / 'artifacts' / name
                    with mock.patch.object(reference.owner, 'owned', side_effect=operation):
                        self.assertEqual(reference.collect(source, destination), 1)
                    report = json.loads((destination / 'capture.json').read_bytes())
                    self.assertFalse(report['passed'])
                    self.assertTrue(report['endpoint_errors'])
                    self.assertEqual(set(report['outputs']), {'logs/ownership.stdout', 'logs/ownership.stderr'})
                    self.assertEqual((destination / 'logs/ownership.stdout').read_bytes(), b'partial stdout\n')
                    row = report['commands'][0]
                    if name == 'failed':
                        self.assertEqual(row['owned_result']['primary_error'], 'synthetic child failure')
                        self.assertIsNone(row['invocation_error'])
                    else:
                        self.assertIsNone(row['owned_result'])
                        self.assertIn('synthetic helper failure', row['invocation_error'])
            self.assertEqual(len(calls), 2)


if __name__ == '__main__':
    unittest.main()

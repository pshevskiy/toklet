"""Actual-reference extraction mutations and original-error retention checks."""
from pathlib import Path
import copy
import hashlib
import io
import json
import sys
import tempfile
import unittest
from unittest import mock
import gemma_head_reference as producer
import check_gemma_head_reference as checker


class HeadMutations(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root, cls.capture = map(Path, sys.argv[1:3])
        cls.baseline = checker.verify(cls.root, cls.capture)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        for name in ('observation.json', 'manifest.json', 'arrays.bin'):
            (self.directory / name).write_bytes(checker.read(self.capture / name, 8 * 1024**2))

    def reject(self):
        with self.assertRaises(ValueError):
            checker.verify(self.root, self.directory)

    def edit(self, change):
        path = self.directory / 'observation.json'
        report = checker.parse(path.read_bytes())
        change(report)
        path.write_text(json.dumps(report, allow_nan=False))
        self.manifest()

    def manifest(self):
        path = self.directory / 'manifest.json'
        value = checker.parse(path.read_bytes())
        value['files'] = {name: checker.identity(self.directory / name) for name in ('observation.json', 'arrays.bin')}
        path.write_text(json.dumps(value))

    def test_actual_complete_reference(self):
        self.assertIs(self.baseline['passed'], True)
        self.assertEqual(self.baseline['output_values'], 1048576)

    def test_wrong_contract(self):
        self.edit(lambda v: v.update(contract='math_fp64'))
        self.reject()

    def test_softcapped_source_name(self):
        self.edit(lambda v: v['arrays'][1].update(name='logits'))
        self.reject()

    def test_boolean_shape(self):
        self.edit(lambda v: v['arrays'][0].update(shape=[True, 1536]))
        self.reject()

    def test_wrong_original_offset(self):
        self.edit(lambda v: v['arrays'][1].update(source_offset=95014916))
        self.reject()

    def test_wrong_binary_offset(self):
        self.edit(lambda v: v['arrays'][1].update(offset=24580))
        self.reject()

    def test_coherent_wrong_complete_output(self):
        path = self.directory / 'arrays.bin'
        raw = bytearray(path.read_bytes())
        raw[24576] ^= 1
        path.write_bytes(raw)
        digest = hashlib.sha256(raw[24576:]).hexdigest()
        self.edit(lambda v: v['arrays'][1].update(sha256=digest))
        self.reject()

    def test_coherent_wrong_input(self):
        path = self.directory / 'arrays.bin'
        raw = bytearray(path.read_bytes())
        raw[0] ^= 1
        path.write_bytes(raw)
        digest = hashlib.sha256(raw[:24576]).hexdigest()
        self.edit(lambda v: v['arrays'][0].update(sha256=digest))
        self.reject()

    def test_truncated_binary(self):
        path = self.directory / 'arrays.bin'
        path.write_bytes(path.read_bytes()[:-4])
        self.manifest()
        self.reject()

    def test_trailing_binary(self):
        path = self.directory / 'arrays.bin'
        path.write_bytes(path.read_bytes() + bytes(4))
        self.manifest()
        self.reject()

    def test_missing_array(self):
        self.edit(lambda v: v['arrays'].pop())
        self.reject()

    def test_incomplete_attempt(self):
        self.edit(lambda v: v.update(array_attempt={'role': 'expected', 'written_bytes': 4}))
        self.reject()

    def test_extra_capability(self):
        self.edit(lambda v: v.update(app_qualified=True))
        self.reject()

    def test_boolean_completed(self):
        self.edit(lambda v: v['arrays'][0].update(completed=1))
        self.reject()

    def test_duplicate_json(self):
        with self.assertRaisesRegex(ValueError, 'duplicate field'):
            checker.parse(b'{"passed":true,"passed":true}')

    def test_nonfinite_json(self):
        with self.assertRaisesRegex(ValueError, 'finite JSON required'):
            checker.parse(b'{"number":NaN}')

    def test_bounded_read_closes(self):
        stream = io.BytesIO(b'12345')
        path = mock.Mock()
        path.open.return_value = stream
        with self.assertRaisesRegex(ValueError, 'bounded file'):
            checker.read(path, 4)
        self.assertTrue(stream.closed)


class FailureRetention(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(sys.argv[1])

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.output = Path(self.temporary.name) / 'capture'

    def test_late_descriptor_failure_retains_first_array(self):
        selected = copy.deepcopy(producer.SELECTED)
        selected[1]['source_offset'] += 4
        with mock.patch.object(producer, 'SELECTED', selected):
            with self.assertRaisesRegex(ValueError, 'selected source descriptor differs'):
                producer.capture(self.root, self.output)
        report = checker.parse((self.output / 'observation.json').read_bytes())
        self.assertIs(report['passed'], False)
        self.assertEqual(len(report['arrays']), 1)
        self.assertIs(report['arrays'][0]['completed'], True)
        self.assertEqual((self.output / 'arrays.bin').stat().st_size, 24576)
        self.assertIs(checker.parse((self.output / 'manifest.json').read_bytes())['passed'], False)

    def test_original_survives_both_publication_failures(self):
        original = OSError('original endpoint failure')
        authenticated = producer.authenticate(self.root)
        with mock.patch.object(producer, 'authenticate', side_effect=[authenticated, original]), mock.patch.object(producer, 'publish', side_effect=OSError('secondary publication failure')):
            with self.assertRaises(OSError) as raised:
                producer.capture(self.root, self.output)
        self.assertIs(raised.exception, original)
        self.assertEqual(len(original.__notes__), 2)
        self.assertEqual((self.output / 'arrays.bin').stat().st_size, 4218880)

    def test_report_failure_keeps_full_binary_and_failed_manifest(self):
        original = OSError('original report publication failure')
        publish = producer.publish
        def fail_report(path, value):
            if path.name == 'observation.json':
                raise original
            return publish(path, value)
        with mock.patch.object(producer, 'publish', side_effect=fail_report):
            with self.assertRaises(OSError) as raised:
                producer.capture(self.root, self.output)
        self.assertIs(raised.exception, original)
        manifest = checker.parse((self.output / 'manifest.json').read_bytes())
        self.assertIs(manifest['passed'], False)
        self.assertEqual(set(manifest['files']), {'arrays.bin'})
        self.assertEqual(manifest['files']['arrays.bin']['bytes'], 4218880)


if __name__ == '__main__':
    if len(sys.argv) != 3:
        raise SystemExit('repository root and actual complete head reference capture')
    unittest.main(argv=[sys.argv[0]], verbosity=2)

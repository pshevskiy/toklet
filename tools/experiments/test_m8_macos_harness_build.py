import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import m8_macos_harness_build as builder


class MacBuilderTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='mac-builder-check-', dir=builder.ROOT / 'artifacts')
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.native = self.root / 'native'
        archive = self.native / 'Toklet.xcframework/macos-arm64/libtoklet.a'
        archive.parent.mkdir(parents=True)
        archive.write_bytes(b'fixture archive, not executable')
        headers = archive.parent / 'Headers'
        headers.mkdir()
        for name in ['toklet.h', 'module.modulemap']:
            (headers / name).write_bytes((builder.ROOT / 'apple/include' / name).read_bytes())
        pins = {str(x.relative_to(builder.ROOT)): builder.package.sha(x) for x in [archive, *headers.iterdir()]}
        self.receipt = self.native / 'build.json'
        self.receipt.write_text(json.dumps(dict(passed=True, primary_error=None, endpoint_errors=[], package_pins=pins)))
        self.expected = builder.package.sha(self.receipt)
        self.out = self.root / 'output'

    def test_existing_output_and_symlink_are_preserved(self):
        for link in [False, True]:
            destination = self.root / ('link' if link else 'existing')
            if link:
                destination.symlink_to(self.root / 'absent')
            else:
                destination.write_bytes(b'keep')
            with self.assertRaises(FileExistsError):
                builder.main(self.native, self.expected, destination)
            self.assertTrue(os.path.lexists(destination))
            if not link:
                self.assertEqual(destination.read_bytes(), b'keep')
        self.assertFalse((self.root / 'absent').exists())

    def test_receipt_and_header_admission_precede_output(self):
        with self.assertRaises(ValueError):
            builder.main(self.native, '0' * 64, self.out)
        self.assertFalse(self.out.exists())
        header = self.native / 'Toklet.xcframework/macos-arm64/Headers/toklet.h'
        header.write_bytes(b'wrong ABI')
        with self.assertRaises(ValueError):
            builder.main(self.native, self.expected, self.out)
        self.assertFalse(self.out.exists())

    def test_fixture_and_environment_admission_precede_output(self):
        with patch.dict(builder.TINY, {'config.json': '0' * 64}):
            with self.assertRaises(ValueError):
                builder.main(self.native, self.expected, self.out)
        self.assertFalse(self.out.exists())
        with patch.object(builder.package, 'build_environment', side_effect=RuntimeError('setup fixture failure')):
            with self.assertRaises(RuntimeError):
                builder.main(self.native, self.expected, self.out)
        self.assertFalse(self.out.exists())

    def test_failed_command_and_missing_endpoint_keep_both_streams(self):
        owned = builder.package.owned

        def failed(command, stdout, stderr, env, timeout):
            record = owned([sys.executable, '-c', "import sys;print('fixture stdout');print('fixture stderr',file=sys.stderr);sys.exit(7)"], stdout, stderr, env, timeout)
            copy = next(Path(x) for x in command if x.endswith('/source-snapshot/apple/Harness/ExpertHarness.swift'))
            copy.unlink()  # Only this test's owned copied source disappears.
            return record

        with patch.object(builder.package, 'owned', side_effect=failed) as invoked:
            self.assertEqual(builder.main(self.native, self.expected, self.out), 1)
        self.assertEqual(invoked.call_count, 1)
        result = json.loads((self.out / 'build.json').read_text())
        self.assertFalse(result['passed'])
        self.assertEqual(result['commands'][0]['exit_code'], 7)
        self.assertEqual(len(result['diagnostic_pins']), 2)
        self.assertTrue(any('FileNotFoundError' in x for x in result['changed_endpoints']))
        for field, text in [('stdout', b'fixture stdout\n'), ('stderr', b'fixture stderr\n')]:
            name = result['commands'][0][field]
            self.assertEqual(result['diagnostic_pins'][name], hashlib.sha256(text).hexdigest())

    def test_missing_one_stream_does_not_hide_other_stream(self):
        owned = builder.package.owned

        def failed(command, stdout, stderr, env, timeout):
            result = owned([sys.executable, '-c', "import sys;print('retained stderr',file=sys.stderr);sys.exit(8)"], stdout, stderr, env, timeout)
            stdout.unlink()
            return result

        with patch.object(builder.package, 'owned', side_effect=failed):
            self.assertEqual(builder.main(self.native, self.expected, self.out), 1)
        result = json.loads((self.out / 'build.json').read_text())
        row = result['commands'][0]
        self.assertEqual(row['exit_code'], 8)
        self.assertTrue(row['capture_errors'])
        self.assertIn(row['stderr'], result['diagnostic_pins'])
        self.assertNotIn(row['stdout'], result['diagnostic_pins'])

    def test_consumed_native_receipt_digest_remains_authoritative(self):
        original = builder.package.sha
        replaced = False

        def replacing(path):
            nonlocal replaced
            if Path(path) == self.receipt and not replaced:
                replaced = True
                self.receipt.write_text('{"replacement":true}')
            return original(path)

        with patch.object(builder.package, 'sha', side_effect=replacing):
            self.assertEqual(builder.main(self.native, self.expected, self.out), 1)
        result = json.loads((self.out / 'build.json').read_text())
        self.assertEqual(result['source_pins'][str(self.receipt.relative_to(builder.ROOT))], self.expected)
        self.assertTrue(any('build.json: hash differs' in x for x in result['changed_endpoints']))
        self.assertFalse(result['commands'])


if __name__ == '__main__':
    unittest.main()

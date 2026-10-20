"""Changed admitted inputs must prevent every child, including compiler probes."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import m7_native_measurement_check as checker


class MeasurementCheckerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # No model, SDK or GPU is needed for these admission-only checks.
        (checker.ROOT/'artifacts').mkdir(exist_ok=True)

    def test_changed_admitted_header_archive_or_helper_never_launches_child(self):
        root = checker.ROOT
        actual_sha = checker.package.sha
        for kind in ('header','archive','helper'):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory(dir=root/'artifacts') as temporary:
                base = Path(temporary); native=base/'native';native.mkdir()
                archive=native/'Toklet.xcframework/macos-arm64/libtoklet.a'
                archive.parent.mkdir(parents=True);archive.write_bytes(b'admitted archive')
                target={'header':root/'apple/include/toklet.h','archive':archive,
                        'helper':Path(checker.integration.__file__).resolve()}[kind]
                name=str(target.relative_to(root));expected=actual_sha(target)
                def changed(path):
                    return '0'*64 if path == target else actual_sha(path)
                with patch.object(checker.integration,'admit_build',return_value={name:expected}), \
                     patch.object(checker.package,'sha',side_effect=changed), \
                     patch.object(checker.package,'owned') as owned, contextlib.redirect_stdout(io.StringIO()):
                    code=checker.main(native,'unused',base/'output')
                self.assertEqual(code,1);owned.assert_not_called()
                record=json.loads((base/'output/checks.json').read_text())
                self.assertFalse(record['passed']);self.assertEqual(record['commands'],[])
                self.assertIsNotNone(record['primary_error'])

    def test_rechecks_map_before_subsequent_child(self):
        # A changed compiled binary is an input before the checker launch too.
        with tempfile.TemporaryDirectory(dir=checker.ROOT/'artifacts') as temporary:
            path=Path(temporary)/'check';path.write_bytes(b'first')
            pins={str(path.relative_to(checker.ROOT)):checker.package.sha(path)}
            checker.verify_inputs(pins)
            path.write_bytes(b'changed')
            with self.assertRaises(ValueError):checker.verify_inputs(pins)

    def test_incomplete_console_is_rejected(self):
        with self.assertRaises(ValueError):checker.validate_output('MEASUREMENT_CHECKS_JSON={"passed":true}\n')


if __name__=='__main__':unittest.main()

import hashlib
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from lfm3_projection_bf16_fixture import overflow_rescue, selected_rows, verify_pins


class ProjectionFixtureAdmission(unittest.TestCase):
    def test_selected_rows_preserve_beginning_middle_and_end(self):
        self.assertEqual(selected_rows(16), [0, 1, 8, 15])
        self.assertEqual(selected_rows(128000), [0, 1, 64000, 127999])
        with self.assertRaisesRegex(ValueError, 'four source rows'):
            selected_rows(3)

    def test_rescue_witness_retains_exact_physical_coordinates(self):
        x, w = overflow_rescue(33)
        self.assertEqual(len(x), 33)
        self.assertEqual((x[0], x[-1], w[0], w[-1]), (0x7f7f, 0x7f7f, 0xbf80, 0x4000))
        self.assertEqual(x[1:-1], [0]*31)
        self.assertEqual(w[1:-1], [0]*31)
        with self.assertRaisesRegex(ValueError, 'two coordinates'):
            overflow_rescue(1)

    def test_changed_pin_rejects_and_preserves_other_inputs(self):
        with tempfile.TemporaryDirectory() as folder:
            a, b = (Path(folder)/name for name in ['a', 'b'])
            a.write_bytes(b'a'); b.write_bytes(b'b')
            pins = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in [a, b]}
            verify_pins(pins, 'Fixture source')
            a.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'Fixture source changed'):
                verify_pins(pins, 'Fixture source')
            self.assertEqual(b.read_bytes(), b'b')

    def test_existing_output_and_bad_contract_reject_before_runtime_or_model(self):
        producer = Path(__file__).with_name('lfm3_projection_bf16_fixture.py')
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); out = root/'fixture'; contract = root/'contract.json'
            out.mkdir(); sentinel = out/'sentinel'; sentinel.write_bytes(b'untouched')
            contract.write_text('{}')
            command = [sys.executable, '-B', str(producer), '--contract', str(contract),
                '--runtime', str(root/'missing-runtime'), '--model', str(root/'missing-model'),
                '--fresh-directory', str(out)]
            done = subprocess.run(command, capture_output=True, text=True, timeout=10)
            self.assertNotEqual(done.returncode, 0)
            self.assertIn('Fresh fixture destination required', done.stderr)
            self.assertEqual(sentinel.read_bytes(), b'untouched')
            sentinel.unlink(); out.rmdir()
            done = subprocess.run(command, capture_output=True, text=True, timeout=10)
            self.assertNotEqual(done.returncode, 0)
            self.assertIn('Frozen projection contract differs', done.stderr)
            self.assertFalse(out.exists())


if __name__ == '__main__':
    unittest.main()

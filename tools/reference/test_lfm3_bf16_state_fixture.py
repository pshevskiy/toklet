import hashlib
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

from lfm3_bf16_state_fixture import packed,verify_pins,visibility


class StateOracleAdmission(unittest.TestCase):
    def test_window_includes_current_and_exactly_255_previous_rows(self):
        self.assertEqual(visibility(255,3,256),[(0,256),(1,257),(2,258)])
        self.assertEqual(visibility(1023,2,256),[(768,1024),(769,1025)])
        self.assertEqual(visibility(255,3,None),[(0,256),(0,257),(0,258)])

    def test_bit_encoding_preserves_zero_signs_subnormals_and_extremes(self):
        words=[0,0x8000,1,0x8001,0x7f7f,0xff7f]
        self.assertEqual(packed(words),b'\x00\x00\x00\x80\x01\x00\x01\x80\x7f\x7f\x7f\xff')
        self.assertEqual(struct.unpack('<6H',packed(words)),tuple(words))

    def test_changed_input_is_rejected_without_modifying_other_inputs(self):
        with tempfile.TemporaryDirectory() as folder:
            a=Path(folder)/'a';b=Path(folder)/'b'
            a.write_bytes(b'first');b.write_bytes(b'second')
            pins={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [a,b]}
            verify_pins(pins,'test')
            a.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'test changed'):verify_pins(pins,'test')
            self.assertEqual(b.read_bytes(),b'second')

    def test_cli_rejects_existing_output_and_altered_contract_before_runtime(self):
        producer=Path(__file__).with_name('lfm3_bf16_state_fixture.py')
        with tempfile.TemporaryDirectory() as folder:
            base=Path(folder);out=base/'fixture.json';contract=base/'contract.json'
            out.write_text('sentinel');contract.write_text('{}')
            command=[sys.executable,'-B',str(producer),'--contract',str(contract),
                '--runtime',str(base/'missing'),'--fresh-fixture',str(out)]
            first=subprocess.run(command,capture_output=True,text=True,timeout=10)
            self.assertNotEqual(first.returncode,0)
            self.assertIn('Fresh fixture destination required',first.stderr)
            self.assertEqual(out.read_text(),'sentinel')
            out.unlink()
            second=subprocess.run(command,capture_output=True,text=True,timeout=10)
            self.assertNotEqual(second.returncode,0)
            self.assertIn('Frozen BF16 state contract differs',second.stderr)
            self.assertFalse(out.exists())


if __name__=='__main__':unittest.main()

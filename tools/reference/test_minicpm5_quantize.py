"""Portable bounds, literal BF16/packing and exact-byte rejection controls."""
import sys
import unittest

import minicpm5_quantize as quant


class Quantization(unittest.TestCase):
    def test_import_does_not_execute_numerical_packages(self):
        self.assertNotIn('numpy', sys.modules)
        self.assertNotIn('mlx.core', sys.modules)

    def test_bounded_complete_row_geometry_is_checked_before_numerical_work(self):
        self.assertEqual(quant.geometry(bytes(256*6144*2), 256, 6144), 256*6144)
        for raw, rows, cols in [(b'', 1, 64), (bytes(128), True, 64), (bytes(128), 1, True),
                                (bytes(128), 0, 64), (bytes(128), 257, 64),
                                (bytes(130), 1, 65), (bytes(128), 1, 6208)]:
            with self.assertRaises(ValueError):quant.geometry(raw, rows, cols)

    def test_literal_controls_preserve_bf16_and_explicit_device_differences(self):
        cases={case['name']:case for case in quant.controls()}
        self.assertEqual(len(cases),6)
        for case in cases.values():
            self.assertEqual(quant.geometry(bytes.fromhex(case['input_bf16']),1,64),64)
            for device in ['cpu','gpu']:
                self.assertEqual(len(bytes.fromhex(case[device]['weight'])),64)
                self.assertEqual(len(bytes.fromhex(case[device]['scales'])),2)
                self.assertEqual(len(bytes.fromhex(case[device]['biases'])),2)
        self.assertEqual(cases['above_fp16_range']['input_bf16'],'8047'*64)
        self.assertEqual(cases['exact_packing']['gpu']['weight'][:8],'00010203')
        self.assertEqual(cases['round_ties']['gpu']['weight'][:10],'0001020304')
        self.assertEqual(cases['round_ties']['cpu']['weight'][:10],'0000020204')
        self.assertEqual(cases['negative_ones']['gpu']['scales'],'813b')
        self.assertEqual(cases['negative_ones']['cpu']['scales'],'d7b3')

    def test_exact_literal_bf16_rejects_rounding_and_nonfinite_values(self):
        self.assertEqual(quant.bf16_exact([-0.,1.,-256.,65536.]).hex(),'0080803f80c38047')
        for value in [0.1,float('inf'),float('nan')]:
            with self.assertRaises(ValueError):quant.bf16_exact([value])

    def test_complete_three_buffer_byte_comparison_never_uses_tolerances(self):
        raw=dict(weight=b'\x00\x01\x02\x03',scales=b'\x00\x40',biases=b'\x80\xc3')
        self.assertTrue(all(row['mismatching_bytes']==0 for row in quant.compare(raw,raw).values()))
        for key in raw:
            changed=dict(raw);changed[key]=bytes([raw[key][0]^1])+raw[key][1:]
            result=quant.compare(changed,raw)
            self.assertEqual(result[key]['mismatching_bytes'],1)
            self.assertNotEqual(result[key]['reconstructed_sha256'],result[key]['expected_sha256'])
        for changed in [dict(raw,weight=b''),dict(weight=raw['weight'],scales=raw['scales']),
                        dict(raw,extra=b''),dict(raw,scales='0040')]:
            with self.assertRaises(ValueError):quant.compare(changed,raw)


if __name__=='__main__':unittest.main()

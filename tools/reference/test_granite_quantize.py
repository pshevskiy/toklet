"""Portable contract controls before any Granite numerical runtime executes."""
import struct
import sys
import unittest

import granite_quantize as quant
import minicpm5_quantize as legacy


class GraniteQuantization(unittest.TestCase):
    def test_cold_import_keeps_numerical_packages_unloaded(self):
        self.assertNotIn('numpy', sys.modules)
        self.assertNotIn('mlx.core', sys.modules)

    def test_wide_geometry_is_bounded_and_does_not_widen_minicpm5(self):
        raw = bytes(256 * 8192 * 2)
        for bits in (4, 8):
            self.assertEqual(quant.geometry(raw, 256, 8192, bits), 256 * 8192)
        with self.assertRaises(ValueError):
            legacy.geometry(raw, 256, 8192)
        for rows, columns, bits in [(True, 64, 4), (1, True, 4), (0, 64, 4),
                                    (257, 64, 4), (1, 65, 4), (1, 8256, 4),
                                    (1, 64, True), (1, 64, 2), (1, 64, 6)]:
            with self.assertRaises(ValueError):
                quant.geometry(bytes(128), rows, columns, bits)
        for raw in (b'', bytes(127), bytes(129), bytearray(128), '00'):
            with self.assertRaises(ValueError):
                quant.geometry(raw, 1, 64, 4)

    def test_bad_geometry_or_device_rejects_before_numerical_access(self):
        for args in [(b'', 1, 64, 4, 'cpu'),
                     (bytes(128), 1, 64, 4, 'implicit'),
                     (bytes(128), 1, 64, 8, None)]:
            with self.assertRaises(ValueError):
                quant.quantize(*args, None, None)

    def test_prospective_inventory_has_six_literal_cases_per_bit_width(self):
        cases = quant.controls()
        self.assertEqual(len(cases), 12)
        self.assertEqual(len({(c['bits'], c['name']) for c in cases}), 12)
        self.assertEqual([c for c in cases if c['bits'] == 8],
                         [dict(c, bits=8) for c in legacy.controls()])
        for case in cases:
            self.assertEqual(len(bytes.fromhex(case['input_bf16'])), 128)
            for device in ('cpu', 'gpu'):
                self.assertEqual(len(bytes.fromhex(case[device]['weight'])), 8 * case['bits'])
                self.assertEqual(len(bytes.fromhex(case[device]['scales'])), 2)
                self.assertEqual(len(bytes.fromhex(case[device]['biases'])), 2)

    def test_literal_four_bit_packing_and_exact_value_controls(self):
        cases = {c['name']: c for c in quant.controls() if c['bits'] == 4}
        for device in ('cpu', 'gpu'):
            ramp = struct.unpack('<64f', quant.literal_reconstruction(cases['exact_packing'], device))
            self.assertEqual(ramp, tuple(range(-8, 8)) * 4)
            for name, value in [('zeros', 0.), ('ones', 1.), ('negative_ones', -1.),
                                ('above_fp16_range', 65536.)]:
                decoded = struct.unpack('<64f', quant.literal_reconstruction(cases[name], device))
                self.assertEqual(decoded, (value,) * 64)
        self.assertEqual(cases['exact_packing']['gpu']['weight'][:16], '1032547698badcfe')
        self.assertEqual(cases['above_fp16_range']['input_bf16'], '8047' * 64)

    def test_four_bit_ties_and_negative_only_maximum_are_device_specific(self):
        cases = {c['name']: c for c in quant.controls() if c['bits'] == 4}
        cpu = struct.unpack('<64f', quant.literal_reconstruction(cases['round_ties'], 'cpu'))
        gpu = struct.unpack('<64f', quant.literal_reconstruction(cases['round_ties'], 'gpu'))
        self.assertEqual(cpu[:5], (-16., -16., -12., -12., -8.))
        self.assertEqual(gpu[:5], (-16., -14., -12., -10., -8.))
        self.assertEqual(cases['negative_ones']['cpu']['scales'], 'd7b3')
        self.assertEqual(cases['negative_ones']['gpu']['scales'], '893d')

    def test_reconstruction_rejects_incomplete_buffers_before_numerical_access(self):
        complete = dict(weight=bytes(32), scales=bytes(2), biases=bytes(2))
        for key in complete:
            broken = dict(complete)
            broken[key] = broken[key][:-1]
            with self.assertRaises(ValueError):
                quant.reconstruct_f32(broken, 1, 64, 4, None)
        for broken in ({}, dict(complete, extra=b''), dict(complete, scales=None)):
            with self.assertRaises(ValueError):
                quant.reconstruct_f32(broken, 1, 64, 4, None)


if __name__ == '__main__':
    unittest.main()

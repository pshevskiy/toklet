import math
import struct
import unittest

import lfm3_rotary_fixture as fixture


class Rounding(unittest.TestCase):
    def test_nearest_even_ties_and_signed_zero(self):
        value = lambda word: struct.unpack('<f', struct.pack('<I', word))[0]
        for word, expected in [(0, 0), (0x80000000, 0x8000), (0x3f808000, 0x3f80),
                               (0x3f818000, 0x3f82), (0xbf808000, 0xbf80),
                               (0xbf818000, 0xbf82), (0x00008000, 0),
                               (0x00008001, 1), (0x00018000, 2)]:
            self.assertEqual(fixture.bf16(value(word)), expected)

    def test_inactive_source_arithmetic_is_not_signed_zero_copy(self):
        self.assertEqual(fixture.scalar_pair(0x8000, 0x8000, 0x3f80, 0), (0, 0x8000))
        self.assertEqual(fixture.scalar_pair(0, 0x8000, 0x3f80, 0), (0, 0))

    def test_inputs_are_finite_and_deterministic(self):
        pool = fixture.input_pool()
        data = fixture.inputs(pool, 511, 17, 16)
        self.assertEqual(len(data), 17 * 16 * 128)
        self.assertTrue(all(math.isfinite(fixture.widen(v)) for v in data))
        self.assertEqual(data, fixture.inputs(pool, 511, 17, 16))
        self.assertNotEqual(data[:128], data[128:256])

    def test_nonfinite_or_overflow_rejects(self):
        for value in [math.inf, -math.inf, math.nan, 3.4e38]:
            with self.assertRaises(ValueError):
                fixture.bf16(value)
        for word in [0x7f80, 0xff80, 0x7fff, -1, 65536]:
            with self.assertRaises(ValueError):
                fixture.widen(word)


if __name__ == '__main__':
    unittest.main()

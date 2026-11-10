import math
import struct
import unittest

import lfm3_bf16_fixture as fixture


class SourceRules(unittest.TestCase):
    def test_nearest_even_and_signed_zero(self):
        value = lambda word: struct.unpack('<f', struct.pack('<I', word))[0]
        for word, expected in [(0, 0), (0x80000000, 0x8000),
                               (0x3f808000, 0x3f80), (0x3f818000, 0x3f82),
                               (0x00008000, 0), (0x00008001, 1)]:
            self.assertEqual(fixture.bf16(value(word)), expected)

    def test_full_finite_bf16_domain(self):
        words = [n for n in range(65536) if n & 0x7f80 != 0x7f80]
        self.assertEqual(len(words), 65280)
        self.assertIn(0, words)
        self.assertIn(0x8000, words)
        self.assertIn(0x7f7f, words)
        self.assertNotIn(0x7f80, words)
        self.assertNotIn(0xffff, words)

    def test_deterministic_bounded_norm_input_domain(self):
        pool = fixture.input_pool()
        self.assertEqual(len(pool), 33538)
        self.assertIn(0x8000, pool)
        self.assertIn(0x4180, pool)
        self.assertIn(0xc180, pool)
        source = fixture.norm_inputs(pool, 17, 17*16, 128)
        values = [struct.unpack('<f', struct.pack('<I', n << 16))[0] for n in source]
        self.assertTrue(all(math.isfinite(v) and abs(v) <= 16 for v in values))
        self.assertEqual(len(source), 17*16*128)
        self.assertNotEqual(source[:128], source[128:256])
        self.assertEqual(source, fixture.norm_inputs(pool, 17, 17*16, 128))

    def test_expected_norm_geometry_and_shared_head_gains(self):
        config = dict(hidden_size=2048, head_dim=128, num_attention_heads=16,
            num_key_value_heads=4, num_hidden_layers=24, norm_eps=1e-5,
            use_sandwich_norm=True, layer_types=['conv']*13+['full_attention']*6+['sliding_attention']*5)
        schema = fixture.norm_schema(config)
        self.assertEqual(len(schema), 119)
        self.assertEqual(sum(width == 2048 for width, _ in schema.values()), 97)
        self.assertEqual(sum(width == 128 for width, _ in schema.values()), 22)
        config['head_dim'] = 64
        with self.assertRaises(ValueError):
            fixture.norm_schema(config)


if __name__ == '__main__':
    unittest.main()

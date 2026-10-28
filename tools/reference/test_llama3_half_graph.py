"""Portable producer controls; no target, MLX, NumPy or GPU execution."""
import importlib.util
import math
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('half_graph', Path(__file__).with_name('llama3_half_graph.py'))
subject = importlib.util.module_from_spec(spec)
spec.loader.exec_module(subject)


class Controls(unittest.TestCase):
    def test_ordered_profiles_and_storage(self):
        values = [2**24, 1, -(2**24)]
        self.assertEqual(subject.dot(values, [1, 1, 1], subject.f32), 0)
        self.assertEqual(subject.dot(values, [1, 1, 1], float), 1)
        self.assertEqual(subject.bits(subject.half(-0.0), 'f16le'), 0x8000)
        self.assertEqual(subject.bits(subject.half(1 + 2**-11), 'f16le'), 0x3c00)
        self.assertEqual(subject.bits(subject.f32(1 + 2**-24), 'f32le'), 0x3f800000)

    def test_gqa_causal_store_and_future_value_control(self):
        query, key = [[0.0] * 32 for _ in range(3)], [[0.0] * 16 for _ in range(3)]
        value = [[2.0] * 8 + [-3.0] * 8 for _ in range(3)]
        for precision in (subject.f32, float):
            scores, probabilities, output = subject.attention(query, key, value, precision)
            self.assertTrue(all(v == 0 for v in subject.flatten(scores)))
            self.assertEqual(probabilities[0][0], [1, 0, 0])
            self.assertEqual(probabilities[0][2], [subject.f32(1 / 3)] * 3)
            self.assertEqual(output[0], [2] * 16 + [-3] * 16)
            altered = [row[:] for row in value]
            altered[2] = [100] * 16
            changed = subject.attention(query, key, altered, precision)[2]
            self.assertEqual(changed[:2], output[:2])
            self.assertNotEqual(changed[2], output[2])

    def test_scaled_rotary_is_not_delayed_until_original_context(self):
        # Pair3 has ordinary inverse.001 and wavelength>64: scales even at pos1.
        self.assertAlmostEqual(subject.canonical_inverse(3, 8, 10000), 0.001 / 8, places=18)
        x = [[0.0] * 8]
        x[0][3] = 1.0
        for wide in (False, True):
            y = subject.rotary(x, 1, 1, wide)[0]
            self.assertLess(abs(y[7] - subject.half(math.sin(0.001 / 8))), 2**-23)
            self.assertNotEqual(y[7], subject.half(math.sin(0.001)))
            self.assertEqual(subject.rotary(x, 1, 0, wide)[0], x[0])

    def test_fixed_inventory(self):
        self.assertEqual(len(subject.CASES), 4)
        self.assertEqual(subject.CASES[-1], ('offset509', [7, 13, 3], 509))
        self.assertEqual(subject.POLICY, dict(atol_floor=1e-4, envelope_factor=2, rtol=1e-3))
        for n in (3, 4):
            plan = subject.contracts(n)
            self.assertEqual(len(plan), 39)
            self.assertEqual(plan['layers.1.attention_scores'], [4, n, n])
            self.assertEqual(plan['logits'], [n, 64])


if __name__ == '__main__':
    unittest.main()

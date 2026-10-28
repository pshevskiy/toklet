"""Portable pointwise source controls, not GPU tests or fixture export."""
import importlib.util
import math
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('gemma3_half_pointwise', Path(__file__).with_name('gemma3_half_pointwise.py'))
subject = importlib.util.module_from_spec(spec)
spec.loader.exec_module(subject)


class Controls(unittest.TestCase):
    def test_policy_shapes_and_counts(self):
        self.assertEqual(subject.POLICY, dict(fatal_reference='main', atol_floor=1e-4, envelope_factor=2, rtol=1e-3))
        self.assertEqual(sum(r * w * (2 if op == 'gelu_gated' else 1) for _, op, r, w, _ in subject.CASES), 152512)
        for _, op, rows, width, _ in subject.CASES:
            words, up, stride = subject.operands(op, rows, width)
            self.assertEqual(stride, width + 3)
            self.assertEqual(len(words), rows * stride + 7)
            for row in range(rows):
                self.assertEqual(words[row * stride + width:(row + 1) * stride], [0x7e55] * 3)
            self.assertEqual(words[-7:], [0x7e55] * 7)
            if up is not None:
                self.assertEqual((len(up), up[-7:]), (rows * width + 7, [0x7e55] * 7))
        for op, rows, width in [('gelu_gated', 0, 128), ('gelu_gated', 33, 128), ('gelu_gated', 1, 640), ('scale', 1, 2048)]:
            with self.assertRaises(ValueError):
                subject.operands(op, rows, width)

    def test_exact_half_intermediate_and_subnormal_words(self):
        for profile in ('main', 'wide'):
            self.assertEqual(subject.gelu_product(0x3400, 0x4500, profile), (0x30ca, 0x39fc))
            self.assertEqual(subject.gelu_product(0x0002, 0x4500, profile), (0x0001, 0x0005))
            self.assertEqual(subject.gelu_product(0x8002, 0x4500, profile), (0x8001, 0x8005))
            for factor, positive in [(8., 8), (25.25, 25)]:
                self.assertEqual(subject.scale_word(1, factor, profile), positive)
                self.assertEqual(subject.scale_word(0x8001, factor, profile), 0x8000 | positive)
        c = subject.ordering_controls()
        self.assertEqual(c['wrong_delayed_gelu_half_product'], 0x39fd)
        self.assertEqual(c['gathered_half_word'], 0x3c00)
        self.assertEqual(c['scaled_half_word'], 0x4e50)
        self.assertEqual(c['wrong_scale_before_gather_half_word'], 0x4e51)
        self.assertNotEqual(subject.half_word(math.sqrt(640)), 0x4e50)

    def test_separate_strides_no_mutation_and_finite_values(self):
        words, up, stride = subject.operands('gelu_gated', 3, 128)
        original = (words[:], up[:])
        for profile in ('main', 'wide'):
            output = subject.evaluate('gelu_gated', 3, 128, words, up, stride, None, profile)
            self.assertEqual(set(output), {'activation', 'product'})
            self.assertTrue(all(len(v) == 384 for v in output.values()))
            self.assertTrue(all(math.isfinite(subject.half_value(x)) for values in output.values() for x in values))
        self.assertEqual((words, up), original)
        with self.assertRaises(ValueError):
            subject.evaluate('gelu_gated', 3, 128, words, [], stride, None, 'main')

    def test_large_finite_and_status_controls(self):
        for profile in ('main', 'wide'):
            self.assertEqual(subject.gelu_product(0x7bff, 0x3c00, profile), (0x7bff, 0x7bff))
            self.assertEqual(subject.half_value(subject.gelu_product(0xfbff, 0x3c00, profile)[0]), 0)
            for item in subject.error_controls():
                with self.assertRaises((ValueError, OverflowError)):
                    if item['operation'] == 'scale':
                        subject.scale_word(item['input_word'], item['factor'], profile)
                    else:
                        subject.gelu_product(item['input_word'], item['up_word'], profile)
        with self.assertRaises(ValueError):
            subject.scale_word(0x3c00, 25.29822, 'main')
        with self.assertRaises(ValueError):
            subject.gelu_unstored(1., 'unknown')


if __name__ == '__main__':
    unittest.main()

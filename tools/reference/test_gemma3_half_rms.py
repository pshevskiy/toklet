"""Portable prospective reference controls; no GPU or source fixture export."""
import importlib.util
import math
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('gemma3_half_rms', Path(__file__).with_name('gemma3_half_rms.py'))
subject = importlib.util.module_from_spec(spec)
spec.loader.exec_module(subject)


class Controls(unittest.TestCase):
    def test_policy_shapes_and_strided_padding(self):
        self.assertEqual(subject.POLICY, dict(atol_floor=1e-4, envelope_factor=2, rtol=1e-3, fatal_reference='main'))
        self.assertEqual(sum(rows * width for _, rows, width, _ in subject.CASES), 54272)
        for _, rows, width, pattern in subject.CASES:
            words, gamma, scales, stride = subject.operands(rows, width, pattern)
            self.assertEqual((len(gamma), len(scales), stride), (width, width, width + 3))
            self.assertEqual(len(words), rows * stride + 7)
            for row in range(rows):
                self.assertEqual(words[row * stride + width:(row + 1) * stride], [0x7e55] * 3)
            self.assertEqual(words[-7:], [0x7e55] * 7)
            self.assertTrue(all(math.isfinite(x) for x in scales))
        for rows, width in ((0, 32), (129, 32), (1, 31), (True, 32)):
            with self.assertRaises(ValueError):
                subject.operands(rows, width, 'dyadic')

    def test_compensation_and_effective_scale(self):
        self.assertEqual(subject.neumaier([2**24, 1, -(2**24)]), 1)
        self.assertEqual(subject.f32(1 + 2**-24), 1)
        scales = subject.effective([0xbf80, 0, 0x3f80, 0x3980, 0xbf7f, 0xbf81])
        self.assertEqual(scales, [0, 1, 2, 1 + 2**-12, 1 / 256, -1 / 128])
        self.assertNotEqual(scales[3], subject.half_value(subject.half_word(scales[3])))
        with self.assertRaises(ValueError):
            subject.effective([0x7fc0])

    def test_exact_rounded_coefficient_witness(self):
        words, _, scales, stride = subject.operands(1, 32, 'rounding')
        original_words, original_scales = words[:], scales[:]
        narrowed = [subject.half_value(subject.half_word(x)) for x in scales]
        for profile in ('main', 'wide'):
            correct = subject.rms(words, scales, 1, 32, stride, profile)
            wrong = subject.rms(words, narrowed, 1, 32, stride, profile)
            self.assertEqual(correct[0], 0x3fa7)
            self.assertEqual(wrong[0], 0x3fa6)
        self.assertEqual(words, original_words)
        self.assertEqual(scales, original_scales)

    def test_zero_large_and_active_errors(self):
        words, _, scales, stride = subject.operands(1, 640, 'dyadic')
        for profile in ('main', 'wide'):
            output = subject.rms(words, scales, 1, 640, stride, profile)
            self.assertTrue(all(subject.half_value(word) == 0 for word in output))
        words, _, scales, stride = subject.operands(1, 640, 'large')
        for profile in ('main', 'wide'):
            self.assertTrue(all(math.isfinite(subject.half_value(word)) for word in subject.rms(words, scales, 1, 640, stride, profile)))
        for control in subject.errors():
            for profile in ('main', 'wide'):
                with self.assertRaises((ValueError, OverflowError)):
                    subject.rms(control['input_words'], subject.effective(control['gamma_words']),
                                control['rows'], control['width'], control['input_stride'], profile)
        with self.assertRaises(ValueError):
            subject.rms([0] * 31, [1.] * 32, 1, 32, 32, 'main')
        with self.assertRaises(ValueError):
            subject.rms([0] * 32, [math.inf] * 32, 1, 32, 32, 'main')


if __name__ == '__main__':
    unittest.main()

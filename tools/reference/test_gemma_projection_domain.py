from fractions import Fraction
import struct
import unittest
import gemma_projection_domain as subject
from check_gemma_affine_admission import rounded


def floats(*values):
    return struct.pack('<'+'f'*len(values), *values)


def bf16(word):
    return struct.pack('<H', word)


class DomainControls(unittest.TestCase):
    def test_raw_significand_quantum_and_signed_zero(self):
        self.assertIsNone(subject.input_grid(floats(0., -0.)))
        for value, expected in [(1., 0), (-1.5, -1), (1+2**-23, -23), (2**-126, -126), (1.5*2**-126, -127)]:
            self.assertEqual(subject.input_grid(floats(value)), expected)

    def test_exact_reconstruction_boundary_and_one_bit_below(self):
        zero = bf16(0)
        self.assertEqual(subject.admit_metadata(zero, bf16(0x0080), 0)['metadata_quantum'], -126)
        with self.assertRaises(subject.DomainError) as caught:
            subject.admit_metadata(zero, bf16(0x00c0), 24)
        self.assertEqual(caught.exception.evidence['metadata_quantum'], -127)
        self.assertEqual(caught.exception.evidence['status'], 'inconclusive')
        # The particular product is normal; a rejected bound is not a numeric failure.
        self.assertEqual(rounded(Fraction(1.5*2**-126)*2**24), 1.5*2**-102)

    def test_exact_product_boundary_and_one_bit_below(self):
        result = subject.admit_metadata(bf16(0), bf16(0x3f80), -126)
        self.assertEqual(result['product_quantum'], -126)
        with self.assertRaises(subject.DomainError) as caught:
            subject.admit_metadata(bf16(0), bf16(0x3f80), -127)
        self.assertEqual(caught.exception.evidence['product_quantum'], -127)

    def test_zero_cases_keep_operand_and_reconstruction_checks(self):
        self.assertTrue(subject.admit_metadata(bf16(0), bf16(0), -149)['admitted'])
        with self.assertRaises(subject.DomainError):
            subject.admit_metadata(bf16(0), bf16(0x00c0), None)
        with self.assertRaises(subject.DomainError):
            subject.input_grid(struct.pack('<I', 1))
        with self.assertRaises(subject.DomainError):
            subject.admit_metadata(bf16(1), bf16(0), None)
        for raw in [struct.pack('<I', 0x7f800000), b'', b'\x00']:
            with self.assertRaises(ValueError):
                subject.input_grid(raw)

    def test_declared_grid_closed_under_rounded_cancellation_and_product_residual(self):
        quantum = Fraction(1, 2**126)
        for a, b in [(3, -2), (2**24+1, -(2**24)), (-7, 11)]:
            left, right = rounded(a*quantum), rounded(b*quantum)
            result = rounded(Fraction(left)+Fraction(right))
            self.assertEqual((Fraction(result)/quantum).denominator, 1)
        weight, value = Fraction(129, 128), Fraction(2**23+1, 2**23)
        product = rounded(weight*value)
        residual = rounded(weight*value-Fraction(product))
        self.assertNotEqual(residual, 0)
        self.assertGreaterEqual(abs(residual), 2**-126)

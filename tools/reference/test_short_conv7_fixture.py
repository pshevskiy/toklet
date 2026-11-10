"""Portable exact-rational rounding and causal layout controls."""
from fractions import Fraction
import sys
import unittest
import short_conv7_fixture as fixture


class Exact(unittest.TestCase):
    def test_import_is_independent_of_numerical_runtimes(self):
        for name in ['numpy','torch','mlx.core']:
            self.assertNotIn(name, sys.modules)

    def test_exact_fp32_roundtrips_and_rounds_ties_to_even(self):
        for value in [0,1,2,0x7fffff,0x800000,0x3f800000,0x3f800001,0x7f7fffff,0xbf123456]:
            self.assertEqual(fixture.rounded(fixture.rational(value)), value)
        for sign in [-1,1]:
            for fraction, expected in [(Fraction(1)+Fraction(1,2**24),0x3f800000),
                (Fraction(1)+Fraction(3,2**24),0x3f800002),
                (Fraction(2)-Fraction(1,2**24),0x40000000),
                (Fraction(1,2**150),0), (Fraction(3,2**150),2)]:
                self.assertEqual(fixture.rounded(sign*fraction), expected | (0x80000000 if sign < 0 else 0))
        for value in [True,-1,2**32,0x7f800000,0x7fc00000]:
            with self.assertRaises(ValueError):
                fixture.rational(value)

    def test_seven_taps_are_oldest_to_current_and_history_stays_raw(self):
        values=[fixture.bits(x) for x in [1.,2.,3.,4.,5.,6.,7.,8.]]
        weights=[fixture.bits(x) for x in [1.,2.,3.,4.,5.,6.,7.]]
        output, histories, final=fixture.oracle(values[6:],weights,values[:6],2,1)
        self.assertEqual(output,[fixture.bits(140.),fixture.bits(168.)])
        self.assertEqual(histories,values[1:7]+values[2:8])
        self.assertEqual(final,values[2:8])
        empty=fixture.oracle([],weights,values[:6],0,1)
        self.assertEqual(empty,([],[],values[:6]))

    def test_geometry_types_and_overflow_reject(self):
        for tokens,channels in [(True,1),(1,0),(33,1)]:
            with self.assertRaises(ValueError):
                fixture.oracle([],[],[],tokens,channels)
        with self.assertRaises(ValueError):
            fixture.rounded(Fraction(2**128))


if __name__=='__main__':
    unittest.main()

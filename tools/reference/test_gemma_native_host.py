import math
import os
import struct
import unittest
from decimal import Decimal, localcontext
import mpmath as mp
import gemma_native_host as subject


def doubles(values):
    return struct.pack('<'+'d'*len(values), *values)


def floats(values):
    return struct.pack('<'+'f'*len(values), *values)


class HostControls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        library = os.environ.get('TOKLET_GEMMA_NATIVE_HOST_LIBRARY')
        if not library:
            raise unittest.SkipTest('Explicit compiled Darwin host library required')
        cls.host = subject.Host(library)

    def test_signed_zeros_and_rounded_pairs_match_high_precision(self):
        self.assertEqual(self.host.sincos(doubles([0., -0.])), bytes.fromhex('000000000000803f000000800000803f'))
        angles = [2**-126, .125, .5, math.pi/2, 31., 255., 433., 511., -511.]
        with mp.workdps(100):
            # Both sides of a FP32 sine midpoint, safely separated in binary64.
            midpoint = (mp.mpf(.5)+mp.mpf(.5+2**-24))/2
            center = mp.asin(midpoint)
            angles += [float(center-mp.mpf(2)**-45), float(center+mp.mpf(2)**-45)]
            expected = [float(operation(mp.mpf(angle))) for angle in angles for operation in [mp.sin, mp.cos]]
        self.assertEqual(self.host.sincos(doubles(angles)), floats(expected))

    def test_provider_uuids_are_live_stable_and_system_owned(self):
        providers = self.host.providers()
        self.assertEqual(providers, self.host.providers())
        for spec in providers.values():
            self.assertTrue(spec['path'].startswith('/usr/lib/'))
            self.assertEqual(len(bytes.fromhex(spec['uuid'])), 16)
            self.assertGreater(spec['address'], spec['image_base'])

    def test_invalid_angle_capacity_nonfinite_and_domain_reject(self):
        for raw in [b'', bytes(9), bytes(65536*8+8), doubles([math.inf]), doubles([math.nan]), doubles([513.])]:
            with self.assertRaises(ValueError):
                self.host.sincos(raw)

    def test_python_sqrt_and_math_provider_controls(self):
        with localcontext() as context:
            context.prec = 200
            for value in [2.**-126, 1., 1+2**-52, 2., 2.**128, float.fromhex('0x1.fffffep+127')**2]:
                expected = float(Decimal.from_float(value).sqrt())
                self.assertEqual(struct.pack('<d', math.sqrt(value)), struct.pack('<d', expected))
        with mp.workdps(100):
            for operation, reference, values in [(math.tanh, mp.tanh, [-8., -.5, 0., .5, 8.]),
                (math.exp, mp.exp, [-30., -1., 0., 1.])]:
                for value in values:
                    expected = float(reference(mp.mpf(value)))
                    self.assertLessEqual(abs(operation(value)-expected), 2*math.ulp(expected))
            for base, exponent in [(10000., -1/128), (1000000., -127/256), (10000., 0.)]:
                expected = float(mp.power(mp.mpf(base), mp.mpf(exponent)))
                self.assertLessEqual(abs(math.pow(base, exponent)-expected), 2*math.ulp(expected))

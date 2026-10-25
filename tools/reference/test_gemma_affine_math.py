"""Portable pre-target exact-dot/admission witnesses; no engine/GPU imports."""
from fractions import Fraction
from pathlib import Path
import copy
import hashlib
import importlib.util
import json
import math
import struct
import tempfile
import unittest


def module(name):
    path = Path(__file__).with_name(name + '.py')
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


m = module('gemma_affine_math')
checker = module('check_gemma_affine_math')


def wire(data, raw):
    body = (json.dumps(data, indent=2, allow_nan=False) + '\n').encode()
    manifest = dict(format='toklet-gemma-metal-affine-manifest-v1', producer_sha256=data['producer_sha256'],
                    files={name: dict(bytes=len(value), sha256=hashlib.sha256(value).hexdigest())
                           for name, value in [('math.json', body), ('math.bin', raw)]},
                    cases=len(data['cases']), native_target_executed=False)
    return body, raw, json.dumps(manifest).encode()


class ProjectionWitness(unittest.TestCase):
    def test_fixed_shapes_and_inventory(self):
        cases = m.cases()
        self.assertEqual(len(cases), 27)
        self.assertTrue({64, 128, 256, 512, 1536, 4096, 6144, 8960, 12288} <= {c['width'] for c in cases})
        self.assertEqual(sum(len(c['math_fp64']) for c in cases), 306)
        self.assertEqual(sum(c['constructor']['admitted'] and c['input_admission']['admitted'] for c in cases), 21)

    def test_decimal400_independently_replays_every_exact_and_ordered_dot(self):
        data, raw = m.document()
        result = checker.verify(*wire(data, raw))
        self.assertTrue(result['passed'])
        self.assertEqual(result['outputs'], 306)
        self.assertEqual(result['admitted_cases'], 21)

    def test_literal_low_nibbles_and_group_boundaries(self):
        data, raw = m.document()
        case = next(c for c in data['cases'] if c['name'] == 'width-256')
        start = case['arrays']['weight']['offset']
        self.assertEqual(raw[start:start + 8], bytes.fromhex('705e3c1af8d6b492'))
        self.assertEqual(raw[start + 32:start + 40], bytes.fromhex('a3816f4d2b09e7c5'))
        codes = m.cases()[2]['codes']
        self.assertEqual(set(codes[:64]), set(range(16)))
        self.assertNotEqual(codes[:64], codes[64:128])

    def test_inputs_are_fp32_and_not_silently_half(self):
        values = m.cases()[0]['input']
        self.assertTrue(all(m.f32(v) == v for v in values))
        self.assertTrue(any(struct.unpack('<e', struct.pack('<e', v))[0] != v for v in values))

    def test_conflicting_oracles_are_retained_and_rejected(self):
        case = next(c for c in m.cases() if c['name'] == 'ordered-exact-conflict')
        self.assertEqual(case['ordered_fp64'], [0.])
        self.assertEqual(case['math_fp64'], [1.])
        self.assertEqual(case['exact_dot_rationals'], [['1', '1']])
        self.assertEqual(case['input_admission']['reason'], 'input-summation')
        self.assertLess(case['input_admission']['absolute_products_bound'], m.MAXIMUM / 4)
        self.assertGreater(abs(case['ordered_fp64'][0] - case['math_fp64'][0]), m.ATOL + m.RTOL)

    def test_finite_exact_result_with_naive_product_overflow_rejects(self):
        case = next(c for c in m.cases() if c['name'] == 'finite-product-overflow')
        self.assertEqual(case['ordered_fp64'], case['math_fp64'])
        self.assertEqual(case['math_fp64'], [0.])
        self.assertEqual(case['input_admission']['reason'], 'input-absolute-products')
        with self.assertRaises(OverflowError):
            m.f32(2 * m.MAXIMUM)

    def test_zero_weight_admits_maximum_input_with_exact_zero_bounds(self):
        case = next(c for c in m.cases() if c['name'] == 'zero-weights-max-input')
        self.assertEqual(case['input_admission'], dict(admitted=True, reason=None, absolute_products_bound=0., ordered_error_bound=0.))
        self.assertEqual(case['math_fp64'], [0.])

    def test_adjacent_fp32_boundary_values_straddle_upward_bound(self):
        a, b = [next(c for c in m.cases() if c['name'] == name) for name in ('boundary-below', 'boundary-above')]
        bits = lambda v: struct.unpack('<I', struct.pack('<f', v))[0]
        self.assertEqual(bits(b['input'][0]), bits(a['input'][0]) + 1)
        self.assertLessEqual(a['input_admission']['ordered_error_bound'], m.ATOL / 16)
        self.assertGreater(b['input_admission']['ordered_error_bound'], m.ATOL / 16)

    def test_constructor_codebook_domain_and_unused_overflow_codes(self):
        for name, reason in [('subnormal-metadata', 'metadata-domain'), ('subnormal-weight', 'weight-domain'), ('large-weight', 'weight-domain')]:
            case = next(c for c in m.cases() if c['name'] == name)
            self.assertFalse(case['constructor']['admitted'])
            self.assertEqual(case['constructor']['reason'], reason)
            self.assertIsNone(case['input_admission'])
        largest_bf16 = struct.unpack('<f', bytes.fromhex('00007f7f'))[0]
        self.assertEqual(m.codebook([largest_bf16], [0.])['reason'], 'weight-domain')
        # The first rejecting code is already too large; a later unused code
        # additionally overflows FP32. Preserve the constructor's check order.
        with self.assertRaises(OverflowError):
            m.f32(2 * largest_bf16)
        self.assertEqual(m.codebook([math.inf], [0.])['reason'], 'metadata-domain')
        with self.assertRaises(ValueError):
            m.codebook([1.], [])

    def test_invalid_envelope_inputs_reject(self):
        for width, weight, values in [(True, 1., [1.]), (63, 1., [1.]), (12352, 1., [1.]), (64, True, [1.]), (64, 1., [True]), (64, 1., [math.inf]), (64, 1., [1.1]), (64, 1., [])]:
            with self.assertRaises((ValueError, OverflowError)):
                m.admission(width, weight, values)

    def test_gamma_bound_and_cancellation_placements(self):
        u = Fraction(1, 2 ** 53)
        for width in (64, 128, 1536, 12288):
            self.assertLessEqual(width * u / (1 - width * u), 2 * width * u)
        cases = [c for c in m.cases() if c['name'].startswith('compensated-')]
        self.assertEqual(len(cases), 3)
        for case in cases:
            self.assertEqual(case['math_fp64'], [2. ** -8])
            self.assertEqual(case['ordered_fp64'], case['math_fp64'])
            self.assertTrue(case['input_admission']['admitted'])
        deep = [c for c in m.cases() if c['name'].startswith('deep-cancel-')]
        self.assertEqual(len(deep), 3)
        for case in deep:
            self.assertEqual(case['math_fp64'], [2. ** -14])
            self.assertTrue(case['input_admission']['admitted'])

    def test_deterministic_exclusive_output_and_corruption_rejection(self):
        with tempfile.TemporaryDirectory(prefix='toklet-affine-reference-') as temporary:
            root = Path(temporary)
            self.assertEqual(m.write(root / 'a'), m.write(root / 'b'))
            for name in ('math.json', 'math.bin', 'manifest.json'):
                self.assertEqual((root / 'a' / name).read_bytes(), (root / 'b' / name).read_bytes())
            with self.assertRaises(FileExistsError):
                m.write(root / 'a')
        data, raw = m.document()
        edited = copy.deepcopy(data)
        next(c for c in edited['cases'] if c['name'] == 'ordered-exact-conflict')['ordered_fp64'] = [1.]
        with self.assertRaises(AssertionError):
            checker.verify(*wire(edited, raw))
        with self.assertRaises(AssertionError):
            checker.verify(*wire(data, bytes([raw[0] ^ 1]) + raw[1:]))


if __name__ == '__main__':
    unittest.main(verbosity=2)

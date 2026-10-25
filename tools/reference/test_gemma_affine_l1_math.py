"""Portable L1 proof, ownership and pre-target numerical witnesses."""
from fractions import Fraction
from pathlib import Path
import hashlib
import json
import math
import struct
import tempfile
import unittest
from unittest.mock import patch

import gemma_affine_l1_math as producer
import check_gemma_affine_l1_math as verifier


class L1(unittest.TestCase):
    def test_directed_sum_retains_positive_small_term_and_signed_zeros(self):
        values = [2.**40, 2.**-20, -0., 0.]
        exact = sum(map(Fraction, values), Fraction())
        self.assertLess(Fraction(sum(values)), exact)
        for function in (producer.upper_sum, verifier.accumulated):
            upper = function(values)
            self.assertGreaterEqual(Fraction(upper), exact)
            self.assertEqual(upper, math.nextafter(2.**40, math.inf))
            self.assertEqual(function([-0., 0., 0.]), 0.)
            self.assertEqual(function([0., 2.**-149, 0.]), 2.**-149)

    def test_sum_and_bounds_reject_invalid_or_nonfinite_terms(self):
        for function in (producer.upper_sum, verifier.accumulated):
            for values in ([True], [-1.], [math.inf], [math.nan]):
                with self.assertRaises(ValueError):function(values)
        for function in (producer.l1_bound, verifier.bound):
            for values in [(True, 1., 1.), (63, 1., 1.), (12352, 1., 1.), (64, True, 1.), (64, 1., math.inf), (64, 2.**33, 1.)]:
                with self.assertRaises(ValueError):function(*values)

    def test_sparse_cancellation_newly_admits_without_changing_dual_gates(self):
        cases = producer.cases()
        cancels = [c for c in cases if c['name'].startswith('sparse-cancel-')]
        self.assertEqual(len(cancels), 4)
        for case in cancels:
            row = case['inputs'][0]
            a = producer.upper_sum(abs(v) for v in row)
            bound = producer.l1_bound(case['width'], 1., a)
            self.assertEqual(bound, verifier.bound(case['width'], 1., a))
            self.assertTrue(bound['admitted'])
            self.assertFalse(producer.prior.bound(case['width'], 1., max(map(abs, row)))['admitted'])
            self.assertEqual(sum(map(Fraction, row), Fraction()), Fraction(1, 256))
            total = 0.
            for value in row:total = producer.prior.f32(total+value)
            self.assertEqual(total, 0.)

    def test_per_token_l1_is_not_the_sum_across_independent_rows(self):
        case = next(c for c in producer.cases() if c['name'] == 'independent-token-rows')
        sums = [producer.upper_sum(abs(v) for v in row) for row in case['inputs']]
        self.assertTrue(producer.l1_bound(case['width'], 1., max(sums))['admitted'])
        self.assertFalse(producer.l1_bound(case['width'], 1., producer.upper_sum(sums))['admitted'])
        self.assertEqual(sums, [verifier.accumulated(abs(v) for v in row) for row in case['inputs']])

    def test_adjacent_l1_cutoff_is_frozen(self):
        values = {c['name']: c for c in producer.cases()}
        first, second = values['cutoff-below'], values['cutoff-above']
        lower = producer.upper_sum(abs(v) for v in first['inputs'][0])
        upper = producer.upper_sum(abs(v) for v in second['inputs'][0])
        self.assertEqual(struct.unpack('<I', struct.pack('<f', upper))[0], struct.unpack('<I', struct.pack('<f', lower))[0]+1)
        self.assertTrue(verifier.bound(12288, 1., lower)['admitted'])
        self.assertFalse(verifier.bound(12288, 1., upper)['admitted'])

    def test_ordered_conflict_and_product_overflow_still_reject(self):
        values = {c['name']: c for c in producer.cases()}
        for name, reason in [('ordered-exact-conflict', 'ordered-error'), ('finite-product-overflow', 'absolute-products')]:
            case = values[name]
            magnitude = producer.upper_sum(abs(v) for v in case['inputs'][0])
            result = producer.l1_bound(case['width'], max(map(abs, case['weights'])), magnitude)
            self.assertFalse(result['admitted'])
            self.assertEqual(result['reason'], reason)
        row = values['ordered-exact-conflict']['inputs'][0]
        ordered = 0.
        for value in row:
            ordered += value
        self.assertEqual(ordered, 0.)
        self.assertEqual(sum(map(Fraction, row), Fraction()), 1)

    def test_zero_weights_and_underflow_inputs_keep_zero_or_tiny_bounds(self):
        maximum = producer.prior.MAXIMUM
        a = producer.upper_sum([maximum, maximum])
        self.assertEqual(producer.l1_bound(64, 0., a), dict(admitted=True, reason=None, absolute_products=0., ordered_error=0.))
        bound = producer.l1_bound(12288, 1., 2.**-149)
        self.assertTrue(bound['admitted'])
        self.assertEqual(bound, verifier.bound(12288, 1., 2.**-149))

    def test_all_case_proofs_enclose_exact_fraction_magnitudes(self):
        cases = producer.cases()
        self.assertEqual(len(cases), 19)
        self.assertEqual(len({c['name'] for c in cases}), 19)
        for case in cases:
            for row in case['inputs']:
                a = producer.upper_sum(abs(v) for v in row)
                exact = sum((Fraction(abs(v)) for v in row), Fraction())
                self.assertGreaterEqual(Fraction(a), exact)
                self.assertEqual(a, verifier.accumulated(abs(v) for v in row))
                w = max(map(abs, case['weights']))
                bound = producer.l1_bound(case['width'], w, a)
                self.assertGreaterEqual(Fraction(bound['absolute_products']), Fraction(w)*exact)
                self.assertEqual(bound, verifier.bound(case['width'], w, a))

    @staticmethod
    def reader(name):
        if name == 'layer.0.context':
            return [float(t+1) for h in range(8) for t in range(4) for _ in range(256)], [1, 8, 4, 256]
        if name == 'a':return [10., 1.]*4, [1, 4, 2]
        if name == 'b':return [1., 10.]*4, [1, 4, 2]
        if name == 'overflow':return [producer.prior.MAXIMUM]*4, [1, 4, 1]
        if name == 'two':return [2.]*4, [1, 4, 1]
        value = 2.**24 if name == 'x' else 1. if name == 'y' else -2.**24
        return [value]*6144, [1, 4, 1536]

    def test_real_context_l1_preserves_token_axis(self):
        for function in (producer.input_rows, verifier.real_rows):
            rows = function('direct', ['layer.0.context'], self.reader)
            self.assertEqual([sum(row) for row in rows], [2048., 4096., 6144., 8192.])

    def test_real_residual_keeps_two_rounded_adds(self):
        first = producer.input_rows('exact-residual', ['x', 'y', 'z'], self.reader)
        second = verifier.real_rows('exact-residual', ['x', 'y', 'z'], self.reader)
        self.assertEqual(first, second)
        self.assertEqual([producer.upper_sum(map(abs, row)) for row in first], [0.]*4)

    def test_coordinate_envelope_l1_preserves_correlation_and_overflow(self):
        for function in (producer.input_rows, verifier.real_rows):
            rows = function('product-envelope', ['a', 'b'], self.reader)
            self.assertEqual([producer.upper_sum(row) for row in rows], [math.nextafter(20., math.inf)]*4)
            self.assertTrue(all(Fraction(producer.upper_sum(row)) >= 20 for row in rows))
            overflow = function('product-envelope', ['overflow', 'two'], self.reader)
            self.assertEqual(overflow, [[None]]*4)
            for values in ([True]*8, [math.nan]*8):
                def malformed(name):
                    return (values if name == 'a' else [1.]*8), [1, 4, 2]
                with self.assertRaises(ValueError):function('product-envelope', ['a', 'b'], malformed)
            def shapes(name):
                return ([1.]*8, [1, 4, 2]) if name == 'a' else ([1.]*4, [1, 4, 1])
            with self.assertRaises(ValueError):function('product-envelope', ['a', 'b'], shapes)

    def test_synthetic_manifest_has_no_native_or_gpu_grant(self):
        document, raw = producer.synthetic()
        self.assertEqual(len(document['cases']), 19)
        self.assertEqual(document['binary']['bytes'], len(raw))
        self.assertEqual(document['gate'], dict(absolute=1e-5, relative=1e-4))
        self.assertFalse(document['engine_admission_changed'])
        self.assertFalse(document['native_target_executed'])
        self.assertFalse(document['gpu_qualified'])

    def test_later_actual_read_failure_retains_completed_and_attempted_records(self):
        # Mock authentication/ledger only; execute write(), actual() and raw reads.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            coordinate, base, trace, arrays, compact = [root/name for name in ('coordinate', 'base', 'trace', 'arrays', 'compact')]
            matrices = [dict(name=name, columns=64, maximum_weight=1., geometry_admitted=True) for name in ('first', 'second')]
            observed = dict(passed=True, base_receipt_sha256=producer.BASE_SHA, matrices=matrices,
                            metadata_sha256='5bbbe6bdcbee8135e271926da1ae7a254dbdb620c9a6ac2c1c25aeeaa01d3947',
                            invocations=[dict(name=name, input_kind='direct', sources=[dict(name=name)], status='proved') for name in ('first', 'second')])
            coordinate.write_text(json.dumps(observed))
            base.write_text(json.dumps(dict(passed=True, matrices=matrices)))
            trace.write_text('{}');compact.write_text('{}')
            raw = struct.pack('<256f', *([1.]*256))
            arrays.write_bytes(raw+raw)
            ledger = {('staged_fp32', name): dict(bytes=len(raw), offset=i*len(raw), shape=[1, 4, 64],
                                                  sha256=hashlib.sha256(raw).hexdigest() if i == 0 else '0'*64)
                      for i, name in enumerate(('first', 'second'))}
            ledger.update({('unused', str(i)): {} for i in range(1424)})
            fixed = dict(zip((coordinate, base, trace, arrays, compact),
                             (producer.COORDINATE_SHA, producer.BASE_SHA, producer.prior.TRACE_SHA, producer.prior.ARRAY_SHA, producer.prior.COMPACT_SHA), strict=True))
            original_sha = producer.prior.sha
            def digest(path):
                return fixed[path] if path in fixed else original_sha(path)
            with patch.object(producer.prior, 'sha', side_effect=digest), patch.object(producer.prior, 'trace_ledger', return_value=ledger):
                output = root/'capture'
                with self.assertRaisesRegex(ValueError, 'raw activation changed'):
                    producer.write(coordinate, base, trace, arrays, compact, output)
            report = json.loads((output/'trace.json').read_text())
            self.assertFalse(report['passed'])
            self.assertEqual(report['primary_error'], 'ValueError: raw activation changed')
            self.assertEqual(report['statuses'], {'proved-legacy': 1})
            self.assertEqual([v['name'] for v in report['records']], ['first', 'second'])
            self.assertEqual([v['completed'] for v in report['records']], [True, False])
            self.assertEqual(report['records'][0]['proposed_status'], 'proved-legacy')
            self.assertTrue(all((output/name).is_file() for name in ('math.bin', 'math.json', 'manifest.json')))


if __name__ == '__main__':
    unittest.main()

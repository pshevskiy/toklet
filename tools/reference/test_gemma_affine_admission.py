"""Offline checks for streamed real-checkpoint admission and independent replay."""
from fractions import Fraction
from pathlib import Path
import copy
import hashlib
import io
import math
import struct
import unittest

import gemma_affine_admission as target
import check_gemma_affine_admission as replay


class Admission(unittest.TestCase):
    def test_exact_rounding_ties_subnormal_and_overflow(self):
        self.assertEqual(replay.rounded(Fraction(1)+Fraction(1, 2**24)), 1.)
        self.assertEqual(replay.rounded(Fraction(1)+Fraction(3, 2**24)), 1.+2.**-22)
        self.assertEqual(replay.rounded(Fraction(1, 2**150)), 0.)
        self.assertEqual(replay.rounded(Fraction(3, 2**150)), 2.**-148)
        self.assertEqual(replay.rounded(Fraction(1, 2**126)), 2.**-126)
        self.assertTrue(math.isinf(replay.rounded(Fraction(2**128))))
        self.assertEqual(replay.rounded(Fraction(-3, 2**150)), -2.**-148)

    def test_exact_rounding_preserves_binary32_samples(self):
        words = [0, 1, 0x7fffff, 0x800000, 0x7f7fffff, 0x3f800001]
        words += [(i*2654435761) & 0x7f7fffff for i in range(1000)]
        for word in words:
            value = struct.unpack('<f', struct.pack('<I', word))[0]
            self.assertEqual(replay.rounded(Fraction(value)), value)
            self.assertEqual(replay.rounded(Fraction(-value)), -value)

    def test_frozen_codebooks_and_all_unused_codes(self):
        root = Path(__file__).resolve().parents[2]/'tests/fixtures/gemma-metal-affine-v1'
        data = target.decode((root/'math.json').read_bytes())
        raw = (root/'math.bin').read_bytes()
        for case in data['cases']:
            scales, biases = [raw[case['arrays'][name]['offset']:case['arrays'][name]['offset']+case['arrays'][name]['bytes']] for name in ('scales', 'biases')]
            maximum, reason = 0., None
            for (a,), (b,) in zip(struct.iter_unpack('<H', scales), struct.iter_unpack('<H', biases), strict=True):
                key = a | b << 16
                first, second = target.codebook(key), replay.coefficients(key)
                self.assertEqual(first, second)
                value, why = first
                if why:
                    reason = why
                    break
                maximum = max(maximum, value)
            self.assertEqual(reason, case['constructor']['reason'])
            self.assertEqual(None if reason else maximum, case['constructor']['max_weight'])

    def test_frozen_bounds_and_both_cutoffs(self):
        root = Path(__file__).resolve().parents[2]/'tests/fixtures/gemma-metal-affine-v1'
        data = target.decode((root/'math.json').read_bytes())
        raw = (root/'math.bin').read_bytes()
        for case in data['cases']:
            if not case['constructor']['admitted']:
                continue
            item = case['arrays']['input']
            maximum = max(abs(v) for (v,) in struct.iter_unpack('<f', raw[item['offset']:item['offset']+item['bytes']]))
            first = target.bound(case['width'], case['constructor']['max_weight'], maximum)
            self.assertEqual(first, replay.bounds(case['width'], case['constructor']['max_weight'], maximum))
            original = case['input_admission']
            self.assertEqual(first['admitted'], original['admitted'])
            self.assertEqual(first['absolute_products'], original['absolute_products_bound'])
            self.assertEqual(first['ordered_error'], original['ordered_error_bound'])

    def test_zero_weights_accept_maximum_input(self):
        self.assertEqual(target.bound(12288, 0., target.MAXIMUM), dict(admitted=True, reason=None, absolute_products=0., ordered_error=0.))
        self.assertEqual(target.bound(12288, 0., target.MAXIMUM), replay.bounds(12288, 0., target.MAXIMUM))

    def test_outward_product_boundary_underflow_and_overflow(self):
        for a, b in [(0., target.MAXIMUM), (1., 1.), (1.+2.**-23, 1.+2.**-23), (2.**-126, 2.**-23), (2.**-149, .5), (target.MAXIMUM, 2.)]:
            first = target.product_ceiling(a, b)
            self.assertEqual(first, replay.envelope(a, b))
            if first is not None:
                self.assertGreaterEqual(Fraction(first), Fraction(a)*Fraction(b))
        self.assertIsNone(target.product_ceiling(target.MAXIMUM, 2.))
        self.assertEqual(target.product_ceiling(2.**-149, .5), 2.**-149)

    def test_bounds_reject_boolean_nonfinite_and_invalid_width(self):
        for function in (target.bound, replay.bounds):
            for args in [(True, 1., 1.), (63, 1., 1.), (12352, 1., 1.), (64, True, 1.), (64, 1., math.nan), (64, 2.**33, 1.)]:
                with self.assertRaises(ValueError):
                    function(*args)
        for function in (target.product_ceiling, replay.envelope):
            for args in [(True, 1.), (math.inf, 0.), (-1., 1.)]:
                with self.assertRaises(ValueError):
                    function(*args)

    @staticmethod
    def reader(name):
        if name.endswith('context'):
            layer = int(name.split('.')[1])
            width = 512 if layer % 5 == 4 else 256
            values = [float(t+1) for h in range(8) for t in range(4) for _ in range(width)]
            return values, [1, 8, 4, width]
        width = 256 if name.endswith(('per_layer_input_gate', 'ple_input')) else 1536
        if name.endswith('input') and not name.endswith('ple_input'):
            value = 2.**24
        elif name.endswith('post_attention_layernorm'):
            value = 1.
        elif name.endswith('post_feedforward_layernorm'):
            value = -2.**24
        else:
            value = 0.
        return [value]*(4*width), [1, 4, width]

    def test_fixed_matrix_inventory_and_metadata_accounting(self):
        first = target.matrix_shapes()
        self.assertEqual(first, replay.inventory())
        self.assertEqual(len(first), 278)
        self.assertEqual(sum(r*c//64*4 for r, c in first.values()), 289267712)
        self.assertEqual(sum(r > 16384 for r, _ in first.values()), 2)

    def test_context_token_axis_and_invocation_inventory(self):
        first = list(target.inputs(self.reader))
        second = list(replay.invocation_inputs(self.reader))
        self.assertEqual(first, second)
        self.assertEqual(len(first), 277)
        self.assertEqual({k: sum(v[1] == k for v in first) for k in ('direct', 'exact-residual', 'product-envelope')}, {'direct': 172, 'exact-residual': 35, 'product-envelope': 70})
        self.assertEqual(len({v[0] for v in first}), 277)
        for name, kind, _, values in first:
            if name.endswith('self_attn.o_proj.weight'):
                self.assertEqual(values, [1., 2., 3., 4.])
            if kind == 'exact-residual':
                self.assertEqual(values, [0.]*4)  # Reassociation would give one.
        self.assertNotIn('embed_tokens_per_layer.weight', {v[0] for v in first})

    @staticmethod
    def small_trace():
        raw = struct.pack('<ff', 1., -2.)
        digest = hashlib.sha256(raw).hexdigest()
        entry = dict(contract='staged_fp32', name='test', shape=[1, 1, 2], dtype='<f4', completed=True, offset=0, bytes=8, sha256=digest)
        compact = dict(arrays=[dict(contract='staged_fp32', name='test', source_shape=[1, 1, 2], source_array_sha256=digest)])
        return dict(arrays=[entry]), compact, raw

    def test_complete_raw_ledger_and_corruptions(self):
        observation, compact, raw = self.small_trace()
        self.assertEqual(len(target.trace_ledger(observation, compact, io.BytesIO(raw))), 1)
        for field, value in [('offset', True), ('bytes', True), ('dtype', '<f8'), ('completed', 1), ('shape', [True, 1, 2]), ('sha256', '0'*64)]:
            bad = copy.deepcopy(observation)
            bad['arrays'][0][field] = value
            with self.assertRaises(ValueError):
                target.trace_ledger(bad, compact, io.BytesIO(raw))
        for changed in (raw[:-1], raw+b'x', b'12345678'):
            with self.assertRaises(ValueError):
                target.trace_ledger(observation, compact, io.BytesIO(changed))
        duplicate = copy.deepcopy(observation)
        duplicate['arrays'] *= 2
        with self.assertRaises(ValueError):
            target.trace_ledger(duplicate, compact, io.BytesIO(raw*2))
        with self.assertRaises(ValueError):
            target.trace_ledger(dict(arrays=[]), compact, io.BytesIO(b''))

    def test_typed_result_rejects_numerically_equal_corruption(self):
        valid = dict(admitted=True, maximum=0., count=1, sources=['a'])
        replay.same(copy.deepcopy(valid), valid)
        for field, value in [('admitted', 1), ('maximum', False), ('count', True), ('sources', ['b']), ('maximum', math.nan)]:
            bad = dict(valid)
            bad[field] = value
            with self.assertRaises(ValueError):
                replay.same(bad, valid)
        with self.assertRaises(ValueError):
            replay.same({**valid, 'extra': 0}, valid)

    def test_json_rejects_duplicate_and_nonfinite_fields(self):
        for raw in (b'{"a":1,"a":2}', b'{"a":NaN}', b'{"a":Infinity}'):
            with self.assertRaises(ValueError):
                target.decode(raw)


if __name__ == '__main__':
    unittest.main()

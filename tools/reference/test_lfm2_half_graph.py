"""Portable source-only controls; no export, Rust, Metal, real model or downloads."""
from pathlib import Path
import math
import struct
import unittest

import lfm2_half_graph as ref


class Controls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = Path.cwd()/'tests/fixtures/lfm2-scalar-v1'
        cls.raw = ref.bounded_bytes(cls.directory/'model.safetensors', 760078)
        cls.matrices, cls.norms, cls.convs = ref.decode(cls.raw)

    def test_direct_rounding_and_subnormal_witnesses(self):
        for integer, exponent, kind, expected in [
            (2049, -11, 'half', 0x3c00), (2051, -11, 'half', 0x3c02),
            (1, -24, 'half', 1), (1, -25, 'half', 0), (3, -25, 'half', 2),
            (-1, -25, 'half', 0x8000), (1, -149, 'f32', 1),
            (1, -150, 'f32', 0), (3, -150, 'f32', 2), (0, -173, 'half', 0)]:
            self.assertEqual(ref.round_dyadic(integer, exponent, kind), expected)
        for integer, exponent, kind in [(65520, 0, 'half'), (1, 128, 'f32')]:
            with self.assertRaises(ValueError):
                ref.round_dyadic(integer, exponent, kind)

    def test_exact_affine_avoids_fp64_double_rounding(self):
        for small, kind, expected, wrong in [(2.**-11, 'half', 0x3c01, 0x3c00),
                                            (2.**-24, 'f32', 0x3f800001, 0x3f800000)]:
            result = ref.Matrix([[1., small, 2.**-60]]).apply([[1., 1., 1.]], kind)[0][0]
            bits = ref.words.half_word if kind == 'half' else ref.words.float_word
            self.assertEqual(bits(result), expected)
            self.assertEqual(bits(math.fsum([1., small, 2.**-60])), wrong)
        self.assertEqual(ref.Matrix([[2.**100, 1., -2.**100]]).apply([[1., 1., 1.]])[0], [1.])
        ordered_fp64 = 0.
        for term in [2.**100, 1., -2.**100]:
            ordered_fp64 = ordered_fp64+term
        self.assertEqual(ordered_fp64, 0.)

    def test_exact_integer_grid_and_common_factor(self):
        for value in [0., -0., 2.**-24, -2.**-24, 1., -2., 65504.]:
            integer = ref.scaled_integer(value, 'half')
            self.assertEqual(ref.stored_dyadic(integer, -24, 'half'), value)
        matrix = ref.Matrix([[.125, -.5, 2.**-12], [0., 1., 2.**-8]])
        for source, integers in zip(matrix.values, matrix.integers, strict=True):
            self.assertEqual([v*2.**matrix.exponent for v in integers], source)
        for value, kind in [(.1, 'half'), (.1, 'f32'), (math.inf, 'half'), (math.nan, 'f32')]:
            with self.assertRaises((ValueError, OverflowError)):
                ref.scaled_integer(value, kind)

    def test_original_inventory_unaligned_rows_and_nibbles(self):
        self.assertEqual((len(self.matrices), len(self.norms), len(self.convs)), (18, 9, 2))
        header = ref.strict_json(self.raw[8:6670])
        self.assertEqual(6670 % 4, 2)
        for name, matrix in self.matrices.items():
            columns = len(matrix.values[0])
            base = name.removesuffix('.weight')
            for row, column in [(0, 0), (len(matrix.values)-1, columns-1)]:
                code_offset = 6670+header[name]['data_offsets'][0]+4*(row*(columns//8)+column//8)
                word = struct.unpack_from('<I', self.raw, code_offset)[0]
                nibble = (word >> (4*(column % 8))) & 15
                group = row*(columns//64)+column//64
                def coefficient(suffix):
                    offset = 6670+header[base+'.'+suffix]['data_offsets'][0]+2*group
                    bits = struct.unpack_from('<H', self.raw, offset)[0]
                    return struct.unpack('<f', struct.pack('<I', bits << 16))[0]
                expected = ref.f32(ref.f32(nibble*coefficient('scales'))+coefficient('biases'))
                self.assertEqual(matrix.values[row][column], expected)

    def test_direct_norm_and_embedding_lookup_has_no_norm(self):
        gains = [0., -.5, 1.5]+[1.]*(ref.D-3)
        source = [[-1., -.75, -.5]+[.25]*(ref.D-3)]
        for profile in ('main', 'wide'):
            result = ref.norm(source, gains, profile)[0]
            self.assertEqual(ref.words.half_word(result[0]), 0x8000)
            self.assertGreater(result[1], 0.)
            self.assertLess(result[2], 0.)
        original = self.matrices['model.embed_tokens.weight'].values[1]
        expected = [[ref.half(value) for value in original]]
        captures = ref.forward(self.matrices, self.norms, self.convs, [1], 'main')
        self.assertEqual(captures['embedding_lookup'], expected[0])
        self.assertNotEqual(expected, ref.norm(expected, self.norms['model.embedding_norm.weight'], 'main'))

    def test_asymmetric_conv_and_raw_history(self):
        u = [[0.]*ref.H for _ in range(3)]
        for row, values in zip(u, [(2., -3.), (6., -2.), (12., 4.)], strict=True):
            row[:2] = values
        taps = [0.]*(3*ref.H)
        taps[:6] = [1., 10., 100., 3., -2., 5.]
        for profile in ('main', 'wide'):
            output, before, after = ref.convolution(u, taps, profile)
            self.assertEqual([row[:2] for row in output], [[200., -15.], [620., -4.], [1262., 15.]])
            self.assertEqual(before[0], [[0.]*ref.H, [0.]*ref.H])
            self.assertEqual(after[-1], u[-2:])
            self.assertEqual(before[1], after[0])
            self.assertNotEqual(after[-1], output[-2:])

    def test_separate_pointwise_stores_and_invalid_zero(self):
        witness = ref.separate_store_witness()
        self.assertEqual(witness['u_conv_cz_affine_words'], [0x3c02, 0x3d02, 0x3d03, 0x3f84])
        self.assertEqual([witness['missing_u_store_conv_word'], witness['missing_conv_store_cz_word'],
                          witness['missing_cz_store_affine_word']], [0x3d03, 0x3d04, 0x3f85])
        value = ref.words.half_value(0x3c01)
        stored = ref.product([[value]], [[value]])[0][0]
        self.assertEqual(ref.words.half_word(stored), 0x3c02)
        self.assertNotEqual(stored, value*value)
        for profile in ('main', 'wide'):
            self.assertEqual(ref.silu([[65504.]], profile), [[65504.]])
            self.assertEqual(ref.words.half_word(ref.silu([[-65504.]], profile)[0][0]), 0x8000)
            self.assertEqual(ref.residual([[1.]], [[-1.]], profile), [[0.]])
        for left, right in [(math.nan, 0.), (0., math.inf), (65504., 2.)]:
            with self.assertRaises((ValueError, OverflowError)):
                ref.product([[left]], [[right]])

    def test_rotary_high_precision_and_absolute_offsets(self):
        frequencies = ref.inverse_frequencies()
        self.assertEqual((len(frequencies), frequencies[0], frequencies[16]), (32, 1., .001))
        inputs = [[ref.half(((r*7+c*11) % 31-15)/16) for c in range(ref.D)] for r in range(3)]
        for profile in ('main', 'wide'):
            full = ref.rotary(inputs, 1, 9, profile)
            self.assertEqual(full, [ref.rotary([row], 1, 9+i, profile)[0] for i, row in enumerate(inputs)])
            self.assertEqual(ref.rotary([inputs[0]], 1, 0, profile), [inputs[0]])
        with self.assertRaises(ValueError):
            ref.rotary(inputs, 1, 10, 'main')

    def test_grouped_heads_masks_and_observable_future_scores(self):
        query = [[.25]*(ref.Q*ref.D) for _ in range(2)]
        keys = [[.125]*(ref.KV*ref.D), [.5]*(ref.KV*ref.D)]
        values = [[.25]*ref.D+[.5]*ref.D for _ in range(2)]
        changed = [keys[0], [.75]*(ref.KV*ref.D)]
        for profile in ('main', 'wide'):
            scores, probs, context = ref.attention(query, keys, values, profile)
            other_scores, other_probs, other_context = ref.attention(query, changed, values, profile)
            self.assertNotEqual(scores[1], other_scores[1])
            self.assertEqual(context[0], other_context[0])
            for head in range(ref.Q):
                self.assertEqual(probs[head*4:head*4+2], [1., 0.])
                self.assertEqual(probs[head*4:head*4+2], other_probs[head*4:head*4+2])
                self.assertEqual(ref.words.float_word(probs[head*4+1]), 0)
                self.assertEqual(context[0][head*ref.D], .25 if head < 2 else .5)

    def test_complete_single_stage_inventory_and_storage(self):
        for profile in ('main', 'wide'):
            captures = ref.forward(self.matrices, self.norms, self.convs, [1], profile)
            self.assertEqual(len(captures), 63)
            self.assertEqual(len(captures['logits']), 64)
            self.assertEqual(len(captures['layers.1.kv_keys_after']), 2*12*64)
            for name, values in captures.items():
                kind = 'f32' if name == 'logits' or name.endswith(('attention_scores', 'attention_probabilities')) else 'half'
                for value in values:
                    self.assertTrue(math.isfinite(value))
                    ref.scaled_integer(value, kind)

    def test_invalid_tokens_profiles_shapes_and_alias_independent_state(self):
        for tokens in [[], [True], [-1], [64], [0]*13]:
            with self.assertRaises(ValueError):
                ref.forward(self.matrices, self.norms, self.convs, tokens, 'main')
        with self.assertRaises(ValueError):
            ref.forward(self.matrices, self.norms, self.convs, [1], 'bad')
        with self.assertRaises(ValueError):
            ref.Matrix([[1., 2.], [3.]])
        with self.assertRaises(ValueError):
            ref.Matrix([[1., 2.]]).apply([[1.]])
        with self.assertRaises(ValueError):
            ref.product([[1.], [1., 2.]], [[1.], [1., 2.]])
        with self.assertRaises(ValueError):
            ref.rotary([[]], 0, 0, 'main')
        snapshots = ref.snapshots([[.5]*(ref.KV*ref.D), [.75]*(ref.KV*ref.D)])
        snapshots[1][0][0][0] = -1.
        self.assertEqual(snapshots[0][0][0][0], .5)

    def test_original_identity_and_strict_json_rejection(self):
        changed = bytearray(self.raw)
        changed[-1] ^= 1
        with self.assertRaises(ValueError):
            ref.decode(bytes(changed))
        for raw in [b'{"a":1,"a":2}', b'{"a":NaN}']:
            with self.assertRaises(ValueError):
                ref.strict_json(raw)
        self.assertEqual((len(ref.original_shapes()), len(self.raw)), (29, 760078))


if __name__ == '__main__':
    unittest.main()

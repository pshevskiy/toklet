"""Portable prospective fixture controls; no graph export or target execution."""
import importlib.util
import json
import math
from pathlib import Path
import struct
import unittest

spec = importlib.util.spec_from_file_location('phi3_graph', Path(__file__).with_name('phi3_graph.py'))
subject = importlib.util.module_from_spec(spec)
spec.loader.exec_module(subject)


class Controls(unittest.TestCase):
    def test_frozen_shapes_cases_and_gates(self):
        self.assertEqual(subject.HALF_POLICY, dict(atol_floor=1e-4, envelope_factor=2, rtol=1e-3))
        self.assertEqual(subject.CPU_GATE, dict(atol=1e-5, rtol=1e-4))
        self.assertEqual(len(subject.CASES), 5)
        self.assertEqual(subject.CASES[0][2][:-1], subject.CASES[1][2][:-1])
        self.assertEqual((subject.CASES[0][2][-1], subject.CASES[1][2][-1]), (53, 59))
        self.assertEqual(subject.CASES[3][3] + len(subject.CASES[3][2]) - 1, 511)
        self.assertEqual(subject.CASES[-1], ('tied', 'tied_control', (1, 7, 13), 0))
        self.assertEqual(sum(len(subject.contracts(len(c[2]))) for c in subject.CASES), 195)
        for n in (3, 17):
            plan = subject.contracts(n)
            self.assertEqual(len(plan), 39)
            self.assertEqual(plan['layers.1.attention_scores'], [2, n, n])
            self.assertEqual(plan['layers.0.q_rotary'], [n, 2, 96])
            self.assertEqual(plan['layers.1.gate_projection'], [n, 256])
            self.assertEqual(plan['logits'], [n, 64])
        c = subject.configuration(False)
        self.assertEqual((c['max_position_embeddings'], c['sliding_window']), (4096, 2047))
        self.assertIsNone(c['rope_scaling'])

    def test_rounding_and_ordered_profile_controls(self):
        values = [2**24, 1, -(2**24)]
        self.assertEqual(subject.dot(values, [1, 1, 1], subject.f32), 0)
        self.assertEqual(subject.dot(values, [1, 1, 1], float), 1)
        self.assertEqual(struct.pack('<e', subject.half(-0.0)), b'\x00\x80')
        self.assertEqual(subject.half(1 + 2**-11), 1)
        self.assertEqual(subject.f32(1 + 2**-24), 1)
        with self.assertRaises(ValueError):
            subject.dot([1], [], float)
        with self.assertRaises(ValueError):
            subject.forward({}, False, [1] * 17, 496, 'main')

    def test_literal_packing_and_distinct_coefficient_rows(self):
        codes, scales, biases = subject.packed_matrix(3, 64, 7)
        expected_first = sum(((7 + 7 * lane) % 16) << (4 * lane) for lane in range(8))
        self.assertEqual(struct.unpack('<I', codes[:4])[0], expected_first)
        self.assertEqual((len(codes), len(scales), len(biases)), (96, 6, 6))
        self.assertNotEqual(scales[:2], scales[2:4])
        self.assertNotEqual(codes[:32], codes[32:64])
        scale = struct.unpack('<e', scales[:2])[0]
        bias = struct.unpack('<e', biases[:2])[0]
        reconstructed = [subject.f32(subject.f32(scale * code) + bias) for code in range(16)]
        self.assertTrue(any(v != subject.half(v) for v in reconstructed))
        with self.assertRaises(ValueError):
            subject.packed_matrix(1, 96, 0)

    def test_original_fused_rows_and_unaligned_artifact(self):
        # Source metadata/payload generation only: no forward graph/export here.
        original = {}
        for tied in (False, True):
            raw = subject.make_artifact(tied)
            physical, start = subject.decode_physical(raw, tied)
            self.assertEqual(start % 4, 1)
            self.assertEqual(len(physical), 14 if tied else 15)
            weights = subject.split_original(physical)
            self.assertEqual(len(weights), 20 if tied else 21)
            self.assertEqual('lm_head.weight' in weights, not tied)
            for layer in range(2):
                p = f'model.layers.{layer}'
                fused = physical[p + '.self_attn.qkv_proj.weight']
                for name, begin in [('q', 0), ('k', 192), ('v', 384)]:
                    split = weights[p + f'.self_attn.{name}_proj.weight']
                    self.assertEqual(split[0], fused[begin])
                    self.assertEqual(split[-1], fused[begin + 191])
                    self.assertIs(split[0], fused[begin])
                gate_up = physical[p + '.mlp.gate_up_proj.weight']
                self.assertEqual(weights[p + '.mlp.gate_proj.weight'][-1], gate_up[255])
                self.assertEqual(weights[p + '.mlp.up_proj.weight'][0], gate_up[256])
                self.assertNotEqual(gate_up[255], gate_up[256])
            embedding = weights['model.embed_tokens.weight']
            self.assertTrue(any(v != subject.half(v) for row in embedding for v in row))
            original[tied] = embedding
            with self.assertRaises(ValueError):
                subject.decode_physical(raw[:-1], tied)
            with self.assertRaises(ValueError):
                subject.decode_physical(raw + b'\x00', tied)
            # Same-size metadata corruption proves decode does not merely trust
            # the generator's known configuration or physical inventory.
            header_size = struct.unpack('<Q', raw[:8])[0]
            header = raw[8:8 + header_size]
            self.assertIn(b'"U32"', header)
            bad = raw[:8] + header.replace(b'"U32"', b'"F32"', 1) + raw[8 + header_size:]
            with self.assertRaises(ValueError):
                subject.decode_physical(bad, tied)
        self.assertEqual(original[False], original[True])

    def test_head96_split_half_rotary_independent_frequency_control(self):
        x = [[0.0] * 192]
        # Pair24 inverse =10000^(-1/2)=.01, in both heads.
        for head in range(2):
            x[0][head * 96 + 24] = 1.0
        for profile in subject.PROFILES:
            self.assertEqual(subject.rotary(x, 2, 0, profile), x)
            output = subject.rotary(x, 2, 5, profile)[0]
            store = subject.f32 if profile == 'cpu' else subject.half
            for head in range(2):
                self.assertAlmostEqual(output[head * 96 + 24], store(math.cos(.05)), delta=2**-10)
                self.assertAlmostEqual(output[head * 96 + 72], store(math.sin(.05)), delta=2**-14)
            self.assertNotEqual(subject.rotary(x, 2, 511, profile), subject.rotary(x, 2, 5, profile))
        with self.assertRaises(ValueError):
            subject.rotary([[0.0] * 95], 1, 0, 'main')

    def test_generic_gqa_causal_probabilities_and_own_value_heads(self):
        q, k = [[0.0] * 384 for _ in range(3)], [[0.0] * 192 for _ in range(3)]
        v = [[2.0] * 96 + [-3.0] * 96 for _ in range(3)]
        for precision, store in ((subject.f32, subject.half), (float, subject.half), (float, subject.f32)):
            scores, probabilities, output = subject.attention(q, k, v, precision, store, 4, 2, 96)
            self.assertTrue(all(value == 0 for value in subject.flatten(scores)))
            self.assertEqual(probabilities[0][0], [1, 0, 0])
            self.assertEqual(probabilities[0][2], [subject.f32(1 / 3)] * 3)
            self.assertEqual(output[0], [2] * 192 + [-3] * 192)
            changed = [row[:] for row in v]
            changed[2] = [100] * 192
            changed_output = subject.attention(q, k, changed, precision, store, 4, 2, 96)[2]
            self.assertEqual(changed_output[:2], output[:2])
            self.assertNotEqual(changed_output[2], output[2])


if __name__ == '__main__':
    unittest.main()

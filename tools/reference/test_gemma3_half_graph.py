"""Portable controls; only execute after independent prospective source review."""
import copy
import math
from pathlib import Path
import struct
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gemma3_half_graph as subject


class Controls(unittest.TestCase):
    def test_fixed_contract_and_input_identities(self):
        self.assertEqual(subject.POLICY, dict(fatal_reference='main', atol_floor=1e-4, envelope_factor=2, rtol=1e-3))
        self.assertEqual(subject.CASES, subject.source.CASES)
        self.assertEqual(subject.POSITION_CAPACITY, 544)
        for _, tokens in subject.CASES:
            shapes = subject.contracts(len(tokens))
            self.assertEqual(len(shapes), 48)
            self.assertEqual(shapes, subject.source.contracts(len(tokens)))
            self.assertEqual(list(shapes)[:2], ['embedding_lookup', 'embedding'])
            self.assertEqual(list(shapes)[-2:], ['final_norm', 'logits'])
        subject.dependencies()
        fixture = Path(__file__).resolve().parents[2] / 'tests/fixtures/gemma3-cpu-v1'
        raw = subject.authenticate(fixture)
        matrices, gamma, effective, _ = subject.source.decode(raw['model.safetensors'])
        self.assertEqual((len(matrices), len(gamma)), (15, 13))
        for name in gamma:
            self.assertEqual(effective[name], [subject.f32(1 + v) for v in gamma[name]])

    def test_separate_projection_rounds_and_tied_head_store(self):
        # Main is explicitly serial separately-rounded FP32; wide is diagnostic.
        x, weights = [[1., 1., 1.]], [[2**24, 1., -(2**24)]]
        self.assertEqual(subject.linear(x, weights, 'main', True), [[0.]])
        self.assertEqual(subject.linear(x, weights, 'wide', True), [[1.]])
        value = 1 + 2**-11
        self.assertEqual(subject.linear([[1.]], [[value]], 'main'), [[1.]])
        self.assertEqual(subject.linear([[1.]], [[value]], 'main', True), [[value]])
        for profile in ('main', 'wide'):
            with self.assertRaises(ValueError):
                subject.linear([[1., 2.]], [[1.]], profile)
        with self.assertRaises(ValueError):
            subject.linear([[1.]], [[1.]], 'other')

    def test_existing_exact_coefficient_and_store_witnesses(self):
        words, _, scales, stride = subject.rms_source.operands(1, 32, 'rounding')
        x = [[subject.half_value(v) for v in words[:32]]]
        for profile in ('main', 'wide'):
            self.assertEqual(subject.half_word(subject.norm(x, scales, profile)[0][0]), 0x3fa7)
            self.assertEqual(subject.pointwise.gelu_product(0x3400, 0x4500, profile), (0x30ca, 0x39fc))
            self.assertEqual(subject.pointwise.gelu_product(0x0002, 0x4500, profile), (0x0001, 0x0005))
            self.assertEqual(subject.pointwise.gelu_product(0x8002, 0x4500, profile), (0x8001, 0x8005))
            self.assertEqual(subject.pointwise.scale_word(0x0001, 8., profile), 8)
            self.assertEqual(subject.pointwise.scale_word(0x8001, 8., profile), 0x8008)
        self.assertEqual(subject.pointwise.ordering_controls()['scaled_half_word'], 0x4e50)
        narrowed = [subject.half(v) for v in scales]
        self.assertEqual(subject.half_word(subject.norm(x, narrowed, 'main')[0][0]), 0x3fa6)

    def test_attention_one_key_and_local_visibility(self):
        q, k, v = [[0.] * 128], [[0.] * 32], [[(i - 16) / 32 for i in range(32)]]
        for profile in ('main', 'wide'):
            scores, probs, context = subject.attention(q, k, v, 4, profile)
            self.assertEqual(scores, [0.] * 4)
            self.assertEqual(probs, [1.] * 4)
            self.assertEqual(context, [v[0] * 4])
            q, k = [[0.] * 128 for _ in range(5)], [[0.] * 32 for _ in range(5)]
            v = [[0.] * 32 for _ in range(5)]
            v[0][0], v[1][0], v[4][0] = 8., 4., 2.
            _, probs, context = subject.attention(q, k, v, 4, profile)
            for head in range(4):
                self.assertEqual(probs[(head * 5 + 4) * 5:(head * 5 + 5) * 5], [0., .25, .25, .25, .25])
                self.assertEqual(context[4][head * 32], 1.5)
            # Restore independent one-key operands for the next profile.
            q, k, v = [[0.] * 128], [[0.] * 32], [[(i - 16) / 32 for i in range(32)]]

    def test_rotary_coverage_math_and_position_domain(self):
        cases = list(subject.rotary_components())
        self.assertEqual(len(cases), 8)
        self.assertEqual(sum(len(c['main']) for c in cases), 11520)
        for case in cases:
            self.assertEqual(case['positions'], [0, 511, 512, 513])
            first = case['heads'] * case['dim']
            for profile in ('main', 'wide'):
                self.assertEqual(case[profile][:first], case['input'][:first])
                self.assertTrue(all(math.isfinite(x) and abs(x - y) <= bound
                    for x, y, bound in zip(case[profile], case['mathematical'], case['math_bound'], strict=True)))
        for profile in ('main', 'wide'):
            subject.rotary([[1.] * 32], 1, 32, 10000, [543], profile)
            for position in (-1, 544, True):
                with self.assertRaises(ValueError):
                    subject.rotary([[1.] * 32], 1, 32, 10000, [position], profile)

    def test_main_only_gate_and_finiteness(self):
        self.assertEqual(subject.gate([1.], [1.])['gate'], dict(atol=1e-4, rtol=1e-3))
        self.assertEqual(subject.gate([1.], [1.125])['gate'], dict(atol=.25, rtol=1e-3))
        for main, wide in (([], []), ([math.nan], [1.]), ([1.], [math.inf]), ([1.], [])):
            with self.assertRaises(ValueError):
                subject.gate(main, wide)
        for value in (math.inf, math.nan):
            with self.assertRaises(ValueError):
                subject.half(value)

    def test_graph_causality_stage_stores_and_immutability(self):
        fixture = Path(__file__).resolve().parents[2] / 'tests/fixtures/gemma3-cpu-v1'
        raw = subject.authenticate(fixture)
        matrices, gamma, effective, _ = subject.source.decode(raw['model.safetensors'])
        before = copy.deepcopy((matrices, gamma, effective))
        for profile in ('main', 'wide'):
            prior = None
            for name, tokens in subject.CASES:
                arrays = subject.forward(matrices, effective, tokens, profile)
                self.assertEqual(list(arrays), list(subject.contracts(len(tokens))))
                for stage, values in arrays.items():
                    self.assertTrue(all(math.isfinite(v) for v in values))
                    if stage != 'logits' and not stage.endswith(('attention_scores', 'attention_probabilities')):
                        self.assertTrue(all(subject.half(v) == v for v in values))
                if name == 'single':
                    for layer in range(2):
                        v = arrays[f'layers.{layer}.v_projection']
                        self.assertEqual(arrays[f'layers.{layer}.attention_context'], v * 4)
                if name == 'wrap12':
                    prior = arrays['logits'][:11 * 64]
                if name == 'future_changed12':
                    self.assertEqual(arrays['logits'][:11 * 64], prior)
            for tokens in ([], [64], [True], list(range(13))):
                with self.assertRaises(ValueError):
                    subject.forward(matrices, effective, tokens, profile)
        self.assertEqual((matrices, gamma, effective), before)


if __name__ == '__main__':
    unittest.main()

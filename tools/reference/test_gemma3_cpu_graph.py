"""Prospective portable source/parser/operator controls; no target graph calls."""
import importlib.util
import json
import math
from pathlib import Path
import struct
import unittest

spec = importlib.util.spec_from_file_location('gemma3_cpu_graph', Path(__file__).with_name('gemma3_cpu_graph.py'))
subject = importlib.util.module_from_spec(spec)
spec.loader.exec_module(subject)


class Controls(unittest.TestCase):
    def test_frozen_protocol_shapes_and_gate(self):
        self.assertEqual(subject.CPU_GATE, dict(atol=1e-5, rtol=1e-4))
        self.assertEqual([len(t) for _, t in subject.CASES], [1, 4, 12, 12])
        self.assertEqual(subject.CASES[2][1][:-1], subject.CASES[3][1][:-1])
        self.assertEqual((subject.CASES[2][1][-1], subject.CASES[3][1][-1]), (10, 19))
        self.assertEqual(sum(len(subject.contracts(len(t))) for _, t in subject.CASES), 192)
        c = subject.configuration()
        self.assertEqual((c['hidden_size'], c['num_attention_heads'] * c['head_dim']), (64, 128))
        self.assertEqual((c['sliding_window'], c['max_position_embeddings']), (4, 12))
        self.assertEqual(subject.contracts(12)['layers.1.attention_scores'], [4, 12, 12])
        self.assertEqual(subject.contracts(12)['layers.0.k_norm'], [12, 1, 32])
        for tokens in ([], [64], [-1], [True], [1] * 13):
            with self.assertRaises(ValueError):
                subject.forward({}, {}, tokens)

    def test_bf16_rounding_offset_gamma_and_embedding(self):
        self.assertEqual(subject.bf16_bits(1 + 2**-8), 0x3f80)
        self.assertEqual(subject.bf16_bits(1 + 3 * 2**-8), 0x3f82)
        self.assertEqual(subject.bf16_bits(-0.0), 0x8000)
        self.assertEqual(subject.embedding_scale(64), 8)
        self.assertEqual(subject.embedding_scale(640), 25.25)
        self.assertNotEqual(subject.embedding_scale(640), subject.f32(math.sqrt(640)))
        gamma = [subject.bf16_value(v) for v in (0xbf80, 0, 0x3f80, 0xbf7f, 0xbf81)]
        effective = [subject.f32(1 + v) for v in gamma]
        self.assertEqual(effective, [0, 1, 2, 1 / 256, -1 / 128])
        x = [1., 2., 3., 4., 5.]
        actual = subject.norm([x], effective)[0]
        inverse = 1 / math.sqrt(11 + subject.f32(1e-6))
        self.assertEqual(actual, [subject.f32(v * inverse * w) for v, w in zip(x, effective)])
        for value in (math.inf, math.nan):
            with self.assertRaises(ValueError):
                subject.bf16_bits(value)
        with self.assertRaises(ValueError):
            subject.bf16_value(0x7fc0)

    def test_literal_packing_and_original_single_owner(self):
        codes, scales, biases = subject.packed_matrix(3, 64, 7)
        self.assertEqual(struct.unpack('<I', codes[:4])[0], sum(((7 + 7 * lane) % 16) << (4 * lane) for lane in range(8)))
        self.assertEqual((len(codes), len(scales), len(biases)), (96, 6, 6))
        self.assertNotEqual(codes[:32], codes[32:64])
        self.assertNotEqual(scales[:2], scales[2:4])
        raw = subject.make_artifact()
        matrices, gamma, effective, start = subject.decode(raw)
        self.assertEqual((len(matrices), len(gamma), len(effective), start % 4), (15, 13, 13, 2))
        self.assertNotIn('lm_head.weight', matrices)
        self.assertEqual(len(matrices['model.layers.0.self_attn.q_proj.weight']), 128)
        self.assertEqual(len(matrices['model.layers.0.self_attn.o_proj.weight'][0]), 128)
        for name, values in gamma.items():
            self.assertEqual(effective[name], [subject.f32(1 + v) for v in values])
        name = 'model.embed_tokens.weight'
        seed = int.from_bytes(subject.hashlib.sha256(name.encode()).digest()[:4], 'little')
        code = (seed + 11 * 2 + 7 * 17 + 3 * (17 // 8)) % 16
        scale = subject.bf16_value(0x3b00 + (seed + 3 * 2) % 97)
        bias = subject.bf16_value(subject.bf16_bits(-7.5 * scale + ((2 + seed) % 7 - 3) / 4096))
        self.assertEqual(matrices[name][2][17], subject.f32(subject.f32(scale * code) + bias))
        with self.assertRaises(ValueError):
            subject.packed_matrix(1, 32, 0)

    def test_parser_rejects_corruption_before_graph(self):
        raw = subject.make_artifact()
        for bad in (raw[:7], raw[:-1], raw + b'\0'):
            with self.assertRaises(ValueError):
                subject.decode(bad)
        size = struct.unpack('<Q', raw[:8])[0]
        header = json.loads(raw[8:8 + size])
        def rebuild(h):
            text = json.dumps(h, separators=(',', ':')).encode()
            return struct.pack('<Q', len(text)) + text + raw[8 + size:]
        first = next(iter(header))
        changed = json.loads(json.dumps(header))
        changed[first]['dtype'] = 'F32'
        with self.assertRaises(ValueError):
            subject.decode(rebuild(changed))
        changed = json.loads(json.dumps(header))
        changed[first]['data_offsets'][0] += 1
        with self.assertRaises(ValueError):
            subject.decode(rebuild(changed))
        changed = json.loads(json.dumps(header))
        changed[first]['shape'][0] = True
        with self.assertRaises(ValueError):
            subject.decode(rebuild(changed))
        with self.assertRaises(ValueError):
            subject.strict_json('{"x":1,"x":2}')
        norm = next(name for name in header if len(header[name]['shape']) == 1)
        begin = 8 + size + header[norm]['data_offsets'][0]
        bad = raw[:begin] + struct.pack('<H', 0x7fc0) + raw[begin + 2:]
        with self.assertRaises(ValueError):
            subject.decode(bad)

    def test_rotary_head_separation_and_theta(self):
        row = [0.0] * 128
        row[8], row[32 + 8] = 1.0, 2.0
        x = [row, row]
        local = subject.rotary(x, 4, 32, 10000)
        global_ = subject.rotary(x, 4, 32, 1000000)
        self.assertEqual(local[0], row)
        self.assertEqual(local[1][8], subject.f32(math.cos(.01)))
        self.assertEqual(local[1][24], subject.f32(math.sin(.01)))
        self.assertEqual(local[1][40], subject.f32(2 * math.cos(.01)))
        self.assertEqual(global_[1][24], subject.f32(math.sin(.001)))
        self.assertNotEqual(local[1], global_[1])
        with self.assertRaises(ValueError):
            subject.rotary([[0.] * 31], 1, 32, 10000)

    def test_window_gqa_and_real_boundary_components(self):
        q, k = [[0.] * 128 for _ in range(6)], [[0.] * 32 for _ in range(6)]
        v = [[float(i)] * 32 for i in range(6)]
        _, probabilities, values = subject.attention(q, k, v, 4, 1, 32, 4, subject.f32(1 / math.sqrt(32)))
        self.assertEqual(probabilities[0][5], [0, 0, .25, .25, .25, .25])
        self.assertEqual(values[5], [3.5] * 128)
        c = subject.components()
        w = c['visibility']
        self.assertEqual(w['local_probabilities'][0], 0)
        self.assertEqual(w['local_probabilities'][1], 1 / 512)
        self.assertEqual(w['local_probabilities'][512], 1 / 512)
        self.assertGreater(w['global_probabilities'][0], 0)
        self.assertEqual(w['local_context'][0], w['expected_local_first'])
        self.assertEqual(w['global_context'][0], w['expected_global_first'])
        self.assertEqual(w['local_context'][1:], [0.] * 255)
        self.assertNotEqual(w['local_context'], w['global_context'])
        with self.assertRaises(ValueError):
            subject.attention(q, k, v, 4, 1, 32, 0, 1.)


if __name__ == '__main__':
    unittest.main()

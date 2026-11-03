import copy
import math
import unittest
import numpy as np
import llama3_original_fp32 as subject


class TinyReader:
    shape = subject.Shape(vocab=11, hidden=8, intermediate=12, layers=2, heads=2, kv_heads=1, head_dim=4)
    def __init__(self):
        self.values = {}
        for name, dims in subject.expected_tensors(self.shape).items():
            value = ((np.arange(math.prod(dims), dtype=np.float32) % 17) - 8) / np.float32(64)
            self.values[name] = np.ones(dims, dtype=np.float32) if 'norm' in name else value.reshape(dims)
    def weight(self, name, start=None, stop=None):
        value = self.values[name]
        return value if start is None else value[start:stop]
    def embedding(self, ids):
        return self.values['model.embed_tokens.weight'][ids].copy()


class OriginalTests(unittest.TestCase):
    def test_bf16_exact_bits_and_nonfinite_rejection(self):
        words = np.array([0, 0x3f80, 0xc000, 0x8000, 1, 0x7f7f], dtype='<u2')
        expected = np.array([0, 0x3f800000, 0xc0000000, 0x80000000, 0x10000, 0x7f7f0000], dtype=np.uint32)
        np.testing.assert_array_equal(subject.widen(words).view(np.uint32), expected)
        for value in [0x7f80, 0x7fc0]:
            with self.assertRaises(ValueError):
                subject.widen(np.array([value], dtype='<u2'))
        with self.assertRaises(ValueError):
            subject.widen(np.array([1.0], dtype=np.float32))

    def test_inventory_rejects_dtype_extent_and_overlap(self):
        reader = TinyReader(); header = {'__metadata__': {'format': 'pt'}}; end = 0
        for name, dims in subject.expected_tensors(reader.shape).items():
            stop = end + math.prod(dims)*2
            header[name] = {'dtype': 'BF16', 'shape': dims, 'data_offsets': [end, stop]}; end = stop
        subject.validate_header(header, end, reader.shape)
        name = 'model.embed_tokens.weight'
        for field, value in [('dtype', 'F16'), ('shape', [11, 7]), ('data_offsets', [1, 177])]:
            changed = copy.deepcopy(header); changed[name][field] = value
            with self.assertRaises(ValueError):
                subject.validate_header(changed, end, reader.shape)
        changed = copy.deepcopy(header); del changed[name]
        with self.assertRaises(ValueError):
            subject.validate_header(changed, end, reader.shape)

    def test_complete_graph_prefix_causality_and_tied_head_tiles(self):
        reader = TinyReader(); ids = [1, 2, 3, 4]
        batch = subject.forward(reader, ids, tile=3)
        self.assertEqual(batch.dtype, np.float32)
        for n in [1, 2, 4]:
            prefix = subject.forward(reader, ids[:n], tile=11)
            np.testing.assert_allclose(batch[n-1], prefix[-1], atol=1e-5, rtol=1e-5)
        changed = subject.forward(reader, [1, 2, 3, 6], tile=3)
        np.testing.assert_allclose(batch[:3], changed[:3], atol=1e-6, rtol=1e-6)
        self.assertGreater(float(np.max(np.abs(batch[-1] - changed[-1]))), 1e-4)

    def test_rotary_matches_independent_decimal_oracle_at_nonzero_positions(self):
        import llama3_rotary
        case = next(c for c in llama3_rotary.export()['cases'] if c['name'] == 'actual_short')
        values = np.array(case['input'], dtype=np.float32).reshape(3,2,64)
        padded = np.concatenate((np.zeros((5,2,64), dtype=np.float32), values), axis=0)
        expected = np.array(case['output'], dtype=np.float64).reshape(3,2,64)
        np.testing.assert_allclose(subject.rotary(padded)[5:], expected, atol=3e-6, rtol=1e-6)

    def test_grouped_attention_matches_scalar_causal_oracle(self):
        q = np.arange(24, dtype=np.float32).reshape(3,2,4)/np.float32(32)
        k = np.arange(12, dtype=np.float32).reshape(3,1,4)/np.float32(16)
        v = np.arange(12, dtype=np.float32).reshape(3,1,4)/np.float32(8)
        expected = np.zeros_like(q)
        for row in range(3):
            for head in range(2):
                scores = [sum(float(q[row,head,d])*float(k[j,0,d]) for d in range(4))/2 for j in range(row+1)]
                weights = [math.exp(s-max(scores)) for s in scores]
                for d in range(4):
                    expected[row,head,d] = sum(w*float(v[j,0,d]) for j,w in enumerate(weights))/sum(weights)
        np.testing.assert_allclose(subject.attention(q,k,v), expected, atol=1e-6, rtol=1e-6)

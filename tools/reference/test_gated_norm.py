"""Portable gated output normalization witnesses; no Apple runtime imported."""
import copy
import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import types
import sys

import numpy as np

import gated_norm as oracle


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


class GatedNorm(unittest.TestCase):
    def test_hand_case_fixes_mean_epsilon_direct_weight_and_gate_order(self):
        h = np.array([[[3.0, 4.0]]], np.float32)
        g = np.array([[[0.0, 1.0]]], np.float32)
        w = np.array([2.0, -0.5], np.float32)
        before = [x.copy() for x in (h, g, w)]
        normalized, activated, output = oracle.math_values(h, g, w, 1.0)
        expected = np.array([[[6 / math.sqrt(13.5), -2 / math.sqrt(13.5)]]])
        np.testing.assert_allclose(normalized, expected, rtol=1e-14, atol=1e-14)
        np.testing.assert_allclose(activated, [[[0.0, 1 / (1 + math.exp(-1))]]], rtol=1e-14, atol=1e-14)
        np.testing.assert_allclose(output, expected * activated, rtol=1e-14, atol=1e-14)
        for actual, expected in zip((h, g, w), before):
            np.testing.assert_array_equal(actual, expected)
        # Normalizing after gating is a different operation, not this witness.
        gated = (h * activated).astype(np.float32)
        reordered = oracle.math_values(gated, np.ones_like(g), w, 1.0)[0]
        self.assertGreater(np.max(np.abs(reordered - output)), 0.1)

    def test_epsilon_dominated_heads_distinguish_zero_and_sum_conventions(self):
        h = np.full((1, 2, 3), 1e-4, np.float32)
        norm, _, _ = oracle.math_values(h, np.ones_like(h), np.ones(3, np.float32), 1e-6)
        x = float(h.flat[0])
        expected = x / math.sqrt(x * x + 1e-6)
        np.testing.assert_allclose(norm, expected, rtol=1e-14, atol=1e-14)
        self.assertGreater(expected, 0.09)
        self.assertGreater(abs(expected - x / math.sqrt(3 * x * x + 1e-6)), 1e-4)

    def test_empty_and_stable_gate_saturation_and_tiny_nonzero(self):
        result = oracle.math_values(np.empty((0, 2, 3), np.float32), np.empty((0, 2, 3), np.float32), np.ones(3, np.float32), 1e-6)
        for value in result:
            self.assertEqual(value.shape, (0, 2, 3))
        h = np.ones((1, 1, 3), np.float32)
        g = np.array([[[-1000, -24, 1000]]], np.float32)
        _, activated, output = oracle.math_values(h, g, np.ones(3, np.float32), 1e-6)
        self.assertEqual(activated[0, 0, 0], 0)
        self.assertEqual(activated[0, 0, 2], 1000)
        self.assertLess(activated[0, 0, 1], 0)
        self.assertAlmostEqual(activated[0, 0, 1] / (-24 * math.exp(-24) / (1 + math.exp(-24))), 1.0, places=14)
        self.assertTrue(np.isfinite(output).all())

    def test_rows_and_head_chunks_are_independent(self):
        h = np.arange(30, dtype=np.float32).reshape(2, 3, 5) * np.float32(0.125)
        g = h - np.float32(2)
        w = np.array([2, -1, 0.5, 3, -2], np.float32)
        full = oracle.math_values(h, g, w, 1e-6)
        for token in range(2):
            for head in range(3):
                single = oracle.math_values(h[token:token+1, head:head+1], g[token:token+1, head:head+1], w, 1e-6)
                for actual, expected in zip(single, full):
                    np.testing.assert_array_equal(actual[0, 0], expected[token, head])

    def test_invalid_input_shapes_types_finiteness_and_epsilon_reject(self):
        valid = [np.ones((1, 2, 3), np.float32), np.ones((1, 2, 3), np.float32), np.ones(3, np.float32)]
        for index in range(3):
            for invalid in (np.nan, np.inf, -np.inf):
                bad = [x.copy() for x in valid]
                bad[index].flat[0] = invalid
                with self.assertRaises(ValueError):
                    oracle.math_values(*bad, 1e-6)
            bad = list(valid)
            bad[index] = bad[index].astype(np.float16)
            with self.assertRaises(ValueError):
                oracle.math_values(*bad, 1e-6)
        for epsilon in (0, -1, np.nan, np.inf, True, np.bool_(True), [], [1], np.array([1.0]), 1+2j, "1"):
            with self.assertRaises(ValueError):
                oracle.math_values(*valid, epsilon)
        for shape in ((1, 0, 3), (1, 4, 3), (3, 1, 3), (1, 2, 0), (1, 2, 129)):
            with self.assertRaises(ValueError):
                oracle.math_values(np.ones(shape, np.float32), np.ones(shape, np.float32), np.ones(shape[-1], np.float32), 1e-6)

    def test_bounded_format_requires_external_hash_and_exact_fp32_inputs(self):
        raw = oracle.export()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'math.json'
            path.write_bytes(raw)
            self.assertEqual(len(oracle.verify(path, digest(raw))['cases']), 5)
            with self.assertRaises(ValueError):
                oracle.verify(path, '0' * 64)
            base = json.loads(raw)
            changes = []
            for key, value in (('native_graph_implemented', True), ('extra', True), ('runtime', dict(numpy='other'))):
                changed = copy.deepcopy(base)
                changed[key] = value
                changes.append(changed)
            changed = copy.deepcopy(base)
            changed['cases'][1]['tokens'] = True
            changes.append(changed)
            for value in (0.1, True, 10**400):
                changed = copy.deepcopy(base)
                changed['cases'][1]['arrays']['hidden']['values'][0] = value
                changes.append(changed)
            changed = copy.deepcopy(base)
            changed['cases'][2]['arrays']['hidden']['shape'][0] = True
            changes.append(changed)
            for changed in changes:
                raw = oracle.encoded(changed)
                path.write_bytes(raw)
                with self.assertRaises(ValueError):
                    oracle.verify(path, digest(raw))
            alias = path.parent / 'alias'
            alias.symlink_to(path)
            with self.assertRaises(ValueError):
                oracle.verify(alias, digest(raw))

    def test_comparison_failure_preserves_the_preceding_numeric_observation(self):
        sources = {
            'qwen3_5.py': b'# inert alias witness',
            'activations.py': b'def precise_swiglu(h, gate, x): return x',
            'qwen3_next.py': b"""class Qwen3NextRMSNormGated:
    def __init__(self, width, eps): self.weight = None
    def __call__(self, h, gate=None): return h
""",
        }
        expected = {name: digest(raw) for name, raw in sources.items()}
        data = json.loads(oracle.export())
        # Match the mocked model's first result exactly so a later failure is
        # exercised independently of the usual fixture numerical comparisons.
        for case in data['cases']:
            case['arrays']['normalized']['values'] = case['arrays']['hidden']['values']
        raw = oracle.encoded(data)
        core = types.ModuleType('mlx.core')
        core.__file__ = 'mock-core'
        core.cpu, core.gpu = 'cpu', 'gpu'
        core.array, core.eval = np.array, lambda value: None
        core.default_device, core.set_default_device = lambda: 'cpu', lambda device: None
        nn = types.ModuleType('mlx.nn')
        nn.__file__ = 'mock-nn'
        def fail_activation(value):
            raise RuntimeError('synthetic activation failure')
        nn.silu = fail_activation
        mlx = types.ModuleType('mlx')
        mlx.core, mlx.nn = core, nn
        admitted = dict(imports={'numpy': np.__file__, 'mlx.core': core.__file__, 'mlx.nn': nn.__file__})
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            reference, destination = root / 'math.json', root / 'out'
            reference.write_bytes(raw)
            destination.mkdir()
            with mock.patch.object(oracle, 'SOURCES', expected), mock.patch.dict(sys.modules, {'mlx': mlx, 'mlx.core': core, 'mlx.nn': nn}), mock.patch.object(oracle.runtime, 'admit_runtime', return_value=admitted):
                # Source fields of this synthetic test remain the original
                # identity; mocked format admission is explicit, not evidence.
                with mock.patch.object(oracle, 'verify', return_value=data):
                    self.assertFalse(oracle.compare(reference, digest(raw), sources, 'cpu', destination, root / 'runtime', '0'*64))
            observation = json.loads((destination / 'observation.json').read_bytes())
            self.assertIn('synthetic activation failure', observation['primary_error'])
            self.assertEqual(set(observation['cases'][1]['arrays']), {'normalized'})
            item = observation['cases'][1]['arrays']['normalized']
            blob = (destination / 'observed.bin').read_bytes()
            self.assertEqual(digest(blob[item['offset']:item['offset']+item['bytes']]), item['sha256'])
            self.assertTrue(observation['comparisons'][0]['passed'])

    def test_selected_definitions_reject_missing_or_ambiguous_sources(self):
        sources = {'qwen3_next.py': b'class Qwen3NextRMSNormGated: pass',
            'activations.py': b'def precise_swiglu(): pass'}
        self.assertEqual(set(oracle.source_nodes(sources)), set(sources))
        for key in sources:
            bad = dict(sources)
            bad[key] = b'pass'
            with self.assertRaises(ValueError):
                oracle.source_nodes(bad)
            bad[key] = sources[key] + b'\n' + sources[key]
            with self.assertRaises(ValueError):
                oracle.source_nodes(bad)


if __name__ == '__main__':
    unittest.main()

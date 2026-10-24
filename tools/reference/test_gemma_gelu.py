"""Portable independent tanh-GELU witnesses; no Apple runtime imported."""
import copy
import hashlib
import json
import math
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

import numpy as np

import gemma_gelu as oracle


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


class GemmaGelu(unittest.TestCase):
    def test_hand_arithmetic_and_actual_gelu_variant_and_gate_order(self):
        gate = np.array([-1, 0, 1], np.float32)
        up = np.array([2, -3, 0.5], np.float32)
        before = [value.copy() for value in (gate, up)]
        activated, output = oracle.math_values(gate, up)
        expected = [0.5 * float(x) * (1 + math.tanh(math.sqrt(2 / math.pi) * (float(x) + 0.044715 * float(x)**3))) for x in gate]
        np.testing.assert_allclose(activated, expected, rtol=1e-14, atol=1e-14)
        np.testing.assert_allclose(output, np.array(expected) * up, rtol=1e-14, atol=1e-14)
        for actual, expected in zip((gate, up), before):
            np.testing.assert_array_equal(actual, expected)
        erf_value = 0.5 * (1 + math.erf(1 / math.sqrt(2)))
        self.assertGreater(abs(activated[2] - erf_value), oracle.ATOL + oracle.RTOL * abs(activated[2]))
        self.assertGreater(abs(activated[2] - 1 / (1 + math.exp(-1))), 0.1)
        reversed_output = oracle.math_values(up, gate)[1]
        self.assertGreater(np.max(np.abs(reversed_output - output)), 1)

    def test_symmetry_and_independent_chunks(self):
        x = np.arange(-64, 65, dtype=np.float32) * np.float32(0.125)
        up = np.linspace(-2, 2, len(x), dtype=np.float32)
        activated, output = oracle.math_values(x, up)
        opposite = oracle.math_values(-x, up)[0]
        np.testing.assert_allclose(activated - opposite, x, rtol=1e-14, atol=1e-14)
        for chunk in (1, 7, 32):
            observed = [oracle.math_values(x[first:first+chunk], up[first:first+chunk]) for first in range(0, len(x), chunk)]
            np.testing.assert_array_equal(np.concatenate([value[0] for value in observed]), activated)
            np.testing.assert_array_equal(np.concatenate([value[1] for value in observed]), output)

    def test_empty_tails_tiny_nonzero_and_finite_fp32_extremes(self):
        for value in oracle.math_values(np.empty(0, np.float32), np.empty(0, np.float32)):
            self.assertEqual(value.shape, (0,))
        gate = np.array([-1000, -1e-8, 0, 1e-8, 1000, -np.finfo(np.float32).max, np.finfo(np.float32).max], np.float32)
        activated, output = oracle.math_values(gate, np.ones_like(gate))
        self.assertTrue(np.isfinite(activated).all())
        self.assertEqual(activated[0], 0)
        self.assertEqual(activated[4], 1000)
        self.assertEqual(activated[5], 0)
        self.assertEqual(activated[6], float(gate[6]))
        self.assertLess(activated[1], 0)
        self.assertAlmostEqual(activated[1] / (0.5 * float(gate[1])), 1.0, places=7)
        np.testing.assert_array_equal(activated, output)

    def test_invalid_shapes_dtypes_and_nonfinite_inputs_reject(self):
        valid = [np.ones(3, np.float32), np.ones(3, np.float32)]
        for index in range(2):
            for invalid in (np.nan, np.inf, -np.inf):
                bad = [value.copy() for value in valid]
                bad[index][1] = invalid
                with self.assertRaises(ValueError):
                    oracle.math_values(*bad)
            bad = list(valid)
            bad[index] = bad[index].astype(np.float16)
            with self.assertRaises(ValueError):
                oracle.math_values(*bad)
        for a, b in ((np.ones((1, 3), np.float32), valid[1]), (valid[0], np.ones(2, np.float32)), (np.ones(130, np.float32), np.ones(130, np.float32))):
            with self.assertRaises(ValueError):
                oracle.math_values(a, b)

    def test_format_external_identity_types_and_exact_fp32_inputs(self):
        raw = oracle.export()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'math.json'
            path.write_bytes(raw)
            self.assertEqual(len(oracle.verify(path, digest(raw))['cases']), 5)
            with self.assertRaises(ValueError):
                oracle.verify(path, '0'*64)
            base = json.loads(raw)
            mutations = []
            for key, value in (('device_qualified', True), ('extra', True), ('runtime', dict(numpy='other'))):
                changed = copy.deepcopy(base)
                changed[key] = value
                mutations.append(changed)
            changed = copy.deepcopy(base)
            changed['cases'][1]['width'] = True
            mutations.append(changed)
            for value in (0.1, True, 10**400):
                changed = copy.deepcopy(base)
                changed['cases'][1]['arrays']['gate']['values'][0] = value
                mutations.append(changed)
            for changed in mutations:
                raw = oracle.encoded(changed)
                path.write_bytes(raw)
                with self.assertRaises(ValueError):
                    oracle.verify(path, digest(raw))
            alias = path.parent / 'alias'
            alias.symlink_to(path)
            with self.assertRaises(ValueError):
                oracle.verify(alias, digest(raw))

    def test_source_definition_is_unique(self):
        good = {'gemma4_text.py': b'def geglu(gate, up): return gate * up'}
        self.assertEqual(set(oracle.source_nodes(good)), {'gemma4_text.py'})
        for source in (b'pass', good['gemma4_text.py'] + b'\n' + good['gemma4_text.py']):
            with self.assertRaises(ValueError):
                oracle.source_nodes({'gemma4_text.py': source})

    def test_later_failure_keeps_completed_activation_bytes(self):
        sources = {'gemma4_text.py': b'def geglu(gate, up): return nn.gelu_approx(gate) * up'}
        expected = {name: digest(raw) for name, raw in sources.items()}
        data = json.loads(oracle.export())
        for case in data['cases']:
            case['arrays']['activated']['values'] = case['arrays']['gate']['values']
        raw = oracle.encoded(data)
        core = types.ModuleType('mlx.core')
        core.__file__ = 'mock-core'
        core.cpu, core.gpu = 'cpu', 'gpu'
        core.array, core.eval = np.array, lambda value: None
        core.default_device, core.set_default_device = lambda: 'cpu', lambda device: None
        nn = types.ModuleType('mlx.nn')
        nn.__file__ = 'mock-nn'
        calls = []
        def activation(value):
            calls.append(True)
            if len(calls) == 2:
                raise RuntimeError('synthetic gated call failure')
            return value
        nn.gelu_approx = activation
        mlx = types.ModuleType('mlx')
        mlx.core, mlx.nn = core, nn
        admitted = dict(imports={'numpy': np.__file__, 'mlx.core': core.__file__, 'mlx.nn': nn.__file__})
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            reference, destination = root/'math.json', root/'out'
            reference.write_bytes(raw)
            destination.mkdir()
            with mock.patch.object(oracle, 'SOURCES', expected), mock.patch.dict(sys.modules, {'mlx': mlx, 'mlx.core': core, 'mlx.nn': nn}), mock.patch.object(oracle.runtime, 'admit_runtime', return_value=admitted), mock.patch.object(oracle, 'verify', return_value=data):
                self.assertFalse(oracle.compare(reference, digest(raw), sources, 'cpu', destination, root/'runtime', '0'*64))
            observation = json.loads((destination/'observation.json').read_bytes())
            self.assertIn('synthetic gated call failure', observation['primary_error'])
            self.assertEqual(set(observation['cases'][1]['arrays']), {'activated'})
            item = observation['cases'][1]['arrays']['activated']
            blob = (destination/'observed.bin').read_bytes()
            self.assertEqual(digest(blob[item['offset']:item['offset']+item['bytes']]), item['sha256'])
            self.assertTrue(observation['comparisons'][0]['passed'])


if __name__ == '__main__':
    unittest.main()

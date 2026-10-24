"""Portable mathematical/format regressions, independent of Apple GPU access."""
import copy
import hashlib
import json
import math
from pathlib import Path
import struct
import tempfile
import unittest

import numpy as np

import causal_conv_fixture as fixture
import causal_conv_oracle as oracle
import causal_conv_upstream as upstream


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


class ConvolutionMath(unittest.TestCase):
    def test_asymmetric_taps_and_channels_have_hand_computed_order(self):
        inputs = np.array([[3, 30], [4, 40]], np.float32)
        weights = np.array([[2, -1, 3], [-2, 0.5, 1]], np.float32)
        initial = np.array([[1, 10], [2, 20]], np.float32)
        before = [x.copy() for x in (inputs, weights, initial)]
        output, activated, history, final = oracle.run(inputs, weights, initial)
        np.testing.assert_array_equal(output, [[9, 20], [13, 15]])
        expected = np.array([[x / (1 + math.exp(-x)) for x in row] for row in [[9, 20], [13, 15]]])
        np.testing.assert_allclose(activated, expected, rtol=1e-14, atol=1e-14)
        np.testing.assert_array_equal(history, [[[2, 20], [3, 30]], [[3, 30], [4, 40]]])
        np.testing.assert_array_equal(final, [[3, 30], [4, 40]])
        for actual, expected in zip((inputs, weights, initial), before):
            np.testing.assert_array_equal(actual, expected)

    def test_chunking_single_steps_and_causality_preserve_history(self):
        for spec in fixture.SPECS:
            row = oracle.make_case(*spec)
            arrays = row['arrays']
            for chunk in (1, 2, 4):
                state = arrays['initial_history'].copy()
                outputs, activations = [], []
                for first in range(0, row['tokens'], chunk):
                    y, a, _, state = oracle.run(arrays['input'][first:first + chunk], arrays['weight'], state)
                    outputs.append(y)
                    activations.append(a)
                if outputs:
                    np.testing.assert_array_equal(np.concatenate(outputs), arrays['output'])
                    np.testing.assert_allclose(np.concatenate(activations), arrays['activated'], rtol=1e-14, atol=1e-14)
                np.testing.assert_array_equal(state, arrays['final_history'])
            if row['tokens'] > 1:
                changed = arrays['input'].copy()
                changed[-1] += np.float32(10)
                y, _, _, _ = oracle.run(changed, arrays['weight'], arrays['initial_history'])
                np.testing.assert_array_equal(y[:-1], arrays['output'][:-1])

    def test_empty_and_kernel_one_are_explicit_mathematical_policies(self):
        inputs = np.empty((0, 2), np.float32)
        initial = np.array([[2, 3], [4, 5]], np.float32)
        y, a, history, final = oracle.run(inputs, np.ones((2, 3), np.float32), initial)
        self.assertEqual(y.shape, (0, 2))
        self.assertEqual(a.shape, y.shape)
        self.assertEqual(history.shape, (0, 2, 2))
        np.testing.assert_array_equal(final, initial)
        y, _, history, final = oracle.run(np.array([[3, 5]], np.float32), np.array([[2], [-1]], np.float32), np.empty((0, 2), np.float32))
        np.testing.assert_array_equal(y, [[6, -5]])
        self.assertEqual(history.shape, (1, 0, 2))
        self.assertEqual(final.shape, (0, 2))

    def test_silu_negative_saturation_and_small_values_remain_finite(self):
        values = np.array([[-1000.0], [-24.0], [0.0], [24.0], [1000.0]], np.float32)
        _, actual, _, _ = oracle.run(values, np.ones((1, 1), np.float32), np.empty((0, 1), np.float32))
        self.assertTrue(np.isfinite(actual).all())
        self.assertEqual(actual[0, 0], 0.0)
        self.assertLess(actual[1, 0], 0.0)
        self.assertAlmostEqual(actual[1, 0] / (-24.0 * math.exp(-24.0) / (1 + math.exp(-24.0))), 1.0, places=14)
        self.assertEqual(actual[2, 0], 0.0)
        self.assertEqual(actual[4, 0], 1000.0)

    def test_selected_source_assignments_reject_missing_or_ambiguous_nodes(self):
        source = b"""class GatedDeltaNet:
    def __init__(self):
        self.conv1d = nn.Conv1d()
    def __call__(self):
        conv_input = mx.concatenate([conv_state, qkv], axis=1)
        if masked:
            cache[0] = mx.take_along_axis(conv_input, positions, axis=1)
        else:
            cache[0] = mx.contiguous(conv_input[:, -n_keep:, :])
        conv_out = nn.silu(self.conv1d(conv_input))
"""
        self.assertEqual(set(upstream.selected_assignments(source)), {'constructor', 'concatenate', 'history', 'activation'})
        for changed in (source.replace(b'conv_out =', b'other ='), source + source,
                source.replace(b'        conv_out =', b'        conv_input = other\n        conv_out =')):
            with self.assertRaises(ValueError):
                upstream.selected_assignments(changed)

    def test_evaluation_failures_keep_prior_raw_arrays_and_unexpected_dtypes(self):
        row, blob = dict(name='mock', arrays={}), bytearray()
        comparisons = []
        def retained(name, actual):
            value, comparison = upstream.retain_array(blob, row, name, actual, np.array([2.0], np.float64))
            comparisons.append(comparison)
            return value
        def steps():
            yield 'raw', np.array([2.0], np.float32)
            raise RuntimeError('mock activation construction failed')
        with self.assertRaisesRegex(RuntimeError, 'activation'):
            upstream.capture_evaluations(steps(), lambda value: value, retained)
        self.assertEqual(set(row['arrays']), {'raw'})
        self.assertEqual(bytes(blob), struct.pack('<f', 2.0))
        self.assertTrue(comparisons[0]['passed'])
        for value in (np.array([2.0], np.float64), np.array([np.nan], np.float32)):
            row, blob = dict(name='mock', arrays={}), bytearray()
            with self.assertRaisesRegex(ValueError, 'dtype/shape/finite'):
                upstream.retain_array(blob, row, 'bad', value, np.array([2.0], np.float64))
            self.assertEqual(set(row['arrays']), {'bad'})
            self.assertEqual(bytes(blob), value.tobytes())
        row, blob = dict(name='prefix', arrays={}), bytearray()
        _, comparison = upstream.retain_array(blob, row, 'prefix_output', np.array([3.0], np.float32), np.array([2.0], np.float64))
        self.assertFalse(comparison['passed'])

    def test_bad_shapes_types_nonfinite_and_arithmetic_failures_reject(self):
        valid = [np.ones((1, 2), np.float32), np.ones((2, 3), np.float32), np.ones((2, 2), np.float32)]
        for index in range(3):
            bad = [x.copy() for x in valid]
            bad[index].flat[0] = np.nan
            with self.assertRaises(ValueError):
                oracle.run(*bad)
            bad = list(valid)
            bad[index] = bad[index].astype(np.float16)
            with self.assertRaises(ValueError):
                oracle.run(*bad)
        with self.assertRaises(ValueError):
            oracle.run(valid[0], np.ones((3, 3)), valid[2])
        with self.assertRaises((ValueError, FloatingPointError)):
            oracle.run(np.array([[1e308]]), np.array([[2.0]]), np.empty((0, 1)))


class ConvolutionFormat(unittest.TestCase):
    def write(self, folder, manifest, binary):
        raw = (json.dumps(manifest, allow_nan=False) + '\n').encode()
        (folder / 'manifest.json').write_bytes(raw)
        (folder / 'arrays.bin').write_bytes(binary)
        return sha(raw), sha(binary)

    def test_exact_export_and_external_identity_are_required(self):
        outputs = oracle.export()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            oracle.publish(outputs, root)
            fixture.verify(root, sha(outputs['manifest.json']), sha(outputs['arrays.bin']))
            with self.assertRaises(ValueError):
                fixture.verify(root, '0' * 64, sha(outputs['arrays.bin']))
            with self.assertRaises(ValueError):
                oracle.publish(outputs, root)
            alias = root.parent / (root.name + '-alias')
            alias.symlink_to(root, target_is_directory=True)
            try:
                with self.assertRaises(ValueError):
                    fixture.verify(alias, sha(outputs['manifest.json']), sha(outputs['arrays.bin']))
            finally:
                alias.unlink()

    def test_mutated_geometry_scope_spans_and_finite_values_reject(self):
        outputs = oracle.export()
        base = json.loads(outputs['manifest.json'])
        mutations = []
        for key, value in [('native_implemented', True), ('runtime', dict(numpy='other')), ('extra', True)]:
            changed = copy.deepcopy(base)
            changed[key] = value
            mutations.append((changed, outputs['arrays.bin']))
        for key, value in [('offset', -1), ('bytes', False), ('shape', [True, 1]), ('dtype', '>f4')]:
            changed = copy.deepcopy(base)
            changed['cases'][1]['arrays']['input'][key] = value
            mutations.append((changed, outputs['arrays.bin']))
        changed = copy.deepcopy(base)
        descriptor = changed['cases'][1]['arrays']['input']
        binary = bytearray(outputs['arrays.bin'])
        start = descriptor['offset']
        binary[start:start + 4] = struct.pack('<f', float('nan'))
        descriptor['sha256'] = sha(binary[start:start + descriptor['bytes']])
        changed['binary']['sha256'] = sha(binary)
        mutations.append((changed, bytes(binary)))
        for manifest, binary in mutations:
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                pins = self.write(root, manifest, binary)
                with self.assertRaises(ValueError):
                    fixture.verify(root, *pins)


if __name__ == '__main__':
    unittest.main()

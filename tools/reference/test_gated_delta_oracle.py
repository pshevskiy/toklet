"""Independent mathematical-reference regression tests."""
from pathlib import Path
import tempfile
import unittest

import numpy as np
import gated_delta_oracle as oracle


class GatedDeltaMath(unittest.TestCase):
    def test_scalar_decay_correction_and_post_update_readout(self):
        q = k = np.ones((2, 1, 1), dtype=np.float64)
        v = np.array([3, 1], dtype=np.float64).reshape(2, 1, 1)
        initial = np.array([[[4.0]]])
        output, states, final = oracle.run_normalized(q, k, v,
            np.array([[0.5], [1.0]]), np.array([[0.5], [1.0]]), initial)
        np.testing.assert_array_equal(output.reshape(-1), [2.5, 1.0])
        np.testing.assert_array_equal(states.reshape(-1), [2.5, 1.0])
        np.testing.assert_array_equal(final, [[[1.0]]])
        np.testing.assert_array_equal(initial, [[[4.0]]])

    def test_group_mapping_is_contiguous_not_alternating(self):
        q = np.array([[[1.0, 0.0], [0.0, 2.0]]])
        k = np.array([[[1.0, 0.0], [0.0, 1.0]]])
        v = np.array([[[1.0], [2.0], [3.0], [4.0]]])
        output, _, _ = oracle.run_normalized(q, k, v, np.zeros((1, 4)),
            np.ones((1, 4)), np.zeros((4, 1, 2)))
        np.testing.assert_array_equal(output.reshape(-1), [1, 2, 6, 8])

    def test_l2_epsilon_and_query_scale_are_explicit(self):
        raw = np.array([[[3.0, 4.0], [0.0, 0.0]]], dtype=np.float32)
        q, k = oracle.normalize_qk(raw, raw, 1.0)
        expected_k = np.array([[[3 / np.sqrt(26), 4 / np.sqrt(26)], [0.0, 0.0]]])
        np.testing.assert_array_equal(k, expected_k)
        np.testing.assert_array_equal(q, expected_k / np.sqrt(2))

    def test_extreme_finite_norms_and_boolean_epsilon_reject(self):
        huge = np.array([[[1e308]]], dtype=np.float64)
        ordinary = np.ones((1, 1, 1), dtype=np.float64)
        for q, k in [(huge, ordinary), (ordinary, huge)]:
            with self.assertRaises(ValueError):
                oracle.normalize_qk(q, k, 1e-6)
        for epsilon in [True, np.bool_(True)]:
            with self.assertRaises(ValueError):
                oracle.normalize_qk(ordinary, ordinary, epsilon)

    def test_scalar_gates_include_saturated_values_without_invalid_numbers(self):
        a = np.array([[0.0, -24.0, 24.0]])
        b = a.copy()
        decay, beta = oracle.gates(a, b, np.zeros(3), np.zeros(3))
        self.assertEqual(decay[0, 0], 0.5)
        self.assertEqual(beta[0, 0], 0.5)
        self.assertTrue(np.isfinite(decay).all() and np.isfinite(beta).all())
        self.assertTrue(((decay >= 0) & (decay <= 1)).all())
        self.assertTrue(((beta >= 0) & (beta <= 1)).all())

    def test_chunk_and_step_state_match_full_sequence(self):
        values = oracle.case('grouped', 7, 2, 4, 8, 3)['arrays']
        q, k, v, g, beta = [values[name] for name in ('normalized_q', 'normalized_k', 'v', 'decay', 'beta')]
        first, _, state = oracle.run_normalized(q[:3], k[:3], v[:3], g[:3], beta[:3], values['initial_state'])
        rest, _, state = oracle.run_normalized(q[3:], k[3:], v[3:], g[3:], beta[3:], state)
        np.testing.assert_array_equal(np.concatenate([first, rest]), values['output'])
        np.testing.assert_array_equal(state, values['final_state'])

    def test_empty_prefix_keeps_independent_initial_state(self):
        values = oracle.case('empty', 0, 1, 1, 8, 3)['arrays']
        self.assertEqual(values['output'].shape, (0, 1, 3))
        self.assertEqual(values['states'].shape, (0, 1, 3, 8))
        np.testing.assert_array_equal(values['final_state'], values['initial_state'])
        self.assertFalse(np.shares_memory(values['final_state'], values['initial_state']))

    def test_bad_shapes_nonfinite_and_unadmitted_gate_values_reject(self):
        q = k = v = np.ones((1, 1, 1), dtype=np.float64)
        state = np.zeros((1, 1, 1))
        for gate in [np.array([[-0.1]]), np.array([[1.1]]), np.array([[np.nan]])]:
            with self.assertRaises(ValueError):
                oracle.run_normalized(q, k, v, gate, np.ones((1, 1)), state)
        with self.assertRaises(ValueError):
            oracle.run_normalized(q, k, np.ones((1, 3, 1)), np.ones((1, 3)), np.ones((1, 3)), np.zeros((3, 2, 1)))
        with self.assertRaises(ValueError):
            two_keys = np.ones((1, 2, 1))
            oracle.run_normalized(two_keys, two_keys, np.ones((1, 3, 1)), np.ones((1, 3)), np.ones((1, 3)), np.zeros((3, 1, 1)))
        with self.assertRaises(ValueError):
            oracle.run_normalized(q.astype(np.complex64), k, v, np.ones((1, 1)), np.ones((1, 1)), state)
        with self.assertRaises(ValueError):
            oracle.normalize_qk(q, k, 0.0)

    def test_publication_is_exclusive_with_fixed_names(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            with self.assertRaises(ValueError):
                oracle.publish({'../escape': b'x'}, folder)
            self.assertEqual(list(folder.iterdir()), [])
            outputs = {'manifest.json': b'{}\n', 'arrays.bin': b''}
            oracle.publish(outputs, folder)
            with self.assertRaises(ValueError):
                oracle.publish(outputs, folder)
            self.assertEqual((folder / 'manifest.json').read_bytes(), b'{}\n')

    def test_symlink_publication_root_cannot_touch_outside_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            outside = root / 'outside'
            outside.mkdir()
            alias = root / 'alias'
            alias.symlink_to(outside, target_is_directory=True)
            with self.assertRaises(ValueError):
                oracle.publish({'manifest.json': b'{}\n', 'arrays.bin': b''}, alias)
            self.assertEqual(list(outside.iterdir()), [])


if __name__ == '__main__':
    unittest.main()

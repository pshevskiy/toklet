"""Offline numerical and evidence-corruption controls without model downloads."""
import json
from pathlib import Path
import tempfile
import unittest
import check_gemma4_heretic_wrap as checker


class WrapCheckerTests(unittest.TestCase):
    def test_literal_state_schema_preserves_integer_and_boolean_types(self):
        expected = dict(schedule=0, overflow_preserved=True, positions=[0,1,None])
        self.assertTrue(checker.exact(expected,dict(expected)))
        for got in [dict(expected,schedule=False),dict(expected,overflow_preserved=1),
                    dict(expected,positions=[False,1,None]),dict(expected,positions=[0,1.0,None])]:
            self.assertFalse(checker.exact(got,expected))

    def test_frozen_elementwise_gate_checks_every_value(self):
        import numpy as np
        expected = np.array([0., 1., -1.], dtype=np.float64)
        tolerance = 1e-5+1e-4*np.abs(expected)
        got = expected+.5*tolerance
        self.assertEqual(checker.statistics(got,expected)['failed_values'], 0)
        got[-1] = expected[-1]+2*tolerance[-1]
        result = checker.statistics(got,expected)
        self.assertEqual(result['failed_values'], 1)
        self.assertAlmostEqual(result['maximum_gate_ratio'], 2.)

    def test_nonfinite_shape_and_empty_arrays_reject(self):
        import numpy as np
        for got, want in [(np.array([float('nan')]),np.array([0.])),
                          (np.array([0.]),np.array([float('inf')])),
                          (np.array([0.]),np.array([0.,0.])),(np.array([]),np.array([]))]:
            with self.assertRaisesRegex(ValueError, 'Finite equal'):
                checker.statistics(got,want)

    def test_memory_guard_rejects_booleans_and_over_budget_values(self):
        checker.memory(6*1024**3)
        for value in [True,0,-1,6*1024**3+1,1.0]:
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'Typed memory'):
                checker.memory(value)

    def test_changed_native_transcript_rejects_before_payload_access(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            receipt = dict(passed=True,exit_code=0,group_absent=True,changed_inputs=[],
                source='quantized-mlx',job='wrap',lifetime_peak_physical_bytes=100,
                files_sha256={'console.log':'0'*64})
            (folder/'checks.json').write_text(json.dumps(receipt))
            (folder/'console.log').write_text('changed\n')
            with self.assertRaisesRegex(ValueError, 'Native transcript differs'):
                checker.native(folder,'quantized-mlx')


if __name__ == '__main__':
    unittest.main()

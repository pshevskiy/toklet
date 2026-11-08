from pathlib import Path
import tempfile
import unittest
import numpy as np
import check_qwen35_real_reference as check


class Controls(unittest.TestCase):
    def test_literal_types_are_distinct_in_nested_contracts(self):
        self.assertFalse(check.exact([True], [1]))
        self.assertFalse(check.exact({'done': 0}, {'done': False}))
        self.assertFalse(check.exact(0, 0.0))
        self.assertTrue(check.exact({'x': [0, 1.0, None]}, {'x': [0, 1.0, None]}))

    def test_owner_token_head_and_canonical_keys_are_strict(self):
        row = dict(token=0, layer=3, stage='attention_k_cache.2')
        self.assertEqual(check.key(row), (3, 'attention_k_cache', 2))
        for change in [dict(token=True), dict(token=1), dict(layer=False), dict(layer=32),
                       dict(stage=1), dict(stage='attention_k_cache.02')]:
            with self.assertRaises(ValueError): check.key({**row, **change})

    def test_original_gate_inclusive_boundary_and_every_value(self):
        zero = np.array([0.], dtype=np.float64)
        self.assertEqual(check.statistics(np.array([1e-5]), zero)['failed_values'], 0)
        self.assertEqual(check.statistics(np.array([np.nextafter(1e-5, np.inf)]), zero)['failed_values'], 1)
        stats = check.statistics(np.array([0., 1., 0.]), np.zeros(3))
        self.assertEqual((stats['values'], stats['failed_values']), (3, 1))
        for a, b in [(np.array([np.nan]), zero), (zero, np.array([np.inf])),
                     (np.zeros(2), zero), (np.array([]), np.array([]))]:
            with self.assertRaises(ValueError): check.statistics(a, b)

    def test_duplicate_nonfinite_overflow_json_reject(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'record.json'
            for text in ['{"x":1,"x":2}', '{"x":NaN}', '{"x":Infinity}', '{"x":1e999}']:
                path.write_text(text)
                with self.assertRaises(ValueError): check.document(path)
            path.write_text('{"x":1e-5}')
            self.assertEqual(check.document(path), {'x': 1e-5})

    def test_evidence_bound_and_symlink_reject(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'file'; path.write_bytes(b'12')
            with self.assertRaises(ValueError): check.read(path, 1)
            link = Path(folder)/'link'; link.symlink_to(path)
            with self.assertRaises(ValueError): check.read(link, 2)
            self.assertEqual(check.read(path, 2), b'12')


if __name__ == '__main__': unittest.main()

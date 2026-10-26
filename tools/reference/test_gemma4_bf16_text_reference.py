"""Offline composition/failure tests; no checkpoint or GPU is accessed."""
from pathlib import Path
from types import SimpleNamespace
import json
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import gemma4_bf16_text_reference as driver

ROOT = Path(__file__).resolve().parents[2]


def model():
    shapes = driver.graph.shapes(driver.graph.E2B)
    return SimpleNamespace(matrices={n: object() for n, s in shapes.items() if len(s) == 2},
                           vectors={n: np.zeros(s, dtype=np.float32) for n, s in shapes.items() if len(s) == 1})


class GraphPreparation(unittest.TestCase):
    def test_composition_retains_matrix_views_and_direct_finite_vectors(self):
        layout = (ROOT / 'tests/fixtures/gemma-unquantized-header-v1/layout.json').read_bytes()
        selected = json.loads(layout)['selected_text']; views = {}
        class View:
            def __init__(self, full):
                shape = selected[full]['shape']; self.rows, self.columns = shape if len(shape) == 2 else (1, shape[0])
                self.value = np.zeros((1, self.columns), dtype=np.float32)
            def read_rows(self, start, end):
                self_test.assertEqual((start, end), (0, 1)); return self.value
        self_test = self
        def resolve(_, full, __, **kwargs):
            if full not in views: views[full] = View(full)
            return views[full]
        with patch.object(driver.adapter.bf16_source, 'official_rows', side_effect=resolve):
            actual = driver.adapter.weights(object(), layout, 1.)
        self.assertEqual((len(actual.matrices), len(actual.vectors)), (278, 262))
        self.assertEqual(sum(v.nbytes for v in actual.vectors.values()), 1143948)
        for name, view in actual.matrices.items(): self.assertIs(view, views[driver.adapter.PREFIX + name])
        for name, value in actual.vectors.items(): self.assertTrue(np.shares_memory(value, views[driver.adapter.PREFIX + name].value))
        def wrong(*args, **kwargs): return SimpleNamespace(rows=1, columns=1)
        with patch.object(driver.adapter.bf16_source, 'official_rows', side_effect=wrong):
            with self.assertRaisesRegex(ValueError, 'row geometry differs'): driver.adapter.weights(object(), layout, 1.)

    def test_partial_graph_error_interrupt_and_endpoint_keep_raw_states_and_primary(self):
        class Owner:
            def __enter__(self): return self
            def __exit__(self, *args): return False
            def audit(self): return {'synthetic_closed': True}
        plan = dict(captures=driver.contract.specifications(), file_bytes=103403520, source_audit={'synthetic_closed': True})
        with tempfile.TemporaryDirectory() as folder:
            for label, error in [('error', OSError('synthetic graph failure')), ('interrupt', KeyboardInterrupt('synthetic interrupt')),
                                 ('endpoint', ValueError('original graph primary'))]:
                output = Path(folder) / label
                def evaluate(_, ids, staged, *, sink):
                    sink('embedding', np.zeros((1, 4, 1536), dtype=np.float64)); raise error
                def endpoint():
                    if label == 'endpoint': raise RuntimeError('secondary endpoint')
                    return {}
                with patch.object(driver.bf16_source, 'Source', return_value=Owner()), \
                     patch.object(driver.adapter, 'weights', return_value=model()), \
                     patch.object(driver.graph, 'evaluate', side_effect=evaluate):
                    self.assertEqual(driver.run(output, plan, {'synthetic': True}, endpoint), 1)
                record = json.loads((output / 'observation.json').read_bytes()); body = (output / 'original.bin').read_bytes()
                self.assertFalse(record['passed']); self.assertFalse(record['capture_complete'])
                self.assertIn(str(error), record['primary_error']); self.assertEqual(len(record['arrays']), 1)
                self.assertTrue(record['arrays'][0]['captured']); self.assertEqual(len(body), 49152)
                self.assertEqual(record['files']['original']['sha256'], driver.hashlib.sha256(body).hexdigest())
                self.assertEqual(record['source_audit'], plan['source_audit'])
                self.assertEqual(bool(record['endpoint_errors']), label == 'endpoint')

    def test_existing_output_refuses_before_owner_access(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(driver.bf16_source, 'Source') as source:
            with self.assertRaises(FileExistsError): driver.run(Path(folder), {}, {}, lambda: {})
            source.assert_not_called()


if __name__ == '__main__': unittest.main()

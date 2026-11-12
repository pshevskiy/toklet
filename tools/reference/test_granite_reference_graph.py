"""Cold geometry/stage controls; numerical fixtures run through admitted CLI."""
import copy
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import granite_reference_graph as graph
import granite_reference_probe as probe


def config():
    return dict(hidden=2560, intermediate=8192, layers=40, vocabulary=100352,
                query_heads=40, kv_heads=8, head_dim=64, theta=10000000,
                rms_eps=1e-5, attention_scale=0.015625, embedding_scale=1,
                residual_scale=1, logits_divisor=1, tied_head=False)


class Geometry(unittest.TestCase):
    def test_full_geometry_and_exact_absolute_boundary(self):
        graph.geometry(config(), [1] * 512, 0)
        graph.geometry(config(), [7, 13, 3], 509)
        for tokens, offset in [([], 0), ([1] * 513, 0), ([7, 13, 3], 510),
                               ([1], -1), ([True], 0), ([100352], 0), ([1], False)]:
            with self.assertRaises(ValueError):
                graph.geometry(config(), tokens, offset)

    def test_invalid_head_mapping_scalars_or_declarations_reject(self):
        changes = [dict(hidden=True), dict(hidden=2561), dict(kv_heads=7),
                   dict(query_heads=39), dict(head_dim=63), dict(layers=41),
                   dict(theta=float('inf')), dict(rms_eps=0), dict(attention_scale=True),
                   dict(residual_scale=2**-127), dict(tied_head=0), dict(extra=1)]
        for change in changes:
            value = config()
            value.update(change)
            with self.assertRaises(ValueError):
                graph.geometry(value, [1], 0)
        value = config()
        del value['logits_divisor']
        with self.assertRaises(ValueError):
            graph.geometry(value, [1], 0)

    def test_every_layer_has_all_sixteen_native_visible_stages(self):
        value = config()
        before = copy.deepcopy(value)
        expected = graph.contracts(value, 128)
        self.assertEqual(value, before)
        self.assertEqual(len(expected), 643)
        self.assertEqual(set(expected) - {f'layers.{i}.{stage}' for i in range(40)
                                         for stage in graph.STAGES},
                         {'embedding', 'final_norm', 'logits'})
        self.assertEqual(expected['layers.39.k'], (128, 512))
        self.assertEqual(expected['layers.39.q'], (128, 2560))
        self.assertEqual(expected['layers.39.down'], (128, 2560))
        self.assertEqual(expected['layers.39.gated'], (128, 8192))
        self.assertEqual(expected['logits'], (128, 100352))

    def test_smaller_independent_fixture_and_tied_head_remain_explicit(self):
        value = config()
        value.update(hidden=32, intermediate=64, layers=2, vocabulary=64,
                     query_heads=4, kv_heads=2, head_dim=8, tied_head=True)
        graph.geometry(value, [1, 7, 13, 3], 0)
        self.assertEqual(len(graph.contracts(value, 4)), 35)

    def test_cold_module_does_not_import_numerical_packages(self):
        self.assertFalse(any(name == package or name.startswith(package + '.')
                             for name in sys.modules for package in ('numpy', 'mlx', 'torch')))

    def test_complete_existing_fixture_is_admitted_to_the_cold_source_inventory(self):
        selected = probe.sources()
        self.assertIn(probe.FIXTURE, selected)
        self.assertEqual(probe.FIXTURE.stat().st_size, 2521660)
        self.assertGreater(probe.FIXTURE_LIMIT, probe.FIXTURE.stat().st_size)

    def cli(self, mode, output):
        return [sys.executable, str(Path(probe.__file__)), '--mode', mode,
                '--environment-root', 'missing-environment', '--runtime-manifest', 'missing-runtime',
                '--runtime-sha256', '0' * 64, '--fresh-output', str(output)]

    def test_existing_output_rejects_before_any_runtime_admission(self):
        with tempfile.TemporaryDirectory() as folder:
            occupied = Path(folder) / 'occupied'
            occupied.write_bytes(b'preserve')
            result = subprocess.run(self.cli('controls', occupied), capture_output=True, timeout=10)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(b'Fresh reference output required', result.stderr)
            self.assertNotIn(b'missing-runtime', result.stderr)
            self.assertEqual(occupied.read_bytes(), b'preserve')

    def test_incomplete_capture_rejects_before_creating_output_or_reading_assets(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / 'fresh'
            result = subprocess.run(self.cli('capture', output), capture_output=True, timeout=10)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(b'Complete externally pinned full-model capture arguments', result.stderr)
            self.assertFalse(output.exists())


if __name__ == '__main__':
    unittest.main()

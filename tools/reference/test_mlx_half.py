"""Portable frozen-oracle identity, storage-boundary and causal checks."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import fp16
import mlx_half


class Oracle(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.folder = mlx_half.DEFAULT
        cls.frozen_manifest_bytes = (cls.folder / 'manifest.json').read_bytes()
        cls.manifest = mlx_half.verify(cls.folder)
        cls.raw = (cls.folder / 'arrays.bin').read_bytes()

    def array(self, name):
        entry = self.manifest['arrays'][name]
        data = self.raw[entry['offset']:entry['offset'] + entry['bytes']]
        return np.frombuffer(data, dtype=entry['dtype']).reshape(entry['shape'])

    def test_frozen_policy_and_half_gather_are_independent_of_fp32_tied_output(self):
        source = json.loads((mlx_half.SOURCE / 'manifest.json').read_text())
        raw = (mlx_half.SOURCE / 'arrays.bin').read_bytes()
        self.assertEqual(len(self.manifest['arrays']), 199)
        for case in self.manifest['cases']:
            key = f'cases.{case}.embedding'
            entry = source['arrays'][key]
            value = np.frombuffer(raw[entry['offset']:entry['offset'] + entry['bytes']], dtype='<f4').reshape(entry['shape'])
            self.assertEqual(self.array(key).dtype, np.dtype('<f2'))
            np.testing.assert_array_equal(self.array(key), value.astype(np.float16))
            # Both MLX backends' actual FP32 logits are checked against the
            # separately frozen storage envelope, not reinterpreted as Half.
            if source['cases'][case]['position_offset'] == 0:
                half = self.array(f'cases.{case}.logits')
                gate = self.manifest['arrays'][f'cases.{case}.logits']['gate_against_source_fp32']
                self.assertEqual(half.dtype, np.dtype('<f4'))
                for backend in ['CPU', 'GPU']:
                    entry = source['arrays'][f'native.{backend}.{case}.logits']
                    actual = np.frombuffer(raw[entry['offset']:entry['offset'] + entry['bytes']], dtype='<f4').reshape(entry['shape'])
                    np.testing.assert_allclose(half, actual, **gate)
        for entry in self.manifest['arrays'].values():
            self.assertEqual(entry['gate']['atol'], max(1e-4, 2 * entry['precision_envelope']))
            self.assertEqual(entry['gate']['rtol'], 1e-3)
            self.assertEqual(entry['gate_against_source_fp32']['atol'], 2 * entry['storage_envelope'] + 1e-5)
            self.assertEqual(entry['gate_against_source_fp32']['rtol'], 1e-4)

    def test_tied_output_uses_unrounded_original_packed_parameters(self):
        source = json.loads((mlx_half.SOURCE / 'manifest.json').read_text())
        raw = (mlx_half.SOURCE / 'arrays.bin').read_bytes()
        entry = source['arrays']['cases.prompt.embedding']
        embedding = np.frombuffer(raw[entry['offset']:entry['offset'] + entry['bytes']], dtype='<f4').reshape(entry['shape'])
        known_ids = source['cases']['prompt']['tokens']
        normalized = self.array('cases.prompt.final_norm')
        logits = self.array('cases.prompt.logits')
        differences = 0
        for row, vector in enumerate(normalized):
            for token, weights in zip(known_ids, embedding):
                original = fp16.dot(vector, weights, np.float32)
                rounded = fp16.dot(vector, weights.astype(np.float16), np.float32)
                self.assertEqual(original.tobytes(), logits[row, token].tobytes())
                differences += original.tobytes() != rounded.tobytes()
        self.assertGreater(differences, 0, 'Fixture must distinguish original tied parameters from a Half table')

    def test_complete_coverage_layouts_and_frozen_manifest(self):
        self.assertEqual(hashlib.sha256(self.frozen_manifest_bytes).hexdigest(),
                         '1b394305deb39cee1076bfc867080dc2106eb05560b3a906e4a0ace526627a27')
        source = json.loads((mlx_half.SOURCE / 'manifest.json').read_text())
        expected = {name for name in source['arrays'] if name.startswith('cases.')}
        expected.update(f'greedy.step_{i}.logits' for i in range(4))
        self.assertEqual(set(self.manifest['arrays']), expected)
        offset = 0
        for name, entry in sorted(self.manifest['arrays'].items(), key=lambda pair: pair[1]['offset']):
            self.assertEqual(entry['offset'], offset)
            dtype = '<f4' if name.endswith(('attention_scores', 'attention_probabilities', 'logits')) else '<f2'
            self.assertEqual(entry['dtype'], dtype)
            shape = source['arrays'][name]['shape'] if name.startswith('cases.') else [64]
            self.assertEqual(entry['shape'], shape)
            self.assertEqual(entry['bytes'], int(np.prod(shape))*np.dtype(dtype).itemsize)
            self.assertTrue(np.isfinite(self.array(name)).all())
            offset += entry['bytes']
        self.assertEqual(offset, len(self.raw))

    def test_changed_future_has_identical_earlier_causal_results(self):
        names = [n for n in self.manifest['arrays'] if n.startswith('cases.prompt.')]
        for name in names:
            if name.endswith(('attention_scores', 'attention_probabilities')):
                continue
            other = name.replace('cases.prompt.', 'cases.future_changed.')
            self.assertEqual(self.array(name)[:3].tobytes(), self.array(other)[:3].tobytes(), name)

    def test_exact_frozen_regeneration_and_selected_producer_identities(self):
        for name, digest in self.manifest['producer_pins'].items():
            self.assertEqual(hashlib.sha256((mlx_half.ROOT / name).read_bytes()).hexdigest(), digest)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'fixture'
            mlx_half.export(output)
            for name in ['arrays.bin', 'manifest.json']:
                self.assertEqual((output / name).read_bytes(), (self.folder / name).read_bytes(), name)

    def test_stdlib_check_and_immutable_export_rejection(self):
        result = subprocess.run([sys.executable, '-S', str(Path(mlx_half.__file__)), '--check', str(self.folder)],
                                capture_output=True, text=True, timeout=30, check=True)
        self.assertEqual(json.loads(result.stdout), dict(verified=True, arrays=199))
        with self.assertRaises(FileExistsError):
            mlx_half.export(self.folder)

    def test_unadmitted_source_manifest_rejects_before_publication(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'manifest.json').write_text('{}')
            with mock.patch.object(mlx_half, 'SOURCE', root):
                with self.assertRaisesRegex(ValueError, 'manifest differs'):
                    mlx_half.export(root / 'new-fixture')
            self.assertFalse((root / 'new-fixture').exists())


if __name__ == '__main__':
    unittest.main()

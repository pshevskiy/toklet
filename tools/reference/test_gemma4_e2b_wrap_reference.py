"""Cold controls for the long oracle's frozen inputs and streaming inventory."""
from pathlib import Path
import hashlib
import json
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import gemma4_e2b_wrap_reference as producer


class WrapReferenceControls(unittest.TestCase):
    def fake_graph(self, weights, ids, staged, *, selected_positions, sink):
        dtype = np.float32 if staged else np.float64
        for owner in range(15):
            dim = 512 if owner % 5 == 4 else 256
            value = np.broadcast_to(np.arange(len(ids), dtype=dtype)[:, None],
                                    (len(ids), dim))
            sink(f'layer.{owner}.self_attn.v_norm', value.reshape(1, len(ids), 1, dim))
            sink(f'layer.{owner}.k_rotary', value.reshape(1, 1, len(ids), dim))
        sink('logits', np.zeros((1, len(selected_positions), 262144), dtype=dtype))

    def run_fake(self, root, graph):
        (root / 'model.safetensors').write_bytes(b'')
        with patch.object(producer.source_graph.source, 'PINS', {}), \
                patch.object(producer.source_graph.source, 'MODEL', root), \
                patch.object(producer.source_graph, 'weights', return_value=object()), \
                patch.object(producer.graph, 'evaluate', side_effect=graph):
            return producer.collect(root / 'output', 'full1025', {}, lambda: {'passed': True})

    def test_frozen_inputs_and_parent_rejection(self):
        ids, positions = producer.inputs('full1025')
        self.assertEqual(len(ids), 1025)
        self.assertEqual(positions, [0, 31, 255, 511, 512, 767, 1023, 1024])
        self.assertEqual(producer.inputs('compat512')[0], producer.prior.inputs('full512')[0])
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            manifest = json.loads((producer.CONTRACT / 'manifest.json').read_text())
            raw = bytearray((producer.CONTRACT / 'tokens.u32le').read_bytes())
            raw[-1] ^= 1
            manifest['tokens']['sha256'] = hashlib.sha256(raw).hexdigest()
            (root / 'manifest.json').write_text(json.dumps(manifest))
            (root / 'tokens.u32le').write_bytes(raw)
            with patch.object(producer, 'CONTRACT', root):
                with self.assertRaises(ValueError):
                    producer.inputs('full1025')

    def test_streamed_absolute_kv_and_selected_rows(self):
        with tempfile.TemporaryDirectory(dir=producer.ROOT / 'artifacts') as name:
            root = Path(name)
            result = self.run_fake(root, self.fake_graph)
            self.assertTrue(result['completed'], result.get('failure'))
            self.assertEqual(len(result['arrays']), 76)
            data = (root / 'output/rows.bin').read_bytes()
            self.assertEqual(len(data), 138522624)
            self.assertEqual(hashlib.sha256(data).hexdigest(), result['rows_sha256'])
            offset = 0
            for item in result['arrays']:
                self.assertEqual(item['offset'], offset)
                raw = data[offset:offset + item['bytes']]
                self.assertEqual(hashlib.sha256(raw).hexdigest(), item['sha256'])
                dtype = '<f4' if item['contract'] == 'staged_fp32' else '<f8'
                values = np.frombuffer(raw, dtype=dtype)
                if item['kind'] != 'logits':
                    values = values.reshape(1025, item['head_dim'])
                    np.testing.assert_array_equal(values[:, 0], np.arange(1025))
                    self.assertEqual(item['retention'], 'global' if item['owner'] % 5 == 4 else 'local512')
                else:
                    self.assertEqual(values.shape, (262144,))
                    self.assertTrue((values == 0).all())
                offset += item['bytes']
            self.assertEqual(offset, len(data))

    def test_missing_duplicate_and_bad_shape_fail(self):
        for mutation in ['missing', 'duplicate', 'shape']:
            def changed(weights, ids, staged, *, selected_positions, sink):
                def mutate(name, value):
                    if name == 'layer.0.k_rotary':
                        if mutation == 'missing':
                            return
                        if mutation == 'shape':
                            value = value[..., :-1]
                        sink(name, value)
                        if mutation == 'duplicate':
                            sink(name, value)
                    else:
                        sink(name, value)
                self.fake_graph(weights, ids, staged, selected_positions=selected_positions, sink=mutate)
            with tempfile.TemporaryDirectory(dir=producer.ROOT / 'artifacts') as name:
                result = self.run_fake(Path(name), changed)
                self.assertFalse(result['completed'])
                self.assertEqual(result['failure']['type'], 'ValueError')


if __name__ == '__main__':
    unittest.main()

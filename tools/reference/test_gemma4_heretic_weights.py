"""CPU-only pinned-header, mixed row binding and permanent source-guard checks."""
import copy
import hashlib
import math
from pathlib import Path
import struct
import tempfile
import time
import unittest
import affine4_tiles
import bf16_source
import gemma4_heretic_weights as adapter

ROOT = Path(__file__).resolve().parents[2]


class HereticWeightsTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.path = Path(self.folder.name) / 'mixed.bin'
        packed = bytes([0x10, 0x32, 0x54, 0x76, 0x98, 0xba, 0xdc, 0xfe]) * 8
        payload = packed + struct.pack('<4H', 0x3f00, 0xc000, 0x3f80, 0x4080)
        payload += struct.pack('<128H', *([0x3f80] * 64 + [0x4000] * 64))
        self.body = b'HEADTEST' + payload
        self.path.write_bytes(self.body)
        self.header = {
            'linear.weight': dict(dtype='U32', shape=[2, 8], data_offsets=[0, 64]),
            'linear.scales': dict(dtype='BF16', shape=[2, 1], data_offsets=[64, 68]),
            'linear.biases': dict(dtype='BF16', shape=[2, 1], data_offsets=[68, 72]),
            'root.weight': dict(dtype='BF16', shape=[2, 64], data_offsets=[72, 328]),
        }
        self.deadline = time.monotonic() + 30
        self.owner = bf16_source.Source(self.path, len(self.body), hashlib.sha256(self.body).hexdigest(), deadline=self.deadline)
        self.addCleanup(self.owner.close)
        self.cursor = adapter.Cursor(self.owner, self.deadline)

    def matrix(self, name='linear.weight', *, bf16=False, inventory=None):
        return adapter.matrix(self.owner, self.cursor, inventory or self.header, 8,
                              name, (2, 64), self.deadline, bf16=bf16)

    def test_packed_metadata_and_original_bf16_have_distinct_row_views(self):
        import numpy as np
        affine, root = self.matrix(), self.matrix('root.weight', bf16=True)
        self.assertIs(type(affine), affine4_tiles.MatrixRows)
        self.assertIs(type(root), bf16_source.Rows)
        expected = np.array([[(c % 16) * .5 + 1 for c in range(64)],
                             [-(c % 16) * 2 + 4 for c in range(64)]], dtype=np.float32)
        np.testing.assert_array_equal(affine.read_rows(0, 2), expected)
        np.testing.assert_array_equal(root.gather([1, 0, 1]), np.array([[2.] * 64, [1.] * 64, [2.] * 64], dtype=np.float32))
        x = np.arange(64, dtype=np.float32)[None] / 8
        actual = affine4_tiles.project_rows(affine, x, np.float64)
        ideal = [[math.fsum(float(a) * float(b) for a, b in zip(x[0], row)) for row in expected]]
        np.testing.assert_array_equal(actual, ideal)

    def test_bf16_root_cannot_be_bound_as_packed(self):
        with self.assertRaisesRegex(ValueError, 'dtype/shape'):
            self.matrix('root.weight')
        with self.assertRaisesRegex(ValueError, 'dtype/shape'):
            self.matrix(bf16=True)

    def test_component_dtype_shape_offsets_and_integer_bounds(self):
        for field, value in [('dtype', 'F16'), ('shape', [2, 7]),
                             ('data_offsets', [-1, 63]), ('data_offsets', [True, 65]),
                             ('data_offsets', [0, 63]), ('data_offsets', [300, 364])]:
            inventory = copy.deepcopy(self.header)
            inventory['linear.weight'][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                self.matrix(inventory=inventory)
        inventory = copy.deepcopy(self.header)
        del inventory['linear.biases']
        with self.assertRaisesRegex(ValueError, 'dtype/shape'):
            self.matrix(inventory=inventory)

    def test_overlapping_affine_components_reject(self):
        inventory = copy.deepcopy(self.header)
        inventory['linear.biases']['data_offsets'] = [64, 68]
        with self.assertRaisesRegex(ValueError, 'overlap'):
            self.matrix(inventory=inventory)

    def test_cursor_bounds_and_read_chain(self):
        for offset in [-1, len(self.body), True]:
            with self.subTest(offset=offset), self.assertRaises(ValueError):
                self.cursor.seek(offset)
        self.cursor.seek(8)
        self.assertEqual(self.cursor.read(2), self.body[8:10])
        self.assertEqual(self.cursor.read(2), self.body[10:12])
        self.assertEqual(self.owner.audit()['reads'], 2)
        with self.assertRaises(ValueError):
            self.cursor.read(len(self.body))

    def test_changed_source_permanently_poisoned(self):
        self.path.write_bytes(b'!' + self.body[1:])
        with self.assertRaises(bf16_source.SourceChanged):
            self.cursor.read(1)
        self.assertTrue(self.owner.poisoned)
        self.path.write_bytes(self.body)
        with self.assertRaises(bf16_source.SourceChanged):
            self.cursor.read(1)

    def test_full_model_binding_requires_pinned_source(self):
        with self.assertRaisesRegex(ValueError, 'Pinned authenticated'):
            adapter.weights(self.owner, b'{}', self.deadline)

    def test_pinned_metadata_and_alterations(self):
        folder = ROOT / 'tests/fixtures/gemma4-heretic-metadata-v1'
        config = (folder / 'config.json').read_bytes()
        raw = (folder / 'header.json').read_bytes()
        prefix = struct.pack('<Q', adapter.HEADER_BYTES)
        value = adapter.header(config, prefix, raw)
        self.assertEqual(value[adapter.PREFIX + adapter.BF16_PROJECTION]['shape'], [8960, 1536])
        for cfg, pre, data in [(config + b' ', prefix, raw),
                               (config, b'\0' * 8, raw),
                               (config, prefix, raw[:-1]),
                               (config, prefix, raw[:-1] + bytes([raw[-1] ^ 1]))]:
            with self.assertRaisesRegex(ValueError, 'Exact Heretic'):
                adapter.header(cfg, pre, data)

    def test_duplicate_metadata_rejects(self):
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            adapter.unique([('tensor', 1), ('tensor', 2)])


if __name__ == '__main__':
    unittest.main()

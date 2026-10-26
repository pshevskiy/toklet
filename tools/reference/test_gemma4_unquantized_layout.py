"""Offline descriptor/geometry controls; no model download or third-party import."""
from pathlib import Path
import copy
import json
import os
import tempfile
import unittest
import gemma4_unquantized_layout as producer
import check_gemma4_unquantized_layout as checker

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / 'tests/fixtures/gemma-unquantized-header-v1'


def tiny():
    return {'__metadata__': {'format': 'pt'},
            'matrix': dict(dtype='BF16', shape=[1, 2], data_offsets=[0, 4]),
            'scalar': dict(dtype='BF16', shape=[], data_offsets=[4, 6])}


def parse(value):
    raw = producer.encoded(value)
    return producer.parse_header(raw, 8 + len(raw) + 6)


class Bounds(unittest.TestCase):
    def test_rank_zero_and_row_major_matrix_cover_exact_payload(self):
        self.assertEqual(parse(tiny()), tiny())

    def test_duplicate_and_nonfinite_json_reject(self):
        for raw in [b'{"x":1,"x":2}', b'{"x":NaN}', b'{"x":Infinity}']:
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                producer.document(raw)

    def test_shapes_reject_boolean_float_zero_negative_and_excess_rank(self):
        for shape in [[True, 2], [1., 2], [0, 2], [-1, 2], [1] * 5, [262145]]:
            value = tiny(); value['matrix']['shape'] = shape
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                parse(value)

    def test_offsets_reject_types_alignment_extent_and_out_of_file(self):
        for pair in [[False, 4], [0., 4], [-2, 2], [1, 5], [0, 2], [0, 8], [4, 0], [0]]:
            value = tiny(); value['matrix']['data_offsets'] = pair
            with self.subTest(pair=pair), self.assertRaises(ValueError):
                parse(value)

    def test_overlap_hole_and_missing_final_extent_reject(self):
        variants = []
        value = tiny(); value['scalar']['data_offsets'] = [2, 4]; variants.append(value)
        value = tiny(); value['scalar']['data_offsets'] = [6, 8]; variants.append(value)
        value = tiny(); del value['scalar']; variants.append(value)
        for value in variants:
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse(value)

    def test_wrong_dtype_descriptor_and_metadata_reject(self):
        value = tiny(); value['matrix']['dtype'] = 'F16'
        with self.assertRaisesRegex(ValueError, 'BF16'): parse(value)
        value = tiny(); value['matrix']['unknown'] = 1
        with self.assertRaisesRegex(ValueError, 'descriptor'): parse(value)
        value = tiny(); value['__metadata__']['format'] = 'mlx'
        with self.assertRaisesRegex(ValueError, 'metadata'): parse(value)

    def test_bounded_sizes_types_and_impossible_file_reject(self):
        raw = producer.encoded(tiny())
        for size in [True, 9, 12 * 1024**3 + 1, len(raw)]:
            with self.subTest(size=size), self.assertRaises(ValueError):
                producer.parse_header(raw, size)
        with self.assertRaises(ValueError): producer.parse_header(b'x' * (producer.MAX_HEADER + 1), 1024)

    def test_regular_read_refuses_symlink_fifo_directory_and_oversize(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); file = root / 'source'; file.write_bytes(b'abc')
            self.assertEqual(producer.read(file, 3), b'abc')
            link = root / 'link'; link.symlink_to(file)
            fifo = root / 'fifo'; os.mkfifo(fifo)
            for path, limit in [(link, 3), (fifo, 3), (root, 3), (file, 2)]:
                with self.subTest(path=path), self.assertRaises((OSError, ValueError)):
                    producer.read(path, limit)


class Captured(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.header = producer.read(FIXTURE / 'header.json', producer.MAX_HEADER)
        cls.config = producer.read(FIXTURE / 'config.json', 8192)
        cls.layout = producer.inspect(cls.header, cls.config)

    def test_actual_full_header_geometry_and_selected_bytes(self):
        value = self.layout
        self.assertEqual(value['counts'], dict(all_tensors=2011, selected_matrices=278,
                         selected_vectors=262, unused_shared_kv=60, excluded_media=1411))
        self.assertEqual(value['bytes']['selected_text'], 9257138758)
        self.assertEqual(value['bytes']['unused_shared_kv'], 37761024)
        ple = value['selected_text']['model.language_model.embed_tokens_per_layer.weight']
        self.assertEqual(ple['shape'], [262144, 8960]); self.assertEqual(ple['bytes'], 4697620480)

    def test_exact_config_and_acquired_header_identity_remain_required(self):
        with self.assertRaisesRegex(ValueError, 'pinned acquired header'):
            producer.inspect(self.header + b' ', self.config)
        config = producer.document(self.config); config['text_config']['max_position_embeddings'] = 32
        with self.assertRaisesRegex(ValueError, 'pinned official config'):
            producer.inspect(self.header, producer.encoded(config))
        with self.assertRaisesRegex(ValueError, 'exact E2B text config'):
            producer.text_shapes(config['text_config'])

    def test_independent_geometry_replay_and_eight_coherent_corruptions(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            def replay(value, index):
                path = root / str(index); raw = producer.encoded(value); path.write_bytes(raw)
                return checker.verify(FIXTURE / 'header.json', FIXTURE / 'config.json', path, producer.sha(raw))
            self.assertTrue(replay(self.layout, 'valid')['passed'])
            for mode in ['missing', 'offset', 'shape', 'unused', 'reclassify', 'flag', 'digest', 'media']:
                value = copy.deepcopy(self.layout); first = next(iter(value['selected_text']))
                if mode == 'missing': del value['selected_text'][first]
                elif mode == 'offset': value['selected_text'][first]['file_offset'] += 2
                elif mode == 'shape': value['selected_text'][first]['shape'][0] += 1
                elif mode == 'unused': value['counts']['unused_shared_kv'] = 0
                elif mode == 'reclassify':
                    name, item = value['unused_shared_kv'].popitem(); value['selected_text'][name] = item
                elif mode == 'flag': value['full_payload_verified'] = True
                elif mode == 'digest': value['advertised_container']['sha256'] = '0' * 64
                else: value['bytes']['excluded_media'] -= 2
                with self.subTest(mode=mode), self.assertRaises(ValueError): replay(value, mode)


if __name__ == '__main__':
    unittest.main()

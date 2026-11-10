"""Portable byte/schema controls; no downloads, model or numerical imports."""
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import lfm3_source as source


def header_fixture(parent=False):
    cursor = 0
    header = {'__metadata__': {'format': 'pt'}}
    for name, shape in source.expected_tensors(parent).items():
        size = 2
        for dimension in shape:
            size *= dimension
        header[name] = dict(dtype='BF16', shape=shape, data_offsets=[cursor, cursor+size])
        cursor += size
    return header, cursor


def admit(header, payload, parent=False):
    raw = json.dumps(header).encode()
    return source.plan(len(raw).to_bytes(8, 'little'), raw, 8+len(raw)+payload, parent)


def tiny():
    rows = [dict(name='dense', dtype='BF16', shape=[2, 2], start=0, end=8),
            dict(name='norm', dtype='BF16', shape=[2], start=8, end=12),
            dict(name='layer.conv.conv.weight', dtype='BF16', shape=[2, 1, 3], start=12, end=24)]
    children = [dict(name='layer.conv.conv.weight', dtype='BF16', shape=[2, 3, 1], start=0, end=12),
                dict(name='dense', dtype='BF16', shape=[2, 2], start=12, end=20),
                dict(name='norm', dtype='BF16', shape=[2], start=20, end=24)]
    header = b'{"fixture":"source"}'
    prefix = len(header).to_bytes(8, 'little')
    payload = bytes(range(24))
    child = prefix+header+payload[12:24]+payload[:8]+payload[8:12]
    return prefix, header, payload, rows, children, child


class Source(unittest.TestCase):
    def test_import_is_stdlib_only(self):
        for name in ['numpy', 'mlx.core', 'torch']:
            self.assertNotIn(name, sys.modules)

    def test_fixed_parameter_tensor_and_normalization_counts(self):
        expected = source.expected_tensors()
        self.assertEqual(len(expected), 286)
        self.assertEqual(sum(len(shape) == 1 for shape in expected.values()), 119)
        self.assertEqual(sum(name.endswith('conv.conv.weight') for name in expected), 13)
        self.assertEqual(len(source.LAYERS), 24)
        self.assertEqual(source.LAYERS.count('conv'), 13)
        self.assertEqual(source.LAYERS.count('full_attention'), 6)
        self.assertEqual(source.LAYERS.count('sliding_attention'), 5)
        for parent in [False, True]:
            header, payload = header_fixture(parent)
            self.assertEqual(payload, 3457603072)
            self.assertEqual(len(admit(header, payload, parent)), 286)

    def test_headers_reject_unexpected_names_types_shapes_and_spans(self):
        header, payload = header_fixture()
        name = 'model.layers.0.conv.conv.weight'
        for field, got in [('dtype', 'F16'), ('shape', [2048, 1, 7]), ('shape', [True, 7, 1]),
                           ('data_offsets', [True, 28672]), ('data_offsets', [0, 28673]),
                           ('data_offsets', [-1, 28671])]:
            damaged = copy.deepcopy(header)
            damaged[name][field] = got
            with self.assertRaises(ValueError):
                admit(damaged, payload)
        for action in ['missing', 'extra', 'metadata']:
            damaged = copy.deepcopy(header)
            if action == 'missing':
                del damaged[name]
            elif action == 'extra':
                damaged['unexpected'] = copy.deepcopy(damaged[name])
            else:
                damaged['__metadata__']['format'] = True
            with self.assertRaises(ValueError):
                admit(damaged, payload)

    def test_duplicate_json_and_header_extent_reject(self):
        raw = b'{"x":1,"x":2}'
        with self.assertRaises(ValueError):
            source.plan(len(raw).to_bytes(8, 'little'), raw, 100)
        for prefix, header, size in [(b'123', b'{}', 20), ((3).to_bytes(8, 'little'), b'{}', 20),
                                     ((2).to_bytes(8, 'little'), b'{}', True)]:
            with self.assertRaises(ValueError):
                source.plan(prefix, header, size)

    def test_singleton_axis_transform_preserves_each_scalar_position(self):
        for channels, taps in [(2, 3), (2048, 7)]:
            for channel in range(channels):
                for tap in range(taps):
                    self.assertEqual((channel*1+0)*taps+tap, (channel*taps+tap)*1+0)
        parent = dict(name='x.conv.conv.weight', shape=[2, 1, 3])
        self.assertTrue(source.compatible(parent, dict(shape=[2, 3, 1])))
        for shape in [[2, 1, 3], [2, 3, 2], [3, 2, 1]]:
            self.assertFalse(source.compatible(parent, dict(shape=shape)))

    def compare(self, parent_data=None, child_data=None, pin=None):
        prefix, header, payload, rows, children, child = tiny()
        data = prefix+header+payload
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'child'
            path.write_bytes(child if child_data is None else child_data)
            fd = os.open(path, os.O_RDONLY)
            journal = io.StringIO()
            try:
                result = source.compare_stream(io.BytesIO(data if parent_data is None else parent_data),
                    prefix, header, len(data), pin or hashlib.sha256(data).hexdigest(),
                    rows, fd, len(prefix+header), children, journal)
            finally:
                os.close(fd)
        return result, [json.loads(line) for line in journal.getvalue().splitlines()]

    def test_actual_byte_comparison_uses_tensor_identity_not_physical_order(self):
        result, journal = self.compare()
        self.assertEqual(result['exact_tensor_count'], 3)
        self.assertEqual(len(journal), 3)
        self.assertTrue(all(row['whole_parent_authenticated'] is False for row in journal))
        self.assertEqual(result['tensors'][0]['sha256'], hashlib.sha256(bytes(range(8))).hexdigest())

    def test_corruption_truncation_tail_header_and_whole_hash_reject(self):
        prefix, header, payload, _, _, child = tiny()
        data = prefix+header+payload
        altered = bytearray(child)
        altered[-1] ^= 1
        with self.assertRaisesRegex(ValueError, 'tensor bytes differ'):
            self.compare(child_data=bytes(altered))
        for got in [data[:-1], data+b'x', bytes([data[0]^1])+data[1:]]:
            with self.assertRaises(ValueError):
                self.compare(parent_data=got)
        with self.assertRaisesRegex(ValueError, 'SHA256 differs'):
            self.compare(pin='0'*64)

    def test_bounded_reader_tolerates_short_fragments_but_not_unbounded_requests(self):
        class Fragmented(io.BytesIO):
            def read(self, count=-1):
                return super().read(min(count, 3))
        self.assertEqual(source.exact(Fragmented(b'0123456789'), 10), b'0123456789')
        for count in [True, -1, source.BLOCK+1]:
            with self.assertRaises(ValueError):
                source.exact(io.BytesIO(b'x'), count)
        with self.assertRaises(ValueError):
            source.exact(io.BytesIO(b'x'), 2)

    def test_multiple_blocks_and_short_tail_compare_and_reject_boundary_corruption(self):
        header = b'{}'
        prefix = len(header).to_bytes(8, 'little')
        payload = bytes(range(256)) * (1025*2048*2//256)
        rows = [dict(name='matrix', dtype='BF16', shape=[1025, 2048], start=0, end=len(payload))]
        whole = prefix+header+payload
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'child'
            path.write_bytes(whole)
            fd = os.open(path, os.O_RDONLY)
            try:
                result = source.compare_stream(io.BytesIO(whole), prefix, header, len(whole),
                    hashlib.sha256(whole).hexdigest(), rows, fd, 10, rows, io.StringIO())
                blocks = result['tensors'][0]['blocks']
                self.assertEqual([b['bytes'] for b in blocks], [source.BLOCK, 4096])
                self.assertEqual([b['parent_offset'] for b in blocks], [10, 10+source.BLOCK])
                self.assertEqual(result['tensors'][0]['sha256'], hashlib.sha256(payload).hexdigest())
                altered = bytearray(whole)
                altered[10+source.BLOCK] ^= 1
                with self.assertRaisesRegex(ValueError, 'tensor bytes differ'):
                    source.compare_stream(io.BytesIO(altered), prefix, header, len(whole),
                        hashlib.sha256(altered).hexdigest(), rows, fd, 10, rows, io.StringIO())
            finally:
                os.close(fd)

    def test_structural_comparison_rejects_gaps_overlaps_duplicate_and_boolean_offsets(self):
        _, _, _, rows, _, _ = tiny()
        for field, value in [('start', 9), ('start', 7), ('start', True), ('name', 'dense')]:
            damaged = copy.deepcopy(rows)
            damaged[1][field] = value
            with self.assertRaises(ValueError):
                source.validate_spans(damaged, 24)
        with self.assertRaises(ValueError):
            source.validate_spans(rows, 26)

    def test_retained_descriptor_rehash_checks_path_identity_and_exact_bytes(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'child'
            data = b'original'
            path.write_bytes(data)
            fd = os.open(path, os.O_RDONLY)
            try:
                with mock.patch.object(source, 'SIZE', len(data)), mock.patch.object(source, 'CHILD_SHA', hashlib.sha256(data).hexdigest()):
                    source.verify_child(fd, path)
                    path.rename(path.with_name('old'))
                    path.write_bytes(data)
                    with self.assertRaisesRegex(ValueError, 'descriptor/path identity'):
                        source.verify_child(fd, path)
            finally:
                os.close(fd)


if __name__ == '__main__':
    unittest.main()

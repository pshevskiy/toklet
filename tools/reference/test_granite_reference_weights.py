"""Portable mutation, exact-read, coverage and ownership controls; no NumPy."""
import copy
import hashlib
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import granite_convert_layout as layout
import granite_reference_weights as reader


def fixture(root):
    recipes = [dict(module='model.embed_tokens', shape=[259, 64], bits=4,
                    group_size=64, mode='affine'),
               dict(module='lm_head', shape=[2, 64], bits=8,
                    group_size=64, mode='affine')]
    planned = layout.output_layout(recipes, [dict(name='model.norm.weight', shape=[4],
                                                dtype='BF16', bytes=8)])
    body = bytearray(planned['body_bytes'])
    tiles = []
    for recipe in recipes:
        name = recipe['module'] + '.weight'
        count, columns = recipe['shape']
        matrix = planned['matrices'][name]
        for first in range(0, count, 256):
            rows = min(256, count - first)
            codes = b''.join(bytes([r % 16 * 17 if recipe['bits'] == 4 else r % 256])
                             * matrix['fields']['weight']['row_stride']
                             for r in range(first, first + rows))
            buffers = dict(weight=codes, scales=bytes.fromhex('803f') * rows,
                           biases=bytes(2 * rows))
            fields = {}
            for field, item in matrix['fields'].items():
                start = planned['header'][item['name']]['data_offsets'][0]
                start += first * item['row_stride']
                raw = buffers[field]
                body[start:start + len(raw)] = raw
                fields[field] = dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
            tiles.append(dict(name=name, first_row=first, rows=rows, columns=columns,
                              bits=recipe['bits'], fields=fields))
    norm = bytes.fromhex('803f004040408040')
    start, end = planned['header']['model.norm.weight']['data_offsets']
    body[start:end] = norm
    norms = [dict(name='model.norm.weight', bytes=8,
                  copied_BF16_sha256=hashlib.sha256(norm).hexdigest())]
    raw = planned['prefix'] + body
    path = root / 'model.safetensors'
    path.write_bytes(raw)
    expected = dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
    return path, planned, expected, tiles, norms


class ConsumedChild(unittest.TestCase):
    def test_complete_components_tail_norm_and_owned_descriptor_close(self):
        with tempfile.TemporaryDirectory() as folder:
            args = fixture(Path(folder))
            with reader.Child(*args) as child:
                descriptor = child.fd
                first = child.tile('model.embed_tokens.weight', 0)
                self.assertEqual(first['weight'][:64], bytes(32) + bytes([0x11]) * 32)
                tail = child.tile('model.embed_tokens.weight', 256)
                self.assertEqual(tail['weight'], bytes(32) + bytes([0x11]) * 32 + bytes([0x22]) * 32)
                head = child.tile('lm_head.weight', 0)
                self.assertEqual(head['weight'], bytes(64) + bytes([1]) * 64)
                self.assertEqual(child.norm('model.norm.weight'), bytes.fromhex('803f004040408040'))
                receipt = child.endpoint()
                self.assertEqual(receipt['component_reads'], 9)
                self.assertEqual(receipt['norm_reads'], 1)
                self.assertEqual(receipt['consumed_bytes'], args[1]['body_bytes'])
            self.assertIsNone(child.fd)
            with self.assertRaises(OSError):
                os.fstat(descriptor)
            with self.assertRaises(ValueError):
                child.__enter__()

    def test_missing_duplicate_mistyped_or_partial_bindings_reject_before_open(self):
        with tempfile.TemporaryDirectory() as folder:
            path, planned, expected, tiles, norms = fixture(Path(folder))
            damaged = []
            damaged.append((tiles[:-1], norms))
            damaged.append((tiles + [tiles[0]], norms))
            damaged.append((tiles, []))
            damaged.append((tiles, norms + norms))
            for key, value in [('first_row', False), ('rows', True), ('columns', 128), ('bits', 8)]:
                changed = copy.deepcopy(tiles)
                changed[0][key] = value
                damaged.append((changed, norms))
            for key, value in [('bytes', True), ('sha256', 'x' * 64)]:
                changed = copy.deepcopy(tiles)
                changed[0]['fields']['weight'][key] = value
                damaged.append((changed, norms))
            for changed, changed_norms in damaged:
                with patch.object(reader.os, 'open') as opened:
                    with self.assertRaises(ValueError):
                        reader.Child(path, planned, expected, changed, changed_norms)
                    opened.assert_not_called()

    def test_header_whole_payload_and_extent_failures_close_the_descriptor(self):
        for damage in ['header', 'payload', 'extent']:
            with self.subTest(damage=damage), tempfile.TemporaryDirectory() as folder:
                args = fixture(Path(folder))
                path = args[0]
                raw = bytearray(path.read_bytes())
                if damage == 'header':
                    raw[0] ^= 1
                elif damage == 'payload':
                    raw[-1] ^= 1
                else:
                    raw.append(0)
                path.write_bytes(raw)
                child = reader.Child(*args)
                with self.assertRaises(ValueError):
                    child.__enter__()
                self.assertIsNone(child.fd)

    def test_actual_tile_mutation_after_whole_admission_is_detected(self):
        with tempfile.TemporaryDirectory() as folder:
            args = fixture(Path(folder))
            with self.assertRaisesRegex(ValueError, 'consumed child component'):
                with reader.Child(*args) as child:
                    item = args[1]['header']['model.embed_tokens.weight']
                    offset = args[1]['origin'] + item['data_offsets'][0]
                    with args[0].open('r+b') as mutation:
                        mutation.seek(offset)
                        mutation.write(b'\xff')
                    child.tile('model.embed_tokens.weight', 0)
            self.assertEqual(child.component_reads, 0)
            self.assertIsNone(child.fd)

    def test_actual_norm_mutation_is_detected_before_bytes_are_returned(self):
        with tempfile.TemporaryDirectory() as folder:
            args = fixture(Path(folder))
            with self.assertRaisesRegex(ValueError, 'consumed norm'):
                with reader.Child(*args) as child:
                    start = args[1]['header']['model.norm.weight']['data_offsets'][0]
                    with args[0].open('r+b') as mutation:
                        mutation.seek(args[1]['origin'] + start)
                        mutation.write(b'\x00')
                    child.norm('model.norm.weight')
            self.assertEqual(child.norm_reads, 0)

    def test_byte_identical_path_replacement_rejects_at_endpoint(self):
        with tempfile.TemporaryDirectory() as folder:
            args = fixture(Path(folder))
            with self.assertRaisesRegex(ValueError, 'identity differs'):
                with reader.Child(*args):
                    raw = args[0].read_bytes()
                    args[0].rename(Path(folder) / 'previous')
                    args[0].write_bytes(raw)

    def test_body_failure_survives_cleanup_and_closed_descriptor(self):
        with tempfile.TemporaryDirectory() as folder:
            args = fixture(Path(folder))
            with self.assertRaisesRegex(RuntimeError, 'primary work failed'):
                with reader.Child(*args) as child:
                    args[0].write_bytes(b'changed')
                    raise RuntimeError('primary work failed')
            self.assertIsNone(child.fd)

    def test_partial_reads_and_exact_eof_remain_bounded(self):
        with tempfile.TemporaryDirectory() as folder:
            args = fixture(Path(folder))
            with args[0].open('rb') as file:
                original = os.pread
                with patch.object(reader.os, 'pread',
                                  side_effect=lambda fd, n, at: original(fd, min(n, 17), at)):
                    self.assertEqual(reader.read_exact(file.fileno(), 100, 0), args[0].read_bytes()[:100])
                for count, offset in [(True, 0), (1, False), (reader.MAX_READ + 1, 0), (1, -1)]:
                    with self.assertRaises(ValueError):
                        reader.read_exact(file.fileno(), count, offset)
                with self.assertRaises(ValueError):
                    reader.read_exact(file.fileno(), 1, args[2]['bytes'])

    def test_symlink_and_fifo_never_admit_a_regular_child(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            args = fixture(root)
            link = root / 'link'
            link.symlink_to(args[0])
            with self.assertRaises(OSError):
                with reader.Child(link, *args[1:]):
                    self.fail('symlink admitted')
            fifo = root / 'fifo'
            os.mkfifo(fifo)
            child = reader.Child(fifo, *args[1:])
            with self.assertRaises(ValueError):
                child.__enter__()
            self.assertIsNone(child.fd)


if __name__ == '__main__':
    unittest.main()

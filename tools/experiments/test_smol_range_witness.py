"""Offline synthetic tests only. This file never opens the real model directory."""
import copy
from contextlib import ExitStack
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import struct
import tempfile
import unittest


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


ROOT = Path(__file__).resolve().parents[2]
p = module('smol_range_witness', ROOT / 'tools/reference/smol_range_witness.py')
r = module('smol_range_replay', Path(__file__).with_name('smol_range_replay.py'))


def fixture(edit=None):
    # Small affine matrix, its metadata, and one norm; dictionary order differs
    # from byte order. No real model data or numerical qualification.
    entries = [('a.weight', 'U32', [1, 16], bytes(range(64))),
               ('a.scales', 'F16', [1, 1], struct.pack('<e', 0.125)),
               ('a.biases', 'F16', [1, 1], struct.pack('<e', -0.5)),
               ('norm.weight', 'F16', [64], struct.pack('<64e', *([1.0] * 64)))]
    header = {'__metadata__': {'format': 'mlx'}}
    payload = bytearray()
    expected = {}
    for name, dtype, shape, data in entries:
        start = len(payload)
        payload.extend(data)
        header[name] = dict(dtype=dtype, shape=shape, data_offsets=[start, len(payload)])
        expected[name] = (dtype, shape)
    if edit:
        edit(header)
    # JSON alphabetical order makes physical traversal differ from name order.
    raw_header = json.dumps(header, sort_keys=True, separators=(',', ':')).encode()
    raw_header += b' ' * ((8 - len(raw_header) % 8) % 8)
    raw = struct.pack('<Q', len(raw_header)) + raw_header + payload
    return bytes(raw), expected


def extract(raw, expected):
    return p.extract_shard(io.BytesIO(raw), (len(raw), hashlib.sha256(raw).hexdigest()), expected)


class Tests(unittest.TestCase):
    def test_regular_file_symlink_fifo_and_size_admission(self):
        with tempfile.TemporaryDirectory(prefix='smol-range-synthetic-') as folder:
            root = Path(folder)
            (root/'regular').write_bytes(b'hi')
            (root/'link').symlink_to(root/'regular')
            os.mkfifo(root/'fifo')
            with ExitStack() as stack:
                directory = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
                stack.callback(os.close, directory)
                regular = r.open_regular(stack, directory, 'regular', 2)
                self.assertEqual(regular.read(), b'hi')
                with self.assertRaises(OSError):
                    r.open_regular(stack, directory, 'link', 2)
                for name, size in [('fifo', 0), ('regular', 3)]:
                    with self.assertRaises(ValueError):
                        r.open_regular(stack, directory, name, size)
                with self.assertRaises(ValueError):
                    r.open_regular(stack, directory, '../regular', 2)

    def test_complete_coverage_exact_prefix_header_and_each_raw_digest(self):
        raw, expected = fixture()
        witness, prefix, header = extract(raw, expected)
        self.assertEqual(prefix + header, raw[:witness['data_start']])
        self.assertEqual(witness['payload_bytes'], 196)
        for tensor in witness['tensors']:
            first, last = tensor['file_offsets']
            self.assertEqual(tensor['sha256'], hashlib.sha256(raw[first:last]).hexdigest())
        independently = r.partition(r.parse(header), len(prefix + header), len(raw), expected)
        self.assertEqual([x[2] for x in independently], [x['name'] for x in witness['tensors']])

    def test_first_authentication_is_bounded_and_does_not_parse_invalid_json(self):
        raw = b'not json at all'
        self.assertEqual(p.authenticate(io.BytesIO(raw), len(raw), p.sha(raw))['bytes'], len(raw))
        for size, digest in [(len(raw)-1, p.sha(raw)), (len(raw)+1, p.sha(raw)), (len(raw), '0'*64)]:
            with self.assertRaises(ValueError):
                p.authenticate(io.BytesIO(raw), size, digest)

    def test_invalid_ranges_dtype_shape_boolean_overflow_and_extra_reject(self):
        edits = [lambda h: h['a.weight']['data_offsets'].__setitem__(0, 1),
                 lambda h: h['a.scales']['data_offsets'].__setitem__(0, 63),
                 lambda h: h['norm.weight']['data_offsets'].__setitem__(1, 197),
                 lambda h: h['a.weight'].__setitem__('dtype', 'F32'),
                 lambda h: h['a.weight'].__setitem__('shape', [True, 16]),
                 lambda h: h['a.weight']['data_offsets'].__setitem__(0, False),
                 lambda h: h['a.weight']['data_offsets'].__setitem__(1, 1 << 70),
                 lambda h: h['a.weight'].__setitem__('extra', 0),
                 lambda h: h.update(unexpected=dict(dtype='F16', shape=[1], data_offsets=[0, 2])),
                 lambda h: h.pop('a.biases')]
        for edit in edits:
            with self.subTest(edit=repr(edit)):
                raw, expected = fixture(edit)
                with self.assertRaises(ValueError):
                    extract(raw, expected)
                header_size = int.from_bytes(raw[:8], 'little')
                with self.assertRaises(ValueError):
                    r.partition(r.parse(raw[8:8+header_size]), 8+header_size, len(raw), expected)

    def test_duplicate_header_keys_and_oversized_prefix_reject(self):
        header = b'{"a":0,"a":1}'
        raw = struct.pack('<Q', len(header)) + header + b'aa'
        with self.assertRaises(ValueError):
            extract(raw, {})
        with self.assertRaises(ValueError):
            r.parse(header)
        raw = struct.pack('<Q', p.HEADER_MAX+1) + b'aa'
        with self.assertRaises(ValueError):
            extract(raw, {})

    def test_gap_overlap_and_tail_reject_independently(self):
        raw, expected = fixture()
        witness, prefix, header = extract(raw, expected)
        for label in ['gap', 'overlap', 'tail']:
            parsed = r.parse(header)
            if label == 'gap':
                parsed['a.biases']['data_offsets'] = [67, 69]
            elif label == 'overlap':
                parsed['a.biases']['data_offsets'] = [65, 67]
            size = len(raw) + (1 if label == 'tail' else 0)
            with self.assertRaises(ValueError):
                r.partition(parsed, len(prefix+header), size, expected)

    def test_consumed_bytes_catch_write_restore_despite_matching_pre_post_hash(self):
        raw, expected = fixture()
        origin = 8 + int.from_bytes(raw[:8], 'little')

        class ABA(io.BytesIO):
            def __init__(self, data):
                super().__init__(data)
                self.mutate_reads = False
            def read(self, count=-1):
                position = self.tell()
                data = super().read(count)
                if self.mutate_reads and position == origin and data:
                    return bytes([data[0] ^ 1]) + data[1:]
                return data

        file = ABA(raw)
        p.authenticate(file, len(raw), p.sha(raw))
        file.mutate_reads = True
        with self.assertRaisesRegex(ValueError, 'Extraction-consumed'):
            p.extract_shard(file, (len(raw), p.sha(raw)), expected)
        file.mutate_reads = False
        self.assertEqual(p.authenticate(file, len(raw), p.sha(raw))['sha256'], p.sha(raw))

    def test_changed_header_after_preauthentication_rejects(self):
        raw, expected = fixture()
        p.authenticate(io.BytesIO(raw), len(raw), p.sha(raw))
        altered = raw.replace(b'"mlx"', b'"xxx"')
        with self.assertRaisesRegex(ValueError, 'Extraction-consumed'):
            p.extract_shard(io.BytesIO(altered), (len(raw), p.sha(raw)), expected)

    def test_repeated_tied_range_is_checked_every_time(self):
        raw, expected = fixture()
        witness, _, _ = extract(raw, expected)
        file = io.BytesIO(raw)
        record = witness['tensors'][0]
        first = p.reread(file, record)
        second = p.reread(file, record)
        self.assertEqual(first, second)
        position = record['file_offsets'][0]
        altered = bytearray(raw)
        altered[position] ^= 1
        with self.assertRaisesRegex(ValueError, 'Repeated consumed range'):
            p.reread(io.BytesIO(altered), record)
        self.assertNotEqual(r.range_hash(io.BytesIO(altered), *record['file_offsets']), first['sha256'])

    def test_late_read_failure_and_truncation_cannot_create_witness(self):
        raw, expected = fixture()
        for truncated in [raw[:7], raw[:-1], raw[:len(raw)//2]]:
            with self.assertRaises(ValueError):
                p.extract_shard(io.BytesIO(truncated), (len(raw), p.sha(raw)), expected)
        class Failure(io.BytesIO):
            def read(self, size=-1):
                if self.tell() > len(raw)//2:
                    raise OSError('synthetic read failure')
                return super().read(size)
        with self.assertRaises(OSError):
            p.extract_shard(Failure(raw), (len(raw), p.sha(raw)), expected)

    def test_independent_witness_schema_rejects_boolean_offsets_and_bad_digest(self):
        raw, expected = fixture()
        witness, _, _ = extract(raw, expected)
        r.witness_ranges(witness['tensors'])
        for field, value in [('payload_offsets', [False, 64]), ('file_offsets', [0.0, 64]),
                             ('bytes', True), ('sha256', 'G'*64), ('shape', [True, 16])]:
            altered = copy.deepcopy(witness['tensors'])
            altered[0][field] = value
            with self.assertRaises(ValueError):
                r.witness_ranges(altered)

    def test_smol_inventory_has_every_affine_metadata_pair_and_tied_head_absence(self):
        inventory = p.expected_smol()
        self.assertEqual(len(inventory), 694)
        self.assertEqual(sum(dtype == 'U32' for dtype, _ in inventory.values()), 211)
        self.assertNotIn('lm_head.weight', inventory)
        for name, (dtype, _) in inventory.items():
            if dtype == 'U32':
                self.assertIn(name.removesuffix('.weight') + '.scales', inventory)
                self.assertIn(name.removesuffix('.weight') + '.biases', inventory)
        config = dict(hidden_size=576, num_hidden_layers=30, intermediate_size=1536, vocab_size=49152,
                      num_attention_heads=9, num_key_value_heads=3, tie_word_embeddings=True,
                      quantization={'bits': 8, 'group_size': 64})
        self.assertEqual(r.derived_inventory(config), inventory)


if __name__ == '__main__':
    unittest.main(verbosity=2)

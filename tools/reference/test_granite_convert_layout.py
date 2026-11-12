"""Portable span, output-layout, partial-write and index rejection controls."""
import copy
import json
import os
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

import granite_convert_layout as convert


def recipe():
    matrices = [dict(module=name, shape=[3, 64], bits=bits, group_size=64, mode='affine')
                for name, bits in [('model.embed_tokens', 4), ('lm_head', 8)]]
    norms = [dict(name='model.norm.weight', shape=[64], dtype='BF16', bytes=128)]
    return matrices, norms


class Layout(unittest.TestCase):
    def test_exact_physical_inventory_offsets_and_dtype_layout(self):
        out = convert.output_layout(*recipe())
        self.assertEqual(out['body_bytes'], 440)
        self.assertEqual(out['origin'] % 8, 0)
        count = struct.unpack('<Q', out['prefix'][:8])[0]
        self.assertEqual(out['origin'], 8 + count)
        self.assertEqual(json.loads(out['prefix'][8:]), out['header'])
        self.assertEqual(out['header']['model.embed_tokens.weight']['shape'], [3, 8])
        self.assertEqual(out['header']['lm_head.weight']['shape'], [3, 16])
        spans = sorted(v['data_offsets'] for k, v in out['header'].items() if k != '__metadata__')
        self.assertEqual(spans[0][0], 0)
        self.assertTrue(all(a[1] == b[0] for a, b in zip(spans, spans[1:])))
        self.assertEqual(spans[-1][1], 440)

    def test_complete_and_partial_row_writes_preserve_all_components_and_norm(self):
        out = convert.output_layout(*recipe())
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'child.safetensors'
            with path.open('w+b') as file:
                file.truncate(out['file_bytes'])
                convert.write_all(file.fileno(), out['prefix'], 0)
                for name, plan in out['matrices'].items():
                    complete = {key: bytes((i % 251 for i in range(3 * item['row_stride'])))
                                for key, item in plan['fields'].items()}
                    for first, rows in [(0, 2), (2, 1)]:
                        tile = {key: raw[first * plan['fields'][key]['row_stride']:
                                         (first + rows) * plan['fields'][key]['row_stride']]
                                for key, raw in complete.items()}
                        convert.write_matrix_tile(file.fileno(), out, name, first, rows, tile)
                    for key, item in plan['fields'].items():
                        start = out['origin'] + out['header'][item['name']]['data_offsets'][0]
                        self.assertEqual(os.pread(file.fileno(), len(complete[key]), start), complete[key])
                norm = b'\x80\x3f' * 64
                convert.write_norm(file.fileno(), out, 'model.norm.weight', norm)
                start = out['origin'] + out['header']['model.norm.weight']['data_offsets'][0]
                self.assertEqual(os.pread(file.fileno(), 128, start), norm)
                self.assertEqual(os.fstat(file.fileno()).st_size, out['file_bytes'])

    def test_all_components_and_row_bounds_are_checked_before_any_write(self):
        out = convert.output_layout(*recipe())
        name = 'model.embed_tokens.weight'
        complete = dict(weight=bytes(32), scales=bytes(2), biases=bytes(2))
        with tempfile.TemporaryDirectory() as folder:
            with (Path(folder) / 'child').open('w+b') as file:
                file.truncate(out['file_bytes'])
                for broken in [dict(complete, scales=b''), dict(complete, biases=None),
                               dict(complete, extra=b'')]:
                    with self.assertRaises(ValueError):
                        convert.write_matrix_tile(file.fileno(), out, name, 0, 1, broken)
                for first, rows in [(-1, 1), (3, 1), (2, 2), (0, True), (0, 257)]:
                    with self.assertRaises(ValueError):
                        convert.write_matrix_tile(file.fileno(), out, name, first, rows, complete)
                self.assertEqual(os.pread(file.fileno(), out['file_bytes'], 0), bytes(out['file_bytes']))

    def test_partial_writes_finish_exactly_and_zero_progress_fails(self):
        with tempfile.TemporaryDirectory() as folder:
            with (Path(folder) / 'child').open('w+b') as file:
                write = os.pwrite

                def partial(fd, raw, offset):
                    return write(fd, raw[:max(1, len(raw) // 2)], offset)

                with patch.object(convert.os, 'pwrite', side_effect=partial):
                    convert.write_all(file.fileno(), b'bounded bytes', 5)
                self.assertEqual(os.pread(file.fileno(), 13, 5), b'bounded bytes')
                with patch.object(convert.os, 'pwrite', return_value=0):
                    with self.assertRaises(ValueError):
                        convert.write_all(file.fileno(), b'x', 0)

    def test_source_header_spans_are_exact_and_must_match_logical_roles(self):
        header = {'x.weight': dict(dtype='BF16', shape=[3, 64], data_offsets=[0, 384])}
        raw = json.dumps(header).encode()
        prefix = struct.pack('<Q', len(raw))
        spans = convert.source_spans(prefix, raw, 8 + len(raw) + 384, {'x.weight': [3, 64]})
        self.assertEqual(spans, [dict(name='x.weight', shape=[3, 64], start=0, end=384)])
        for expected in [{}, {'x.weight': [3, 128]}]:
            with self.assertRaises(ValueError):
                convert.source_spans(prefix, raw, 8 + len(raw) + 384, expected)
        for changed in [dict(dtype='F16'), dict(data_offsets=[1, 385]), dict(shape=[True, 64])]:
            bad = copy.deepcopy(header)
            bad['x.weight'].update(changed)
            data = json.dumps(bad).encode()
            with self.assertRaises(ValueError):
                convert.source_spans(struct.pack('<Q', len(data)), data, 8 + len(data) + 384,
                                     {'x.weight': [3, 64]})
        with self.assertRaises(ValueError):
            convert.source_spans(prefix, raw, 8 + len(raw) + 385, {'x.weight': [3, 64]})

    def test_stale_index_total_mapping_and_duplicate_source_ownership_reject(self):
        objects = [('a.safetensors', [dict(name='x.weight')]),
                   ('b.safetensors', [dict(name='norm.weight')])]
        correct = dict(metadata=dict(total_size=512),
                       weight_map={'x.weight': 'a.safetensors', 'norm.weight': 'b.safetensors'})
        self.assertEqual(convert.validate_index(json.dumps(correct), objects, 512), correct['weight_map'])
        parameter_count = dict(correct, metadata=dict(total_size=512, total_parameters=256))
        self.assertEqual(convert.validate_index(json.dumps(parameter_count), objects, 512), correct['weight_map'])
        for count in [255, True, '256']:
            with self.assertRaisesRegex(ValueError, 'parameter count differs'):
                convert.validate_index(json.dumps(dict(correct, metadata=dict(total_size=512, total_parameters=count))),
                                       objects, 512)
        for metadata in [dict(total_size=384), dict(total_size=True), dict(total_size=512, extra=1)]:
            bad = dict(correct, metadata=metadata)
            with self.assertRaises(ValueError):
                convert.validate_index(json.dumps(bad), objects, 512)
        with self.assertRaises(ValueError):
            convert.validate_index(json.dumps(dict(correct, weight_map={'x.weight': 'b.safetensors'})),
                                   objects, 512)
        with self.assertRaises(ValueError):
            convert.validate_index(json.dumps(correct), objects + [objects[0]], 512)

    def test_invalid_recipes_duplicates_and_output_budget_reject_without_payload_allocation(self):
        matrices, norms = recipe()
        for changed in [dict(shape=[True, 64]), dict(shape=[3, 65]), dict(bits=6),
                        dict(group_size=32), dict(mode='mxfp4')]:
            bad = [dict(matrices[0], **changed), matrices[1]]
            with self.assertRaises(ValueError):
                convert.output_layout(bad, norms)
        with self.assertRaises(ValueError):
            convert.output_layout(matrices + [matrices[0]], norms)
        with self.assertRaises(ValueError):
            convert.output_layout(matrices, [dict(norms[0], dtype='F32')])
        huge = [dict(module='m' + str(i), shape=[131072, 8192], bits=8, group_size=64, mode='affine')
                for i in range(7)]
        with self.assertRaisesRegex(ValueError, 'engine ceiling'):
            convert.output_layout(huge, [])


if __name__ == '__main__':
    unittest.main()

"""Portable selected-layout, complete-header, index and bounded-I/O controls."""
import copy
import hashlib
import json
import os
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

import ministral3_convert_layout as layout

FIXTURES = Path(__file__).resolve().parents[2] / 'tests/fixtures/ministral3-conversion-v1'


def fixture(name):
    manifest = json.loads((FIXTURES / 'manifest.json').read_bytes())
    raw = (FIXTURES / name).read_bytes()
    pin = manifest['files'][name]
    if len(raw) != pin['bytes'] or hashlib.sha256(raw).hexdigest() != pin['sha256']:
        raise ValueError('Tracked metadata fixture differs')
    return raw


def tiny_recipe():
    matrices = [dict(module='language_model.model.' + name, shape=[3, 64],
                     bits=bits, group_size=64, mode='affine')
                for name, bits in [('embed_tokens', 8), ('layers.0.self_attn.q_proj', 4)]]
    norms = [dict(name='language_model.model.norm.weight', shape=[64], dtype='BF16', bytes=128)]
    return matrices, norms


def source_header(shape=None):
    shape = shape or [2, 3, 2, 2]
    count = 1
    for value in shape:
        count *= value
    name = 'vision_tower.patch_conv.weight'
    declared = {name: dict(dtype='BF16', shape=shape, data_offsets=[0, 2 * count])}
    inventory = {name: dict(dtype='BF16', shape=shape)}
    return declared, inventory


def spans_for(declared, inventory, extra=0):
    raw = json.dumps(declared).encode()
    end = max(item['data_offsets'][1] for item in declared.values())
    return layout.source_spans(struct.pack('<Q', len(raw)), raw,
                               8 + len(raw) + end + extra, inventory)


class SelectedLayout(unittest.TestCase):
    def test_exact_selected_recipe_inventory_and_prefix_without_payload_allocation(self):
        matrices, norms = layout.selected_recipe()
        out = layout.output_layout(matrices, norms)
        expected = layout.EXPECTED_OUTPUT
        self.assertEqual((len(matrices), len(norms), len(out['header']) - 1), (183, 53, 602))
        self.assertEqual(sum(item['bits'] == 4 for item in matrices), 130)
        self.assertEqual(sum(item['bits'] == 8 for item in matrices), 53)
        self.assertEqual(out['body_bytes'], expected['body_bytes'])
        self.assertEqual(out['file_bytes'], expected['file_bytes'])
        self.assertEqual(hashlib.sha256(out['prefix']).hexdigest(), expected['header_sha256'])
        self.assertEqual(sum((item['shape'][0] + 223) // 224 for item in matrices), 4252)
        self.assertEqual(sum(item['shape'][0] * item['shape'][1] for item in matrices), 3428843520)
        inventory = layout.selected_inventory()
        self.assertEqual(len(inventory), 458)
        self.assertEqual(sum(name not in out['logical'] for name in inventory), 222)
        self.assertNotIn('language_model.lm_head.weight', out['logical'])
        q = out['matrices']['language_model.model.layers.0.self_attn.q_proj.weight']
        self.assertEqual(q['shape'], [4096, 3072])
        self.assertEqual(out['origin'] % 8, 0)

    def test_both_genuine_complete_headers_and_indexes_bind_all458_tensors(self):
        manifest = json.loads((FIXTURES / 'manifest.json').read_bytes())
        inventory = layout.selected_inventory()
        for source in manifest['sources']:
            objects, total, names = [], 0, set()
            for container in source['containers']:
                prefix = fixture(container['files'][0]['file'])
                header = fixture(container['files'][1]['file'])
                spans = layout.source_spans(prefix, header, container['expected_bytes'], inventory)
                self.assertTrue(names.isdisjoint(item['name'] for item in spans))
                names.update(item['name'] for item in spans)
                total += sum(item['end'] - item['start'] for item in spans)
                objects.append((container['container'], spans))
            index = fixture(source['label'] + '-model.safetensors.index.json')
            if source['label'] == 'ministral_original':
                # Preserve the genuine index and reject its inconsistent count.
                # The numerical difference does not establish writer intent.
                with self.assertRaisesRegex(ValueError, 'parameter count differs'):
                    layout.validate_index(index, objects, total, inventory)
                mapping = json.loads(index)['weight_map']
                self.assertEqual(json.loads(index)['metadata']['total_size'], total)
            else:
                mapping = layout.validate_index(index, objects, total, inventory)
            self.assertEqual(set(mapping), set(inventory))
            self.assertEqual(names, set(inventory))
            self.assertEqual(len(mapping), 458)

    def test_rank4_span_is_validated_without_reading_or_allocating_payload(self):
        header, inventory = source_header()
        self.assertEqual(spans_for(header, inventory), [dict(name='vision_tower.patch_conv.weight',
                         dtype='BF16', shape=[2, 3, 2, 2], start=0, end=48)])
        for shape in ([2, 3, 2], [True], [0], [-1], [131073], [131072] * 4):
            declared, expected = source_header(shape)
            with self.assertRaises(ValueError):
                spans_for(declared, expected)

    def test_source_shapes_types_unknowns_metadata_and_tight_spans_reject(self):
        header, inventory = source_header()
        name = next(iter(header))
        for change in [dict(dtype='F16'), dict(shape=[2, 3, 2, True]),
                       dict(data_offsets=[1, 49]), dict(data_offsets=[0, 47]),
                       dict(data_offsets=[False, 48]), dict(extra=1)]:
            broken = copy.deepcopy(header)
            broken[name].update(change)
            with self.assertRaises(ValueError):
                spans_for(broken, inventory)
        with self.assertRaises(ValueError):
            spans_for(header, {'other.weight': inventory[name]})
        with self.assertRaises(ValueError):
            spans_for(header, inventory, 1)
        raw = json.dumps(header).encode()
        with self.assertRaises(ValueError):
            layout.source_spans(struct.pack('<Q', len(raw) + 1), raw, 8 + len(raw) + 48, inventory)
        duplicate = b'{"x":1,"x":2}'
        with self.assertRaises(ValueError):
            layout.source_spans(struct.pack('<Q', len(duplicate)), duplicate, len(duplicate) + 10, inventory)
        broken = dict(header, __metadata__={'format': 1})
        raw = json.dumps(broken).encode()
        with self.assertRaises(ValueError):
            layout.source_spans(struct.pack('<Q', len(raw)), raw, len(raw) + 8 + 48, inventory)

    def test_holes_overlaps_and_duplicate_cross_shard_owners_reject(self):
        inv = {name: dict(dtype='BF16', shape=[2]) for name in ['a.weight', 'b.weight']}
        for first, second in [([0, 4], [5, 9]), ([0, 4], [3, 7]), ([1, 5], [5, 9])]:
            header = {name: dict(dtype='BF16', shape=[2], data_offsets=span)
                      for name, span in zip(inv, [first, second])}
            with self.assertRaises(ValueError):
                spans_for(header, inv)
        raw = json.dumps(dict(metadata=dict(total_size=8), weight_map={'a.weight': 'a.safetensors'}))
        objects = [('a.safetensors', [dict(name='a.weight')]), ('b.safetensors', [dict(name='a.weight')])]
        with self.assertRaises(ValueError):
            layout.validate_index(raw, objects, 8, inv)

    def test_index_stale_total_parameter_count_mapping_alias_and_missing_inventory_reject(self):
        objects = [('a.safetensors', [dict(name='a.weight')]), ('b.safetensors', [dict(name='b.weight')])]
        mapping = {'a.weight': 'a.safetensors', 'b.weight': 'b.safetensors'}
        correct = dict(metadata=dict(total_size=8, total_parameters=4), weight_map=mapping)
        self.assertEqual(layout.validate_index(json.dumps(correct), objects, 8, dict.fromkeys(mapping)), mapping)
        for change in [dict(metadata=dict(total_size=7)), dict(metadata=dict(total_size=True)),
                       dict(metadata=dict(total_size=8, total_parameters=3)),
                       dict(metadata=dict(total_size=8, total_parameters=True)),
                       dict(weight_map=dict(mapping, **{'a.weight': 'b.safetensors'})),
                       dict(weight_map={'a.weight': 'a.safetensors'})]:
            with self.assertRaises(ValueError):
                layout.validate_index(json.dumps(dict(correct, **change)), objects, 8)
        for filename in ['../a.safetensors', '.', '..', 'a/b', 'a\\b']:
            with self.assertRaises(ValueError):
                layout.validate_index(json.dumps(correct), [(filename, objects[0][1]), objects[1]], 8)
        with self.assertRaises(ValueError):
            layout.validate_index(json.dumps(correct), objects, 8, {'a.weight': None})
        with self.assertRaises(ValueError):
            layout.validate_index(json.dumps(correct), [('a.safetensors', objects[0][1]),
                                                       ('a.safetensors', objects[1][1])], 8)

    def test_tiny_layout_writes_all_components_exactly_in_irregular_row_tiles(self):
        out = layout.output_layout(*tiny_recipe())
        self.assertLess(out['file_bytes'], 4096)
        with tempfile.TemporaryDirectory() as folder:
            with (Path(folder) / 'child').open('w+b') as file:
                file.truncate(out['file_bytes'])
                layout.write_all(file.fileno(), out['prefix'], 0)
                for name, plan in out['matrices'].items():
                    complete = {key: bytes(index % 251 for index in range(3 * item['row_stride']))
                                for key, item in plan['fields'].items()}
                    for first, rows in [(0, 2), (2, 1)]:
                        tile = {key: value[first * plan['fields'][key]['row_stride']:
                                           (first + rows) * plan['fields'][key]['row_stride']]
                                for key, value in complete.items()}
                        layout.write_matrix_tile(file.fileno(), out, name, first, rows, tile)
                    for key, item in plan['fields'].items():
                        offset = out['origin'] + out['header'][item['name']]['data_offsets'][0]
                        self.assertEqual(os.pread(file.fileno(), len(complete[key]), offset), complete[key])
                norm = b'\x80\x3f' * 64
                layout.write_norm(file.fileno(), out, 'language_model.model.norm.weight', norm)
                self.assertEqual(os.fstat(file.fileno()).st_size, out['file_bytes'])

    def test_bad_component_or_tile_bounds_preflight_without_any_write(self):
        out = layout.output_layout(*tiny_recipe())
        name = 'language_model.model.embed_tokens.weight'
        buffers = dict(weight=bytes(64), scales=bytes(2), biases=bytes(2))
        with tempfile.TemporaryDirectory() as folder:
            with (Path(folder) / 'child').open('w+b') as file:
                file.truncate(out['file_bytes'])
                for bad in [dict(buffers, scales=b''), dict(buffers, biases=None), dict(buffers, extra=b'')]:
                    with self.assertRaises(ValueError):
                        layout.write_matrix_tile(file.fileno(), out, name, 0, 1, bad)
                for first, rows in [(-1, 1), (3, 1), (2, 2), (False, 1), (0, True), (0, 225)]:
                    with self.assertRaises(ValueError):
                        layout.write_matrix_tile(file.fileno(), out, name, first, rows, buffers)
                self.assertEqual(os.pread(file.fileno(), out['file_bytes'], 0), bytes(out['file_bytes']))

    def test_partial_write_finishes_and_zero_progress_or_short_readback_rejects(self):
        out = layout.output_layout(*tiny_recipe())
        with tempfile.TemporaryDirectory() as folder:
            with (Path(folder) / 'child').open('w+b') as file:
                file.truncate(out['file_bytes'])
                write = os.pwrite
                with patch.object(layout.generic.os, 'pwrite', side_effect=lambda fd, raw, off: write(fd, raw[:3], off)):
                    layout.write_all(file.fileno(), b'complete bytes', 0)
                self.assertEqual(os.pread(file.fileno(), 14, 0), b'complete bytes')
                with patch.object(layout.generic.os, 'pwrite', return_value=0):
                    with self.assertRaises(ValueError):
                        layout.write_all(file.fileno(), b'x', 0)
                with patch.object(layout.generic.os, 'pread', return_value=b''):
                    with self.assertRaises(ValueError):
                        layout.write_norm(file.fileno(), out, 'language_model.model.norm.weight', bytes(128))

    def test_recipe_types_duplicates_column_limit_and_budget_reject(self):
        matrices, norms = tiny_recipe()
        for change in [dict(shape=[True, 64]), dict(shape=[3, 65]), dict(shape=[3, 9280]),
                       dict(bits=True), dict(bits=6), dict(group_size=True), dict(mode='mxfp4'),
                       dict(module='vision_tower.weight'), dict(extra=1)]:
            with self.assertRaises(ValueError):
                layout.output_layout([dict(matrices[0], **change)], norms)
        with self.assertRaises(ValueError):
            layout.output_layout(matrices + matrices, norms)
        with self.assertRaises(ValueError):
            layout.output_layout(matrices, [dict(norms[0], bytes=True)])
        huge = [dict(module='language_model.model.matrix' + str(index), shape=[131072, 9216],
                     bits=8, group_size=64, mode='affine') for index in range(6)]
        with self.assertRaisesRegex(ValueError, 'engine ceiling'):
            layout.output_layout(huge, [])


if __name__ == '__main__':
    unittest.main()

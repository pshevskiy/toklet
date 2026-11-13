"""Portable selected-source, typed-storage and child identity controls."""
import copy
import hashlib
import json
import os
from pathlib import Path
import struct
import tempfile
import unittest

import qwen35_2b_layout as layout

FIXTURE = Path(__file__).resolve().parents[2] / 'tests/fixtures/qwen35-2b-source-v1'


def fixture(name):
    manifest = json.loads((FIXTURE / 'manifest.json').read_bytes())
    raw = (FIXTURE / name).read_bytes()
    pin = manifest['files'][name]
    if len(raw) != pin['bytes'] or hashlib.sha256(raw).hexdigest() != pin['sha256']:
        raise ValueError('Frozen source fixture changed')
    return raw


def tiny_recipe():
    matrices = [dict(module=layout.PREFIX + 'tiny', shape=[3, 64], bits=4,
                     group_size=64, mode='affine')]
    small = [dict(name=layout.PREFIX + 'norm.weight', shape=[64], dtype='BF16', bytes=128),
             dict(name=layout.PREFIX + 'tiny.A_log', shape=[2], dtype='F32', bytes=8),
             dict(name=layout.PREFIX + 'tiny.conv1d.weight', shape=[4, 4, 1], dtype='BF16', bytes=32)]
    return matrices, small


def manifest():
    plan = layout.selected_layout()
    config = layout.converted_config(fixture('source-config.json'))
    return layout.child_manifest(layout.artifact_pins(config, plan, '1' * 64), '2' * 64, '3' * 64)


class Layout(unittest.TestCase):
    def test_complete_selected_typed_inventory_and_all_affine4_extents(self):
        matrices, small = layout.selected_recipe()
        plan = layout.selected_layout()
        self.assertEqual((len(matrices), len(small), len(plan['logical']), len(plan['header']) - 1),
                         (187, 133, 320, 694))
        self.assertEqual(plan['body_bytes'], 1059315904)
        self.assertEqual(sum(s['dtype'] == 'F32' for s in small), 18)
        self.assertTrue(all(s['name'].endswith('.A_log') and s['shape'] == [16]
                            for s in small if s['dtype'] == 'F32'))
        self.assertEqual(sum(s['dtype'] == 'BF16' and len(s['shape']) == 1 for s in small), 97)
        self.assertEqual(sum(len(s['shape']) == 3 for s in small), 18)
        self.assertTrue(all(m['bits'] == 4 and m['group_size'] == 64 for m in matrices))
        spans = [v['data_offsets'] for k, v in plan['header'].items() if k != '__metadata__']
        self.assertEqual(spans[0][0], 0)
        self.assertTrue(all(a[1] == b[0] for a, b in zip(spans, spans[1:])))
        self.assertEqual(spans[-1][1], plan['body_bytes'])
        self.assertEqual(plan['header']['__metadata__'], {'format': 'mlx'})
        self.assertEqual(plan['origin'] % 8, 0)

    def test_exact_unaligned_source_header_covers_text_and_excluded_vision(self):
        raw = fixture('source-header.bin')
        spans = layout.source_spans(struct.pack('<Q', len(raw)), raw, layout.SOURCE_BYTES)
        self.assertEqual((len(spans), sum(s['selected'] for s in spans)), (617, 320))
        self.assertEqual(len(raw) + 8, 75241)
        self.assertEqual({s['dtype'] for s in spans if s['name'].endswith('.A_log')}, {'F32'})
        self.assertTrue(all(s['name'].startswith('vision_tower.') for s in spans if not s['selected']))
        for prefix, header, size in [(bytes(8), raw, layout.SOURCE_BYTES),
                                     (struct.pack('<Q', len(raw)), raw + b' ', layout.SOURCE_BYTES),
                                     (struct.pack('<Q', len(raw)), raw, layout.SOURCE_BYTES + 1)]:
            with self.assertRaises(ValueError):
                layout.source_spans(prefix, header, size)

    def test_source_config_preserves_full_wrapper_with_only_explicit_top_recipe(self):
        source = json.loads(fixture('source-config.json'))
        result = json.loads(layout.converted_config(fixture('source-config.json')))
        self.assertEqual(result.pop('quantization'), dict(group_size=64, bits=4, mode='affine'))
        self.assertEqual(source, result)
        self.assertNotIn('quantization_config', result)
        for raw in (fixture('source-config.json') + b' ', b'{}', b'{"a":1,"a":2}'):
            with self.assertRaises(ValueError):
                layout.converted_config(raw)

    def test_tiny_rows_and_typed_small_bytes_keep_non_bf16_f32_information(self):
        matrices, small = tiny_recipe()
        plan = layout.output_layout(matrices, small)
        values = struct.pack('<II', 0x3f800001, 0xbf000123)
        with tempfile.TemporaryDirectory() as folder:
            with (Path(folder) / 'child').open('w+b') as stream:
                stream.truncate(plan['file_bytes'])
                layout.write_small(stream.fileno(), plan, small[1], values)
                spec = plan['header'][small[1]['name']]
                self.assertEqual(os.pread(stream.fileno(), 8, plan['origin'] + spec['data_offsets'][0]), values)
                with self.assertRaises(ValueError):
                    layout.write_small(stream.fileno(), plan, small[1], values[:4])
        self.assertEqual(plan['header'][small[1]['name']]['dtype'], 'F32')

    def test_finite_small_originals_and_typed_extents_are_required(self):
        _, small = tiny_recipe()
        for item, raw in [(small[1], struct.pack('<ff', float('inf'), 1.0)),
                          (small[0], struct.pack('<H', 0x7fc1) * 64),
                          (small[0], bytes(126))]:
            with self.assertRaises(ValueError):
                layout.validate_small(item, raw)

    def test_duplicate_bits_geometry_and_untyped_small_declarations_reject(self):
        matrices, small = tiny_recipe()
        candidates = [(matrices + matrices, small), (matrices, small + small),
                      ([dict(matrices[0], bits=8)], small),
                      ([dict(matrices[0], shape=[True, 64])], small),
                      ([dict(matrices[0], shape=[3, 65])], small),
                      (matrices, [dict(small[1], dtype='BF16')])]
        for a, b in candidates:
            with self.assertRaises(ValueError):
                layout.output_layout(a, b)

    def test_actual_child_identity_manifest_and_external_sha_roundtrip(self):
        value = manifest()
        raw = (json.dumps(value, indent=2) + '\n').encode()
        self.assertEqual(layout.capture_child_manifest(raw, hashlib.sha256(raw).hexdigest()), value)
        with self.assertRaises(ValueError):
            layout.capture_child_manifest(raw, '4' * 64)
        self.assertFalse(value['qualification']['model_numerics_quality_fit_performance_adoption_qualified'])

    def test_identity_manifest_rejects_type_confusion_unknown_keys_and_changed_pins(self):
        value = manifest()
        paths = [('passed', False), ('lineage_certificate_sha256', '0' * 64),
                 ('conversion_manifest_sha256', None), ('other', 0)]
        candidates = [dict(value, **{k: v}) for k, v in paths]
        for section, key, changed in [('storage', 'bits', True), ('source', 'revision', 'a' * 40),
                ('artifact_pins', 'container_bytes', True), ('artifact_pins', 'header_bytes', 1),
                ('artifact_pins', 'config_sha256', '0' * 64), ('qualification', 'diagnostic_only', 1)]:
            bad = copy.deepcopy(value)
            bad[section][key] = changed
            candidates.append(bad)
        for bad in candidates:
            with self.assertRaises(ValueError):
                layout.validate_child_manifest(bad)
        raw = json.dumps(value).encode().replace(b'"passed": true', b'"passed": true,"passed": true')
        with self.assertRaises(ValueError):
            layout.capture_child_manifest(raw, hashlib.sha256(raw).hexdigest())


if __name__ == '__main__':
    unittest.main()

"""Cold tiny typed-byte controls; full generation uses the owned heavy queue."""
import hashlib
import copy
import json
import math
from pathlib import Path
import struct
import tempfile
import unittest

import qwen35_2b_fixture as fixture

FOLDER = fixture.ROOT / 'tests/fixtures/qwen35-2b-cpu-graph-v1'
MANIFEST_SHA = 'c0254e4b68ad37dedfb8c30493527a27f09d322b281ba7b8264c7470f45ee955'


def need(value, message):
    if not value: raise ValueError(message)


def observations(token):
    """Independent complete stored-stage shapes, including growing KV arrays."""
    output = [(token, None, 'embedding', 64)]
    for layer in range(4):
        output.append((token, layer, 'input_norm', 64))
        if layer % 4 != 3:
            stages = [('delta_qkv', 192), ('delta_z', 64), ('delta_a', 2), ('delta_b', 2),
                      ('delta_convolved', 192), ('delta_q', 64), ('delta_k', 64), ('delta_v', 64),
                      ('delta_decay', 2), ('delta_beta', 2), ('delta_state', 2048), ('delta_history', 576),
                      ('delta_output', 64), ('delta_norm', 64), ('delta_gated', 64)]
        else:
            stages = [('attention_q_gate', 128), ('attention_q', 64), ('attention_k', 32), ('attention_v', 32),
                      ('attention_q_norm', 64), ('attention_k_norm', 32), ('attention_q_rot', 64), ('attention_k_rot', 32)]
            stages += [(f'attention_{kind}_cache.{head}', (token + 1) * 16) for head in range(2) for kind in ('k', 'v')]
            stages += [(f'attention_{kind}.{head}', token + 1) for head in range(4) for kind in ('scores', 'probabilities')]
            stages += [('attention_context', 64), ('attention_gated', 64)]
        stages += [('operator_output', 64), ('operator_residual', 64), ('post_norm', 64),
                   ('mlp_gate', 128), ('mlp_up', 128), ('mlp_activated', 128), ('mlp_output', 64), ('layer_output', 64)]
        output.extend((token, layer, stage, count) for stage, count in stages)
    output.extend([(token, None, 'final_norm', 64), (token, None, 'logits', 32)])
    return output


def validate(manifest, weights, reference):
    need(manifest['format'] == 'toklet-qwen35-cpu-graph-v1' and manifest['geometry'] == fixture.GEOMETRY
         and manifest['gate'] == dict(atol=1e-5, rtol=1e-4) and manifest['fixture_only'] is True
         and manifest['real_model_qualified'] is False and manifest['typed_F32_A_log'] is True, 'Frozen typed fixture policy differs')
    need(len(weights) == manifest['weights_bytes'] and hashlib.sha256(weights).hexdigest() == manifest['weights_sha256'], 'Weight object differs')
    need(len(reference) == manifest['reference_bytes'] and hashlib.sha256(reference).hexdigest() == manifest['reference_sha256'], 'Reference object differs')
    expected, spans, backing = fixture.specs(), [], 0
    need(set(manifest['weights']) == set(expected), 'Complete weight inventory required')
    for name, (kind, shape) in expected.items():
        item = manifest['weights'][name]
        need(item['kind'] == kind and item['shape'] == shape, 'Exact typed weight descriptor required')
        if kind == 'matrix':
            need(set(item) == {'kind', 'shape', 'codes', 'scales', 'biases'}, 'Complete packed component map required')
            fields = [item[key] for key in ('codes', 'scales', 'biases')]
            need([v['bytes'] for v in fields] == [math.prod(shape) // 2, math.prod(shape) // 32, math.prod(shape) // 32], 'Packed extents differ')
            backing += sum(v['bytes'] for v in fields)
        else:
            need(set(item) == {'kind', 'shape', 'data'} and item['data']['bytes'] == math.prod(shape) * (4 if kind == 'f32' else 2), 'Typed small extent differs')
            fields = [item['data']]; backing += math.prod(shape) * 4
        for field in fields:
            first, count = field['offset'], field['bytes']
            need(type(first) is int and type(count) is int and 0 <= first < first + count <= len(weights), 'Weight range differs')
            raw = weights[first:first + count]
            need(hashlib.sha256(raw).hexdigest() == field['sha256'], 'Weight component differs')
            spans.append((first, first + count))
            if kind == 'bf16' or (kind == 'matrix' and field is not item['codes']):
                need(all(word & 0x7f80 != 0x7f80 for (word,) in struct.iter_unpack('<H', raw)), 'Finite BF16 parameter required')
            if kind == 'f32':
                need(all(word & 65535 for (word,) in struct.iter_unpack('<I', raw)), 'A_log fixture must retain non-BF16 precision')
                need(all(math.isfinite(v[0]) for v in struct.iter_unpack('<f', raw)), 'Finite F32 parameter required')
    spans.sort()
    need(spans[0][0] == 0 and spans[-1][1] == len(weights)
         and all(a[1] == b[0] for a,b in zip(spans,spans[1:])) and backing == manifest['backing_bytes'], 'Complete tight typed weight coverage required')
    need(len(manifest['cases']) == 4, 'All frozen cases required')
    cursor = 0
    for case, (name, tokens, offset) in zip(manifest['cases'], fixture.arithmetic.CASES, strict=True):
        need(case['name'] == name and case['tokens'] == list(tokens) and all(type(v) is int for v in case['tokens']) and type(case['position_offset']) is int
             and case['position_offset'] == offset and set(case['profiles']) == {'ideal_fp64', 'staged_fp32'}, 'Frozen case/profile identities differ')
        expected_rows = [row for token in range(len(tokens)) for row in observations(token)]
        for profile, width, code in [('ideal_fp64', 8, '<d'), ('staged_fp32', 4, '<f')]:
            arrays = case['profiles'][profile]
            need(len(arrays) == len(expected_rows), 'Every state/activation array required')
            for array, (token, layer, stage, count) in zip(arrays, expected_rows, strict=True):
                need(set(array) == {'token', 'layer', 'stage', 'offset', 'bytes', 'count', 'sha256'}
                     and type(array['token']) is int and array['token'] == token
                     and (array['layer'] is None if layer is None else type(array['layer']) is int and array['layer'] == layer)
                     and array['stage'] == stage and type(array['count']) is int and array['count'] == count
                     and type(array['offset']) is int and array['offset'] == cursor
                     and type(array['bytes']) is int and array['bytes'] == count * width, 'Complete ordered typed observation differs')
                raw = reference[cursor:cursor + count * width]
                need(len(raw) == count * width and hashlib.sha256(raw).hexdigest() == array['sha256']
                     and all(math.isfinite(v[0]) for v in struct.iter_unpack(code, raw)), 'Finite exact reference array required')
                cursor += len(raw)
    need(cursor == len(reference), 'No reference bytes may be omitted or appended')
    return True

class TypedFixture(unittest.TestCase):
    def test_ratio_one_and_group64_geometry(self):
        g = fixture.GEOMETRY
        self.assertEqual((g['key_heads'], g['value_heads']), (2, 2))
        self.assertEqual((g['key_dim'], g['value_dim']), (32, 32))
        self.assertTrue(all(shape[1] % 64 == 0 for kind, shape in fixture.specs().values() if kind == 'matrix'))
        self.assertEqual(fixture.GATE, dict(atol=1e-5, rtol=1e-4))

    def test_exact_F32_A_log_values_cannot_be_roundtripped_through_BF16(self):
        raw, table, _, vectors, _ = fixture.artifact()
        selected = [(name, value) for name, value in table.items() if value['kind'] == 'f32']
        self.assertEqual(len(selected), 3)
        for name, value in selected:
            self.assertTrue(name.endswith('linear_attn.A_log'))
            self.assertEqual(value['shape'], [2])
            source = value['data']
            encoded = raw[source['offset']:source['offset'] + source['bytes']]
            self.assertEqual(len(encoded), 8)
            self.assertEqual(hashlib.sha256(encoded).hexdigest(), source['sha256'])
            self.assertEqual(list(struct.unpack('<2f', encoded)), vectors[name])
            for word in struct.unpack('<2I', encoded): self.assertNotEqual(word & 65535, 0)
            self.assertTrue(all(fixture.arithmetic.bf16(v)[1] != v for v in vectors[name]))

    def test_typed_backing_count_and_all_byte_extents(self):
        raw, table, _, vectors, backing = fixture.artifact()
        expected = 0
        spans = []
        for name, value in table.items():
            if value['kind'] == 'matrix':
                fields = [value[key] for key in ('codes', 'scales', 'biases')]
                expected += sum(v['bytes'] for v in fields)
            else:
                fields = [value['data']]
                expected += math.prod(value['shape']) * 4
                self.assertEqual(len(vectors[name]), math.prod(value['shape']))
                self.assertEqual(value['data']['bytes'], math.prod(value['shape']) * (4 if value['kind'] == 'f32' else 2))
            for field in fields:
                spans.append((field['offset'], field['offset'] + field['bytes']))
                self.assertEqual(hashlib.sha256(raw[field['offset']:field['offset'] + field['bytes']]).hexdigest(), field['sha256'])
        spans.sort()
        self.assertEqual(spans[0][0], 0)
        self.assertTrue(all(left[1] == right[0] for left, right in zip(spans, spans[1:])))
        self.assertEqual(spans[-1][1], len(raw))
        self.assertEqual(backing, expected)

    def test_existing_output_rejects_before_numerical_generation(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'occupied'
            path.write_bytes(b'preserve')
            with self.assertRaises(ValueError): fixture.generate(path)
            self.assertEqual(path.read_bytes(), b'preserve')


class FrozenFixture(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = json.loads((FOLDER / 'manifest.json').read_text())
        cls.weights = (FOLDER / 'weights.bin').read_bytes()
        cls.reference = (FOLDER / 'reference.bin').read_bytes()

    def test_complete_ordered_dual_profile_activation_and_state_coverage(self):
        self.assertEqual(hashlib.sha256((FOLDER / 'manifest.json').read_bytes()).hexdigest(), MANIFEST_SHA)
        self.assertTrue(validate(self.manifest, self.weights, self.reference))
        self.assertEqual(sum(len(v) for c in self.manifest['cases'] for v in c['profiles'].values()), 11024)

    def test_actual_producer_and_independent_arithmetic_pins_match(self):
        for name, pin in self.manifest['producers'].items():
            raw = (fixture.ROOT / name).read_bytes()
            self.assertEqual(dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest()), pin)

    def test_corrupted_raw_weight_or_reference_byte_rejects(self):
        for kind in ('weights', 'reference'):
            weights, reference = self.weights, self.reference
            if kind == 'weights': weights = bytes([weights[0] ^ 1]) + weights[1:]
            else: reference = reference[:-1] + bytes([reference[-1] ^ 1])
            with self.subTest(kind=kind), self.assertRaises(ValueError): validate(self.manifest, weights, reference)

    def test_BF16_narrowing_cannot_hide_behind_updated_byte_hashes(self):
        value, raw = copy.deepcopy(self.manifest), bytearray(self.weights)
        name = next(name for name, item in value['weights'].items() if item['kind'] == 'f32')
        field = value['weights'][name]['data']
        for offset in range(field['offset'], field['offset'] + field['bytes'], 4):
            number = struct.unpack('<f', raw[offset:offset + 4])[0]
            raw[offset:offset + 4] = struct.pack('<f', fixture.arithmetic.bf16(number)[1])
        field['sha256'] = hashlib.sha256(raw[field['offset']:field['offset'] + field['bytes']]).hexdigest()
        value['weights_sha256'] = hashlib.sha256(raw).hexdigest()
        with self.assertRaises(ValueError): validate(value, bytes(raw), self.reference)

    def test_missing_duplicate_wrong_count_or_wrong_F32_kind_rejects(self):
        for change in ('missing', 'duplicate', 'count', 'kind'):
            value = copy.deepcopy(self.manifest)
            arrays = value['cases'][1]['profiles']['staged_fp32']
            if change == 'missing': arrays.pop()
            elif change == 'duplicate': arrays[-1] = arrays[-2]
            elif change == 'count': arrays[0]['count'] = True
            else: next(v for v in value['weights'].values() if v['kind'] == 'f32')['kind'] = 'bf16'
            with self.subTest(change=change), self.assertRaises(ValueError): validate(value, self.weights, self.reference)

    def test_nonfinite_typed_small_parameters_reject_with_updated_hashes(self):
        for kind, encoded in [('f32', struct.pack('<I', 0x7fc00001)), ('bf16', struct.pack('<H', 0x7f80))]:
            value, raw = copy.deepcopy(self.manifest), bytearray(self.weights)
            item = next(v for v in value['weights'].values() if v['kind'] == kind)
            field = item['data']; first = field['offset']
            raw[first:first + len(encoded)] = encoded
            field['sha256'] = hashlib.sha256(raw[first:first + field['bytes']]).hexdigest()
            value['weights_sha256'] = hashlib.sha256(raw).hexdigest()
            with self.subTest(kind=kind), self.assertRaises(ValueError): validate(value, bytes(raw), self.reference)

    def test_changed_gate_case_offset_and_absent_profile_reject(self):
        for change in ('gate', 'offset', 'profile'):
            value = copy.deepcopy(self.manifest)
            if change == 'gate': value['gate']['rtol'] = 0.001
            elif change == 'offset': value['cases'][2]['position_offset'] += 1
            else: del value['cases'][0]['profiles']['ideal_fp64']
            with self.subTest(change=change), self.assertRaises(ValueError): validate(value, self.weights, self.reference)

    def test_future_token_change_preserves_exact_first_sixteen_token_arrays(self):
        original, changed = (self.manifest['cases'][index] for index in (1, 3))
        for profile in ('ideal_fp64', 'staged_fp32'):
            a = [v for v in original['profiles'][profile] if v['token'] < 16]
            b = [v for v in changed['profiles'][profile] if v['token'] < 16]
            self.assertEqual([(v['token'], v['layer'], v['stage'], v['count'], v['sha256']) for v in a],
                             [(v['token'], v['layer'], v['stage'], v['count'], v['sha256']) for v in b])

if __name__ == '__main__': unittest.main()

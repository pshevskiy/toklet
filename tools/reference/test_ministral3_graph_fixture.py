"""Portable authored arithmetic, geometry, byte-reader and frozen-gate controls."""
import copy
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

import ministral3_convert_layout as layout
import ministral3_graph_fixture as graph
import ministral3_graph_protocol as protocol
import ministral3_graph_reference as reference


class Geometry(unittest.TestCase):
    def test_query_width_is_independent_of_hidden(self):
        shapes = protocol.contracts(protocol.REAL, 512)
        self.assertEqual(len(shapes), 419)
        self.assertEqual(shapes['layers.25.q'], [512, 4096])
        self.assertEqual(shapes['layers.25.k'], [512, 1024])
        self.assertEqual(shapes['layers.25.attention_context'], [512, 4096])
        self.assertEqual(shapes['layers.25.attention_projection'], [512, 3072])
        self.assertEqual(shapes['layers.25.down'], [512, 3072])
        self.assertEqual(shapes['layers.25.gated'], [512, 9216])
        self.assertEqual(shapes['logits'], [512, 131072])

    def test_context_boundary_and_diagnostic_positions_are_distinct(self):
        protocol.geometry(protocol.REAL, [1] * 512, 0)
        protocol.geometry(protocol.TINY, [1, 7, 13], 509)
        protocol.geometry(protocol.TINY, [1, 7], 16384, True)
        for tokens, offset, diagnostic in [([], 0, False), ([1] * 513, 0, False),
                ([1, 7, 13], 510, False), ([1], -1, False), ([True], 0, False),
                ([64], 0, False), ([1], False, False), ([1], 16384, False),
                ([1, 7], 262143, True), ([1], 0, 1)]:
            with self.subTest(tokens=len(tokens), offset=offset, diagnostic=diagnostic):
                with self.assertRaises(ValueError):
                    protocol.geometry(protocol.TINY, tokens, offset, diagnostic)

    def test_schema_geometry_and_scalars_reject(self):
        for change in [dict(hidden=True), dict(hidden=65), dict(intermediate=9217),
                       dict(head_dim=64), dict(kv_heads=3), dict(query_heads=33),
                       dict(layers=27), dict(theta=1000000), dict(rms_eps=1e-6),
                       dict(tied_head=1), dict(tied_head=False), dict(extra=1)]:
            value = dict(protocol.TINY, **change)
            with self.subTest(change=change):
                with self.assertRaises(ValueError):
                    protocol.geometry(value, [1], 0)

    def test_module_imports_are_cold(self):
        result = subprocess.run([sys.executable, '-s', '-B', '-c',
            "import sys; import ministral3_graph_fixture, ministral3_graph_reference; assert not any(k.split('.')[0] in ('numpy','mlx','torch') for k in sys.modules)"],
            cwd=Path(graph.__file__).parent, capture_output=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)


class Arithmetic(unittest.TestCase):
    def test_authored_mixed_inventory_and_tied_head(self):
        weights = graph.authored_weights()
        matrices = [value for value in weights.values() if 'bits' in value]
        self.assertEqual(len(weights), 11)
        self.assertEqual(sum(value['bits'] == 4 for value in matrices), 5)
        self.assertEqual(sum(value['bits'] == 8 for value in matrices), 3)
        self.assertNotIn('lm_head.weight', weights)
        self.assertEqual(weights['model.layers.0.self_attn.o_proj.weight']['shape'], [64, 256])
        self.assertEqual(weights['model.layers.0.mlp.down_proj.weight']['shape'], [64, 128])

    def test_low_bit_order_and_separate_metadata_reconstruction(self):
        raw = graph.authored_weights()
        values = graph.decode_weights(raw)
        for name, row, column in [('model.embed_tokens.weight', 2, 63),
                ('model.layers.0.self_attn.q_proj.weight', 13, 17),
                ('model.layers.0.self_attn.o_proj.weight', 7, 191)]:
            record = raw[name]
            bits, width = record['bits'], record['shape'][1]
            word = record['weight_words'][row * (width * bits // 32) + column // (32 // bits)]
            code = (word >> ((column % (32 // bits)) * bits)) & ((1 << bits) - 1)
            group = row * (width // 64) + column // 64
            expected = graph.f32(graph.f32(code * graph.bf16(record['scales_bf16'][group]))
                                 + graph.bf16(record['biases_bf16'][group]))
            self.assertEqual(values[name][row][column], expected)

    def test_malformed_packed_records_reject(self):
        for change in [('bits', 6), ('bits', True), ('group_size', 32), ('shape', [128, 64]),
                       ('weight_words', [0]), ('scales_bf16', [0x7f80]), ('extra', 1)]:
            weights = graph.authored_weights()
            weights['model.layers.0.self_attn.q_proj.weight'][change[0]] = change[1]
            with self.subTest(change=change[0]):
                with self.assertRaises(ValueError):
                    graph.decode_weights(weights)

    def test_norm_bf16_half_and_finite_boundaries(self):
        self.assertEqual(graph.bf16_word(1.0), 0x3f80)
        self.assertEqual(graph.bf16_word(-0.5), 0xbf00)
        self.assertEqual(graph.bf16(0x3f80), 1.0)
        self.assertEqual(graph.norm([[3.0, 4.0]], [1.0, 1.0], float), [[graph.half(3 / (12.5 + graph.f32(1e-5))**0.5), graph.half(4 / (12.5 + graph.f32(1e-5))**0.5)]])
        for word in [True, -1, 65536, 0x7f80, 0xff80, 0x7fc1]:
            with self.assertRaises(ValueError):
                graph.bf16(word)

    def test_policy_is_the_existing_profile_only_envelope(self):
        envelope, gate = protocol.gate([1.0, 0.0], [1.01, 0.0])
        self.assertAlmostEqual(envelope, 0.01)
        self.assertAlmostEqual(gate['atol'], 0.02)
        self.assertEqual(gate['rtol'], 1e-3)
        self.assertTrue(protocol.member(1.0, 1.01, gate))
        self.assertFalse(protocol.member(1.2, 1.01, gate))
        for main, wide in [([], []), ([1], []), ([float('nan')], [1]), ([True], [1])]:
            with self.assertRaises(ValueError):
                protocol.gate(main, wide)


def saved_case(folder):
    matrix = dict(module='language_model.model.layers.0.self_attn.q_proj',
                  shape=[225, 64], bits=4, group_size=64, mode='affine')
    norm = dict(name='language_model.model.norm.weight', shape=[64], dtype='BF16', bytes=128)
    planned = layout.output_layout([matrix], [norm])
    body = bytearray(planned['body_bytes'])
    # Distinct deterministic bytes provide component-hash witnesses; this is
    # byte-reader qualification only and imports no numerical runtime.
    for index in range(len(body)):
        body[index] = (index * 17 + 31) & 255
    raw = planned['prefix'] + body
    path = Path(folder) / 'model.safetensors'
    path.write_bytes(raw)
    name = matrix['module'] + '.weight'
    records = []
    for first, count in [(0, 224), (224, 1)]:
        fields = {}
        for field, declaration in planned['matrices'][name]['fields'].items():
            item = planned['header'][declaration['name']]
            offset = planned['origin'] + item['data_offsets'][0] + first * declaration['row_stride']
            data = raw[offset:offset + count * declaration['row_stride']]
            fields[field] = dict(name=declaration['name'], dtype=item['dtype'], shape=[count, item['shape'][1]],
                                 offset=offset, bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
        records.append(dict(name=name, first_row=first, rows=count, columns=64, bits=4, fields=fields))
    item = planned['header'][norm['name']]
    start, end = item['data_offsets']
    data = raw[planned['origin'] + start:planned['origin'] + end]
    digest = hashlib.sha256(data).hexdigest()
    norms = [dict(name=norm['name'], bytes=len(data), copied_BF16_sha256=digest,
                  child=dict(name=norm['name'], dtype='BF16', shape=[64],
                             offset=planned['origin'] + start, bytes=len(data), sha256=digest))]
    return path, planned, dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest()), records, norms


class ChildBytes(unittest.TestCase):
    def test_exact_224_row_tail_complete_hash_and_component_consumption(self):
        with tempfile.TemporaryDirectory() as folder:
            args = saved_case(folder)
            with reference.Child(*args) as child:
                for first in (0, 224):
                    self.assertEqual(set(child.tile('language_model.model.layers.0.self_attn.q_proj.weight', first)), {'weight', 'scales', 'biases'})
                self.assertEqual(len(child.norm('language_model.model.norm.weight')), 128)
                receipt = child.endpoint()
                self.assertEqual(receipt['component_reads'], 6)
                self.assertEqual(receipt['norm_reads'], 1)

    def test_missing_duplicate_or_legacy_256_stripes_reject_before_open(self):
        with tempfile.TemporaryDirectory() as folder:
            path, planned, pin, rows, norms = saved_case(folder)
            for changed in [rows[:1], rows + [rows[0]], [dict(rows[0], rows=225), rows[1]],
                            [rows[0], dict(rows[1], first_row=256)]]:
                with self.assertRaises(ValueError):
                    reference.Child(path, planned, pin, changed, norms)

    def test_rich_component_and_norm_layout_metadata_are_bound(self):
        with tempfile.TemporaryDirectory() as folder:
            path, planned, pin, rows, norms = saved_case(folder)
            for field, value in [('offset', 0), ('dtype', 'F32'), ('shape', [224, 9]),
                                 ('name', 'other'), ('bytes', True), ('extra', 1)]:
                changed = copy.deepcopy(rows)
                changed[0]['fields']['weight'][field] = value
                with self.assertRaises(ValueError):
                    reference.Child(path, planned, pin, changed, norms)
            changed = copy.deepcopy(norms)
            changed[0]['child']['offset'] += 2
            with self.assertRaises(ValueError):
                reference.Child(path, planned, pin, rows, changed)

    def test_source_replacement_and_independent_component_hash_fail(self):
        with tempfile.TemporaryDirectory() as folder:
            args = saved_case(folder)
            with self.assertRaises(ValueError):
                with reference.Child(*args) as child:
                    args[0].rename(Path(folder) / 'old')
                    args[0].write_bytes((Path(folder) / 'old').read_bytes())
                    child.endpoint()
            path, planned, _, rows, norms = saved_case(folder)
            raw = bytearray(path.read_bytes())
            raw[planned['origin']] ^= 1
            path.write_bytes(raw)
            pin = dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
            with reference.Child(path, planned, pin, rows, norms) as child:
                # Whole-object SHA matches the supplied test pin; the retained
                # conversion component receipt independently rejects its bytes.
                with self.assertRaises(ValueError):
                    child.tile('language_model.model.layers.0.self_attn.q_proj.weight', 0)


class FrozenFixture(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = graph.load_fixture(graph.ROOT / 'tests/fixtures/ministral3-graph-v1')

    def test_complete_ordinary_and_high_position_fixture_inventory(self):
        self.assertEqual(len(self.fixture['cases']), 6)
        self.assertEqual(sum(case['diagnostic_offset'] for case in self.fixture['cases']), 1)
        self.assertTrue(all(len(case['arrays']) == 19 for case in self.fixture['cases']))

    def test_every_reference_coordinate_has_frozen_both_profile_gate(self):
        for case in self.fixture['cases']:
            for stage, array in case['arrays'].items():
                decode = graph.rotary.from_f32 if stage == 'logits' else graph.rotary.from_half
                main = [decode(word) for word in array['main_bits']]
                wide = [decode(word) for word in array['wide_bits']]
                envelope, gate = protocol.gate(main, wide)
                self.assertEqual((envelope, gate), (array['precision_envelope'], array['gate']))
                self.assertTrue(all(protocol.member(a, b, gate) for a, b in zip(main, wide, strict=True)))

    def test_gate_growth_missing_stages_and_changed_policy_reject(self):
        for change in ('gate', 'envelope', 'missing', 'policy', 'wrong_kind'):
            value = copy.deepcopy(self.fixture)
            array = value['cases'][0]['arrays']['logits']
            if change == 'gate':
                array['gate']['atol'] *= 2
            elif change == 'envelope':
                array['precision_envelope'] += 1
            elif change == 'missing':
                del value['cases'][0]['arrays']['layers.0.q_rotary']
            elif change == 'policy':
                value['policy']['rtol'] = 0.1
            else:
                high = value['cases'][-1]['arrays']['layers.0.q_rotary']
                value['composition_control']['wrong_keys_kind'] = {key: high[key] for key in ('main_bits', 'wide_bits')}
            with self.subTest(change=change):
                with self.assertRaises(ValueError):
                    graph.validate_fixture(value)

    def test_all_stage_causal_prefixes_are_exact_in_both_profiles(self):
        prompt, changed = self.fixture['cases'][1:3]
        for stage, array in prompt['arrays'].items():
            width = array['shape'][1]
            for profile in ('main_bits', 'wide_bits'):
                self.assertEqual(array[profile][:3 * width], changed['arrays'][stage][profile][:3 * width])

    def test_fresh_output_rejects_before_arithmetic(self):
        with tempfile.TemporaryDirectory() as folder:
            occupied = Path(folder) / 'occupied'
            occupied.write_bytes(b'preserve')
            result = subprocess.run([sys.executable, '-s', '-B', str(Path(graph.__file__)), '--fresh-directory', str(occupied)],
                                    capture_output=True, timeout=10)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(b'Fresh fixture directory required', result.stderr)
            self.assertEqual(occupied.read_bytes(), b'preserve')


if __name__ == '__main__':
    unittest.main()

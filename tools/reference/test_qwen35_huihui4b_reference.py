"""Independent fixed-workload and conservative-envelope controls, no model I/O.

Pure literals/exact rational arithmetic cannot qualify a source, numerical graph,
FP32 adoption or a real reference capture. No fake full-model positive fixture.
"""
import copy
from fractions import Fraction
import json
import math
from pathlib import Path
import struct
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import qwen35_huihui4b_reference as reference
import qwen35_huihui4b_envelope as envelope

TOKENS17 = (1, 7, 13, 3, 9, 5, 11, 2, 17, 4, 6, 10, 19, 8, 21, 23, 29)
SOURCE = dict(container_bytes=2367237149,
    container_sha256='244b47c46b759664c223a996046857720826a388bdf3969620a8f12ebb6e14e0',
    config_bytes=3024, config_sha256='13a514e981196ba683d8d53312789384ff5d9b9e73c7bfa137010397212c4af3',
    header_bytes=119829, header_sha256='7284f68df36a7761569b48321066e92d616306e33b4a9487029673a0ff344cb2',
    payload_origin=119837)
GEOMETRY = dict(hidden=2560, intermediate=9216, vocab=248320, layers=32,
    query_heads=16, kv_heads=4, head_dim=256, rotated_dims=64, key_heads=16,
    value_heads=32, key_dim=128, value_dim=128, max_positions=262144,
    rope_theta=10000000.0, rms_epsilon=1e-6)


def exact_gate(value, binary_constants=False):
    center = Fraction.from_float(value)
    if binary_constants:
        radius = Fraction.from_float(1e-5) + Fraction.from_float(1e-4) * abs(center)
    else:
        radius = Fraction(1, 100000) + Fraction(1, 10000) * abs(center)
    return center - radius, center + radius


def steps(value, direction, count):
    for _ in range(count):
        value = math.nextafter(value, direction)
    return value


class FixedSourceReferenceTests(unittest.TestCase):
    def test_frozen_seventeen_ids_repeat_to_256_at_absolute_offset_zero(self):
        tokens, offset, values = reference.plan()
        self.assertEqual(reference.TOKENS, TOKENS17)
        self.assertEqual(tokens, [TOKENS17[i % 17] for i in range(256)])
        self.assertEqual(tokens[-1], 1)
        self.assertIs(type(offset), int)
        self.assertEqual(offset, 0)
        self.assertEqual(values, {128: 18180608, 256: 20310528})
        tokens[0] = 99
        values[128] = 0
        self.assertEqual(reference.plan()[0][0], 1)
        self.assertEqual(reference.plan()[2], {128: 18180608, 256: 20310528})

    def test_only_both_full_checkpoints_are_kept_and_profiles_cannot_change(self):
        self.assertEqual(reference.CHECKPOINTS, (128, 256))
        self.assertEqual([reference.checkpoint(i) for i in [0, 126, 127, 128, 254, 255]],
                         [None, None, 128, None, None, 256])
        self.assertEqual([i for i in range(256) if reference.checkpoint(i) is not None], [127, 255])
        self.assertEqual(reference.PROFILES, (('ideal_fp64', False), ('staged_fp32', True)))
        self.assertTrue(all(type(staged) is bool for _, staged in reference.PROFILES))
        self.assertEqual(reference.ARRAYS, 1051)
        self.assertEqual(reference.VALUES, 16067328)
        self.assertEqual(reference.GEOMETRY, GEOMETRY)
        self.assertEqual(reference.graph.GATE, dict(atol=1e-5, rtol=1e-4))

    def test_checkpoint_rejects_noninteger_subclass_boolean_and_invalid_coordinates(self):
        class Coordinate(int):
            pass
        for token in [True, False, -1, 256, 1.0, '127', None, Coordinate(127)]:
            with self.subTest(token=token), self.assertRaises(ValueError):
                reference.checkpoint(token)

    def test_literal_capture_and_resource_plan_include_helper_without_fit_claim(self):
        self.assertEqual(reference.ROWS_BYTES, 461893632)
        self.assertEqual(sum(reference.plan()[2].values()) * 12, 461893632)
        self.assertEqual(reference.MAX_CAPTURE, 536870912)
        self.assertLess(reference.ROWS_BYTES, reference.MAX_CAPTURE)
        self.assertEqual(reference.DEADLINE_SECONDS, 21600)
        self.assertEqual(reference.PROCESS_LIMIT, 6442450944)
        self.assertEqual(reference.HELPER_BYTES, 33554432)
        self.assertEqual(reference.LEDGER, dict(python_state_and_temporaries=2147483648,
            interpreter_and_runtime=536870912, small_weight_vectors=67108864,
            tile_and_source_buffers=134217728, capture=536870912,
            guard_helper=33554432, controller_and_logs=67108864))
        self.assertTrue(all(type(n) is int and n > 0 for n in reference.LEDGER.values()))
        self.assertLessEqual(sum(reference.LEDGER.values()), 6442450944)

    def test_new_source_pins_are_distinct_without_rewriting_old_identity(self):
        old = (reference.base.REPOSITORY, reference.base.REVISION,
               reference.base.pin.FILE_SHA, reference.base.pin.CONFIG_SHA)
        self.assertEqual(reference.REPOSITORY, 'TheWirelessPhoenix/Qwen3.5-4B-abliterated_4bit')
        self.assertEqual(reference.REVISION, '916d9f7c2e27ed41e2b9ed39f3a2badccb2bbc7c')
        self.assertEqual(reference.SOURCE_PINS, SOURCE)
        self.assertEqual((reference.FILE_BYTES, reference.FILE_SHA),
                         (2367237149, SOURCE['container_sha256']))
        self.assertEqual((reference.CONFIG_BYTES, reference.CONFIG_SHA), (3024, SOURCE['config_sha256']))
        self.assertEqual((reference.HEADER_BYTES, reference.HEADER_SHA, reference.ORIGIN),
                         (119829, SOURCE['header_sha256'], 119837))
        self.assertNotEqual(reference.FILE_SHA, old[2])
        self.assertNotEqual(reference.CONFIG_SHA, old[3])
        reference.plan()
        reference.checkpoint(127)
        self.assertEqual((reference.base.REPOSITORY, reference.base.REVISION,
                          reference.base.pin.FILE_SHA, reference.base.pin.CONFIG_SHA), old)

    def test_graph_proof_and_extras_are_bounded_source_metadata_not_model_execution(self):
        proof = reference.graph_provenance()
        self.assertEqual(set(proof), {'original_sha256', 'vector_sha256', 'helper_sha256',
                                     'builder_sha256', 'inverse_patch_exact'})
        self.assertIs(proof['inverse_patch_exact'], True)
        for key in ['original_sha256', 'vector_sha256', 'helper_sha256', 'builder_sha256']:
            self.assertRegex(proof[key], r'^[0-9a-f]{64}$')
        paths = reference.extras()
        self.assertEqual(len(paths), len(set(paths)))
        self.assertTrue(all(isinstance(path, Path) and path.is_absolute() and path.is_file() for path in paths))
        self.assertEqual({path.name for path in paths}, {
            'qwen35_real_weights.py', 'check_qwen35_payload.py', 'qwen35_cpu_graph.py',
            'affine4_tiles.py', 'qwen35_vector_graph.py', 'qwen35_recurrent_vector.py',
            'build_qwen35_vector_graph.py', 'bf16_source.py', 'qwen35_huihui4b_reference.py'})
        wrong = dict(proof, inverse_patch_exact=False)
        with mock.patch.object(reference.builder, 'validate', return_value=wrong):
            with self.assertRaises(ValueError):
                reference.graph_provenance()
        self.assertIn('staged_fp32', reference.ARITHMETIC_EQUIVALENCE['historical_scope'])
        self.assertIn('not an ideal_fp64', reference.ARITHMETIC_EQUIVALENCE['historical_scope'])

    def _metadata_owner(self, **changes):
        # An uninitialized real class instance carries metadata only. No file is
        # created/opened and no Source constructor or authenticated read occurs.
        owner = object.__new__(reference.bf16_source.Source)
        owner.size = 2367237149
        owner.sha256 = SOURCE['container_sha256']
        owner.closed = False
        owner.reason = None
        for name, value in changes.items():
            setattr(owner, name, value)
        return owner

    def test_wrong_owner_identity_closed_or_poisoned_rejects_before_config_or_read(self):
        fake = SimpleNamespace(size=2367237149, sha256=SOURCE['container_sha256'],
                               closed=False, poisoned=False)
        owners = [fake, self._metadata_owner(size=2367237148),
                  self._metadata_owner(sha256='0' * 64), self._metadata_owner(closed=True),
                  self._metadata_owner(reason='synthetic poison')]
        with mock.patch.object(reference.bf16_source.Source, 'read', side_effect=AssertionError('no model read')) as read:
            for owner in owners:
                with self.subTest(owner=type(owner)), self.assertRaisesRegex(ValueError, 'Authenticated open fixed new4'):
                    reference.weights_binding(owner, b'invalid config', 1e100)
            read.assert_not_called()

    def test_wrong_config_type_length_or_hash_rejects_before_header_or_base_binding(self):
        owner = self._metadata_owner()
        with mock.patch.object(reference.bf16_source.Source, 'read', side_effect=AssertionError('no model read')) as read, \
             mock.patch.object(reference.base, 'bind', side_effect=AssertionError('no graph binding')) as bind:
            for config in [None, '', bytearray(3024), bytes(3023), bytes(3024)]:
                with self.subTest(config=type(config)), self.assertRaisesRegex(ValueError, 'Pinned new4 config'):
                    reference.weights_binding(owner, config, 1e100)
            read.assert_not_called()
            bind.assert_not_called()

    def test_preopen_failure_retains_distinct_schema_and_unqualified_endpoint_record(self):
        calls = []
        with tempfile.TemporaryDirectory(prefix='toklet-new4-reference-control-') as folder:
            output = Path(folder) / 'capture'
            # Only the version attribute is reached before the injected source
            # proof refusal. No array function, config or model is consumed.
            with mock.patch.dict(sys.modules, {'numpy': SimpleNamespace(__version__='2.3.2')}), \
                 mock.patch.object(reference, 'graph_provenance', side_effect=ValueError('independent preopen refusal')), \
                 mock.patch.object(reference.runtime_owner.runtime, 'captured', side_effect=AssertionError('no config read')) as config:
                result = reference.collect(output, Path(folder) / 'no-model', {'synthetic': True},
                                           lambda: calls.append('endpoint') or {'synthetic': 'endpoint'})
            config.assert_not_called()
            self.assertEqual(calls, ['endpoint'])
            self.assertEqual(result['format'], 'toklet-qwen35-huihui4b-context256-reference-v1')
            self.assertEqual(result['reference_variant'], 'ordered_vector_recurrence_context256')
            self.assertEqual(result['source_pins'], SOURCE)
            self.assertEqual(result['primary_error']['message'], 'independent preopen refusal')
            self.assertFalse(result['completed'])
            self.assertIsNone(result['source'])
            self.assertIsNone(result['source_endpoint_sha256'])
            self.assertIsNone(result['rows_sha256'])
            self.assertEqual(result['endpoint_errors'], [])
            self.assertEqual(result['runtime_endpoint'], {'synthetic': 'endpoint'})
            for key in ['engine_executed', 'contextual_numerical_qualified', 'request_metrics_qualified',
                        'quality_qualified', 'managed_adoption_qualified']:
                self.assertIs(result[key], False)
            self.assertEqual(sorted(path.name for path in output.iterdir()), ['result.json'])
            self.assertEqual(json.loads((output / 'result.json').read_bytes()), result)


class ConservativeEnvelopeTests(unittest.TestCase):
    def test_zero_has_four_radius_ulps_plus_outward_endpoint_guards(self):
        bits = lambda value: struct.unpack('<Q', struct.pack('<d', value))[0]
        for value in [0.0, -0.0]:
            lo, hi = envelope.interval_bounds(value)
            self.assertLess(lo, -1e-5)
            self.assertGreater(hi, 1e-5)
            self.assertEqual(bits(hi) - bits(1e-5), 5)
            self.assertEqual(bits(-lo) - bits(1e-5), 5)

    def test_exact_decimal_and_binary_original_intervals_are_subsets(self):
        values = [0.0, -0.0, math.nextafter(0.0, math.inf), math.nextafter(0.0, -math.inf),
                  1e-300, -1e-300, 0.25, -0.25, 1.0, -1.0, 1e100, -1e100,
                  float.fromhex('0x1.abcdef1234567p+900'), float.fromhex('-0x1.abcdef1234567p+900')]
        for value in values:
            with self.subTest(value=value):
                bounds = envelope.interval_bounds(value)
                self.assertIs(type(bounds), tuple)
                self.assertEqual(len(bounds), 2)
                lo, hi = bounds
                self.assertIs(type(lo), float)
                self.assertIs(type(hi), float)
                self.assertTrue(math.isfinite(lo) and math.isfinite(hi))
                self.assertLess(lo, value)
                self.assertGreater(hi, value)
                for binary in [False, True]:
                    exact_lo, exact_hi = exact_gate(value, binary)
                    self.assertLessEqual(Fraction.from_float(lo), exact_lo)
                    self.assertGreaterEqual(Fraction.from_float(hi), exact_hi)

    def test_identical_adjacent_and_overlapping_profiles_do_not_false_reject(self):
        pairs = [(0.0, -0.0), (0.0, 1e-6), (-1e-5, 1e-5), (-1.0, -1.0001),
                 (float(Fraction(-1, 99990)), float(Fraction(1, 99990)))]
        for value in [0.0, -0.0, 1e-30, -1e-30, 1.0, -1.0, 1e100, -1e100]:
            pairs.append((value, value))
            pairs.append((value, math.nextafter(value, math.inf)))
        for ideal, staged in pairs:
            with self.subTest(ideal=ideal, staged=staged):
                self.assertIs(envelope.disjoint(ideal, staged), False)
                self.assertIs(envelope.disjoint(staged, ideal), False)
        # False means no disjoint-envelope proof; it never establishes FP32,
        # numerical/quality, source or default adoption.

    def test_literal_separated_profiles_are_symmetric_with_exact_fraction_gap(self):
        for ideal, staged in [(0.0, 1e-3), (1.0, 1.001), (-1.0, -1.001), (-1e-4, 1e-4)]:
            with self.subTest(ideal=ideal, staged=staged):
                self.assertIs(envelope.disjoint(ideal, staged), True)
                self.assertIs(envelope.disjoint(staged, ideal), True)
                a, b = exact_gate(ideal), exact_gate(staged)
                self.assertTrue(a[1] < b[0] or b[1] < a[0])
                guarded_a = tuple(Fraction.from_float(x) for x in envelope.interval_bounds(ideal))
                guarded_b = tuple(Fraction.from_float(x) for x in envelope.interval_bounds(staged))
                self.assertTrue(guarded_a[1] < guarded_b[0] or guarded_b[1] < guarded_a[0])

    def test_touching_threshold_and_neighboring_floats_preserve_conservative_direction(self):
        # For zero vspositive x, exactdecimalintervals touch at
        # x*(1-1/10000)=2/100000, hence x=1/49995 independently.
        center = float(Fraction(1, 49995))
        below = steps(center, 0.0, 64)
        above = steps(center, math.inf, 64)
        self.assertIs(envelope.disjoint(0.0, below), False)
        self.assertIs(envelope.disjoint(0.0, center), False)
        self.assertIs(envelope.disjoint(0.0, above), True)
        self.assertIs(envelope.disjoint(above, 0.0), True)
        self.assertGreater(exact_gate(above)[0] - exact_gate(0.0)[1], 0)
        self.assertIs(envelope.disjoint(-center, -0.0), False)
        self.assertIs(envelope.disjoint(-above, -0.0), True)
        # Negative-center contact, avoiding a test that covers onlyabs(v)>=0.
        negative = float(Fraction(-1, 4) + Fraction(7, 100010))
        self.assertIs(envelope.disjoint(-0.25, negative), False)
        self.assertIs(envelope.disjoint(-0.25, steps(negative, math.inf, 64)), True)

    def test_exact_float_types_nonfinite_and_overflow_are_rejected(self):
        class NotPlainFloat(float):
            pass
        for value in [0, 1, True, False, None, '0', NotPlainFloat(1.0),
                      float('nan'), float('inf'), -float('inf')]:
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    envelope.interval_bounds(value)
                with self.assertRaises(ValueError):
                    envelope.disjoint(value, 0.0)
                with self.assertRaises(ValueError):
                    envelope.disjoint(0.0, value)
        for value in [sys.float_info.max, -sys.float_info.max]:
            with self.subTest(overflow=value):
                with self.assertRaises(ValueError):
                    envelope.interval_bounds(value)
                with self.assertRaises(ValueError):
                    envelope.disjoint(value, 0.0)
                with self.assertRaises(ValueError):
                    envelope.disjoint(0.0, value)

    def test_partial_or_counter_only_inventory_cannot_be_admitted_as_full_capture(self):
        row = dict(checkpoint=128, profile='ideal_fp64', token=127, layer=None,
                   stage='embedding', count=2560, offset=0, bytes=20480, sha256='a' * 64)
        for descriptors in [None, {}, {'completed': True, 'arrays': 4204}, [], [row]]:
            with self.subTest(descriptors=type(descriptors)), self.assertRaises(ValueError):
                envelope.validate_inventory(descriptors)

    def test_duplicate_or_overlong_descriptor_counts_cannot_fake_full_geometry(self):
        row = dict(checkpoint=128, profile='ideal_fp64', token=127, layer=None,
                   stage='embedding', count=2560, offset=0, bytes=20480, sha256='a' * 64)
        for count in [4204, 4205]:
            with self.subTest(count=count), self.assertRaises(ValueError):
                envelope.validate_inventory([copy.deepcopy(row) for _ in range(count)])


if __name__ == '__main__':
    unittest.main()

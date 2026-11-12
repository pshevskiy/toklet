"""Portable strict-coverage/hash/replay controls; no NumPy, model or GPU import."""
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest import mock

import check_granite_reference as checker

PIN = 'a' * 64


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def protocol():
    cases = checker.fixed_cases()
    count = sum(sum(shape[0] * shape[1] * (4 if name == 'logits' else 2)
                    for name, shape in checker.contracts(len(case['tokens'])).items()) for case in cases)
    return dict(format='toklet-granite-full-graph-prospective-v1', cases=cases,
                geometry=dict(checker.GEOMETRY), numerical_policy=dict(atol_floor=1e-4, envelope_factor=2, rtol=1e-3),
                native=dict(capacity=512, projection='MlxAffine', normalization='Compensated', attention='Reference',
                            chunks=[128, 32, 1], effective_chunk='min(requested,512-position_offset)'),
                model=dict(data_sha256=PIN, config_sha256=PIN, conversion_manifest_sha256=PIN,
                           conversion_journal_sha256=PIN), source_inputs_sha256={'source.py': PIN},
                memory_and_disk=dict(engine_limit_bytes=6 * 1024**3, reference_setup_limit_bytes=512 * 1024**2,
                                     helper_reserve_bytes=32 * 1024**2, disk_floor_bytes=2 * 1024**3,
                                     raw_bytes_per_reference_profile_or_lossless_native_capture=count))


def reference_fixture(value):
    arrays, cases, total = {}, [], 0
    for case in value['cases']:
        for profile in ('main', 'wide'):
            cases.append(dict(name=case['name'], profile=profile, arrays=643, last_logits_sha256=PIN))
            for stage, shape in checker.contracts(len(case['tokens'])).items():
                dtype = '<f4' if stage == 'logits' else '<f2'
                size = shape[0] * shape[1] * checker.DTYPES[dtype]
                arrays[case['name'] + '.' + profile + '.' + stage] = dict(file=case['name'] + '/' + profile + '/' + stage + '.bin',
                    shape=shape, dtype=dtype, bytes=size, sha256=PIN)
                total += size
    return dict(format='toklet-granite-independent-reference-v1', mode='capture', passed=True,
                protocol_sha256=PIN, conversion_manifest_sha256=PIN, contract_sha256=PIN,
                source_consumption=dict(file_sha256=PIN, component_reads=1, norm_reads=1, consumed_bytes=1),
                geometry=value['geometry'], numerical_policy=value['numerical_policy'],
                arrays=arrays, cases=cases, published_bytes=total, native_compared=False,
                graph_qualified=False, quality_qualified=False, performance_qualified=False,
                managed_adoption_qualified=False, whole_goal_complete=False)


def native_fixture(value):
    cases = []
    for case in value['cases']:
        for chunk in (128, 32, 1):
            effective = min(chunk, 512 - case['offset'])
            arrays = {}
            for stage, shape in checker.contracts(len(case['tokens'])).items():
                dtype = '<f4' if stage == 'logits' else '<f2'
                arrays[stage] = dict(shape=shape, dtype=dtype, bytes=shape[0] * shape[1] * checker.DTYPES[dtype],
                                     sha256=PIN, baseline_file=stage + '.bin', mismatching_bytes=0, patches=[])
            checks = {'exact_chunk_schedule', 'diagnostic_last_row_matches_logits',
                      'full_logit_greedy_selection', 'schedule_all_stage_bytes_equal',
                      'fully_cached_prompt_zero_work_unchanged_logits', 'capacity_rejection_preserves_prefix',
                      'reset_clears_history_KV_prefix_and_distribution',
                      'reset_replay_exact_schedule_and_final_logits',
                      'greedy_token_readback_equals_full_logit_selection'}
            if len(case['tokens']) < 512 - case['offset']:
                checks.add('pending_sampled_token_is_uncommitted_and_discardable')
            if case['name'] == 'short':
                checks.add('changed_prefix_replay_equals_fresh_session')
            trace = dict(folder=f"{case['name']}/chunk{chunk}", baseline_folder=case['name'] + '/chunk128',
                         effective_chunk=effective,
                         effective_chunks=[min(effective, len(case['tokens']) - first)
                                           for first in range(0, len(case['tokens']), effective)],
                         arrays=arrays, final_logits_sha256=PIN, final_greedy_token=0,
                         diagnostic_last_row_matches_logits=True)
            entry = dict(name=case['name'], requested_chunk=chunk, position_offset=case['offset'],
                         tokens=case['tokens'], **trace)
            if chunk == 128:
                checks.add('reset_replay_all_stage_bytes_equal')
                entry['reset_replay'] = dict(trace, folder=trace['folder'] + '/reset_replay')
            entry['state_checks'] = dict.fromkeys(checks, True)
            cases.append(entry)
    return dict(format='toklet-granite-native-graph-v1', protocol_sha256=PIN,
                model_data_sha256=PIN, model_config_sha256=PIN, cases=cases,
                complete=True, error=None, state_checks_passed=True)


def saved(folder, name, raw):
    path = folder / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return dict(bytes=len(raw), sha256=digest(raw))


class StrictJSON(unittest.TestCase):
    def test_imports_remain_numerical_package_independent(self):
        for name in ('numpy', 'mlx', 'mlx.core', 'torch'):
            self.assertNotIn(name, sys.modules)

    def test_duplicate_nonfinite_overflow_and_trailing_json_reject(self):
        for raw in (b'{"x":1,"x":2}', b'{"x":NaN}', b'{"x":Infinity}', b'{"x":1e999}',
                    b'{"x":9223372036854775808}', b'{"x":-9223372036854775808}', b'{} {}'):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                checker.parsed(raw)
        self.assertEqual(checker.parsed(b'{"x":1.5,"y":9223372036854775807}'),
                         dict(x=1.5, y=9223372036854775807))

    def test_concrete_types_and_sha_pins_reject_aliases(self):
        for value in (True, 1., '1', None, -1, 1 << 63):
            with self.assertRaises(ValueError):
                checker.integer(value)
        for pin in (None, '', 'A' * 64, 'a' * 63, 'g' * 64):
            with self.assertRaises(ValueError):
                checker.sha(pin)
        with self.assertRaises(ValueError):
            checker.exact([True], [1], 'boolean alias')

    def test_only_canonical_relative_paths_are_admitted(self):
        for path in ('', '/abs', '../x', 'a/../x', './x', 'a//x', 'a/', 'a\\x', 'a\x00x'):
            with self.subTest(path=path), self.assertRaises(ValueError):
                checker.relative(path)
        self.assertEqual(checker.relative('single/main/layers.0.q.bin'), ['single', 'main', 'layers.0.q.bin'])

    def test_external_json_hash_extent_and_object_type_are_verified(self):
        with tempfile.TemporaryDirectory(dir='/private/tmp') as tmp:
            folder = Path(tmp)
            path = folder / 'input.json'
            pin = saved(folder, path.name, b'{"ok":true}')['sha256']
            self.assertEqual(checker.decoded(path, pin), dict(ok=True))
            with self.assertRaises(ValueError):
                checker.decoded(path, '0' * 64)
            with self.assertRaises(ValueError):
                checker.decoded(path, pin, 1)
            path.write_bytes(b'[]')
            with self.assertRaises(ValueError):
                checker.decoded(path, digest(b'[]'))


class Coverage(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = protocol()
        cls.reference = reference_fixture(cls.protocol)
        cls.native = native_fixture(cls.protocol)

    def test_complete_frozen_case_stage_byte_coverage(self):
        self.assertEqual(checker.validate_protocol(self.protocol), self.protocol)
        checker.validate_reference(self.reference, self.protocol, PIN)
        indexed = checker.validate_native(self.native, self.protocol, PIN)
        self.assertEqual((len(self.reference['arrays']), len(indexed)), (12860, 30))
        self.assertEqual(len(checker.contracts(512)), 643)
        self.assertEqual(self.protocol['memory_and_disk']['raw_bytes_per_reference_profile_or_lossless_native_capture'],
                         5279121408)
        self.assertEqual(indexed['offset509', 128]['effective_chunks'], [3])
        self.assertEqual(indexed['context512', 128]['effective_chunks'], [128] * 4)
        self.assertEqual(sum(len(entry['state_checks']) for entry in indexed.values()), 304)
        self.assertEqual(sum('reset_replay' in entry for entry in indexed.values()), 10)
        self.assertEqual(sum(1 + ('reset_replay' in entry) for entry in indexed.values()), 40)
        for name in ('offset509', 'context512', 'context128_offset384'):
            self.assertNotIn('pending_sampled_token_is_uncommitted_and_discardable', indexed[name, 128]['state_checks'])

    def test_protocol_tokens_offsets_geometry_limits_and_policy_cannot_drift(self):
        for category, field, changed in (('geometry', 'hidden', 2561), ('geometry', 'layers', True),
                                        ('native', 'chunks', [128, 32]), ('numerical_policy', 'rtol', .002),
                                        ('memory_and_disk', 'engine_limit_bytes', 7 * 1024**3)):
            value = copy.deepcopy(self.protocol)
            value[category][field] = changed
            with self.subTest(field=field), self.assertRaises(ValueError):
                checker.validate_protocol(value)
        value = copy.deepcopy(self.protocol)
        value['cases'][0]['tokens'] = [100282]
        with self.assertRaises(ValueError):
            checker.validate_protocol(value)
        value = copy.deepcopy(self.protocol)
        value['cases'][3]['offset'] = True
        with self.assertRaises(ValueError):
            checker.validate_protocol(value)

    def test_reference_missing_empty_extra_profiles_and_receipts_never_pass(self):
        for change in ('empty', 'missing', 'extra', 'receipt'):
            value = dict(self.reference)
            value['arrays'] = dict(value['arrays'])
            if change == 'empty':
                value['arrays'] = {}
            elif change == 'missing':
                del value['arrays']['single.wide.embedding']
            elif change == 'extra':
                value['arrays']['extra'] = value['arrays']['single.main.embedding']
            else:
                value['cases'] = value['cases'][:-1]
            with self.subTest(change=change), self.assertRaises(ValueError):
                checker.validate_reference(value, self.protocol, PIN)

    def test_reference_storage_types_extents_source_and_protocol_are_checked(self):
        for field, changed in (('shape', [True, 2560]), ('dtype', '<f4'), ('bytes', 5121),
                               ('sha256', 'z' * 64), ('file', 'single/main/../embedding.bin')):
            value = dict(self.reference)
            value['arrays'] = dict(value['arrays'])
            key = 'single.main.embedding'
            value['arrays'][key] = dict(value['arrays'][key], **{field: changed})
            with self.subTest(field=field), self.assertRaises(ValueError):
                checker.validate_reference(value, self.protocol, PIN)
        for field, changed in (('passed', 1), ('protocol_sha256', 'b' * 64),
                               ('conversion_manifest_sha256', 'b' * 64), ('published_bytes', True),
                               ('source_consumption', dict(file_sha256='b' * 64))):
            with self.subTest(field=field), self.assertRaises(ValueError):
                checker.validate_reference(dict(self.reference, **{field: changed}), self.protocol, PIN)

    def test_native_missing_empty_duplicate_case_and_stages_never_pass(self):
        for change in ('empty', 'missing', 'duplicate', 'stage', 'empty_stages'):
            value = dict(self.native)
            value['cases'] = list(value['cases'])
            if change == 'empty':
                value['cases'] = []
            elif change == 'missing':
                value['cases'] = value['cases'][:-1]
            elif change == 'duplicate':
                value['cases'][1] = value['cases'][0]
            else:
                value['cases'][0] = dict(value['cases'][0])
                value['cases'][0]['arrays'] = dict(value['cases'][0]['arrays'])
                if change == 'stage':
                    del value['cases'][0]['arrays']['layers.39.output']
                else:
                    value['cases'][0]['arrays'] = {}
            with self.subTest(change=change), self.assertRaises(ValueError):
                checker.validate_native(value, self.protocol, PIN)

    def test_native_tokens_schedule_folders_and_concrete_state_booleans(self):
        for field, changed in (('tokens', [100282]), ('position_offset', True), ('requested_chunk', True),
                               ('effective_chunk', 127), ('effective_chunks', [True]),
                               ('folder', 'single/chunk32'), ('baseline_folder', '../baseline'),
                               ('state_checks', {}), ('state_checks', dict(committed_length=1))):
            value = dict(self.native)
            value['cases'] = list(value['cases'])
            value['cases'][0] = dict(value['cases'][0], **{field: changed})
            with self.subTest(field=field, changed=changed), self.assertRaises(ValueError):
                checker.validate_native(value, self.protocol, PIN)
        value = dict(self.native)
        value['cases'] = list(value['cases'])
        value['cases'][0] = dict(value['cases'][0], state_checks=dict(value['cases'][0]['state_checks'],
                                                                  exact_chunk_schedule=False))
        self.assertFalse(checker.validate_native(value, self.protocol, PIN)['single', 128]['state_checks']['exact_chunk_schedule'])
        checker.validate_native(dict(self.native, state_checks_passed=False), self.protocol, PIN)

    def test_native_source_protocol_and_full_baseline_patch_constraints(self):
        for field in ('model_data_sha256', 'model_config_sha256', 'protocol_sha256'):
            with self.subTest(field=field), self.assertRaises(ValueError):
                checker.validate_native(dict(self.native, **{field: 'b' * 64}), self.protocol, PIN)
        value = copy.deepcopy(self.native)
        value['cases'][0]['arrays']['embedding']['mismatching_bytes'] = 1
        with self.assertRaises(ValueError):
            checker.validate_native(value, self.protocol, PIN)

    def test_supplied_trace_receipts_and_native_completion_have_concrete_types(self):
        entry = self.native['cases'][0]
        for field, changed in (('final_greedy_token', True), ('final_greedy_token', 100352),
                               ('final_logits_sha256', 'z' * 64), ('diagnostic_last_row_matches_logits', 1)):
            value = dict(self.native)
            value['cases'] = list(value['cases'])
            value['cases'][0] = dict(entry, **{field: changed})
            with self.subTest(field=field), self.assertRaises(ValueError):
                checker.validate_native(value, self.protocol, PIN)
        for field, changed in (('complete', False), ('complete', 1), ('error', 'failed'), ('state_checks_passed', 1)):
            with self.subTest(field=field), self.assertRaises(ValueError):
                checker.validate_native(dict(self.native, **{field: changed}), self.protocol, PIN)

    def test_reset_replay_is_separate_complete_and_schedule_bound(self):
        entry = self.native['cases'][0]
        reset = dict(entry['reset_replay'], folder='single/reset')
        value = dict(self.native)
        value['cases'] = list(value['cases'])
        value['cases'][0] = dict(entry, reset_replay=reset)
        checker.validate_native(value, self.protocol, PIN)
        for field, changed in (('arrays', {}), ('effective_chunks', [2]), ('folder', entry['folder']),
                               ('baseline_folder', 'short/chunk128')):
            value['cases'][0] = dict(entry, reset_replay=dict(reset, **{field: changed}))
            with self.subTest(field=field), self.assertRaises(ValueError):
                checker.validate_native(value, self.protocol, PIN)

    def test_every_required_state_boolean_is_mandatory(self):
        removed = 0
        for index, entry in enumerate(self.native['cases']):
            for check in entry['state_checks']:
                value = dict(self.native, cases=list(self.native['cases']))
                checks = dict(entry['state_checks'])
                del checks[check]
                value['cases'][index] = dict(entry, state_checks=checks)
                with self.subTest(case=entry['name'], chunk=entry['requested_chunk'], check=check), \
                        self.assertRaisesRegex(ValueError, 'Missing required native state checks'):
                    checker.validate_native(value, self.protocol, PIN)
                removed += 1
        self.assertEqual(removed, 304)

    def test_all_ten_full_stage_resets_are_mandatory_and_repeat_resets_reject(self):
        removed = 0
        for index, entry in enumerate(self.native['cases']):
            value = dict(self.native, cases=list(self.native['cases']))
            if entry['requested_chunk'] == 128:
                changed = dict(entry)
                del changed['reset_replay']
                removed += 1
            else:
                changed = dict(entry, reset_replay=self.native['cases'][index - index % 3]['reset_replay'])
            value['cases'][index] = changed
            with self.subTest(case=entry['name'], chunk=entry['requested_chunk']), self.assertRaises(ValueError):
                checker.validate_native(value, self.protocol, PIN)
        self.assertEqual(removed, 10)

    def test_every_primary_and_reset_final_row_receipt_field_is_mandatory(self):
        removed = 0
        for index, entry in enumerate(self.native['cases']):
            for reset in ((False, True) if entry['requested_chunk'] == 128 else (False,)):
                for field in ('final_logits_sha256', 'final_greedy_token', 'diagnostic_last_row_matches_logits'):
                    value = dict(self.native, cases=list(self.native['cases']))
                    changed = dict(entry)
                    receipt = dict(entry['reset_replay']) if reset else changed
                    del receipt[field]
                    if reset:
                        changed['reset_replay'] = receipt
                    value['cases'][index] = changed
                    with self.subTest(case=entry['name'], chunk=entry['requested_chunk'], reset=reset, field=field), \
                            self.assertRaises(ValueError):
                        checker.validate_native(value, self.protocol, PIN)
                    removed += 1
        self.assertEqual(removed, 120)

    def test_top_level_completion_summary_and_reset_schedule_must_be_explicit(self):
        for field in ('complete', 'error', 'state_checks_passed'):
            value = dict(self.native)
            del value[field]
            with self.subTest(field=field), self.assertRaises(ValueError):
                checker.validate_native(value, self.protocol, PIN)
        entry = self.native['cases'][0]
        value = dict(self.native, cases=list(self.native['cases']))
        reset = dict(entry['reset_replay'])
        del reset['effective_chunk']
        value['cases'][0] = dict(entry, reset_replay=reset)
        with self.assertRaises(ValueError):
            checker.validate_native(value, self.protocol, PIN)


class Completion(unittest.TestCase):
    @staticmethod
    def complete_result():
        return dict(arrays=19290, required_native_state_checks=304, unavailable_native_state_checks=0,
                    reset_replayed_arrays=6430, reset_replayed_cases=[case['name'] for case in checker.fixed_cases()],
                    reset_unavailable_cases=[], final_logit_receipts_checked=40, final_logit_receipts_verified=40,
                    failed_profile_coordinates=0, exact_schedule_mismatching_bytes=0, reset_mismatching_bytes=0,
                    failed_native_state_checks=0, native_receipt_failures=[], native_state_summary=True,
                    exact_native_causal_prefix=dict(mismatching_bytes=0))

    def test_missing_required_coverage_cannot_complete_or_pass_even_with_zero_numerical_failures(self):
        complete = self.complete_result()
        self.assertEqual(checker.comparison_status(complete), dict(completed=True, passed=True))
        for field, changed in (('arrays', 19289), ('required_native_state_checks', 303),
                               ('unavailable_native_state_checks', 1), ('reset_replayed_arrays', 6429),
                               ('reset_replayed_cases', complete['reset_replayed_cases'][:-1]),
                               ('reset_replayed_cases', complete['reset_replayed_cases'][:-1] + ['single']),
                               ('reset_unavailable_cases', ['single']), ('final_logit_receipts_checked', 39),
                               ('final_logit_receipts_verified', 39)):
            with self.subTest(field=field, changed=changed):
                result = dict(complete, **{field: changed})
                self.assertEqual(result['failed_profile_coordinates'], 0)
                self.assertEqual(checker.comparison_status(result), dict(completed=False, passed=False))

    def test_present_false_state_summary_checks_or_other_failures_remain_completed_but_do_not_pass(self):
        for field, changed in (('native_state_summary', False), ('native_state_summary', None),
                               ('failed_native_state_checks', 1), ('failed_profile_coordinates', 212),
                               ('exact_schedule_mismatching_bytes', 1), ('reset_mismatching_bytes', 1),
                               ('native_receipt_failures', [dict(field='final_greedy_token')]),
                               ('exact_native_causal_prefix', dict(mismatching_bytes=1))):
            with self.subTest(field=field):
                self.assertEqual(checker.comparison_status(dict(self.complete_result(), **{field: changed})),
                                 dict(completed=True, passed=False))

    def test_mandatory_receipts_are_compared_to_raw_hash_and_lowest_id_without_optional_skips(self):
        receipt = dict(final_logits_sha256=PIN, final_greedy_token=7, diagnostic_last_row_matches_logits=True)
        actual = dict(final_logits_sha256=PIN, final_greedy_token=7)
        self.assertEqual(checker.check_trace_receipt(receipt, actual), [])
        for field, changed in (('final_logits_sha256', 'b' * 64), ('final_greedy_token', 8),
                               ('diagnostic_last_row_matches_logits', False)):
            with self.subTest(field=field):
                failures = checker.check_trace_receipt(dict(receipt, **{field: changed}), actual)
                self.assertEqual(len(failures), 1)
                self.assertEqual(failures[0]['field'], field)


class RawAndReplay(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir='/private/tmp')
        self.folder = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_original_raw_hash_size_truncation_and_trailing_bytes(self):
        pin = saved(self.folder, 'base.bin', b'original')
        with checker.Files(self.folder) as files:
            with files.open('base.bin', **pin) as source:
                self.assertEqual(b''.join(raw for _, raw in checker.original_blocks(source)), b'original')
                with self.assertRaises(ValueError):
                    source.read(7, 2)
            with self.assertRaises(ValueError):
                files.open('base.bin', 7, pin['sha256'])
        with checker.Files(self.folder) as files:
            with files.open('base.bin', pin['bytes'], '0' * 64) as source:
                with self.assertRaises(ValueError):
                    source.verify()
        (self.folder / 'base.bin').write_bytes(b'original!')
        with checker.Files(self.folder) as files, self.assertRaises(ValueError):
            files.open('base.bin', **pin)

    def test_symlink_file_ancestor_and_hardlink_aliases_reject(self):
        pin = saved(self.folder, 'base.bin', b'original')
        (self.folder / 'link.bin').symlink_to('base.bin')
        with checker.Files(self.folder) as files, self.assertRaises(OSError):
            files.open('link.bin', **pin)
        (self.folder / 'real').mkdir()
        (self.folder / 'alias').symlink_to('real', target_is_directory=True)
        saved(self.folder, 'real/value.bin', b'original')
        with checker.Files(self.folder) as files, self.assertRaises(OSError):
            files.open('alias/value.bin', **pin)
        os.link(self.folder / 'base.bin', self.folder / 'hard.bin')
        with checker.Files(self.folder) as files, self.assertRaises(ValueError):
            files.open('base.bin', **pin)

    def test_descriptor_mutation_replacement_and_duplicate_claims_reject(self):
        pin = saved(self.folder, 'base.bin', b'original')
        with checker.Files(self.folder) as files:
            with files.open('base.bin', **pin) as source:
                with self.assertRaises(ValueError):
                    files.open('base.bin', **pin)
                (self.folder / 'base.bin').write_bytes(b'changed!')
                with self.assertRaises(ValueError):
                    source.endpoint()
        saved(self.folder, 'base.bin', b'original')
        with checker.Files(self.folder) as files:
            with files.open('base.bin', **pin) as source:
                (self.folder / 'base.bin').unlink()
                saved(self.folder, 'base.bin', b'original')
                with self.assertRaises(ValueError):
                    source.endpoint()

    def test_patches_require_typed_sorted_nonoverlapping_inbounds_intervals(self):
        patch = dict(offset=1, bytes=2, file='part.patch', sha256=PIN)
        checker.validate_patches([patch, dict(patch, offset=3)], 5)
        for patches in ([dict(patch, offset=True)], [dict(patch, bytes=0)], [dict(patch, bytes=5)],
                        [dict(patch, file='../part')], [dict(patch, sha256='z' * 64)],
                        [patch, dict(patch, offset=2)], [dict(patch, offset=3), patch],
                        [dict(patch, extra=1)]):
            with self.subTest(patches=patches), self.assertRaises(ValueError):
                checker.validate_patches(patches, 5)

    def replay(self, actual=b'aXYdefZhi', patches=None, mismatches=3, sha256=None):
        base = saved(self.folder, 'base.bin', b'abcdefghi')
        if patches is None:
            first = saved(self.folder, 'repeat/first.patch', b'XY')
            second = saved(self.folder, 'repeat/second.patch', b'Z')
            patches = [dict(offset=1, file='first.patch', **first), dict(offset=6, file='second.patch', **second)]
        spec = dict(sha256=sha256 or digest(actual), mismatching_bytes=mismatches, patches=patches)
        with checker.Files(self.folder) as files:
            with files.open('base.bin', **base) as baseline:
                baseline.verify()
                return b''.join(raw for _, raw in checker.Replay(files, baseline, 'repeat', spec).blocks())

    def test_sparse_replay_is_complete_and_counted_from_actual_bytes(self):
        self.assertEqual(self.replay(), b'aXYdefZhi')
        with self.assertRaises(ValueError):
            self.replay(mismatches=2)
        with self.assertRaises(ValueError):
            self.replay(sha256='0' * 64)

    def test_hash_only_repeat_still_replays_complete_authenticated_baseline(self):
        self.assertEqual(self.replay(actual=b'abcdefghi', patches=[], mismatches=0), b'abcdefghi')
        with self.assertRaises(ValueError):
            self.replay(patches=[], mismatches=0)
        with self.assertRaises(ValueError):
            self.replay(actual=b'abcdefghi', patches=[], mismatches=1)

    def test_patch_payload_hash_extent_redundancy_and_alias_are_not_self_reported(self):
        self.replay()
        for pin in (dict(bytes=1, sha256=digest(b'X')), dict(bytes=2, sha256='0' * 64)):
            with self.assertRaises(ValueError):
                self.replay(patches=[dict(offset=1, file='first.patch', **pin)])
        saved(self.folder, 'repeat/same.patch', b'bc')
        with self.assertRaises(ValueError):
            self.replay(actual=b'abcdefghi', mismatches=0,
                        patches=[dict(offset=1, bytes=2, file='same.patch', sha256=digest(b'bc'))])
        with self.assertRaises(ValueError):
            self.replay(patches=[dict(offset=1, bytes=2, file='first.patch', sha256=digest(b'XY')),
                                 dict(offset=4, bytes=2, file='first.patch', sha256=digest(b'XY'))])

    def test_streaming_buffers_are_bounded_for_large_patch(self):
        raw = bytes(range(256)) * (checker.BLOCK // 256 + 2)
        base = saved(self.folder, 'base.bin', bytes(len(raw)))
        patch = saved(self.folder, 'repeat/all.patch', raw)
        spec = dict(sha256=digest(raw), mismatching_bytes=sum(v != 0 for v in raw),
                    patches=[dict(offset=0, file='all.patch', **patch)])
        with checker.Files(self.folder) as files, files.open('base.bin', **base) as baseline:
            chunks = list(checker.Replay(files, baseline, 'repeat', spec).blocks())
        self.assertEqual(b''.join(block for _, block in chunks), raw)
        self.assertLessEqual(max(len(block) for _, block in chunks), checker.BLOCK)

    def test_exact_causal_prefix_uses_actual_bytes_and_complete_scalar_extent(self):
        left_pin = saved(self.folder, 'left', b'abcdefgh')
        right_pin = saved(self.folder, 'right', b'abcdXXgh')
        with checker.Files(self.folder) as files, files.open('left', **left_pin) as left, \
                files.open('right', **right_pin) as right:
            left.verify()
            right.verify()
            self.assertEqual(checker.prefix_difference(left, right, 4, 2, 2)['mismatching_bytes'], 0)
            row = checker.prefix_difference(left, right, 8, 2, 2)
            self.assertEqual(row['mismatching_bytes'], 2)
            self.assertEqual(row['examples'][0]['coordinate'], [1, 0])
            with self.assertRaises(ValueError):
                checker.prefix_difference(left, right, 7, 2, 2)


class OriginalGate(unittest.TestCase):
    def test_gate_floor_factor_and_rtol_are_immutable_per_array(self):
        self.assertEqual(checker.immutable_gate(0), dict(atol=1e-4, rtol=1e-3))
        self.assertEqual(checker.immutable_gate(.25), dict(atol=.5, rtol=1e-3))
        for value in (True, -1, math.inf, math.nan):
            with self.assertRaises(ValueError):
                checker.immutable_gate(value)

    def test_boundary_is_inclusive_without_a_native_derived_enlargement(self):
        gate = checker.immutable_gate(0)
        self.assertTrue(checker.scalar_member(1e-4, 0., gate))
        self.assertFalse(checker.scalar_member(math.nextafter(1e-4, math.inf), 0., gate))
        self.assertTrue(checker.scalar_member(-1e-4, 0., gate))
        self.assertFalse(checker.scalar_member(math.nextafter(-1e-4, -math.inf), 0., gate))
        for value in (math.inf, -math.inf, math.nan):
            self.assertFalse(checker.scalar_member(value, 0., gate))

    def test_both_profiles_are_required_even_with_their_own_relative_limits(self):
        gate = checker.immutable_gate(0)
        actual, main, wide = 1., 1., 1.01
        self.assertTrue(checker.scalar_member(actual, main, gate))
        self.assertFalse(checker.scalar_member(actual, wide, gate))
        self.assertEqual(gate, dict(atol=1e-4, rtol=1e-3))


class CommandLine(unittest.TestCase):
    def test_failed_inputs_retain_fresh_failure_report_and_return_one(self):
        with tempfile.TemporaryDirectory(dir='/private/tmp') as tmp:
            folder = Path(tmp)
            args = ['checker']
            for name, value in dict(environment_root=folder, runtime_manifest=folder / 'runtime.json',
                                    runtime_sha256=PIN, protocol=folder / 'missing.json', protocol_sha256=PIN,
                                    reference=folder / 'reference', reference_sha256=PIN,
                                    native=folder / 'native', native_sha256=PIN,
                                    fresh_output=folder / 'report').items():
                args.extend(['--' + name.replace('_', '-'), str(value)])
            with mock.patch.object(sys, 'argv', args), mock.patch('builtins.print'):
                self.assertEqual(checker.main(), 1)
            report = json.loads((folder / 'report/manifest.json').read_bytes())
            self.assertIs(report['passed'], False)
            self.assertIs(report['completed'], False)
            self.assertEqual(report['failure']['type'], 'FileNotFoundError')
            with mock.patch.object(sys, 'argv', args), self.assertRaises(ValueError):
                checker.main()


def numeric_controls(np):
    """Root-queued, admitted NumPy controls; ordinary unittest never calls this.

    Exercises both storage dtypes, native nonfinites, numerical failures against
    both profiles, and lossless decoding across deliberately unaligned patches.
    """
    records = []
    with tempfile.TemporaryDirectory(dir='/private/tmp') as tmp:
        folder = Path(tmp)
        for dtype, encoding in (('<f2', 'e'), ('<f4', 'f')):
            width = checker.DTYPES[dtype]
            for changed in (False, True):
                main_values = [0., 1., -2., 3.]
                wide_values = [0., 1., -2., 3.]
                actual_values = [0., 1., -2., 3.] if not changed else [1., math.inf, -2., -8.]
                pack = lambda values: b''.join(struct.pack('<' + encoding, v) for v in values)
                a, b, raw = pack(main_values), pack(wide_values), pack(actual_values)
                sub = folder / (dtype[-1] + str(changed))
                sub.mkdir()
                pins = {name: saved(sub, name, value) for name, value in [('a', a), ('b', b), ('base', raw)]}
                spec = dict(dtype=dtype, shape=[2, 2], bytes=len(raw), sha256=digest(raw))
                with checker.Files(sub) as files, files.open('a', **pins['a']) as main, \
                        files.open('b', **pins['b']) as wide, files.open('base', **pins['base']) as baseline:
                    gate, spread = checker.gate_from_files(main, wide, dtype, np)
                    result = checker.compare_blocks(checker.original_blocks(baseline), main, wide, spec, gate, np,
                                                    logits=dtype == '<f4')
                    checker.need(gate == dict(atol=1e-4, rtol=1e-3) and spread == 0,
                                 'Equal-profile control must retain the exact floor gate: ' + dtype)
                    checker.need(result['nonfinite_native_values'] == int(changed),
                                 'Native nonfinite control count differs: ' + dtype)
                    checker.need(all(row['values'] == 4 and row['failed_values'] == (3 if changed else 0)
                                     for row in result['references'].values()),
                                 'Both-profile control coordinate/failure counts differ: ' + dtype)
                    if dtype == '<f4':
                        checker.need(result['final_logits_sha256'] == digest(raw[len(raw) // 2:]),
                                     'F32 control final-logit SHA256 differs from actual final-row bytes')
                        checker.need(result['final_greedy_token'] == (None if changed else 1),
                                     'F32 control lowest-ID greedy token/nonfinite rejection differs')
                    records.append(dict(dtype=dtype, changed=changed, **result))
            sub = folder / (dtype[-1] + '-independent-envelope')
            sub.mkdir()
            a, b, raw = pack([0., 0., 0., 0.]), pack([.25, .25, .25, .25]), pack([.5, .75, 0., .25])
            pins = {name: saved(sub, name, value) for name, value in [('a', a), ('b', b), ('base', raw)]}
            spec = dict(dtype=dtype, shape=[2, 2], bytes=len(raw), sha256=digest(raw))
            with checker.Files(sub) as files, files.open('a', **pins['a']) as main, \
                    files.open('b', **pins['b']) as wide, files.open('base', **pins['base']) as baseline:
                gate, spread = checker.gate_from_files(main, wide, dtype, np)
                result = checker.compare_blocks(checker.original_blocks(baseline), main, wide, spec, gate, np)
                checker.need(gate == dict(atol=.5, rtol=1e-3) and spread == .25,
                             'Independent-envelope control gate differs: ' + dtype)
                checker.need(result['references']['main']['failed_values'] == 1,
                             'Independent-envelope main-profile failure count differs: ' + dtype)
                checker.need(result['references']['wide']['failed_values'] == 0,
                             'Independent-envelope wide-profile failure count differs: ' + dtype)
                records.append(dict(dtype=dtype, separate_profile_predicates=True, gate=gate, **result))
            # A byte patch splits one native word and a later patch crosses a
            # word boundary. Decode must exactly match original reconstructed values.
            base = pack([0., 0., 0., 0.])
            raw = pack([1., -2., 3., 4.])
            sub = folder / (dtype[-1] + '-unaligned')
            sub.mkdir()
            pins = {name: saved(sub, name, value) for name, value in [('a', raw), ('b', raw), ('base', base)]}
            patches = []
            for first, end in ((1, 3), (3, len(raw))):
                pin = saved(sub, f'repeat/at{first}', raw[first:end])
                patches.append(dict(offset=first, file=f'at{first}', **pin))
            # All omitted bytes equal baseline (the first little-endian byte is zero).
            checker.need(raw[0] == base[0], 'Unaligned-patch control omitted a changed first byte: ' + dtype)
            spec = dict(dtype=dtype, shape=[2, 2], bytes=len(raw), sha256=digest(raw), patches=patches,
                        mismatching_bytes=sum(a != b for a, b in zip(raw, base, strict=True)))
            with checker.Files(sub) as files, files.open('a', **pins['a']) as main, \
                    files.open('b', **pins['b']) as wide, files.open('base', **pins['base']) as baseline:
                gate, _ = checker.gate_from_files(main, wide, dtype, np)
                result = checker.compare_blocks(checker.Replay(files, baseline, 'repeat', spec).blocks(),
                                                main, wide, spec, gate, np)
                checker.need(all(row['failed_values'] == 0 for row in result['references'].values()),
                             'Lossless unaligned-patch control fails a profile comparison: ' + dtype)
                records.append(dict(dtype=dtype, unaligned_patch=True, **result))
    return dict(passed=True, arrays=len(records), records=records, numerical_runtime_imported_by_caller=True)


if __name__ == '__main__':
    unittest.main()

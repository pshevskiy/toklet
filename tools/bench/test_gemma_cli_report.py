"""Independent data-only controls: no producer import,model,codec or GPU."""
import contextlib
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest

import gemma_cli_report as consumer
import request_cost_report as legacy

FIXTURE = Path(__file__).resolve().parents[2] / 'tests/fixtures/gemma-cli-v1'


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


class GemmaCliReportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='toklet-gemma-cli-report-')
        self.root = Path(self.temp.name)
        self.rows = [json.loads(line) for line in (FIXTURE / 'complete.jsonl').read_bytes().splitlines()]

    def tearDown(self):
        self.temp.cleanup()

    def raw(self, raw):
        path = self.root / 'input.jsonl'
        path.write_bytes(raw)
        return path

    def write(self, rows):
        return self.raw(b''.join((json.dumps(row, separators=(',', ':')) + '\n').encode() for row in rows))

    def report(self, rows):
        path = self.write(rows)
        return consumer.report(path, expected_sha256=sha(path.read_bytes()))

    def reject(self, rows):
        with self.assertRaises((ValueError, TypeError)):
            self.report(rows)

    def one(self, row):
        footer = copy.deepcopy(self.rows[-1])
        footer['retained'] = 1
        return [copy.deepcopy(self.rows[0]), row, footer]

    def test_hand_authored_complete_fixture_and_fixed_arithmetic(self):
        expected = json.loads((FIXTURE / 'expected.json').read_bytes())
        raw = (FIXTURE / 'complete.jsonl').read_bytes()
        self.assertEqual(sha(raw), expected['complete_sha256'])
        result = consumer.report(FIXTURE / 'complete.jsonl', expected_sha256=expected['complete_sha256'])
        for key in ('retained_records', 'completed_core_reports', 'unavailable_core_reports', 'retained_outcomes'):
            self.assertEqual(result[key], expected[key])
        for key in ('request_total_ms', 'prefill_ms', 'prefill_tokens_per_s', 'decode_tokens_per_s',
                    'decode_evaluations_per_s', 'ttft_event_ms', 'token_interval_mean_ms'):
            self.assertEqual(result['metrics'][key]['p50'], expected[key + '_p50'])
        self.assertTrue(result['validated_complete_export'])
        self.assertTrue(result['telemetry_complete'])
        self.assertEqual(result['attempted_requests'], 4)
        self.assertEqual(result['attempted_outcome_rate_unit'], 'fraction_of_attempted_requests')
        self.assertEqual(result['attempted_outcome_rates'], dict(cancelled=0, execution_error=0,
                         output_error=0, rejected=.25, stop=.25, success=.5))
        self.assertIsNone(result['attempted_outcome_rates_unavailable_reason'])
        for key in ('source_model_runtime_independently_verified', 'numerical_quality_qualified',
                    'observer_overhead_qualified', 'adoption_qualified'):
            self.assertIs(result[key], False)

    def test_empty_complete_export_does_not_invent_rates_or_outcomes(self):
        path = FIXTURE / 'empty.jsonl'
        result = consumer.report(path, expected_sha256=sha(path.read_bytes()))
        self.assertEqual(result['retained_records'], 0)
        self.assertEqual(result['attempted_requests'], 0)
        self.assertTrue(all(v is None for v in result['attempted_outcome_rates'].values()))
        self.assertEqual(result['attempted_outcome_rates_unavailable_reason'], 'no_attempted_requests')
        self.assertEqual(sum(result['retained_outcomes'].values()), 0)
        self.assertTrue(all(row['p50'] is None and row['retained_observations'] == 0
                            for row in result['metrics'].values()))

    def test_header_route_identity_and_provenance_clock_gaps_are_explicit(self):
        result = self.report(self.rows)
        header = result['manifest']
        self.assertEqual(header['model_profile'], 'gemma4-bf16')
        self.assertEqual(header['model_revision'], '70af34e20bd4b7a91f0de6b22675850c43922a03')
        self.assertEqual(header['engine_crate_version'], '0.1.0')
        self.assertEqual(header['weight_dtype'], 'bf16')
        self.assertEqual(header['kv_dtype'], 'fp32')
        self.assertFalse(header['thinking_enabled'])
        self.assertEqual(header['external_provenance_reason'], 'external_capture_manifest_required')
        for field in ('engine_revision', 'build_flags', 'device_identity', 'device_power',
                      'model_content_sha256', 'tokenizer_content_sha256', 'conversion_ms',
                      'compile_ms', 'process_start_to_ready_ms'):
            self.assertIsNone(header[field])
        self.assertIs(result['source_model_runtime_independently_verified'], False)
        self.assertIn('process startup', header['route_start_to_ready_scope'])
        self.assertIn('not_observed', header['process_start_to_ready_availability_reason'])

    def test_collector_resource_facts_are_logical_storage_and_not_a_second_admission(self):
        result = self.report(self.rows)
        header = result['manifest']
        self.assertEqual(header['collector_fixed_bytes'], 8192)
        self.assertEqual(header['retained_ring_capacity_bytes'], 1032192)
        self.assertEqual(header['collector_peak_bytes'], 1040384)
        self.assertEqual(header['telemetry_admitted_bytes'], 1064960)
        self.assertEqual(header['additional_owner_and_external_admitted_bytes'], 1246773248)
        self.assertEqual(header['session_admitted_bytes'], 1299202048)
        self.assertIn('after construction, fixed until drop', header['collector_memory_scope'])
        self.assertIn('not RSS or process peak', header['collector_memory_scope'])
        self.assertIsNone(self.rows[1]['ordinary_observations']['process_footprint_bytes'])
        # Actual logical storage already fits inside the telemetry allowance.
        # Adding it again to whole-owner admission must fail the owner partition.
        bad = copy.deepcopy(self.rows)
        peak = bad[0]['collector_peak_bytes']
        bad[0]['additional_owner_and_external_admitted_bytes'] += peak
        bad[0]['session_admitted_bytes'] += peak
        for row in bad[1:-1]:
            row['memory']['session_admitted_bytes'] += peak
        self.reject(bad)

    def test_collector_resource_boundaries_and_zero_ring_failed_reservation(self):
        header, footer = copy.deepcopy(self.rows[0]), copy.deepcopy(self.rows[-1])
        footer['retained'] = 0
        header.update(collector_fixed_bytes=16384, retained_ring_capacity_bytes=1032192,
                      collector_peak_bytes=1048576)
        result = self.report([header, footer])
        self.assertTrue(result['telemetry_complete'])
        self.assertEqual(result['retained_records'], 0)
        self.assertEqual(result['manifest']['collector_peak_bytes'], 1048576)
        # Independent synthetic lower boundary; no platform Rust size is imported.
        header.update(collector_fixed_bytes=1, retained_ring_capacity_bytes=0,
                      collector_peak_bytes=1)
        footer['telemetry_dropped'] = 1
        result = self.report([header, footer])
        self.assertFalse(result['telemetry_complete'])
        self.assertIsNone(result['attempted_requests'])
        self.assertEqual(result['manifest']['retained_ring_capacity_bytes'], 0)
        self.assertEqual(result['export_status']['telemetry_dropped'], 1)

    def test_collector_resource_integer_types_bounds_and_equations_reject(self):
        changes = [('collector_fixed_bytes', value) for value in (True, -1, 0, 16385, 2**64)]
        changes += [('retained_ring_capacity_bytes', value)
                    for value in (False, -1, 1, 1032191, 1032193, 2**64)]
        changes += [('collector_peak_bytes', value)
                    for value in (True, -1, 1040383, 1040385, 1048577, 2**64)]
        changes += [('telemetry_admitted_bytes', value)
                    for value in (True, -1, 0, 1048576, 1064961, 2**64)]
        for field, value in changes:
            with self.subTest(field=field, value=value):
                bad = copy.deepcopy(self.rows)
                bad[0][field] = value
                self.reject(bad)

    def test_collector_resource_fields_scope_and_cancel_reason_are_required(self):
        fields = ('collector_fixed_bytes', 'retained_ring_capacity_bytes', 'collector_peak_bytes',
                  'telemetry_admitted_bytes', 'collector_memory_scope',
                  'cancel_to_terminal_availability_reason')
        for field in fields:
            with self.subTest(missing=field):
                bad = copy.deepcopy(self.rows)
                del bad[0][field]
                self.reject(bad)
        for field in ('collector_memory_scope', 'cancel_to_terminal_availability_reason'):
            for value in ('measured process peak or signal latency', None, True):
                with self.subTest(field=field, value=value):
                    bad = copy.deepcopy(self.rows)
                    bad[0][field] = value
                    self.reject(bad)

    def test_record_telemetry_allowance_must_match_header_admission(self):
        bad = copy.deepcopy(self.rows)
        # Keep the existing additional-owner sum coherent so the mismatched
        # telemetry allowance itself cannot hide behind another owner.
        bad[1]['memory']['telemetry_allowance_bytes'] += 1
        bad[1]['memory']['owner_reserved_bytes'] -= 1
        self.reject(bad)

    def test_cancelled_request_keeps_null_signal_latency_with_explicit_reason(self):
        rows = [json.loads(line) for line in (FIXTURE / 'logical-history.jsonl').read_bytes().splitlines()]
        selected = [copy.deepcopy(rows[0]), copy.deepcopy(rows[1]), copy.deepcopy(rows[-1])]
        selected[-1]['retained'] = 1
        result = self.report(selected)
        self.assertEqual(result['retained_outcomes']['cancelled'], 1)
        self.assertIsNone(selected[1]['timing']['cancel_to_terminal_ms'])
        self.assertEqual(result['manifest']['cancel_to_terminal_availability_reason'],
                         'signal_timestamp_not_observed; cancellation_outcome_and_state_do_not_measure_signal_to_ack_latency')
        for value in (0, 1.0, False):
            bad = copy.deepcopy(selected)
            bad[1]['timing']['cancel_to_terminal_ms'] = value
            with self.subTest(value=value):
                self.reject(bad)

    def test_header_cannot_relabel_route_or_invent_external_identity_and_clock_observations(self):
        changes = [('model_profile', 'qwen'), ('backend', 'cpu'), ('model_revision', '0' * 40),
                   ('weight_dtype', 'f16'), ('kv_dtype', 'f16'), ('sampling', 'temperature'),
                   ('thinking_enabled', True), ('thinking_enabled', 0),
                   ('instrumentation', 'disabled'), ('engine_revision', '0' * 40),
                   ('model_content_sha256', '0' * 64), ('device_power', 'AC'),
                   ('compile_ms', 0), ('process_start_to_ready_ms', 20),
                   ('external_provenance_reason', 'authenticated_by_shape'),
                   ('engine_crate_version', ''), ('engine_crate_version', 'latest'),
                   ('engine_crate_version', True), ('engine_crate_version', '1' * 65)]
        for field, value in changes:
            bad = copy.deepcopy(self.rows)
            bad[0][field] = value
            with self.subTest(field=field, value=value):
                self.reject(bad)

    def test_explicit_input_pin_is_required_and_authenticates_full_bytes(self):
        path = FIXTURE / 'complete.jsonl'
        for pin in (None, '', 'A' * 64, 'a' * 63, '0' * 64, True):
            with self.subTest(pin=pin), self.assertRaises(ValueError):
                consumer.report(path, expected_sha256=pin)
        raw = path.read_bytes()
        changed = self.raw(raw.replace(b'100.0', b'101.0', 1))
        with self.assertRaises(ValueError):
            consumer.report(changed, expected_sha256=sha(raw))

    def test_footer_is_mandatory_terminal_and_cannot_be_partial(self):
        for rows in (self.rows[:-1], self.rows + [self.rows[1]], self.rows + [self.rows[-1]]):
            with self.subTest(rows=len(rows)):
                self.reject(rows)
        bad = copy.deepcopy(self.rows)
        bad[-1]['complete'] = False
        self.reject(bad)
        bad[-1]['complete'] = 1
        self.reject(bad)

    def test_footer_counters_are_typed_and_retained_inventory_is_exact(self):
        for key, value in [('retained', 3), ('retained', True), ('invalid_records', -1),
                           ('telemetry_dropped', 2**64), ('counter_overflow', 1)]:
            bad = copy.deepcopy(self.rows)
            bad[-1][key] = value
            with self.subTest(key=key):
                self.reject(bad)

    def test_drops_invalid_export_errors_and_saturation_remain_explicit(self):
        for field in ('telemetry_dropped', 'invalid_records', 'export_errors', 'counter_overflow'):
            rows = copy.deepcopy(self.rows)
            rows[-1][field] = True if field == 'counter_overflow' else 1
            result = self.report(rows)
            self.assertTrue(result['validated_complete_export'])
            self.assertFalse(result['telemetry_complete'])
            self.assertEqual(result['retained_records'], 4)
            self.assertEqual(result['export_status'][field], rows[-1][field])
            self.assertIsNone(result['attempted_requests'])
            self.assertTrue(all(v is None for v in result['attempted_outcome_rates'].values()))
            self.assertEqual(result['attempted_outcome_rates_unavailable_reason'], 'telemetry_incomplete')
        rows[-1].update(telemetry_dropped=2**64 - 1, invalid_records=2**64 - 1,
                        export_errors=2**64 - 1, counter_overflow=True)
        self.assertFalse(self.report(rows)['telemetry_complete'])

    def test_loss_free_identity_gaps_cannot_manufacture_an_attempt_denominator(self):
        for field in ('request_id', 'attempt_id'):
            rows = copy.deepcopy(self.rows)
            for row in rows[1:-1]:
                row[field] += 1
            self.reject(rows)
            rows[-1]['telemetry_dropped'] = 1
            result = self.report(rows)
            self.assertIsNone(result['attempted_requests'])
            self.assertEqual(result['retained_outcomes'],
                             dict(cancelled=0, execution_error=0, output_error=0,
                                  rejected=1, stop=1, success=2))

    def test_unassignable_tail_identity_makes_rates_unknown_even_with_contiguous_records(self):
        rows = copy.deepcopy(self.rows)
        rows[-1]['counter_overflow'] = True
        result = self.report(rows)
        self.assertEqual(result['retained_records'], 4)
        self.assertIsNone(result['attempted_requests'])
        self.assertTrue(all(value is None for value in result['attempted_outcome_rates'].values()))

    def test_append_continue_and_partial_cancel_use_logical_admitted_history(self):
        rows = [json.loads(line) for line in (FIXTURE / 'logical-history.jsonl').read_bytes().splitlines()]
        result = self.report(rows)
        self.assertEqual(result['completed_core_reports'], 3)
        self.assertEqual(result['attempted_requests'], 3)
        self.assertAlmostEqual(result['attempted_outcome_rates']['cancelled'], 1 / 3)
        self.assertAlmostEqual(result['attempted_outcome_rates']['success'], 2 / 3)
        self.assertAlmostEqual(result['metrics']['prefill_tokens_per_s']['p50'], 600 / 7)
        #Continue includes its five historical tokens; Append adds two to the
        #existing seven-token history. Neither prompt count is just the suffix.
        for index, suffix_count in ((2, 0), (3, 2)):
            bad = copy.deepcopy(rows)
            bad[index]['work']['prompt_tokens'] = suffix_count
            self.reject(bad)
        # Even coherent work/after-state cannot fabricate new Continue input.
        bad = copy.deepcopy(rows)
        bad[2]['work'].update(prompt_tokens=6, reused_prefix_tokens=3)
        bad[2]['after'] = dict(history_tokens=8, committed_tokens=7, pending_tokens=1)
        bad[2]['memory']['kv_valid_positions'] = 7
        self.reject(bad)

    def test_completed_history_pending_decode_and_replayed_rows_are_not_inferred(self):
        rows = [json.loads(line) for line in (FIXTURE / 'logical-history.jsonl').read_bytes().splitlines()]
        for field, value in (('prompt_tokens', None), ('reused_prefix_tokens', 0),
                              ('replayed_rows', 4)):
            bad = copy.deepcopy(rows)
            bad[3]['work'][field] = value
            self.reject(bad)
        bad = copy.deepcopy(rows)
        bad[3]['after'] = dict(history_tokens=11, committed_tokens=10, pending_tokens=1)
        bad[3]['memory']['kv_valid_positions'] = 10
        self.reject(bad)
        bad[3]['after'] = dict(history_tokens=10, committed_tokens=10, pending_tokens=0)
        self.reject(bad)
        bad = copy.deepcopy(rows)
        bad[3]['work']['decode_evaluations'] = 1
        bad[3]['timing']['decode_ms'] = 5
        bad[3]['decode_evaluations_per_s'] = 200
        self.reject(bad)

    def test_partial_cancel_and_output_error_do_not_require_finished_prompt_coverage(self):
        rows = [json.loads(line) for line in (FIXTURE / 'logical-history.jsonl').read_bytes().splitlines()]
        row = rows[1]
        self.assertEqual(row['work']['prompt_tokens'], 5)
        self.assertEqual(row['work']['prefill_rows'], 2)
        self.assertEqual(row['after']['pending_tokens'], 3)
        partial = [copy.deepcopy(rows[0]), copy.deepcopy(row), copy.deepcopy(rows[-1])]
        partial[-1]['retained'] = 1
        self.assertTrue(self.report(partial)['telemetry_complete'])
        error = copy.deepcopy(row)
        error.update(terminal_status='output_error', failed_phase='output',
                     error_category='output', replay_required=True)
        partial[1] = error
        self.assertEqual(self.report(partial)['retained_outcomes']['output_error'], 1)
        bad = copy.deepcopy(row)
        bad['work']['prompt_tokens'] = 1
        partial[1] = bad
        self.reject(partial)

    def test_json_duplicate_nonfinite_invalid_utf8_and_bool_schema_reject(self):
        raw = (FIXTURE / 'complete.jsonl').read_bytes()
        cases = [raw.replace(b'"capacity":512', b'"capacity":512,"capacity":512', 1),
                 raw.replace(b'"prefill_ms":40.0', b'"prefill_ms":NaN', 1),
                 raw.replace(b'"prefill_ms":40.0', b'"prefill_ms":1e999', 1),
                 raw.replace(b'"schema_version":1', b'"schema_version":true', 1),
                 b'\xff\n']
        for value in cases:
            path = self.raw(value)
            with self.subTest(raw=value[:60]), self.assertRaises(ValueError):
                consumer.report(path, expected_sha256=sha(value))

    def test_line_file_newline_and_blank_record_bounds_are_enforced(self):
        raw = (FIXTURE / 'complete.jsonl').read_bytes()
        for value in (raw[:-1], raw + b'\n', b' ' * 4096 + b'\n', b' ' * (1024 * 1024 + 8193)):
            path = self.raw(value)
            with self.subTest(size=len(value)), self.assertRaises(ValueError):
                consumer.report(path, expected_sha256=sha(value))

    @unittest.skipUnless(hasattr(os, 'O_NOFOLLOW') and hasattr(os, 'mkfifo'), 'Unix file admission')
    def test_symlink_fifo_and_directory_are_rejected_without_blocking_open(self):
        target = FIXTURE / 'complete.jsonl'
        link = self.root / 'link'
        link.symlink_to(target)
        fifo = self.root / 'fifo'
        os.mkfifo(fifo)
        for path in (link, fifo, self.root):
            with self.subTest(path=path), self.assertRaises((ValueError, OSError)):
                consumer.report(path, expected_sha256=sha(target.read_bytes()))

    def test_state_conservation_capacity_and_rejection_preservation(self):
        for change in (('history_tokens', 3), ('committed_tokens', 513), ('pending_tokens', True)):
            bad = copy.deepcopy(self.rows)
            bad[1]['after'][change[0]] = change[1]
            self.reject(bad)
        bad = copy.deepcopy(self.rows)
        bad[2]['after'] = dict(history_tokens=3, committed_tokens=3, pending_tokens=0)
        self.reject(bad)

    def test_partial_core_report_and_callback_work_fabrication_reject(self):
        for field, value in [('prefill_rows', None), ('emitted_tokens', 1), ('sampled_tokens', 0),
                             ('decode_evaluations', 3), ('callback_tokens', True)]:
            bad = copy.deepcopy(self.rows)
            bad[1]['work'][field] = value
            self.reject(bad)
        bad = copy.deepcopy(self.rows)
        bad[2]['work']['prefill_rows'] = 0
        self.reject(bad)

    def test_clock_nesting_interval_coverage_and_phase_totals(self):
        for field, value in [('prefill_ms', -1), ('decode_ms', 70), ('native_total_ms', 101),
                             ('last_token_event_ms', 101), ('ttft_event_ms', 71),
                             ('token_interval_sum_ms', 19), ('token_interval_count', 2),
                             ('first_text_publication_ms', 101)]:
            bad = copy.deepcopy(self.rows)
            bad[1]['timing'][field] = value
            self.reject(bad)

    def test_two_decode_denominators_and_units_cannot_be_relabelled(self):
        #2callbacks over20ms =>50tokens/s;1numerical step over10ms =>100steps/s.
        for field, value in [('decode_tokens_per_s', 100), ('decode_evaluations_per_s', 50),
                             ('prefill_tokens_per_s', .05), ('token_interval_mean_ms', .02)]:
            bad = copy.deepcopy(self.rows)
            bad[1][field] = value
            self.reject(bad)

    def test_zero_single_and_zero_duration_rates_are_unavailable(self):
        row = copy.deepcopy(self.rows[1])
        row['work'].update(callback_tokens=1, emitted_tokens=1, sampled_tokens=1, decode_evaluations=0)
        row['after'] = dict(history_tokens=3, committed_tokens=2, pending_tokens=1)
        row['memory']['kv_valid_positions'] = 2
        row['timing'].update(decode_ms=0, last_token_event_ms=50, token_interval_count=0,
                             token_interval_min_ms=None, token_interval_max_ms=None,
                             token_interval_sum_ms=0)
        row.update(decode_tokens_per_s=None, decode_evaluations_per_s=None, token_interval_mean_ms=None)
        result = self.report(self.one(row))
        self.assertIsNone(result['metrics']['decode_tokens_per_s']['p50'])
        row = copy.deepcopy(self.rows[1])
        row['timing'].update(prefill_ms=0, decode_ms=0, last_token_event_ms=50,
                             token_interval_min_ms=0, token_interval_max_ms=0, token_interval_sum_ms=0)
        row.update(prefill_tokens_per_s=None, decode_tokens_per_s=None,
                   decode_evaluations_per_s=None, token_interval_mean_ms=0)
        self.assertEqual(self.report(self.one(row))['metrics']['token_interval_mean_ms']['p50'], 0)
        row['decode_tokens_per_s'] = 0
        self.reject(self.one(row))

    def test_unknown_observations_are_null_with_exact_reasons_not_zero(self):
        for field in ('gpu_ms', 'command_count', 'process_footprint_bytes', 'ui_first_visible_ms'):
            bad = copy.deepcopy(self.rows)
            bad[1]['ordinary_observations'][field] = 0
            self.reject(bad)
        bad = copy.deepcopy(self.rows)
        bad[1]['availability']['ui_first_visible_ms'] = 'stdout is visible'
        self.reject(bad)

    def test_logical_owner_inventory_and_exact_sum_are_required(self):
        for field in ('history_bytes', 'logits_bytes', 'metadata_bytes', 'metadata_allowance_bytes'):
            bad = copy.deepcopy(self.rows)
            del bad[0]['logical_memory'][field]
            self.reject(bad)
        for field in ('positional_bytes', 'admitted_bytes'):
            bad = copy.deepcopy(self.rows)
            bad[0]['logical_memory'][field] += 1
            self.reject(bad)
        bad = copy.deepcopy(self.rows)
        bad[1]['memory']['weights_owned_bytes'] += 1
        self.reject(bad)

    def test_additional_admission_counts_each_owner_once_and_preserves6gib(self):
        rows = self.one(copy.deepcopy(self.rows[1]))
        header, row = rows[0], rows[1]
        limit = 6442450944
        additional = 1246773248
        logical = {key: 0 for key in header['logical_memory']}
        logical.update(weights_bytes=limit - additional, admitted_bytes=limit - additional)
        header.update(logical_memory=logical, numerical_session_admitted_bytes=limit - additional,
                      load_admitted_bytes=limit, session_admitted_bytes=limit)
        row['memory'].update(weights_owned_bytes=limit - additional, kv_allocated_bytes=0,
                             scratch_bytes=0, load_admitted_bytes=limit, session_admitted_bytes=limit)
        self.assertTrue(self.report(rows)['validated_complete_export'])
        header['session_admitted_bytes'] += 1
        row['memory']['session_admitted_bytes'] += 1
        self.reject(rows)
        bad = copy.deepcopy(self.rows)
        bad[1]['memory']['helper_reserve_bytes'] = 0
        self.reject(bad)

    def test_duplicate_attempts_session_mismatch_and_owner_generation_reject(self):
        for field, value in [('request_id', 1), ('attempt_id', 1), ('owner_generation', 2),
                             ('session_id', 'another-session')]:
            bad = copy.deepcopy(self.rows)
            bad[2][field] = value
            self.reject(bad)

    def test_output_error_can_preserve_completed_core_after_terminal_sink_failure(self):
        row = copy.deepcopy(self.rows[1])
        row.update(terminal_status='output_error', failed_phase='output', error_category='output',
                   replay_required=True)
        result = self.report(self.one(row))
        self.assertEqual(result['completed_core_reports'], 1)
        self.assertEqual(result['retained_outcomes']['output_error'], 1)
        self.assertEqual(result['export_status']['retained'], 1)
        row['error_category'] = None
        self.reject(self.one(row))

    def test_partial_execution_error_keeps_callbacks_but_never_invents_report(self):
        row = copy.deepcopy(self.rows[1])
        row.update(terminal_status='output_error', failed_phase='output', error_category='output',
                   replay_required=True, prefill_tokens_per_s=None, decode_evaluations_per_s=None)
        for field in ('reused_prefix_tokens', 'prefill_rows', 'decode_evaluations',
                      'sampled_tokens', 'emitted_tokens', 'replayed_rows'):
            row['work'][field] = None
        for field in ('native_total_ms', 'prefill_ms', 'decode_ms', 'sampling_ms'):
            row['timing'][field] = None
        result = self.report(self.one(row))
        self.assertEqual(result['unavailable_core_reports'], 1)
        self.assertEqual(result['metrics']['decode_tokens_per_s']['p50'], 50)
        self.assertIsNone(result['metrics']['decode_evaluations_per_s']['p50'])

    def test_stable_schema_rejects_legacy_or_unexpected_payload_fields(self):
        for where, field, value in [(0, 'schema_version', 2), (1, 'prompt', 'private text'),
                                     (1, 'token_ids', [1, 2]), (1, 'event', 'token')]:
            bad = copy.deepcopy(self.rows)
            bad[where][field] = value
            self.reject(bad)

    def test_existing_swift_reader_cannot_silently_accept_cli_schema(self):
        with self.assertRaises((ValueError, KeyError)):
            legacy.summaries(FIXTURE / 'complete.jsonl', 'unrelated-build')

    def test_cli_creates_only_a_new_bounded_report_after_validation(self):
        source = FIXTURE / 'complete.jsonl'
        target = self.root / 'report.json'
        args = [str(source), '--input-sha256', sha(source.read_bytes()), '--output', str(target)]
        self.assertEqual(consumer.main(args), 0)
        self.assertLessEqual(target.stat().st_size, 65536)
        original = target.read_bytes()
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(consumer.main(args), 2)
        self.assertEqual(target.read_bytes(), original)
        bad_target = self.root / 'must-not-be-created.json'
        args[-1] = str(bad_target)
        args[2] = '0' * 64
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(consumer.main(args), 2)
        self.assertFalse(bad_target.exists())


if __name__ == '__main__':
    unittest.main(verbosity=2)

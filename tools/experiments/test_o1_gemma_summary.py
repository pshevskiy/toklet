import argparse
import copy
import inspect
import json
import random
import re
import unittest
from pathlib import Path
from unittest import mock

import o1_gemma_summary as summary
from o1_gemma_summary import comparison


def cohort(measured_pairs=10, timings=None):
    rows = []
    for trial in range(3 + measured_pairs):
        modes = ('disabled', 'summary') if trial % 2 == 0 else ('summary', 'disabled')
        for mode in modes:
            latency = timings(trial - 3, mode) if timings is not None else (100 if mode == 'disabled' else 102)
            start = 1_000_000_000 + trial * 1_000_000_000_000 + (mode == 'summary') * 100_000_000_000
            rows.append(dict(request_id=f'00000000-0000-4000-8000-{2 * trial + (mode == "summary"):012d}',
                trial=trial, warmup=trial < 3, mode=mode, ttft_ms=latency, total_ms=10 * latency,
                submitted_uptime_ns=start, completed_uptime_ns=start + round(10 * latency * 1e6),
                tokens=[1, 2], history=4, committed=3, pending=1))
    return rows


def capture_parts(rows, measured_pairs=10):
    manifest = dict(kind='request_summary_manifest', schemaVersion=2,
                    droppedRecords=max(0, measured_pairs + 3 - 32), buildIdentity='synthetic')
    header = manifest if measured_pairs == 10 else dict(kind='benchmark_request_summary_capture', schemaVersion=1,
        nativeManifest=manifest, recordCount=measured_pairs + 3, ringCapacity=32,
        captureByteLimit=512 * 1024, outsideTimedPairs=True)
    records = [dict(sequence=index, requestID=row['request_id'], processLifetimePeakBytes=1024**2,
        callbackTokens=len(row['tokens']), status=0, after=dict(history=row['history'],
        committed=row['committed'], pending=row['pending']))
        for index, row in enumerate(r for r in rows if r['mode'] == 'summary')]
    return header, records


def wire(header, records):
    return b''.join((json.dumps(row, separators=(',', ':'), allow_nan=False) + '\n').encode()
                    for row in [header, *records])


class OverheadAnalysisTests(unittest.TestCase):
    def rows(self):
        return cohort()

    def test_known_two_percent_cost_is_not_hidden_by_units(self):
        result = comparison(self.rows())
        for value in result.values():
            self.assertAlmostEqual(value['median_ratio_overhead_pct'], 2)
            self.assertAlmostEqual(value['paired_median_pct'], 2)
            self.assertGreater(value['paired_median_bootstrap_95_pct'][0], 1)

    def test_mismatched_work_is_rejected(self):
        rows = self.rows()
        rows[-1]['pending'] = 0
        with self.assertRaises(ValueError):
            comparison(rows)

    def test_xcode_summary_source_membership(self):
        project = (Path(__file__).resolve().parents[2] / 'apple/TokletHarness.xcodeproj/project.pbxproj').read_text()
        file_id = re.search(r'(\w{24}) = \{ isa = PBXFileReference;[^\n]*path = "Harness/NativeRequestSummary.swift"', project).group(1)
        build_id = re.search(r'(\w{24}) = \{ isa = PBXBuildFile; fileRef = ' + file_id, project).group(1)
        self.assertRegex(project, r'PBXSourcesBuildPhase;[^\n]*files = \([^)]*' + build_id)
        self.assertRegex(project, r'PBXGroup; children = \([^)]*' + file_id)

    def test_ascii_cli_bounds_and_no_numeric_coercion(self):
        for text, expected in [('10', 10), ('11', 11), ('100', 100), ('00010', 10), ('0' * 5000 + '100', 100)]:
            self.assertEqual(summary.parse_measured_pairs(text), expected)
        for text in ['', '9', '101', '-10', '+10', ' 10', '10 ', '10\n', '10.0', '1e2',
                     '١٠', '１０', '9' * 5000, True, 10, 10.0, b'10', None]:
            with self.subTest(text=repr(text)[:40]), self.assertRaises(argparse.ArgumentTypeError):
                summary.parse_measured_pairs(text)

    def test_original_defaults_fields_and_constant_pair_bootstrap(self):
        self.assertEqual(summary.WARMUP_PAIRS, 3)
        self.assertEqual(inspect.signature(comparison).parameters['measured_pairs'].default, 10)
        implicit = comparison(self.rows())
        self.assertEqual(implicit, comparison(self.rows(), measured_pairs=10))
        self.assertEqual(set(implicit), {'ttft_ms', 'total_ms'})
        for metric, factor in [('ttft_ms', 1), ('total_ms', 10)]:
            stats = implicit[metric]
            self.assertEqual(set(stats), {'disabled_median', 'summary_median', 'median_ratio_overhead_pct',
                'paired_overhead_pct', 'paired_median_pct', 'paired_median_bootstrap_95_pct',
                'disabled_range', 'summary_range'})
            self.assertEqual(stats['disabled_median'], 100 * factor)
            self.assertEqual(stats['summary_median'], 102 * factor)
            self.assertEqual(stats['disabled_range'], [100 * factor] * 2)
            self.assertEqual(stats['summary_range'], [102 * factor] * 2)
            self.assertEqual(len(stats['paired_overhead_pct']), 10)
            for endpoint in stats['paired_median_bootstrap_95_pct']:
                self.assertAlmostEqual(endpoint, 2.)

    def test_all_hundred_pairs_including_last_ninety_affect_statistics(self):
        def timings(index, mode):
            return 100 if mode == 'disabled' or index < 10 else 102
        result = comparison(cohort(100, timings), measured_pairs=100)
        for stats in result.values():
            self.assertEqual(len(stats['paired_overhead_pct']), 100)
            self.assertEqual(stats['paired_overhead_pct'][:10], [0.] * 10)
            self.assertAlmostEqual(stats['paired_overhead_pct'][-1], 2.)
            self.assertAlmostEqual(stats['median_ratio_overhead_pct'], 2.)
            self.assertAlmostEqual(stats['paired_median_pct'], 2.)
            for endpoint in stats['paired_median_bootstrap_95_pct']:
                self.assertAlmostEqual(endpoint, 2.)

    def test_warmup_times_are_excluded_but_work_remains_validated(self):
        rows = self.rows()
        baseline = comparison(rows)
        for row in rows:
            if row['warmup']:
                row['ttft_ms'] = 1e6 if row['mode'] == 'disabled' else 1e9
                row['total_ms'] = 10 * row['ttft_ms']
                row['completed_uptime_ns'] = row['submitted_uptime_ns'] + round(row['total_ms'] * 1e6)
        self.assertEqual(comparison(rows), baseline)

    def test_point_ratio_and_paired_median_are_distinct_statistics(self):
        def timings(index, mode):
            if index < 0:
                return 100
            return (1 if index < 5 else 100) if mode == 'disabled' else (2 if index < 5 else 99)
        result = comparison(cohort(timings=timings))
        for stats in result.values():
            self.assertEqual(stats['median_ratio_overhead_pct'], 0.)
            self.assertAlmostEqual(stats['paired_median_pct'], 49.5)
            self.assertAlmostEqual(stats['paired_median_bootstrap_95_pct'][1], 100.)

    def test_high_point_cost_is_not_hidden_by_zero_paired_upper_bound(self):
        def timings(index, mode):
            off = index + 1 if index >= 0 else 1
            return 1000 if mode == 'summary' and index == 4 else off
        result = comparison(cohort(timings=timings))
        for stats in result.values():
            self.assertAlmostEqual(stats['median_ratio_overhead_pct'], 100 * (6.5 / 5.5 - 1))
            self.assertGreater(stats['median_ratio_overhead_pct'], 1.)
            self.assertEqual(stats['paired_median_pct'], 0.)
            self.assertEqual(stats['paired_median_bootstrap_95_pct'], [0., 0.])
        # comparison reports statistics; the caller's qualification expression
        # is intentionally not reimplemented as a test-side substitute.

    def test_exact_bootstrap_seed_draw_count_and_no_global_rng_change(self):
        original_random = random.Random
        calls = []
        class ObservedRandom:
            def __init__(self, seed):
                self.rng = original_random(seed)
                self.record = dict(seed=seed, draws=0, sizes=set())
                calls.append(self.record)
            def choices(self, population, *, k):
                self.record['draws'] += 1
                self.record['sizes'].add((len(population), k))
                return self.rng.choices(population, k=k)
        before = random.getstate()
        with mock.patch.object(summary.random, 'Random', ObservedRandom):
            comparison(self.rows())
        self.assertEqual(calls, [dict(seed=1, draws=10000, sizes={(10, 10)})] * 2)
        self.assertEqual(random.getstate(), before)

    def test_row_order_does_not_change_trial_pairing(self):
        rows = self.rows()
        self.assertEqual(comparison(rows), comparison(list(reversed(rows))))

    def test_invalid_requested_counts_and_cohort_container(self):
        for count in [True, False, 9, 101, 0, -1, 10., '10', None]:
            with self.subTest(count=count), self.assertRaises(ValueError):
                comparison(self.rows(), measured_pairs=count)
        for rows in [tuple(self.rows()), iter(self.rows()), {}, None]:
            with self.assertRaises(ValueError):
                comparison(rows)

    def test_missing_duplicate_or_out_of_range_trial_mode(self):
        for count in (10, 100):
            original = cohort(count)
            changes = [original[:-1], original + [copy.deepcopy(original[-1])]]
            duplicate = copy.deepcopy(original); duplicate[-1] = copy.deepcopy(duplicate[-2]); changes.append(duplicate)
            for rows in changes:
                with self.assertRaises(ValueError):
                    comparison(rows, count)
            for field, values in [('trial', [True, 3., '3', None, -1, count + 3]),
                                  ('mode', ['observed', 'Summary', 'legacy', True, None])]:
                for value in values:
                    rows = copy.deepcopy(original); rows[-1][field] = value
                    with self.subTest(count=count, field=field, value=value), self.assertRaises(ValueError):
                        comparison(rows, count)

    def test_warmup_flags_are_strict_and_match_trial(self):
        for index in (0, 6):
            for flag in [0, 1, None, 'false', not (index == 0)]:
                rows = self.rows(); rows[index]['warmup'] = flag
                with self.subTest(index=index, flag=flag), self.assertRaises(ValueError):
                    comparison(rows)

    def test_invalid_metrics_rejected_even_in_warmups(self):
        for trial in (0, 3):
            for metric in ('ttft_ms', 'total_ms'):
                for value in [float('nan'), float('inf'), -float('inf'), 0., -1., True, '1', None, 10**1000]:
                    rows = self.rows()
                    for row in rows:
                        if row['trial'] == trial:
                            row[metric] = value
                    with self.subTest(trial=trial, metric=metric, value=repr(value)[:20]), self.assertRaises(ValueError):
                        comparison(rows)

    def test_exact_tokens_and_state_parity_in_warmups_and_measurements(self):
        for trial in (0, 3, 12):
            for field, value in [('tokens', [1, 3]), ('tokens', [2, 1]), ('history', 5), ('committed', 2), ('pending', 0)]:
                rows = self.rows()
                next(row for row in rows if row['trial'] == trial and row['mode'] == 'summary')[field] = value
                with self.subTest(trial=trial, field=field), self.assertRaises(ValueError):
                    comparison(rows)

    def test_matching_but_invalid_tokens_or_state_are_not_parity(self):
        changes = [('tokens', value) for value in [None, (1, 2), '1,2', [True, 2], [1., 2], [-1, 2]]]
        changes += [(key, value) for key in ('history', 'committed', 'pending') for value in [True, -1, 1., '1', None]]
        for field, value in changes:
            rows = self.rows()
            for row in rows:
                if row['trial'] == 3:
                    row[field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                comparison(rows)

    def test_matching_but_impossible_timing_or_state_is_rejected(self):
        for update in (dict(ttft_ms=2000, total_ms=1000), dict(history=5, committed=3, pending=1)):
            rows = self.rows()
            for row in rows:
                if row['trial'] == 3:
                    row.update(update)
            with self.subTest(update=update), self.assertRaises(ValueError):
                comparison(rows)

    def test_missing_required_fields_reject(self):
        for field in ('trial', 'warmup', 'mode', 'ttft_ms', 'total_ms', 'tokens', 'history', 'committed', 'pending'):
            rows = self.rows(); rows[-1].pop(field)
            with self.subTest(field=field), self.assertRaises(ValueError):
                comparison(rows)


class SummaryCaptureTests(unittest.TestCase):
    def validate(self, rows, count=10, header=None, records=None):
        default_header, default_records = capture_parts(rows, count)
        return summary.validate_summary_capture(wire(default_header if header is None else header,
            default_records if records is None else records), rows, count)

    def test_original_ten_pairs_native_manifest_and_fourteen_lines(self):
        rows = cohort(); header, records = capture_parts(rows)
        raw = wire(header, records)
        self.assertEqual(len(raw.splitlines()), 14)
        self.assertEqual(summary.validate_summary_capture(raw, rows), records)
        self.assertEqual(self.validate(rows, 10), records)

    def test_hundred_pair_external_capture_keeps_all_warmups_and_native_drops(self):
        rows = cohort(100); header, records = capture_parts(rows, 100)
        raw = wire(header, records)
        self.assertEqual((header['recordCount'], header['ringCapacity'], header['nativeManifest']['droppedRecords']), (103, 32, 71))
        self.assertEqual(len(raw.splitlines()), 104)
        result = summary.validate_summary_capture(raw, rows, 100)
        self.assertEqual(result, records)
        self.assertEqual({r['requestID'] for r in result}, {r['request_id'] for r in rows if r['mode'] == 'summary'})

    def test_external_header_is_exact_typed_and_preserves_ring_accounting(self):
        rows = cohort(100); header, records = capture_parts(rows, 100)
        for update in [dict(kind='request_summary_manifest'), dict(schemaVersion=0), dict(schemaVersion=2),
                       dict(schemaVersion=True), dict(schemaVersion=1.), dict(recordCount=102), dict(recordCount=True),
                       dict(ringCapacity=103), dict(ringCapacity=32.), dict(captureByteLimit=524289),
                       dict(outsideTimedPairs=False), dict(outsideTimedPairs=1), dict(extra='unknown')]:
            with self.subTest(update=update), self.assertRaises(ValueError):
                self.validate(rows, 100, {**header, **update}, records)
        missing_version = copy.deepcopy(header); missing_version.pop('schemaVersion')
        with self.assertRaises(ValueError):
            self.validate(rows, 100, missing_version, records)
        for dropped in (0, 70, 72, True, 71., -1):
            changed = copy.deepcopy(header); changed['nativeManifest']['droppedRecords'] = dropped
            with self.subTest(dropped=dropped), self.assertRaises(ValueError):
                self.validate(rows, 100, changed, records)

    def test_default_native_manifest_and_dropped_records_reject_tampering(self):
        rows = cohort(); header, records = capture_parts(rows)
        for changed in [{**header, 'kind': 'benchmark_request_summary_capture'},
                        {**header, 'droppedRecords': 1}, {**header, 'droppedRecords': False},
                        {**header, 'droppedRecords': 0.}]:
            with self.assertRaises(ValueError):
                self.validate(rows, header=changed, records=records)

    def test_capture_records_are_bijective_including_first_warmup(self):
        rows = cohort(100); header, original = capture_parts(rows, 100)
        variants = [original[1:], original + [original[-1]]]
        for index, identity in [(0, original[1]['requestID']), (50, 'foreign'), (102, None)]:
            changed = copy.deepcopy(original); changed[index]['requestID'] = identity; variants.append(changed)
        for records in variants:
            with self.assertRaises(ValueError):
                self.validate(rows, 100, header, records)
        changed_rows = copy.deepcopy(rows)
        next(r for r in changed_rows if r['mode'] == 'summary' and r['trial'] == 0)['request_id'] = 'foreign'
        with self.assertRaises(ValueError):
            self.validate(changed_rows, 100, header, original)

    def test_bad_source_summary_ids_reject(self):
        rows = cohort(); header, records = capture_parts(rows)
        for value in (None, True, 1, ''):
            changed = copy.deepcopy(rows)
            next(r for r in changed if r['mode'] == 'summary')['request_id'] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.validate(changed, header=header, records=records)
        changed = copy.deepcopy(rows)
        on = [r for r in changed if r['mode'] == 'summary']
        on[1]['request_id'] = on[0]['request_id']
        with self.assertRaises(ValueError):
            self.validate(changed, header=header, records=records)

    def test_process_lifetime_peak_inclusive_limit_and_invalid_values(self):
        rows = cohort(); header, records = capture_parts(rows)
        records[0]['processLifetimePeakBytes'] = 6 * 1024**3
        self.assertEqual(self.validate(rows, header=header, records=records), records)
        for value in (6 * 1024**3 + 1, -1, True, 1., None):
            changed = copy.deepcopy(records); changed[-1]['processLifetimePeakBytes'] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.validate(rows, header=header, records=changed)

    def test_per_record_raw_byte_bound_and_complete_external_byte_bound(self):
        rows = cohort(100); header, records = capture_parts(rows, 100)
        base = len(json.dumps(records[0], separators=(',', ':')).encode())
        # Adding this field costs13 bytes in compact JSON: ,"padding":"".
        records[0]['padding'] = 'x' * (4096 - base - 13)
        self.assertEqual(len(json.dumps(records[0], separators=(',', ':')).encode()), 4096)
        self.assertEqual(len(self.validate(rows, 100, header, records)), 103)
        records[0]['padding'] += 'x'
        with self.assertRaises(ValueError):
            self.validate(rows, 100, header, records)
        raw = wire(*capture_parts(rows, 100))
        with self.assertRaises(ValueError):
            summary.validate_summary_capture(raw + b' ' * (524289 - len(raw)), rows, 100)

    def test_empty_wrong_type_malformed_and_nonfinite_wire_reject(self):
        rows = cohort()
        for raw in (b'', 'not bytes', b'not json\n', b'[]\n', b'{"kind":NaN}\n'):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                summary.validate_summary_capture(raw, rows)


if __name__ == '__main__':
    unittest.main()

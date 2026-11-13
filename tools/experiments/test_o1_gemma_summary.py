import argparse
import copy
import hashlib
import inspect
import json
import random
import re
import struct
import tempfile
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


def workload_descriptor():
    conversation = '[{"role":"user","content":"Public café identity fixture."}]'
    ids = [0x10200 + index for index in range(384)]
    return dict(schema_version=1, workload_id='public_384_128', conversation_json=conversation,
        conversation_sha256=hashlib.sha256(conversation.encode()).hexdigest(), input_token_count=len(ids),
        input_tokens_sha256=hashlib.sha256(b''.join(struct.pack('<I', value) for value in ids)).hexdigest(),
        maximum_output_tokens=128, capacity=512, chunk=32, cache_policy='reset_chat_each_request',
        expected_output=dict(tokens=list(range(1000, 1128)), history=512, committed=511,
                             pending=1, reason=1, stop_token=2**32 - 1))


def workload_rows(workload, *, count=10, preflight=False, expected=None):
    expected = expected or workload['expected_output']
    rows = cohort(count)
    if preflight:
        rows = rows[:2]
        for row in rows:
            row.update(trial=0, warmup=False)
    for row in rows:
        row.update(copy.deepcopy(expected), workload_sha256=workload['_sha256'],
            conversation_sha256=workload['conversation_sha256'], input_tokens=workload['input_token_count'],
            input_tokens_sha256=workload['input_tokens_sha256'], maximum_output_tokens=workload['maximum_output_tokens'],
            capacity=512, chunk=32, cache_policy='reset_chat_each_request', status=0,
            model_instance=37, session_instance=43,
            admission=dict(operation=3, base_history=0, input=384, history_after_input=384,
                           output=128, required=512, remaining=0))
    return rows


def workload_capture(rows, workload, count=10):
    header, records = capture_parts(rows, count)
    for record, row in zip(records, (r for r in rows if r['mode'] == 'summary'), strict=True):
        state = dict(modelInstance=37, sessionInstance=43, history=row['history'],
                     committed=row['committed'], pending=row['pending'])
        record.update(schemaVersion=2, after=state, before=copy.deepcopy(state), capacity=512, chunk=32,
            promptTokens=384, requestedOutputTokens=128, nativeEmittedTokens=len(row['tokens']),
            processPhysicalBytes=768*1024, sessionAdmittedBytes=512*1024,
            submittedUptimeNanoseconds=row['submitted_uptime_ns'] + 100,
            firstCallbackNanoseconds=row['submitted_uptime_ns'] + round(row['ttft_ms'] * 1_000_000),
            completedUptimeNanoseconds=row['completed_uptime_ns'] - 100,
            outcome='outputLimit' if row['reason'] == 1 else 'stopToken', nativeMetricStatus='completed',
            nativeMetrics=dict(sessionInstance=43, reusedTokens=0, prefillTokens=384,
                sampledTokens=row['history'] - 384, emittedTokens=len(row['tokens']),
                prefillSeconds=.05, decodeSeconds=.1, samplingSeconds=.001, totalSeconds=.2,
                firstTokenSeconds=.05, prefillTokensPerSecond=7680.))
    return header, records


class WorkloadIdentityTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / 'workload.json'

    def load(self, descriptor=None, *, preflight=False, raw=None):
        if raw is None:
            raw = json.dumps(workload_descriptor() if descriptor is None else descriptor,
                             ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode()
        self.path.write_bytes(raw)
        return summary.load_workload(self.path, preflight_only=preflight)

    def test_exact_wire_identity_and_independent_little_endian_reference(self):
        descriptor = workload_descriptor()
        raw = json.dumps(descriptor, ensure_ascii=False, indent=2).encode()
        loaded = self.load(raw=raw)
        self.assertEqual(loaded, {**descriptor, '_sha256': hashlib.sha256(raw).hexdigest()})
        ids = [0x10200 + index for index in range(384)]
        reference = b''.join(bytes((v & 255, v >> 8 & 255, v >> 16 & 255, v >> 24 & 255)) for v in ids)
        self.assertEqual(descriptor['input_tokens_sha256'], hashlib.sha256(reference).hexdigest())
        rows = workload_rows(loaded)
        for row in rows:
            row['input_tokens_sha256'] = hashlib.sha256(b''.join(struct.pack('>I', v) for v in ids)).hexdigest()
        with self.assertRaises(ValueError):
            summary.validate_workload_rows(rows, loaded)

    def test_manifest_wire_and_utf8_conversation_inclusive_bounds(self):
        descriptor = workload_descriptor()
        prefix, suffix = '[{"role":"user","content":"', '"}]'
        descriptor['conversation_json'] = prefix + 'a' * (8192 - len(prefix + suffix)) + suffix
        descriptor['conversation_sha256'] = hashlib.sha256(descriptor['conversation_json'].encode()).hexdigest()
        raw = json.dumps(descriptor, separators=(',', ':')).encode()
        padded = raw + b' ' * (16384 - len(raw))
        self.assertEqual(self.load(raw=padded)['_sha256'], hashlib.sha256(padded).hexdigest())
        for raw in (padded + b' ', b'', b'\xff', b'[]', b'{', b'null'):
            with self.subTest(raw=repr(raw[:20])), self.assertRaises(ValueError):
                self.load(raw=raw)
        descriptor['conversation_json'] += ' '
        descriptor['conversation_sha256'] = hashlib.sha256(descriptor['conversation_json'].encode()).hexdigest()
        with self.assertRaises(ValueError):
            self.load(descriptor)

    def test_duplicate_nonfinite_unknown_and_missing_manifest_fields_reject(self):
        descriptor = workload_descriptor()
        raw = json.dumps(descriptor, separators=(',', ':')).encode()
        for changed in (raw[:-1] + b',"schema_version":1}', raw.replace(b'"schema_version":1', b'"schema_version":NaN')):
            with self.assertRaises(ValueError):
                self.load(raw=changed)
        for field in descriptor:
            changed = copy.deepcopy(descriptor); changed.pop(field)
            with self.subTest(missing=field), self.assertRaises(ValueError):
                self.load(changed)
        for extra in ('unknown', '_sha256'):
            with self.assertRaises(ValueError):
                self.load({**descriptor, extra: 'injected'})
        for conversation in ('{}', 'null', '[NaN]', '[{"role":"user","role":"assistant","content":"x"}]'):
            changed = copy.deepcopy(descriptor); changed['conversation_json'] = conversation
            changed['conversation_sha256'] = hashlib.sha256(conversation.encode()).hexdigest()
            with self.subTest(conversation=conversation), self.assertRaises(ValueError):
                self.load(changed)

    def test_exact_manifest_types_counts_hashes_and_window_admission(self):
        changes = {
            'schema_version': [True, 1., 0, 2], 'workload_id': ['', '_bad', 'UPPER', 'a/b', 'é', 'x' * 65, True],
            'conversation_json': ['', [], True], 'conversation_sha256': ['0' * 64, 'A' * 64, True],
            'input_token_count': [True, 0, 513, 384., '384', 2**64],
            'input_tokens_sha256': ['', 'a' * 63, 'G' * 64, 'A' * 64, True],
            'maximum_output_tokens': [True, 0, 129, 257, 128., '128'],
            'capacity': [True, 511, 513, 512.], 'chunk': [True, 31, 33, 32.],
            'cache_policy': ['reuse', 'reset', None],
        }
        for field, values in changes.items():
            for value in values:
                changed = workload_descriptor(); changed[field] = value
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    self.load(changed)

    def test_expected_output_is_exact_typed_and_not_just_paired_agreement(self):
        mutations = [('tokens', []), ('tokens', [True]), ('tokens', [2**32]), ('tokens', [-1]),
                     ('tokens', [1.] * 128), ('history', True), ('history', 513), ('committed', 510),
                     ('pending', 0), ('pending', True), ('reason', True), ('reason', 3), ('stop_token', 106)]
        for field, value in mutations:
            changed = workload_descriptor(); changed['expected_output'][field] = value
            with self.subTest(field=field, value=repr(value)[:30]), self.assertRaises(ValueError):
                self.load(changed)
        for action in ('missing', 'extra'):
            changed = workload_descriptor()
            if action == 'missing': changed['expected_output'].pop('reason')
            else: changed['expected_output']['extra'] = 0
            with self.assertRaises(ValueError):
                self.load(changed)
        loaded = self.load(); rows = workload_rows(loaded)
        for row in rows: row['tokens'][0] += 1
        with self.assertRaises(ValueError):
            summary.validate_workload_rows(rows, loaded)

    def test_null_output_only_preflight_and_input_identity_always_required(self):
        descriptor = workload_descriptor(); descriptor['expected_output'] = None
        with self.assertRaises(ValueError): self.load(descriptor)
        loaded = self.load(descriptor, preflight=True)
        expected = workload_descriptor()['expected_output']
        rows = workload_rows(loaded, preflight=True, expected=expected)
        self.assertEqual(summary.validate_workload_rows(rows, loaded, preflight_only=True), expected)
        for field in ('input_token_count', 'input_tokens_sha256'):
            changed = copy.deepcopy(descriptor); changed[field] = None
            with self.subTest(field=field), self.assertRaises(ValueError): self.load(changed, preflight=True)
        for changed in (rows[:1], rows + [rows[0]], [rows[0], copy.deepcopy(rows[0])]):
            with self.assertRaises(ValueError): summary.validate_workload_rows(changed, loaded, preflight_only=True)

    def test_freeze_input_only_roundtrip_preserves_identity_and_exclusive_file(self):
        descriptor = workload_descriptor(); descriptor['expected_output'] = None
        loaded = self.load(descriptor, preflight=True); original = copy.deepcopy(loaded)
        expected = workload_descriptor()['expected_output']
        output = self.path.parent / 'frozen.json'
        frozen = summary.freeze_workload(output, loaded, expected)
        raw = output.read_bytes(); wire_fields = json.loads(raw)
        self.assertEqual(wire_fields, {**descriptor, 'expected_output': expected})
        self.assertNotIn('_sha256', wire_fields)
        self.assertEqual(frozen, summary.load_workload(output))
        self.assertEqual(frozen['_sha256'], hashlib.sha256(raw).hexdigest())
        self.assertEqual(loaded, original)
        with self.assertRaises(FileExistsError): summary.freeze_workload(output, loaded, expected)
        self.assertEqual(output.read_bytes(), raw)
        self.assertEqual(loaded, original)

    def test_freeze_rejects_valid_input_that_expands_beyond_wire_limit_without_file(self):
        descriptor = workload_descriptor(); descriptor['expected_output'] = None
        encode = lambda value: json.dumps(value, ensure_ascii=False, separators=(',', ':')).encode()
        # An inner JSON document can end in whitespace. Each added newline costs
        # one UTF8 byte there and two escaped bytes in the outer wire document.
        padding = min((16384 - len(encode(descriptor))) // 2,
                      8192 - len(descriptor['conversation_json'].encode()))
        descriptor['conversation_json'] += '\n' * padding
        descriptor['conversation_sha256'] = hashlib.sha256(descriptor['conversation_json'].encode()).hexdigest()
        raw = encode(descriptor)
        self.assertLessEqual(len(raw), 16384)
        self.assertLessEqual(len(descriptor['conversation_json'].encode()), 8192)
        loaded = self.load(raw=raw, preflight=True); original = copy.deepcopy(loaded)
        expected = workload_descriptor()['expected_output']
        projected = {**descriptor, 'expected_output': expected}
        self.assertGreater(len(json.dumps(projected, ensure_ascii=False, sort_keys=True, indent=2).encode()), 16384)
        output = self.path.parent / 'must-not-exist.json'
        with self.assertRaises(ValueError): summary.freeze_workload(output, loaded, expected)
        self.assertFalse(output.exists())
        self.assertEqual(loaded, original)

    def test_old_fifteen_token_rows_cannot_claim384_or512(self):
        loaded = self.load()
        legacy = cohort()
        for row in legacy:
            row.update(tokens=[818, 5279, 529, 7001, 563, 5213, 50429, 84750],
                       history=24, committed=23, pending=1)
        with self.assertRaises(ValueError): summary.validate_workload_rows(legacy, loaded)
        for actual_count in (15, 512):
            rows = workload_rows(loaded)
            for row in rows:
                row['input_tokens'] = actual_count
                row['admission'].update(input=actual_count, history_after_input=actual_count,
                                        required=actual_count + 128, remaining=512 - actual_count - 128)
            with self.subTest(actual_count=actual_count), self.assertRaises(ValueError):
                summary.validate_workload_rows(rows, loaded)

    def test_valid_full_cohort_preserves_original_statistics_and_warmup_validation(self):
        loaded = self.load()
        for count in (10, 100):
            rows = workload_rows(loaded, count=count)
            self.assertEqual(summary.validate_workload_rows(rows, loaded), loaded['expected_output'])
            self.assertEqual(comparison(rows, count), comparison(cohort(count), count))
            changed = copy.deepcopy(rows); changed[0]['conversation_sha256'] = '0' * 64
            with self.assertRaises(ValueError): summary.validate_workload_rows(changed, loaded)

    def test_actual_driver_rejects_legacy_capture_for_custom_workload(self):
        self.load()
        archive = self.path.parent / 'archive.a'; archive.write_bytes(b'mocked archive')
        output = self.path.parent / 'fresh-output'
        legacy_rows = cohort(); legacy_wire = wire(*capture_parts(legacy_rows))
        measured_commands = []
        def child(command, **_kwargs):
            if '-o' in command:
                Path(command[command.index('-o') + 1]).write_bytes(b'mocked Swift executable')
            elif Path(command[0]).name == 'summary-check':
                if '--conformance-output' in command:
                    Path(command[-1]).write_bytes(legacy_wire)
                else:
                    measured_commands.append(command)
                    Path(command[2]).write_text(json.dumps(legacy_rows))
                    Path(str(command[2]) + '.summaries.jsonl').write_bytes(legacy_wire)
            return mock.Mock(returncode=0)
        argv = ['o1_gemma_summary.py', 'mocked-model', str(archive), str(output), '--workload', str(self.path)]
        with mock.patch('sys.argv', argv), mock.patch.object(summary.subprocess, 'run', side_effect=child), \
                mock.patch.object(summary.subprocess, 'check_output', return_value='synthetic'):
            with self.assertRaises(ValueError): summary.main()
        self.assertTrue(measured_commands, 'Identity failure must follow the mocked measured capture')
        if (output / 'result.json').exists():
            failure = json.loads((output / 'result.json').read_text())
            self.assertFalse(failure.get('passed'))
            self.assertFalse(failure.get('overhead_qualified', False))

    def test_actual_admission_fields_are_exact_even_when_both_modes_match(self):
        loaded = self.load()
        changes = dict(operation=1, base_history=512, input=15, history_after_input=15,
                       output=12, required=396, remaining=116, extra=0)
        for field, value in changes.items():
            rows = workload_rows(loaded)
            for row in rows: row['admission'][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError): summary.validate_workload_rows(rows, loaded)
        for value in (None, [], True):
            rows = workload_rows(loaded); rows[0]['admission'] = value
            with self.assertRaises(ValueError): summary.validate_workload_rows(rows, loaded)
        rows = workload_rows(loaded); rows[0]['admission']['operation'] = True
        with self.assertRaises(ValueError): summary.validate_workload_rows(rows, loaded)

    def test_row_identity_outcomes_state_and_host_clocks_cannot_be_fabricated(self):
        loaded = self.load()
        changes = dict(workload_sha256='0'*64, conversation_sha256='0'*64, input_tokens=True,
            input_tokens_sha256='0'*64, maximum_output_tokens=12, capacity=513, chunk=16,
            cache_policy='reuse', status=1, reason=2, stop_token=106, model_instance=0,
            session_instance=True, history=24, committed=23, pending=0,
            submitted_uptime_ns=True, completed_uptime_ns=0)
        for field, value in changes.items():
            rows = workload_rows(loaded)
            for row in rows: row[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError): summary.validate_workload_rows(rows, loaded)
        rows = workload_rows(loaded)
        for row in rows: row['completed_uptime_ns'] += 1_000_000_000
        with self.assertRaises(ValueError): summary.validate_workload_rows(rows, loaded)

    def test_capture_joins_actual_summary_work_and_allows_prior_reset_history(self):
        loaded = self.load()
        for count in (10, 100):
            rows = workload_rows(loaded, count=count); header, records = workload_capture(rows, loaded, count)
            self.assertGreater(records[0]['before']['history'], 0)
            self.assertEqual(summary.validate_workload_capture(wire(header, records), rows, loaded,
                                                              measured_pairs=count), records)
        descriptor = workload_descriptor(); descriptor['expected_output'] = None
        loaded = self.load(descriptor, preflight=True)
        rows = workload_rows(loaded, preflight=True, expected=workload_descriptor()['expected_output'])
        header, records = workload_capture(rows, loaded)
        self.assertEqual(summary.validate_workload_capture(wire(header, records), rows, loaded,
                                                          preflight_only=True), records)

    def test_capture_wrong_native_work_status_session_and_state_reject(self):
        loaded = self.load(); rows = workload_rows(loaded); header, records = workload_capture(rows, loaded)
        mutations = [('promptTokens', 15), ('requestedOutputTokens', 12), ('callbackTokens', 127),
            ('nativeEmittedTokens', 127), ('status', 1), ('outcome', 'stopToken'),
            ('nativeMetricStatus', 'notStarted'), ('capacity', 513), ('chunk', 16),
            ('processLifetimePeakBytes', 6*1024**3+1)]
        for field, value in mutations:
            changed = copy.deepcopy(records); changed[0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                summary.validate_workload_capture(wire(header, changed), rows, loaded)
        native_changes = [('sessionInstance', 44), ('reusedTokens', 1), ('reusedTokens', False),
                          ('prefillTokens', 15), ('prefillTokens', 384.),
                          ('sampledTokens', 127), ('sampledTokens', 128.), ('emittedTokens', 127)]
        for field, value in native_changes:
            changed = copy.deepcopy(records); changed[0]['nativeMetrics'][field] = value
            with self.subTest(native=field), self.assertRaises(ValueError):
                summary.validate_workload_capture(wire(header, changed), rows, loaded)
        changed = copy.deepcopy(records); changed[0]['after']['history'] -= 1
        with self.assertRaises(ValueError): summary.validate_workload_capture(wire(header, changed), rows, loaded)
        changed = copy.deepcopy(records); changed[0]['after']['modelInstance'] += 1
        with self.assertRaises(ValueError): summary.validate_workload_capture(wire(header, changed), rows, loaded)

    def test_capture_schema_memory_and_nested_host_clock_boundaries(self):
        loaded = self.load(); rows = workload_rows(loaded); header, records = workload_capture(rows, loaded)
        bounded = copy.deepcopy(records)
        for key in ('processLifetimePeakBytes', 'processPhysicalBytes', 'sessionAdmittedBytes'):
            bounded[0][key] = 6 * 1024**3
        self.assertEqual(summary.validate_workload_capture(wire(header, bounded), rows, loaded), bounded)
        fields = ['schemaVersion', 'processLifetimePeakBytes', 'processPhysicalBytes', 'sessionAdmittedBytes',
                  'submittedUptimeNanoseconds', 'firstCallbackNanoseconds', 'completedUptimeNanoseconds']
        for key in fields:
            changed = copy.deepcopy(records); changed[0].pop(key)
            with self.subTest(missing=key), self.assertRaises(ValueError):
                summary.validate_workload_capture(wire(header, changed), rows, loaded)
        for key in ('processLifetimePeakBytes', 'processPhysicalBytes', 'sessionAdmittedBytes'):
            for value in (0, -1, True, 1024., 6 * 1024**3 + 1):
                changed = copy.deepcopy(records); changed[0][key] = value
                with self.subTest(memory=key, value=value), self.assertRaises(ValueError):
                    summary.validate_workload_capture(wire(header, changed), rows, loaded)
        for value in (1, True, 2., None):
            changed = copy.deepcopy(records); changed[0]['schemaVersion'] = value
            with self.subTest(schema=value), self.assertRaises(ValueError):
                summary.validate_workload_capture(wire(header, changed), rows, loaded)
        source = next(row for row in rows if row['mode'] == 'summary')
        changes = [('submittedUptimeNanoseconds', source['submitted_uptime_ns'] - 1),
                   ('completedUptimeNanoseconds', source['completed_uptime_ns'] + 1),
                   ('submittedUptimeNanoseconds', records[0]['firstCallbackNanoseconds'] + 1),
                   ('completedUptimeNanoseconds', records[0]['firstCallbackNanoseconds'] - 1),
                   ('firstCallbackNanoseconds', records[0]['firstCallbackNanoseconds'] + 1000)]
        changes += [(key, True) for key in fields[-3:]]
        for key, value in changes:
            changed = copy.deepcopy(records); changed[0][key] = value
            with self.subTest(clock=key, value=value), self.assertRaises(ValueError):
                summary.validate_workload_capture(wire(header, changed), rows, loaded)

    def test_hidden_stop_sampling_is_not_emitted_count_or_requested_cap(self):
        descriptor = workload_descriptor()
        expected = dict(tokens=[31, 32, 33], history=388, committed=387, pending=1, reason=2, stop_token=106)
        descriptor['expected_output'] = expected
        loaded = self.load(descriptor); rows = workload_rows(loaded)
        self.assertEqual(summary.validate_workload_rows(rows, loaded), expected)
        header, records = workload_capture(rows, loaded)
        self.assertEqual(records[0]['nativeMetrics']['sampledTokens'], 4)
        self.assertEqual(summary.validate_workload_capture(wire(header, records), rows, loaded), records)
        records[0]['nativeMetrics']['sampledTokens'] = 3
        with self.assertRaises(ValueError): summary.validate_workload_capture(wire(header, records), rows, loaded)


class DecodeEvaluationCaptureTests(unittest.TestCase):
    def capture(self, count=1, *, sampled=1, emitted=1, status='completed'):
        header = dict(kind='request_summary_manifest', nativeGenerationMetricsVersion=2)
        native = dict(decodeEvaluations=count, sampledTokens=sampled, emittedTokens=emitted)
        record = dict(nativeMetricStatus=status, nativeMetrics=native)
        return header, [record]

    def test_known_zero_and_completed_post_decode_cancel_are_actual_counts(self):
        for count, sampled, emitted in [(0, 0, 0), (1, 1, 1), (32768, 32768, 32768)]:
            header, records = self.capture(count, sampled=sampled, emitted=emitted)
            self.assertEqual(summary.validate_native_work_capture(wire(header, records), records, required=True),
                             dict(version=2, available=1, unavailable=0))

    def test_legacy_remains_unavailable_and_cannot_qualify_current_build(self):
        rows = cohort(); header, records = capture_parts(rows)
        result = summary.validate_native_work_capture(wire(header, records), records)
        self.assertEqual(result, dict(version=1, available=0, unavailable=13))
        with self.assertRaises(ValueError): summary.validate_native_work_capture(wire(header, records), records, required=True)
        records[0]['nativeMetrics'] = dict(decodeEvaluations=0)
        with self.assertRaises(ValueError): summary.validate_native_work_capture(wire(header, records), records)

    def test_native_version_is_exact_and_nested_capture_supported(self):
        header, records = self.capture()
        wrapped = dict(kind='benchmark_request_summary_capture', nativeManifest=header)
        self.assertEqual(summary.validate_native_work_capture(wire(wrapped, records), records, required=True)['available'], 1)
        for version in [None, True, False, 2., '2', 0, 3]:
            changed = dict(header, nativeGenerationMetricsVersion=version)
            with self.subTest(version=version), self.assertRaises(ValueError):
                summary.validate_native_work_capture(wire(changed, records), records)

    def test_missing_coerced_overflow_or_excess_decode_counts_reject(self):
        header, records = self.capture()
        for value in [None, True, False, -1, 1., '1', 2, 32769, 2**64]:
            changed = copy.deepcopy(records); changed[0]['nativeMetrics']['decodeEvaluations'] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                summary.validate_native_work_capture(wire(header, changed), changed)
        changed = copy.deepcopy(records); changed[0]['nativeMetrics'].pop('decodeEvaluations')
        with self.assertRaises(ValueError): summary.validate_native_work_capture(wire(header, changed), changed)
        for field, value in [('sampledTokens', True), ('sampledTokens', 32769), ('emittedTokens', 2)]:
            changed = copy.deepcopy(records); changed[0]['nativeMetrics'][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                summary.validate_native_work_capture(wire(header, changed), changed)

    def test_unavailable_outcomes_never_publish_known_zero_or_stale_count(self):
        header, records = self.capture()
        for status in ['notStarted', 'generationFailed', 'invalidRecord']:
            changed = [dict(nativeMetricStatus=status)]
            self.assertEqual(summary.validate_native_work_capture(wire(header, changed), changed)['unavailable'], 1)
            for native in [dict(decodeEvaluations=0), records[0]['nativeMetrics']]:
                changed[0]['nativeMetrics'] = native
                with self.subTest(status=status), self.assertRaises(ValueError):
                    summary.validate_native_work_capture(wire(header, changed), changed)
        for status in [None, '', 0, 'unknown']:
            changed = [dict(nativeMetricStatus=status)]
            with self.assertRaises(ValueError): summary.validate_native_work_capture(wire(header, changed), changed)

    def test_raw_identity_duplicates_nonfinite_and_byte_bound_reject(self):
        header, records = self.capture(); raw = wire(header, records)
        for changed in [raw.replace(b'"decodeEvaluations":1', b'"decodeEvaluations":1,"decodeEvaluations":1'),
                        raw.replace(b'"decodeEvaluations":1', b'"decodeEvaluations":NaN'),
                        b' ' * (summary.CAPTURE_BYTE_LIMIT + 1), b'{}', raw.decode()]:
            with self.subTest(type=type(changed)), self.assertRaises(ValueError):
                summary.validate_native_work_capture(changed, records)
        changed = copy.deepcopy(records); changed[0]['nativeMetrics']['decodeEvaluations'] = 0
        with self.assertRaises(ValueError): summary.validate_native_work_capture(raw, changed)

    def test_wire_counter_types_cannot_be_hidden_by_python_value_equality(self):
        for field in ['decodeEvaluations', 'sampledTokens', 'emittedTokens']:
            header, records = self.capture(0, sampled=0, emitted=0)
            captured = copy.deepcopy(records); captured[0]['nativeMetrics'][field] = False
            self.assertEqual(captured, records)  # Python equality alone loses type.
            with self.subTest(field=field), self.assertRaises(ValueError):
                summary.validate_native_work_capture(wire(header, captured), records, required=True)

    def test_clean_reset_limit_and_hidden_stop_check_actual_evaluated_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'workload.json'
            for expected in [workload_descriptor()['expected_output'],
                             dict(tokens=[31, 32, 33], history=388, committed=387, pending=1, reason=2, stop_token=106)]:
                descriptor = workload_descriptor(); descriptor['expected_output'] = expected
                path.write_text(json.dumps(descriptor)); loaded = summary.load_workload(path)
                rows = workload_rows(loaded); header, records = workload_capture(rows, loaded)
                header['nativeGenerationMetricsVersion'] = 2
                count = expected['committed'] - 384
                for record in records: record['nativeMetrics']['decodeEvaluations'] = count
                self.assertEqual(summary.validate_workload_capture(wire(header, records), rows, loaded,
                                 require_native_work=True), records)
                for wrong in [count - 1, count + 1]:
                    changed = copy.deepcopy(records); changed[0]['nativeMetrics']['decodeEvaluations'] = wrong
                    with self.subTest(expected=count, wrong=wrong), self.assertRaises(ValueError):
                        summary.validate_workload_capture(wire(header, changed), rows, loaded, require_native_work=True)


if __name__ == '__main__':
    unittest.main()

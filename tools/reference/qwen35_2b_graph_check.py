"""Bounded full-coordinate checker for the frozen Qwen3.5-2B graph protocol.

Every array/profile/key/byte is mandatory. Initial-short raw captures are fully
replayable; a pass does not qualify context, quality, performance or adoption.
"""
from collections import defaultdict
from contextlib import ExitStack
import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import bf16_source
import qwen35_2b_graph_protocol as p
import qwen35_2b_weights as weights

need = p.need


def document(path, limit=p.MAX_JSON, expected=None):
    return weights.decode(weights.captured(path, limit, expected))


def common(record, role, phase, source_sha, pins):
    fields = {'format', 'role', 'phase', 'capture_mode', 'protocol_sha256', 'source_manifest_sha256',
        'artifact_pins', 'cases', 'geometry', 'gate', 'profiles', 'arrays', 'values_per_profile',
        'rows_bytes', 'rows_sha256', 'journal_bytes', 'journal_sha256', 'completed', 'qualifications'}
    if role == 'native':
        fields.add('controls')
    need(type(record) is dict and set(record) == fields, 'Exact completed capture result required')
    plan = p.plan(phase)
    need(record['capture_mode'] in ('raw', 'stream'), 'Explicit capture storage mode required')
    fixed = dict(format=p.FORMAT, role=role, phase=phase, protocol_sha256=p.protocol_sha256(),
        source_manifest_sha256=p.sha(source_sha), artifact_pins=pins, cases=p.cases(phase),
        geometry=p.GEOMETRY, gate=p.GATE, profiles=list(p.PROFILES[role]),
        arrays=plan['arrays_per_profile'] * len(p.PROFILES[role]),
        values_per_profile=plan['values_per_profile'],
        rows_bytes=plan[role + '_rows_bytes'], completed=True, qualifications=p.QUALIFICATIONS)
    for key, expected in fixed.items():
        need(p.exact(record[key], expected), 'Frozen capture contract differs: ' + key)
    for key in ('rows_sha256', 'journal_sha256'):
        p.sha(record[key])
    need(type(record['journal_bytes']) is int and 0 < record['journal_bytes'] <= p.journal_limit(phase, role),
         'Bounded complete journal required')
    if role == 'native':
        control = record['controls']
        need(type(control) is dict and set(control) == {'bytes', 'sha256'}
             and type(control['bytes']) is int and 0 < control['bytes'] <= 1024**2,
             'Bounded native controls journal required')
        p.sha(control['sha256'])
    return record


class Capture:
    def __init__(self, folder, role, phase, expected_result_sha, source_sha, pins):
        self.folder, self.role, self.phase = Path(folder), role, phase
        self.record = common(document(self.folder / 'result.json', expected=expected_result_sha),
                             role, phase, source_sha, pins)
        need(self.record['capture_mode'] == 'raw' and phase == 'initial_short', 'Raw checker admits initial short phase only')
        raw = weights.captured(self.folder / 'arrays.jsonl', p.journal_limit(phase, role),
                               self.record['journal_sha256'])
        need(len(raw) == self.record['journal_bytes'] and raw.endswith(b'\n'), 'Exact closed array journal required')
        self.ordered, self.arrays = [], {}
        offset, totals = 0, defaultdict(int)
        for line in raw.splitlines():
            need(0 < len(line) <= 4096, 'Bounded nonempty descriptor line required')
            row = weights.decode(line)
            key = p.observation(row, role, phase)
            need(key not in self.arrays and row['offset'] == offset, 'Duplicate/aliased/noncontiguous observation')
            self.arrays[key] = row
            self.ordered.append(row)
            offset += row['bytes']
            totals[row['profile']] += row['count']
        wanted = {(profile, *key) for profile in p.PROFILES[role] for key in p.inventory(phase)}
        need(set(self.arrays) == wanted and len(self.ordered) == self.record['arrays']
             and offset == self.record['rows_bytes']
             and dict(totals) == {profile: self.record['values_per_profile'] for profile in p.PROFILES[role]},
             'Complete array/key/profile/byte inventory required')
        self.owner = None

    def __enter__(self):
        self.owner = bf16_source.Source(self.folder / 'rows.bin', self.record['rows_bytes'],
            self.record['rows_sha256'], deadline=time.monotonic() + 600)
        return self

    def raw(self, row):
        need(self.owner is not None, 'Open capture descriptor required')
        raw = self.owner.read(row['offset'], row['bytes'], deadline=time.monotonic() + 600)
        need(hashlib.sha256(raw).hexdigest() == row['sha256'], 'Complete array SHA256 differs')
        p.finite(raw, row['dtype'])
        return raw

    def __exit__(self, kind, error, traceback):
        if self.owner is not None:
            self.owner.close()
        return False


def controls(native):
    """Reconstruct every complete canonical trace/state witness from raw bytes."""
    pin = native.record['controls']
    raw = weights.captured(native.folder / 'controls.jsonl', 1024**2, pin['sha256'])
    need(len(raw) == pin['bytes'] and raw.endswith(b'\n'), 'Exact closed controls journal required')
    expected, lookup = {}, {case['name']: case for case in p.cases(native.phase)}
    for case, source in lookup.items():
        trace, state = hashlib.sha256(), hashlib.sha256()
        arrays, values, logits = 0, 0, None
        for row in native.ordered:
            if row['case'] != case:
                continue
            block = native.raw(row)
            trace.update(block)
            arrays += 1
            values += row['count']
            if row['token'] == len(source['tokens']) - 1:
                if row['stage'] in p.STATE_STAGES:
                    state.update(block)
                if row['stage'] == 'logits':
                    logits = hashlib.sha256(block).hexdigest()
        need(logits is not None and arrays == 669 * len(source['tokens']), 'Complete canonical state/trace required')
        for schedule in ('chunk1', 'chunk3', 'reset'):
            expected[case, schedule] = dict(case=case, schedule=schedule, arrays=arrays,
                values=values, trace_sha256=trace.hexdigest(), state_sha256=state.hexdigest(),
                logits_sha256=logits, history=source['tokens'])
    seen = set()
    for line in raw.splitlines():
        row = weights.decode(line)
        need(type(row) is dict and type(row.get('case')) is str and type(row.get('schedule')) is str,
             'Typed native control identity required')
        key = row['case'], row['schedule']
        need(key in expected and key not in seen and p.exact(row, expected[key]),
             'Complete independently reconstructed schedule/reset control differs')
        seen.add(key)
    need(seen == set(expected), 'Every native schedule/reset control required')
    return len(seen)


def prefix_equal(capture):
    pairs = [('prefix4', 'future_changed4', 3)]
    comparisons = 0
    for left, right, count in pairs:
        for row in capture.ordered:
            if row['case'] == left and row['token'] < count:
                key = (row['profile'], right, row['token'], row['layer'], row['stage'], row['head'])
                other = capture.arrays[key]
                need(capture.raw(row) == capture.raw(other), 'Full causal prefix bytes differ')
                comparisons += 1
    return comparisons


def check(native_folder, reference_folder, manifest, manifest_sha256,
          native_result_sha256, reference_result_sha256, phase='initial_short'):
    source = weights.artifact_manifest(manifest, manifest_sha256)
    totals = dict(arrays=0, values=0, failed_arrays=0, failed_values=0, max_abs=0.0)
    failures = []
    with ExitStack() as stack:
        native = stack.enter_context(Capture(native_folder, 'native', phase,
            native_result_sha256, manifest_sha256, source['artifact_pins']))
        reference = stack.enter_context(Capture(reference_folder, 'reference', phase,
            reference_result_sha256, manifest_sha256, source['artifact_pins']))
        control_count = controls(native)
        prefix_count = prefix_equal(native) + prefix_equal(reference)
        for row in reference.ordered:
            key = ('native_fp32', row['case'], row['token'], row['layer'], row['stage'], row['head'])
            stats = p.compare(native.raw(native.arrays[key]), reference.raw(row), row['dtype'])
            totals['arrays'] += 1
            totals['values'] += stats['values']
            totals['failed_values'] += stats['failed_values']
            totals['failed_arrays'] += bool(stats['failed_values'])
            totals['max_abs'] = max(totals['max_abs'], stats['max_abs'])
            if stats['failed_values']:
                failures.append(dict(profile=row['profile'], case=row['case'], token=row['token'],
                    layer=row['layer'], stage=row['stage'], head=row['head'], **stats))
    return dict(format='toklet-qwen35-2b-graph-check-v1', phase=phase,
        passed=totals['failed_values'] == 0, protocol_sha256=p.protocol_sha256(),
        source_manifest_sha256=manifest_sha256, native_result_sha256=native_result_sha256,
        reference_result_sha256=reference_result_sha256, gate=p.GATE, totals=totals,
        failures=failures, schedule_reset_controls=control_count,
        causal_prefix_arrays=prefix_count, qualifications=p.QUALIFICATIONS,
        limitations='Initial short graph diagnostic only. Private cancellation/failure tests, representative context, source lineage, quality and performance need separate gates.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--child-manifest', type=Path, required=True)
    parser.add_argument('--expected-child-manifest-sha256', required=True)
    parser.add_argument('--expected-native-result-sha256', required=True)
    parser.add_argument('--expected-reference-result-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = check(args.native, args.reference, args.child_manifest,
        args.expected_child_manifest_sha256, args.expected_native_result_sha256,
        args.expected_reference_result_sha256)
    with args.output.open('x') as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(result['totals']))
    raise SystemExit(0 if result['passed'] else 1)


class InventoryCursor:
    """Exact per-token coverage using at most one token's 669 identities."""
    def __init__(self, phase):
        self.cases, self.case, self.token, self.seen = p.cases(phase), 0, 0, set()

    def accept(self, case, token, layer, stage, head):
        need(self.case < len(self.cases) and type(case) is str and type(token) is int
             and case == self.cases[self.case]['name'] and token == self.token,
             'Complete ordered case/token inventory required')
        key = layer, stage, head
        wanted = p.shapes(token + 1)
        need(key in wanted and key not in self.seen, 'Unique complete stage/head inventory required')
        self.seen.add(key)
        if len(self.seen) == len(wanted):
            self.seen.clear()
            self.token += 1
            if self.token == len(self.cases[self.case]['tokens']):
                self.case += 1
                self.token = 0

    def finish(self):
        need(self.case == len(self.cases) and not self.seen and self.token == 0,
             'Missing case/token/stage/head at inventory endpoint')


class StreamComparison:
    """One native stream against one wholly independent reference profile.

    Retains complete digest/statistics journals and bounded failure arrays. A
    discarded passing array cannot be numerically replayed from its digest;
    replay requires reexecuting the externally pinned native/reference cohort.
    """
    def __init__(self, stream, output, profile, phase, source_sha, pins):
        need(profile in ('ideal_fp64', 'staged_fp32'), 'Explicit independent stream profile required')
        self.stream, self.output = stream, Path(output)
        self.profile, self.phase, self.source_sha, self.pins = profile, phase, p.sha(source_sha), pins
        self.output.mkdir(exist_ok=False)
        self.journal = (self.output / 'comparisons.jsonl').open('xb')
        self.journal_hash, self.rows_hash, self.descriptors_hash = (hashlib.sha256() for _ in range(3))
        self.descriptor_bytes = self.offset = self.sequence = self.values = self.failed_values = self.failed_arrays = 0
        self.max_abs = 0.0
        self.failure_bytes = 0
        self.failure_limit = 16 * 1024**2
        self.witnesses = {case['name']: dict(trace=hashlib.sha256(), state=hashlib.sha256(),
            arrays=0, values=0, logits_sha256=None, history=case['tokens']) for case in p.cases(phase)}
        self.cursor = InventoryCursor(phase)
        self.prefixes = {}
        self.prefix_checks = 0
        self.closed = False

    def read_exact(self, count):
        need(type(count) is int and 0 <= count <= p.MAX_FRAME_BYTES, 'Bounded exact stream read required')
        raw = bytearray()
        while len(raw) < count:
            block = self.stream.read(count - len(raw))
            need(type(block) is bytes and len(block) > 0, 'Truncated native stream')
            raw.extend(block)
        return bytes(raw)

    def save(self, case, reference_row):
        import struct
        size = struct.unpack('<I', self.read_exact(4))[0]
        need(0 < size <= p.MAX_FRAME_JSON, 'Bounded native array frame required')
        encoded = self.read_exact(size)
        row = weights.decode(encoded)
        key = p.observation(row, 'native', self.phase)
        stage, head = p.split_stage(reference_row['stage'])
        expected = ('native_fp32', case['name'], reference_row['token'], reference_row['layer'], stage, head)
        self.cursor.accept(*key[1:])
        need(key == expected and row['offset'] == self.offset and row['bytes'] <= p.MAX_FRAME_BYTES,
             'Ordered complete native/reference frame identity differs')
        native = self.read_exact(row['bytes'])
        need(hashlib.sha256(native).hexdigest() == row['sha256'], 'Native frame payload digest differs')
        dtype = 'F64' if self.profile == 'ideal_fp64' else 'F32'
        values = reference_row['values']
        reference = struct.pack('<' + ('d' if dtype == 'F64' else 'f') * len(values), *values)
        need(len(values) == row['count'], 'Complete independent stream array required')
        stats = p.compare(native, reference, dtype)
        ref_sha = hashlib.sha256(reference).hexdigest()
        descriptor_line = encoded + b'\n'
        self.descriptors_hash.update(descriptor_line)
        self.descriptor_bytes += len(descriptor_line)
        self.rows_hash.update(native)
        witness = self.witnesses[case['name']]
        witness['trace'].update(native)
        witness['arrays'] += 1
        witness['values'] += row['count']
        if row['token'] == len(case['tokens']) - 1:
            if row['stage'] in p.STATE_STAGES:
                witness['state'].update(native)
            if row['stage'] == 'logits':
                witness['logits_sha256'] = row['sha256']
        # Exact causal prefix proof uses complete-array digests computed from the
        # actual consumed native AND independently computed reference values.
        pair = ('prefix4', 'future_changed4', 3) if self.phase == 'initial_short' else (
            ('context17', 'future_changed17', 16) if self.phase == 'context17' else None)
        if pair is not None and row['token'] < pair[2]:
            prefix_key = row['token'], row['layer'], row['stage'], row['head']
            if case['name'] == pair[0]:
                self.prefixes[prefix_key] = row['sha256'], ref_sha
            elif case['name'] == pair[1]:
                need(self.prefixes.get(prefix_key) == (row['sha256'], ref_sha), 'Complete native/reference causal prefix differs')
                self.prefix_checks += 1
        record = dict(sequence=self.sequence, profile=self.profile, key=list(key[1:]),
            shape=row['shape'], count=row['count'], native_sha256=row['sha256'],
            reference_sha256=ref_sha, **stats)
        if stats['failed_values'] and self.failure_bytes + len(native) + len(reference) <= self.failure_limit:
            stem = f'failure-{self.sequence}'
            (self.output / (stem + '.native.f32le')).write_bytes(native)
            (self.output / (stem + '.reference.' + dtype.lower() + 'le')).write_bytes(reference)
            self.failure_bytes += len(native) + len(reference)
            record['retained_failure'] = dict(native=stem + '.native.f32le',
                reference=stem + '.reference.' + dtype.lower() + 'le')
        line = (json.dumps(record, separators=(',', ':'), allow_nan=False) + '\n').encode()
        need(len(line) <= 2048 and self.journal.tell() + len(line) <= p.journal_limit(self.phase, 'reference'),
             'Complete comparison journal exceeds admitted bound')
        need(self.journal.write(line) == len(line), 'Short comparison journal write')
        self.journal_hash.update(line)
        self.sequence += 1
        self.offset += row['bytes']
        self.values += stats['values']
        self.failed_values += stats['failed_values']
        self.failed_arrays += bool(stats['failed_values'])
        self.max_abs = max(self.max_abs, stats['max_abs'])

    def finish(self, native_folder):
        import struct
        need(not self.closed, 'Stream comparison already closed')
        self.cursor.finish()
        need(self.read_exact(4) == b'\0' * 4, 'Native stream has omitted/extra arrays or no trailer')
        size = struct.unpack('<I', self.read_exact(4))[0]
        need(0 < size <= 65536, 'Bounded native stream trailer required')
        trailer = common(weights.decode(self.read_exact(size)), 'native', self.phase, self.source_sha, self.pins)
        need(trailer['capture_mode'] == 'stream' and self.stream.read(1) == b'', 'Closed streamed native EOF required')
        fixed = dict(rows_bytes=self.offset, rows_sha256=self.rows_hash.hexdigest(),
            journal_bytes=self.descriptor_bytes, journal_sha256=self.descriptors_hash.hexdigest(), arrays=self.sequence)
        for key, value in fixed.items():
            need(p.exact(trailer[key], value), 'Complete native stream trailer differs: ' + key)
        plan = p.plan(self.phase)
        need(self.sequence == plan['arrays_per_profile'] and self.values == plan['values_per_profile'],
             'Every phase coordinate required')
        expected_prefix = 669 * (3 if self.phase == 'initial_short' else 16 if self.phase == 'context17' else 0)
        need(self.prefix_checks == expected_prefix, 'All causal prefix coordinates required')
        folder = Path(native_folder)
        result_raw = weights.captured(folder / 'result.json', p.MAX_JSON)
        need(p.exact(weights.decode(result_raw), trailer), 'Native saved result/trailer differ')
        # Authenticate complete retained native descriptor bytes without loading
        # a potentially large deep-phase journal into process memory.
        with bf16_source.Source(folder / 'arrays.jsonl', trailer['journal_bytes'],
            trailer['journal_sha256'], deadline=time.monotonic() + 600):
            pass
        pin = trailer['controls']
        controls_raw = weights.captured(folder / 'controls.jsonl', 1024**2, pin['sha256'])
        need(len(controls_raw) == pin['bytes'] and controls_raw.endswith(b'\n'), 'Exact native stream controls required')
        expected = {}
        for case, witness in self.witnesses.items():
            need(witness['logits_sha256'] is not None, 'Complete final logits required')
            for schedule in ('chunk1', 'chunk3', 'reset'):
                expected[case, schedule] = dict(case=case, schedule=schedule,
                    arrays=witness['arrays'], values=witness['values'], trace_sha256=witness['trace'].hexdigest(),
                    state_sha256=witness['state'].hexdigest(), logits_sha256=witness['logits_sha256'],
                    history=witness['history'])
        seen = set()
        for line in controls_raw.splitlines():
            row = weights.decode(line)
            need(type(row) is dict and type(row.get('case')) is str and type(row.get('schedule')) is str,
                 'Exact native stream control identity required')
            key = row['case'], row['schedule']
            need(key in expected and key not in seen and p.exact(row, expected[key]), 'Native stream full-state schedule/reset differs')
            seen.add(key)
        need(seen == set(expected), 'Missing native stream schedule/reset control')
        self.journal.close()
        self.closed = True
        return dict(format='toklet-qwen35-2b-stream-profile-v1', phase=self.phase, profile=self.profile,
            completed=True, passed=self.failed_values == 0, protocol_sha256=p.protocol_sha256(),
            source_manifest_sha256=self.source_sha, artifact_pins=self.pins,
            native_result_sha256=hashlib.sha256(result_raw).hexdigest(), native_trace_sha256=self.rows_hash.hexdigest(),
            journal_bytes=(self.output / 'comparisons.jsonl').stat().st_size,
            journal_sha256=self.journal_hash.hexdigest(), arrays=self.sequence, values=self.values,
            failed_arrays=self.failed_arrays, failed_values=self.failed_values, max_abs=self.max_abs,
            retained_failure_bytes=self.failure_bytes, retained_failure_limit_bytes=self.failure_limit,
            gate=p.GATE, qualifications=p.QUALIFICATIONS, schedule_reset_controls=len(seen),
            causal_prefix_arrays=self.prefix_checks,
            replay_limit='Discarded passing values require reexecution of the externally pinned native/reference cohort; hashes are not numeric replay data.')

    def close(self):
        if not self.closed:
            self.journal.close()
            self.closed = True


def pair_stream_profiles(ideal, staged):
    """The root runner independently seals both owned process/runtime cohorts."""
    for result, profile in [(ideal, 'ideal_fp64'), (staged, 'staged_fp32')]:
        need(type(result) is dict and result.get('format') == 'toklet-qwen35-2b-stream-profile-v1'
             and result.get('profile') == profile and result.get('completed') is True,
             'Both complete independent streamed profiles required')
        phase = result.get('phase')
        plan = p.plan(phase)
        need(result.get('protocol_sha256') == p.protocol_sha256()
             and p.exact(result.get('gate'), p.GATE) and p.exact(result.get('qualifications'), p.QUALIFICATIONS),
             'Unchanged stream qualification/gate contract required')
        p.sha(result.get('source_manifest_sha256'))
        p.sha(result.get('native_trace_sha256'))
        p.sha(result.get('journal_sha256'))
        p.source_pins(result.get('artifact_pins'))
        for field, expected in [('arrays', plan['arrays_per_profile']), ('values', plan['values_per_profile']),
                                ('schedule_reset_controls', len(p.cases(phase)) * 3),
                                ('causal_prefix_arrays', 669 * (3 if phase == 'initial_short' else 16 if phase == 'context17' else 0))]:
            need(p.exact(result.get(field), expected), 'Complete paired phase inventory differs: ' + field)
        need(type(result.get('failed_values')) is int and 0 <= result['failed_values'] <= result['values']
             and type(result.get('failed_arrays')) is int and 0 <= result['failed_arrays'] <= result['arrays']
             and (result['failed_values'] == 0) == (result['failed_arrays'] == 0)
             and type(result.get('passed')) is bool and result['passed'] == (result['failed_values'] == 0),
             'Consistent complete stream numerical decision required')
    for field in ('phase', 'protocol_sha256', 'source_manifest_sha256', 'artifact_pins',
                  'native_trace_sha256', 'arrays', 'values', 'schedule_reset_controls', 'causal_prefix_arrays'):
        need(p.exact(ideal[field], staged[field]), 'Complete native replay/profile identity differs: ' + field)
    return dict(passed=ideal['passed'] is True and staged['passed'] is True,
        phase=ideal['phase'], profiles=['ideal_fp64', 'staged_fp32'],
        arrays=ideal['arrays'] + staged['arrays'], values=ideal['values'] + staged['values'],
        failed_values=ideal['failed_values'] + staged['failed_values'],
        native_trace_sha256=ideal['native_trace_sha256'], qualifications=p.QUALIFICATIONS)


def audit_stream_profile(folder, expected_result_sha256):
    """Replay complete retained metadata and any saved failing numerical arrays.

    Statistics of discarded passing values remain an owned-execution claim;
    cryptographic digests cannot recover or numerically recheck those values.
    """
    folder = Path(folder)
    result = document(folder / 'result.json', expected=expected_result_sha256)
    need(type(result) is dict and result.get('completed') is True
         and result.get('format') == 'toklet-qwen35-2b-stream-profile-v1', 'Complete streamed result required')
    phase, profile = result['phase'], result['profile']
    need(profile in ('ideal_fp64', 'staged_fp32'), 'Independent stream profile required')
    p.plan(phase)
    cursor = InventoryCursor(phase)
    sequence, values, failures, failed_arrays, max_abs, retained = 0, 0, 0, 0, 0.0, 0
    size = result['journal_bytes']
    need(type(size) is int and 0 < size <= p.journal_limit(phase, 'reference'), 'Bounded complete comparison journal required')
    base_fields = {'sequence', 'profile', 'key', 'shape', 'count', 'native_sha256',
                   'reference_sha256', 'values', 'failed_values', 'max_abs'}
    cases = {case['name']: case for case in p.cases(phase)}
    def consume(line):
        nonlocal sequence, values, failures, failed_arrays, max_abs, retained
        need(0 < len(line) <= 2048, 'Bounded complete comparison record required')
        row = weights.decode(line)
        need(type(row) is dict and set(row) in (base_fields, base_fields | {'retained_failure'})
             and p.exact(row['sequence'], sequence) and row['profile'] == profile
             and type(row['key']) is list and len(row['key']) == 5, 'Exact ordered comparison record required')
        case, token, layer, stage, head = row['key']
        need(type(case) is str and case in cases and type(token) is int
             and 0 <= token < len(cases[case]['tokens']), 'Exact comparison case/token required')
        count = row['count']
        descriptor = dict(profile='native_fp32', case=case, token=token,
            position=cases[case]['position_offset'] + token, layer=layer, stage=stage, head=head,
            shape=row['shape'], dtype='F32', count=count, offset=values * 4,
            bytes=count * 4 if type(count) is int else -1, sha256=row['native_sha256'])
        p.observation(descriptor, 'native', phase)
        p.sha(row['reference_sha256'])
        cursor.accept(*row['key'])
        need(p.exact(row['values'], count) and type(row['failed_values']) is int
             and 0 <= row['failed_values'] <= count and type(row['max_abs']) in (int, float)
             and math.isfinite(row['max_abs']) and row['max_abs'] >= 0,
             'Complete finite comparison statistics required')
        if 'retained_failure' in row:
            pin = row['retained_failure']
            stem = f'failure-{sequence}'
            dtype = 'F64' if profile == 'ideal_fp64' else 'F32'
            expected = dict(native=stem + '.native.f32le', reference=stem + '.reference.' + dtype.lower() + 'le')
            need(row['failed_values'] > 0 and p.exact(pin, expected), 'Exact local failing-array paths required')
            native = weights.captured(folder / pin['native'], p.MAX_FRAME_BYTES, row['native_sha256'])
            reference = weights.captured(folder / pin['reference'], 2 * p.MAX_FRAME_BYTES, row['reference_sha256'])
            need(len(native) == count * 4 and len(reference) == count * (8 if dtype == 'F64' else 4),
                 'Complete saved failing-array bytes required')
            stats = p.compare(native, reference, dtype)
            need(all(p.exact(row[k], stats[k]) for k in stats), 'Saved failing-array numerical replay differs')
            retained += len(native) + len(reference)
        sequence += 1
        values += count
        failures += row['failed_values']
        failed_arrays += bool(row['failed_values'])
        max_abs = max(max_abs, row['max_abs'])
    with bf16_source.Source(folder / 'comparisons.jsonl', size, result['journal_sha256'],
                             deadline=time.monotonic() + 600) as owner:
        pending = b''
        for offset in range(0, size, bf16_source.HASH_BLOCK):
            raw = owner.read(offset, min(bf16_source.HASH_BLOCK, size - offset), deadline=time.monotonic() + 600)
            lines = (pending + raw).split(b'\n')
            pending = lines.pop()
            need(len(pending) <= 2048, 'Comparison record exceeds bound')
            for line in lines:
                consume(line)
        need(pending == b'', 'Closed comparison journal required')
    cursor.finish()
    for key, expected in [('arrays', sequence), ('values', values), ('failed_values', failures),
                          ('failed_arrays', failed_arrays), ('max_abs', max_abs),
                          ('retained_failure_bytes', retained), ('retained_failure_limit_bytes', 16 * 1024**2)]:
        need(p.exact(result[key], expected), 'Complete comparison result/journal differs: ' + key)
    need(retained <= result['retained_failure_limit_bytes'], 'Saved failure limit exceeded')
    return result

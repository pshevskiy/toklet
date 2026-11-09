"""Qualify complete MiniCPM5 native captures against both frozen CPU profiles."""
from pathlib import Path
import argparse
import hashlib
import json
import math
import sys

import minicpm5_checked as reference

CHUNKS = [1, 3, 32]
VOCAB = 130560
MAXIMUM = 256 * 1024 ** 2
need = reference.need


def duration_ns(value):
    # At <=2**20 seconds, binary64 spacing is <0.25 ns. Convert the
    # serialized duration to nearest ns (ties up) with exact integer arithmetic;
    # quantization is <=0.5 ns per field. Absolute GPU timestamps are separate.
    need(type(value) in (int, float) and 0 <= value <= 2 ** 20,
         'Duration outside bounded nanosecond conversion')
    numerator, denominator = value.as_integer_ratio()
    return (2 * numerator * 10 ** 9 + denominator) // (2 * denominator)


def validate_load_timing(timing, elapsed_ns, work_ns):
    need(type(timing) is dict
         and set(timing) == {'load_seconds', 'conversion_seconds', 'compilation_seconds'},
         'Completed load fields invalid')
    clocks = {key: duration_ns(value) for key, value in timing.items()}
    load = clocks['load_seconds']
    need(0 < load <= elapsed_ns
         and clocks['conversion_seconds'] + clocks['compilation_seconds'] <= load,
         'Completed load/conversion/compilation metrics invalid')
    need(load + work_ns <= elapsed_ns, 'Disjoint load and completed work clocks exceed owned lifetime')


def expected_memory(requested_chunk, position_offset):
    # Independent fixed-graph ledger: H=2048, Q=16*128, KV=2*128,
    # I=6144, V=130560, L=42; seven scalar guards per GPU array.
    # Pinned 64-bit Apple metadata layouts: Views=120, Cache=32, Snapshot=72 B.
    capacity = 512 - position_offset
    chunk = min(requested_chunk, capacity)
    need(requested_chunk in CHUNKS and position_offset in [0, 5, 509], 'Unknown capture owner shape')
    h, k, m, v = 2048, 256, 6144, 130560
    scratch = (12 * (chunk * h + 7) + 6 * (chunk * k + 7)
               + 8 * (chunk * m + 7) + 4 * (chunk * v + 7)
               + 8 * (16 * chunk * capacity + 7) + 4 * (chunk + 7)
               + 32 + 2 * (v + 7))
    # 50 Half snapshots: two H plus 3*(9H+3KV+4I); one FP32 V.
    half_columns = 2 * h + 3 * (9 * h + 3 * k + 4 * m)
    scratch += (2 * (chunk * half_columns + 7 * 50) + 4 * (chunk * v + 7)
                + 4 * chunk * (half_columns + v) + 51 * (2 + 72))
    result = dict(weights_bytes=2674218150, kv_bytes=42 * 2 * (2 * capacity * 128 + 7) * 2,
                  scratch_bytes=scratch, history_bytes=capacity * 4, logits_bytes=v * 8,
                  metadata_bytes=chunk * 120 + 42 * 32, metadata_allowance_bytes=64 * 1024 ** 2)
    result['admitted_bytes'] = sum(result.values())
    return result


def inventory():
    expected = {}
    arrays = reference.expected_arrays()
    for case, spec in reference.CASES.items():
        prefix = 'cases.' + case + '.'
        stages = {name[len(prefix):]: shape for name, (shape, _) in arrays.items()
                  if name.startswith(prefix) and not name.endswith(('attention_scores', 'attention_probabilities'))}
        need(len(stages) == 51, 'Fixed native stage coverage differs')
        for requested in CHUNKS:
            chunk = min(requested, 512 - spec['position_offset'])
            for start in range(0, len(spec['tokens']), chunk):
                rows = min(chunk, len(spec['tokens']) - start)
                for stage, shape in stages.items():
                    expected[(f'chunk{requested}.cases.{case}.{stage}', start)] = [rows, math.prod(shape[1:])]
    for group in ['greedy', 'native_generated']:
        for step in range(4):
            expected[(f'{group}.step_{step}.logits', 0)] = [1, VOCAB]
    for count in range(1, 4):
        expected[(f'scoring.prefix_{count}.logits', 0)] = [1, VOCAB]
    for chunk in [3, 32]:
        expected[(f'boundary.chunk{chunk}.logits', 511)] = [1, VOCAB]
    return expected


def bounded_json(path, maximum):
    with Path(path).open('rb') as stream:
        raw = stream.read(maximum + 1)
    need(len(raw) <= maximum, 'Oversize numerical metadata')
    return json.loads(raw, object_pairs_hook=reference.unique)


def validate_chunk_metrics(rows, owned):
    need(type(owned) is dict and type(owned.get('started_ns')) is int
         and type(owned.get('completed_ns')) is int
         and 0 < owned['started_ns'] < owned['completed_ns'], 'Owned clock interval invalid')
    expected = {}
    for case, spec in reference.CASES.items():
        for requested in CHUNKS:
            chunk = min(requested, 512 - spec['position_offset'])
            for start in range(0, len(spec['tokens']), chunk):
                expected[(case, requested, start)] = min(chunk, len(spec['tokens']) - start)
    need(type(rows) is list and len(rows) == len(expected), 'Chunk metric coverage differs')
    total, commands, gpu_known = 0, 0, 0
    for row in rows:
        need(type(row) is dict and set(row) == {'case', 'requested_chunk', 'rows', 'position_offset',
             'committed_before', 'committed_after', 'pending_after', 'wall_nanoseconds',
             'completed_timing', 'session_memory'}, 'Chunk metric fields differ')
        for key in ['requested_chunk', 'rows', 'position_offset', 'committed_before',
                    'committed_after', 'pending_after', 'wall_nanoseconds']:
            need(type(row[key]) is int and row[key] >= 0, 'Untyped/negative chunk work or clock')
        count = expected.pop((row['case'], row['requested_chunk'], row['committed_before']), None)
        need(count == row['rows'] and count is not None and row['wall_nanoseconds'] > 0,
             'Chunk work/clock differs')
        spec = reference.CASES[row['case']]
        need(row['position_offset'] == spec['position_offset']
             and row['committed_after'] == row['committed_before'] + count
             and row['pending_after'] == len(spec['tokens']) - row['committed_after'], 'Committed/pending chunk state differs')
        timing = row['completed_timing']
        fields = ['command_creation_seconds', 'clock_calibration_seconds', 'commit_seconds',
                  'host_encode_seconds', 'wait_seconds', 'gpu_start_seconds', 'gpu_end_seconds', 'readback_seconds']
        need(type(timing) is dict and set(timing) == set(fields) | {'command_count'}
             and all(reference.number(timing[key]) and timing[key] >= 0 for key in fields)
             and type(timing['command_count']) is int and timing['command_count'] > 0,
             'Completed phase timing/counters invalid')
        wall = row['wall_nanoseconds']
        host_ns = [duration_ns(timing[key]) for key in fields if not key.startswith('gpu_')]
        need(sum(host_ns) <= wall, 'Disjoint host phases exceed completed chunk wall time')
        begin, end = timing['gpu_start_seconds'], timing['gpu_end_seconds']
        need(end >= begin and (end - begin) * 10 ** 9 <= wall, 'GPU interval exceeds completed chunk wall time')
        gpu_known += end > begin > 0
        memory = row['session_memory']
        need(reference.same(memory, expected_memory(row['requested_chunk'], row['position_offset']))
             and memory['admitted_bytes'] + 32 * 1024 ** 2 <= 6 * 1024 ** 3,
             'Session admission/owner ledger differs from fixed graph and capture shape')
        total += wall
        commands += timing['command_count']
    need(not expected and total <= owned['completed_ns'] - owned['started_ns'],
         'Disjoint completed chunk clocks exceed owned lifetime')
    return dict(completed_chunks=len(rows), completed_rows=sum(row['rows'] for row in rows),
                command_count=commands, completed_chunk_wall_nanoseconds=total,
                owned_elapsed_ns=owned['completed_ns'] - owned['started_ns'],
                gpu_timestamp_intervals_available=gpu_known, ordinary_request_overhead_qualified=False)


def check(source, external_sha, folder, owned):
    import numpy as np
    source, folder = Path(source), Path(folder)
    oracle = reference.verify(source, external_sha)
    with (folder / 'capture.json').open('rb') as stream:
        capture_bytes = stream.read(4 * 1024 ** 2 + 1)
    need(len(capture_bytes) <= 4 * 1024 ** 2
         and reference.digest(owned.get('capture_manifest_sha256'))
         and hashlib.sha256(capture_bytes).hexdigest() == owned['capture_manifest_sha256'],
         'Capture manifest does not match completed owned receipt')
    capture = json.loads(capture_bytes, object_pairs_hook=reference.unique)
    need(type(capture) is dict and set(capture) == {'format', 'repository', 'revision',
         'source_manifest_sha256', 'device', 'declared_positions', 'implemented_positions',
         'profile', 'requested_chunks', 'effective_chunk', 'weights_bytes', 'positional_bytes',
         'load_timing', 'maximum_admitted_bytes', 'application_allowance_bytes',
         'native_selected_on_teacher_forced_prefixes', 'array_bytes', 'arrays',
         'native_generated_ids', 'native_generation_scope', 'chunk_metrics', 'chunk_metric_scope',
         'held_out_quality_qualified', 'app_qualified', 'iphone_qualified',
         'numerical_comparison_pending'}, 'Native report fields differ')
    need(type(owned.get('exit_code')) is int and owned['exit_code'] == 0
         and owned.get('passed') is True and owned.get('owned_group_absent') is True
         and owned.get('failure') is None, 'Native owned job was not successful and closed')
    need(capture['format'] == 'toklet-minicpm5-checkpoint-capture-v1'
         and capture['repository'] == reference.REPO and capture['revision'] == reference.REVISION
         and capture['source_manifest_sha256'] == external_sha
         and capture['profile'] == 'MlxAffine/Compensated/Reference/Half'
         and type(capture['implemented_positions']) is int and capture['implemented_positions'] == 512
         and type(capture['declared_positions']) is int and capture['declared_positions'] == 131072
         and reference.same(capture['requested_chunks'], CHUNKS)
         and capture['effective_chunk'] == 'min(requested,512-position_offset)'
         and capture['device'] == 'Apple M1 Pro'
         and capture['native_generation_scope'] == 'four raw IDs; no text/stop/quality qualification'
         and capture['chunk_metric_scope'] == 'completed evaluate_next_chunk only; excludes diagnostic array copy/serialization; instrumented numerical diagnostic, not ordinary request/telemetry overhead'
         and capture['numerical_comparison_pending'] is True
         and all(capture[k] is False for k in ['held_out_quality_qualified', 'app_qualified', 'iphone_qualified']),
         'Fixed native identity/profile/qualification differs')
    need(type(capture['maximum_admitted_bytes']) is int and 0 < capture['maximum_admitted_bytes'] <= 6 * 1024 ** 3
         and type(capture['application_allowance_bytes']) is int and capture['application_allowance_bytes'] == 32 * 1024 ** 2
         and type(capture['weights_bytes']) is int and capture['weights_bytes'] == 2674218150
         and type(capture['positional_bytes']) is int and capture['positional_bytes'] == 0,
         'Native admission/ordinary positional owner differs')
    chunk_metrics = validate_chunk_metrics(capture['chunk_metrics'], owned)
    need(capture['maximum_admitted_bytes'] >= max(row['session_memory']['admitted_bytes']
         + 32 * 1024 ** 2 for row in capture['chunk_metrics']), 'Maximum admission undercounts chunk owners')
    validate_load_timing(capture['load_timing'], chunk_metrics['owned_elapsed_ns'],
                         chunk_metrics['completed_chunk_wall_nanoseconds'])
    expected = inventory()
    need(type(capture['arrays']) is list and len(capture['arrays']) == len(expected), 'Native array coverage differs')
    cursor = 0
    for item in capture['arrays']:
        need(type(item) is dict and set(item) == {'event', 'name', 'row_start', 'shape', 'dtype', 'offset', 'bytes', 'sha256'},
             'Native descriptor fields differ')
        need(type(item['row_start']) is int and item['event'] == 'completed', 'Native row/completion invalid')
        shape = expected.pop((item['name'], item['row_start']), None)
        need(shape is not None and reference.same(item['shape'], shape)
             and type(item['offset']) is int and item['offset'] == cursor
             and type(item['bytes']) is int and item['bytes'] == math.prod(shape) * 4
             and item['dtype'] == '<f4' and reference.digest(item['sha256']), 'Native descriptor layout differs')
        cursor += item['bytes']
    need(not expected and cursor <= MAXIMUM and type(capture['array_bytes']) is int
         and cursor == capture['array_bytes'] == (folder / 'arrays.bin').stat().st_size, 'Native raw extent differs')
    with (folder / 'arrays.jsonl').open('rb') as stream:
        journal_raw = stream.read(8 * 1024 ** 2 + 1)
    need(len(journal_raw) <= 8 * 1024 ** 2, 'Oversize native journal')
    journal = [json.loads(row, object_pairs_hook=reference.unique) for row in journal_raw.splitlines()]
    need(len(journal) == 2 * len(capture['arrays']), 'Incomplete native journal')
    for i, item in enumerate(capture['arrays']):
        start = dict(event='started', name=item['name'], row_start=item['row_start'],
                     shape=item['shape'], offset=item['offset'], expected_bytes=item['bytes'])
        need(reference.same(journal[2 * i], start) and reference.same(journal[2 * i + 1], item), 'Journal/final descriptor differs')
    metrics, logits, observed, failures = [], {}, 0, []
    source_streams = {p: (source / (p + '.bin')).open('rb') for p in ['fp32', 'fp64']}
    try:
        with (folder / 'arrays.bin').open('rb') as stream:
            for item in capture['arrays']:
                raw = stream.read(item['bytes'])
                need(len(raw) == item['bytes'] and hashlib.sha256(raw).hexdigest() == item['sha256'], 'Native array digest differs')
                actual = np.frombuffer(raw, dtype='<f4').reshape(item['shape'])
                need(np.isfinite(actual).all(), 'Nonfinite native array')
                name = key = item['name']
                if key.startswith('chunk'):
                    schedule, key = key.split('.', 1)
                    if key.endswith('.logits'):
                        logits.setdefault((key.split('.')[1], schedule), []).append(actual.copy())
                elif key.endswith('.logits'):
                    logits[(key, '')] = actual.copy()
                if key.startswith(('cases.', 'greedy.', 'scoring.')):
                    for profile in ['fp32', 'fp64']:
                        spec = oracle['arrays'][profile + '.' + key]
                        columns = math.prod(spec['shape'][1:]) if len(spec['shape']) > 1 else spec['shape'][0]
                        width = np.dtype(spec['dtype']).itemsize
                        source_streams[profile].seek(spec['offset'] + item['row_start'] * columns * width)
                        data = source_streams[profile].read(actual.size * width)
                        need(len(data) == actual.size * width, 'Source slice truncated')
                        target = np.frombuffer(data, dtype=spec['dtype']).reshape(actual.shape).astype(np.float64)
                        error = np.abs(actual.astype(np.float64) - target)
                        gate = oracle['gates'][spec['gate_group']]
                        limits = gate['atol'] + gate['rtol'] * np.abs(target)
                        bad = error > limits
                        row = dict(name=name, row_start=item['row_start'], profile=profile,
                                   values=actual.size, failed=int(np.count_nonzero(bad)),
                                   max_absolute_error=float(np.max(error)), max_gate_fraction=float(np.max(error / limits)))
                        metrics.append(row)
                        if row['failed'] and len(failures) < 20:
                            coordinate = np.argwhere(bad)[0]
                            r, c = int(coordinate[0]), int(coordinate[1])
                            failures.append(dict(name=name, profile=profile, row=item['row_start'] + r,
                                column=c, actual=float(actual[r, c]), target=float(target[r, c]), limit=float(limits[r, c])))
                else:
                    observed += actual.size
            need(not stream.read(1), 'Trailing native data')
    finally:
        for stream in source_streams.values():
            stream.close()
    schedules = True
    for case in reference.CASES:
        full = np.concatenate(logits[(case, 'chunk32')])
        for chunk in [1, 3]:
            schedules &= bool(np.array_equal(np.concatenate(logits[(case, f'chunk{chunk}')]), full))
    causal = all(np.array_equal(np.concatenate(logits[('short', f'chunk{chunk}')])[:3],
                                np.concatenate(logits[('future_changed', f'chunk{chunk}')])[:3]) for chunk in CHUNKS)
    boundary = np.array_equal(logits[('boundary.chunk3.logits', '')], logits[('boundary.chunk32.logits', '')])
    initial = np.array_equal(logits[('greedy.step_0.logits', '')], logits[('native_generated.step_0.logits', '')])
    selections = True
    for group, field in [('greedy', 'native_selected_on_teacher_forced_prefixes'), ('native_generated', 'native_generated_ids')]:
        selected = [int(np.argmax(logits[(f'{group}.step_{i}.logits', '')])) for i in range(4)]
        selections &= reference.same(capture[field], selected)
    for profile in ['fp32', 'fp64']:
        need(reference.model.sha(source / (profile + '.bin')) == oracle['files'][profile + '.bin']['sha256'], 'Source bits changed during comparison')
    failed = sum(row['failed'] for row in metrics)
    return dict(format='toklet-minicpm5-short-comparison-v1', passed=failed == 0 and schedules and causal and boundary and initial and selections,
        source_manifest_sha256=external_sha, arrays=len(capture['arrays']), source_compared_arrays=len(metrics),
        compared_values=sum(row['values'] for row in metrics), failed_values=failed,
        max_absolute_error=max(row['max_absolute_error'] for row in metrics),
        max_gate_fraction=max(row['max_gate_fraction'] for row in metrics), first_failures=failures,
        exact_numeric_logits_schedules=schedules, exact_causal_prefix=causal, exact_boundary_schedules=bool(boundary),
        exact_initial_distribution=bool(initial), exact_recorded_native_selections=selections,
        raw_state_observed_values=observed, chunk_metrics=chunk_metrics, metrics=metrics,
        full_quality_qualified=False, physical_iphone_qualified=False, managed_adoption_qualified=False,
        scope='Both original CPU profile gates; full512 boundary is functional state only, not source numerical or adoption quality')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('source_sha')
    parser.add_argument('capture', type=Path)
    parser.add_argument('owned', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    result = check(args.source, args.source_sha, args.capture, bounded_json(args.owned, 1024 ** 2))
    with args.output.open('x') as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({key: result[key] for key in ['passed', 'compared_values', 'failed_values', 'max_gate_fraction']}))
    sys.exit(0 if result['passed'] else 1)

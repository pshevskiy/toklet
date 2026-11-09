"""Strict offline reader for the fixed MiniCPM5 raw request experiment."""
import math
import hashlib
import json
from pathlib import Path

REPOSITORY = 'bumblebuttpow/MiniCPM5-2B-heretic-abliterated-MLX-8bit'
REVISION = '50ab1b226834c290cdf68e460eb899e5cff7c61d'
PROMPT = [0, 7, 13, 3]
PROFILE = 'MlxAffine/Compensated/Reference/Half'
TIMING_FIELDS = ['command_creation_seconds', 'clock_calibration_seconds', 'commit_seconds',
                 'host_encode_seconds', 'wait_seconds', 'gpu_start_seconds', 'gpu_end_seconds', 'readback_seconds']


def need(value, message):
    if not value:
        raise ValueError(message)


def finite(value):
    if type(value) not in (int, float):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


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


def exact(actual, expected):
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return set(actual) == set(expected) and all(exact(actual[k], expected[k]) for k in expected)
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(exact(a, b) for a, b in zip(actual, expected))
    return actual == expected


def interval(owned):
    need(type(owned) is dict and type(owned.get('started_ns')) is int
         and type(owned.get('completed_ns')) is int
         and 0 < owned['started_ns'] < owned['completed_ns'], 'Invalid positive monotonic owned interval')
    need(owned.get('passed') is True and type(owned.get('exit_code')) is int
         and owned['exit_code'] == 0 and owned.get('failure') is None
         and owned.get('owned_group_absent') is True, 'Owned job is not successful and closed')
    return owned['completed_ns'] - owned['started_ns']


def unique(pairs):
    result = {}
    for key, value in pairs:
        need(key not in result, 'Duplicate benchmark JSON field')
        result[key] = value
    return result


def read_report(path, owned):
    with open(path, 'rb') as stream:
        raw = stream.read(2 * 1024 ** 2 + 1)
    need(len(raw) <= 2 * 1024 ** 2 and hashlib.sha256(raw).hexdigest()
         == owned.get('benchmark_report_sha256'), 'Benchmark report differs from completed owned receipt')
    return json.loads(raw, object_pairs_hook=unique)


def validate(report, owned, mode, expected_ids, expected_memory, source_sha):
    elapsed = interval(owned)
    need(type(source_sha) is str and len(source_sha) == 64
         and all(c in '0123456789abcdef' for c in source_sha), 'Source manifest SHA invalid')
    need(mode in ['full-logits', 'greedy-token'], 'Unexpected readback control')
    need(type(expected_ids) is list and len(expected_ids) == 4
         and all(type(v) is int and 0 <= v < 130560 for v in expected_ids), 'Expected native history invalid')
    need(type(report) is dict and set(report) == {'format', 'repository', 'revision', 'source_manifest_sha256',
         'device', 'profile', 'readback', 'context_capacity', 'prefill_chunk', 'position_offset', 'prompt',
         'teacher_tokens', 'warmups', 'measured_samples', 'cache_state', 'stop_policy', 'load_timing',
         'session_memory', 'application_allowance_bytes', 'request_clock_scope', 'decision_ttft_scope',
         'diagnostics_enabled', 'operator_counts_available', 'allocation_count', 'user_visible_ttft_nanoseconds',
         'samples', 'model_quality_qualified', 'managed_adoption_qualified',
         'ordinary_telemetry_overhead_qualified'}, 'Benchmark report fields differ')
    need(report['format'] == 'toklet-minicpm5-raw-request-bench-v1'
         and report['repository'] == REPOSITORY and report['revision'] == REVISION
         and report['source_manifest_sha256'] == source_sha and report['device'] == 'Apple M1 Pro'
         and report['profile'] == PROFILE and report['readback'] == mode
         and exact(report['prompt'], PROMPT) and exact(report['teacher_tokens'], expected_ids),
         'Benchmark identity/profile/workload differs')
    for key, expected in [('context_capacity', 512), ('prefill_chunk', 32), ('position_offset', 0),
                          ('warmups', 3), ('measured_samples', 10), ('application_allowance_bytes', 33554432)]:
        need(type(report[key]) is int and report[key] == expected, 'Benchmark capacity/work count differs')
    for field in ['diagnostics_enabled', 'operator_counts_available', 'model_quality_qualified',
                  'managed_adoption_qualified', 'ordinary_telemetry_overhead_qualified']:
        need(report[field] is False, 'Benchmark relabels diagnostics or qualification')
    need(report['allocation_count'] is None and report['user_visible_ttft_nanoseconds'] is None,
         'Unavailable observation was fabricated')
    need(report['cache_state'] == 'loaded-model;reset-empty-history-per-request;warm-filesystem'
         and report['stop_policy'] == 'fixed-four-teacher-tokens-including-stop-id;last-token-pending'
         and report['request_clock_scope'] == 'set_tokens through final pending append; completed evaluation and selection; excludes reset and JSON'
         and report['decision_ttft_scope'] == 'internal raw token decision; not streamed text or visible UI',
         'Benchmark cache/stop/clock scope differs')
    memory = report['session_memory']
    need(exact(memory, expected_memory) and all(type(v) is int and v >= 0 for v in memory.values())
         and memory['admitted_bytes'] == sum(v for k, v in memory.items() if k != 'admitted_bytes')
         and memory['admitted_bytes'] + 33554432 <= 6442450944, 'Session owner/admission ledger differs')
    need(type(report['samples']) is list and len(report['samples']) == 13, 'Incomplete request coverage')
    derived, total_request = [], 0
    for index, sample in enumerate(report['samples']):
        need(type(sample) is dict and set(sample) == {'sample', 'warmup', 'request_nanoseconds', 'steps',
             'committed_after', 'pending_after', 'history_after', 'outcome'}, 'Request fields differ')
        need(type(sample['sample']) is int and sample['sample'] == index
             and sample['warmup'] is (index < 3)
             and sample['outcome'] == 'fixed_teacher_continuation_completed', 'Request order/outcome differs')
        for key, value in [('committed_after', 7), ('pending_after', 1), ('history_after', 8)]:
            need(type(sample[key]) is int and sample[key] == value, 'Final sampled/committed state differs')
        request = sample['request_nanoseconds']
        need(type(request) is int and 0 < request <= elapsed, 'Request clock invalid')
        need(type(sample['steps']) is list and len(sample['steps']) == 4, 'Incomplete phase coverage')
        previous, phase_total, sampling_total, gpu_total, commands = 0, 0, 0, 0., 0
        for step, row in enumerate(sample['steps']):
            need(type(row) is dict and set(row) == {'step', 'rows', 'committed_after', 'pending_after',
                 'wall_nanoseconds', 'sampling_nanoseconds', 'decision_nanoseconds', 'selected',
                 'full_logits_available', 'completed_timing'}, 'Phase fields differ')
            for key, value in [('step', step), ('rows', 4 if step == 0 else 1),
                               ('committed_after', 4 + step), ('pending_after', 0), ('selected', expected_ids[step])]:
                need(type(row[key]) is int and row[key] == value, 'Completed work/state/selection differs')
            need(row['full_logits_available'] is (mode == 'full-logits'), 'Readback availability differs')
            wall, sampling, event = (row[k] for k in ['wall_nanoseconds', 'sampling_nanoseconds', 'decision_nanoseconds'])
            need(type(wall) is int and wall > 0 and type(sampling) is int and sampling >= 0
                 and type(event) is int and previous < event <= request
                 and wall + sampling <= event - previous, 'Disjoint phase/selection/event clocks invalid')
            timing = row['completed_timing']
            need(type(timing) is dict and set(timing) == set(TIMING_FIELDS) | {'command_count'}
                 and all(finite(timing[k]) and timing[k] >= 0 for k in TIMING_FIELDS)
                 and type(timing['command_count']) is int and timing['command_count'] > 0,
                 'Completed GPU/host timing counters invalid')
            host_ns = [duration_ns(timing[key]) for key in TIMING_FIELDS if not key.startswith('gpu_')]
            need(sum(host_ns) <= wall, 'Disjoint host phases exceed completed wall clock')
            begin, end = timing['gpu_start_seconds'], timing['gpu_end_seconds']
            need(0 < begin < end and (end - begin) * 10 ** 9 <= wall, 'GPU timestamp coverage/clock invalid')
            previous = event
            phase_total += wall
            sampling_total += sampling
            gpu_total += end - begin
            commands += timing['command_count']
        need(phase_total + sampling_total <= request, 'Disjoint request clocks exceed request lifetime')
        events = [row['decision_nanoseconds'] for row in sample['steps']]
        derived.append(dict(sample=index, warmup=index < 3, request_ns=request,
            decision_ttft_ns=events[0], decode_interval_ns=events[-1] - events[0],
            decode_tokens_per_second=3 * 10 ** 9 / (events[-1] - events[0]),
            prefill_tokens_per_second=4 * 10 ** 9 / sample['steps'][0]['wall_nanoseconds'],
            completed_phase_ns=phase_total, selection_accessor_ns=sampling_total,
            gpu_seconds=gpu_total, command_count=commands,
            per_depth_wall_ns=[row['wall_nanoseconds'] for row in sample['steps']]))
        total_request += request
    need(total_request <= elapsed, 'Disjoint request clocks exceed owned process lifetime')
    validate_load_timing(report['load_timing'], elapsed, total_request)
    return dict(all_samples=derived, measured=[row for row in derived if not row['warmup']],
                completed_requests=13, measured_requests=10, warmup_requests=3,
                completed_prompt_tokens=52, completed_decode_tokens=39, sampled_tokens=52,
                total_request_ns=total_request, owned_elapsed_ns=elapsed,
                ordinary_telemetry_overhead_qualified=False, model_quality_adoption_qualified=False)


if __name__ == '__main__':
    import argparse
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('report', type=Path)
    cli.add_argument('owned', type=Path)
    cli.add_argument('contract', type=Path)
    cli.add_argument('mode', choices=['full-logits', 'greedy-token'])
    cli.add_argument('output', type=Path)
    args = cli.parse_args()
    def metadata(path):
        with path.open('rb') as stream:
            data = stream.read(2 * 1024 ** 2 + 1)
        need(len(data) <= 2 * 1024 ** 2, 'Oversize owned/contract metadata')
        return json.loads(data, object_pairs_hook=unique)
    owner, contract = metadata(args.owned), metadata(args.contract)
    result = validate(read_report(args.report, owner), owner, args.mode,
                      contract['teacher_tokens'], contract['expected_session_memory'][args.mode],
                      contract['source_manifest_sha256'])
    # Power/thermal eligibility belongs to the separate raw environment envelope.
    result['environment_evaluated'] = False
    with args.output.open('x') as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(validated_requests=13, environment_evaluated=False,
                          model_quality_adoption_qualified=False)))

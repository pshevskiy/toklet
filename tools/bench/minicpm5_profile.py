"""Validate and rank bounded, instrumented MiniCPM5 GPU stage observations."""
import hashlib
import json
import math
from pathlib import Path
import statistics

from minicpm5_performance import (PROMPT, REPOSITORY, REVISION, TIMING_FIELDS, duration_ns, exact, finite,
                                  interval, need, unique, validate_load_timing)

ROLES = [('input_norm', 'rms'), ('q', 'mlx_product'), ('k', 'mlx_product'),
         ('v', 'mlx_product'), ('q_rotary', 'rotary'), ('k_rotary', 'rotary'),
         ('cache', 'cache'), ('attention', 'attention'),
         ('attention_projection', 'mlx_product'), ('attention_residual', 'residual'),
         ('post_attention_norm', 'rms'), ('gate', 'mlx_product'), ('silu', 'silu'),
         ('up', 'mlx_product'), ('gated', 'gating'), ('down', 'mlx_product'),
         ('output_residual', 'residual')]
INVENTORY = [(None, 'embedding', 'mlx_gather')]
INVENTORY += [(layer, role, kernel) for layer in range(42) for role, kernel in ROLES]
INVENTORY += [(None, 'final_norm', 'rms'), (None, 'head', 'mlx_product'),
              (None, 'validate', 'validate_float')]


BASE_MEMORY = dict(weights_bytes=2674218150, kv_bytes=22021272, scratch_bytes=21478868,
                   history_bytes=2048, logits_bytes=1044480, metadata_bytes=5184,
                   metadata_allowance_bytes=67108864, admitted_bytes=2785878866)
PROFILE_MEMORY = dict(BASE_MEMORY, metadata_allowance_bytes=134217728, admitted_bytes=2852987730)


def validate_contract(contract):
    need(type(contract) is dict, 'Profile contract must be a typed object')
    fixed = dict(format='toklet-minicpm5-profile-contract-v1',
        engine_base='d7ed9038aec127661b5998579712dad1d5e7908f', repository=REPOSITORY, revision=REVISION,
        source_manifest_sha256='360cf36c6f743fb4af37c07dbd0e0c78d22e0c9922653629f0707131c88657b3',
        prompt=PROMPT, teacher_tokens=[6, 978, 44004, 130073], capacity=512, prefill_chunk=32,
        profile='MlxAffine/Compensated/Reference/Half/FullLogits',
        cohorts=['baseline-first', 'profile-first'], warmups=3, measured_samples=10, steps=4,
        events_per_phase=718, exact_logits_values_per_cohort=6789120,
        expected_baseline_memory=BASE_MEMORY, expected_profile_memory=PROFILE_MEMORY,
        combined_admitted_bytes=2998202878, written_before_first_native_profile=True)
    need(all(key in contract for key in fixed)
         and exact({key: contract[key] for key in fixed}, fixed), 'Fixed prospective profile contract differs')
    need(type(contract.get('expected_logits_sha256')) is list and len(contract['expected_logits_sha256']) == 4
         and all(digest(v) for v in contract['expected_logits_sha256']), 'Prior numerical capture hashes invalid')


def digest(value):
    return type(value) is str and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def uint(value):
    return type(value) is int and 0 <= value < 2 ** 64


def timestamp_grain_ns(begin, finish):
    grain = .5 * (math.ulp(begin) + math.ulp(finish))
    need(finite(grain) and 0 <= grain <= 1022e-9,
         'GPU command timestamp grain exceeds bounded conversion policy')
    return math.ceil(grain * 10 ** 9) + 2


def timing(row):
    need(type(row) is dict and set(row) == {'wall_nanoseconds', 'completed_timing'}, 'Phase fields differ')
    wall, raw = row['wall_nanoseconds'], row['completed_timing']
    need(uint(wall) and wall > 0, 'Completed wall clock invalid')
    need(type(raw) is dict and set(raw) == set(TIMING_FIELDS) | {'command_count'}
         and all(finite(raw[k]) and raw[k] >= 0 for k in TIMING_FIELDS)
         and type(raw['command_count']) is int and raw['command_count'] == 1, 'Completed timing fields invalid')
    host = sum(duration_ns(raw[k]) for k in TIMING_FIELDS if not k.startswith('gpu_'))
    begin, end = raw['gpu_start_seconds'], raw['gpu_end_seconds']
    need(host <= wall and 0 < begin < end and duration_ns(end - begin) <= wall,
         'Completed host/GPU interval exceeds phase wall clock')
    return wall


def counters(row, wall):
    start, end = row['clock_start'], row['clock_end']
    for pair in [start, end]:
        need(type(pair) is dict and set(pair) == {'cpu_nanoseconds', 'gpu_ticks'}
             and all(uint(v) and v > 0 for v in pair.values()), 'Calibration pair invalid')
    cpu = end['cpu_nanoseconds'] - start['cpu_nanoseconds']
    gpu = end['gpu_ticks'] - start['gpu_ticks']
    need(0 < cpu <= wall and gpu > 0, 'Calibration span exceeds completed wall')
    bracket = duration_ns(row['host_clock_sample_bracket_seconds'])
    need(bracket <= wall and type(row['stages']) is list and len(row['stages']) == 718,
         'Counter coverage/bracket differs')
    completed = row['instrumented']['completed_timing']
    # A rounded endpoint loses <=0.5 ns; each interval has <=1 ns grain.
    need(bracket <= duration_ns(completed['clock_calibration_seconds']) + 1,
         'Initial sample bracket exceeds completed clock calibration')
    previous_host, previous_gpu = bracket, start['gpu_ticks']
    roles, ticks, host_sum = {}, 0, 0
    for index, (raw, expected) in enumerate(zip(row['stages'], INVENTORY)):
        layer, role, kernel = expected
        need(type(raw) is dict and set(raw) == {'index', 'layer', 'role', 'sample'}
             and exact({k: raw[k] for k in ['index', 'layer', 'role']},
                       dict(index=index, layer=layer, role=role)), 'Stage identity/coverage differs')
        event = raw['sample']
        need(type(event) is dict and set(event) == {'name', 'host_begin_seconds', 'host_end_seconds',
             'gpu_begin_ticks', 'gpu_end_ticks', 'gpu_begin_seconds', 'gpu_end_seconds'}
             and event['name'] == kernel, 'Stage kernel inventory differs')
        hb, he = (duration_ns(event[k]) for k in ['host_begin_seconds', 'host_end_seconds'])
        gb, ge = event['gpu_begin_ticks'], event['gpu_end_ticks']
        need(gb != 2 ** 64 - 1 and ge != 2 ** 64 - 1, 'Metal counter-error sentinel')
        need(previous_host <= hb <= he <= wall and uint(gb) and uint(ge)
             and previous_gpu <= gb <= ge <= end['gpu_ticks'], 'Stage clocks overlap, reorder or exceed calibration')
        for tick, key in [(gb, 'gpu_begin_seconds'), (ge, 'gpu_end_seconds')]:
            value = event[key]
            # Reproduce the public paired-clock conversion, permitting only
            # bounded floating arithmetic grain, not uncalibrated GPU ns.
            expected_seconds = (tick - start['gpu_ticks']) / gpu * cpu / 10 ** 9
            need(finite(value) and abs(value - expected_seconds) <= 8 * math.ulp(expected_seconds) + 1e-15,
                 'Exported GPU seconds differ from paired-clock conversion')
        host_sum += he - hb
        ticks += ge - gb
        roles[role] = roles.get(role, 0) + ge - gb
        previous_host, previous_gpu = he, ge
    need(ticks <= gpu, 'Disjoint stage ticks exceed calibration span')
    need(host_sum <= duration_ns(completed['host_encode_seconds']) + len(INVENTORY) + 1,
         'Host stage durations exceed completed encode interval')
    host_span = duration_ns(row['stages'][-1]['sample']['host_end_seconds']) - duration_ns(
        row['stages'][0]['sample']['host_begin_seconds'])
    need(host_span <= duration_ns(completed['host_encode_seconds']) + 2,
         'Host stage span and gaps exceed completed encode interval')
    # Round the sum once from exact integer tick/paired-CPU ratios. Propagate
    # both absolute command-time binary64 ULPs; reject imprecise epochs rather
    # than allowing an unbounded tolerance. GPU and host scopes still overlap.
    begin, finish = completed['gpu_start_seconds'], completed['gpu_end_seconds']
    gpu_grain_ns = timestamp_grain_ns(begin, finish)
    stage_ns = (2 * ticks * cpu + gpu) // (2 * gpu)
    need(stage_ns <= duration_ns(finish - begin) + gpu_grain_ns,
         'Sampled GPU stages exceed completed command interval')
    return dict(role_gpu_seconds={key: value / gpu * cpu / 10 ** 9 for key, value in roles.items()},
                sampled_stage_gpu_seconds=ticks / gpu * cpu / 10 ** 9,
                calibration_span_ns=cpu, events=718)


def validate(records, owned, contract, first_order):
    validate_contract(contract)
    elapsed = interval(owned)
    need(first_order in ['baseline-first', 'profile-first'], 'Profile order invalid')
    need(type(records) is list and len(records) == 54, 'Profile record coverage differs')
    header = records[0]
    need(type(header) is dict and set(header) == {'event', 'format', 'repository', 'revision',
         'source_manifest_sha256', 'device', 'profile', 'capacity', 'prefill_chunk', 'prompt',
         'teacher_tokens', 'first_order', 'warmups', 'measured_samples', 'events_per_phase',
         'baseline_memory', 'profile_memory', 'shared_weights_bytes', 'application_allowance_bytes',
         'combined_admitted_bytes', 'load_timing', 'ordinary_request_clock_available', 'clock_scope'},
         'Profile header fields differ')
    fixed = dict(event='ready', format='toklet-minicpm5-profile-v1', repository=contract['repository'],
        revision=contract['revision'], source_manifest_sha256=contract['source_manifest_sha256'],
        device='Apple M1 Pro', profile=contract['profile'], capacity=512, prefill_chunk=32,
        prompt=contract['prompt'], teacher_tokens=contract['teacher_tokens'], first_order=first_order,
        warmups=3, measured_samples=10, events_per_phase=718,
        baseline_memory=contract['expected_baseline_memory'], profile_memory=contract['expected_profile_memory'],
        shared_weights_bytes=2674218150, application_allowance_bytes=33554432,
        combined_admitted_bytes=contract['combined_admitted_bytes'], ordinary_request_clock_available=False,
        clock_scope='completed evaluate_next_chunk only; excludes comparison, selection accessor, counter resolve and JSON')
    need(exact({k: header[k] for k in fixed}, fixed), 'Profile identity, policy or owner ledger differs')
    need(0 < header['combined_admitted_bytes'] <= 6442450944
         and header['combined_admitted_bytes'] == header['baseline_memory']['admitted_bytes']
         + header['profile_memory']['admitted_bytes'] - header['shared_weights_bytes'] + 33554432,
         'Combined admission double-counts shared weights or underreports sessions')
    derived, total = [], 0
    previous_cpu, previous_gpu, previous_command_gpu_end = 0, 0, 0.
    first_cpu, first_command_gpu = None, None
    for index, row in enumerate(records[1:-1]):
        sample, step = divmod(index, 4)
        need(type(row) is dict and set(row) == {'event', 'sample', 'warmup', 'step', 'rows', 'profile_first',
             'baseline', 'instrumented', 'committed_after', 'pending_after', 'history_after', 'selected',
             'all_logits_exact', 'logits_sha256', 'clock_start', 'clock_end',
             'host_clock_sample_bracket_seconds', 'stages'}, 'Profile phase fields differ')
        expected = dict(event='phase', sample=sample, step=step, warmup=sample < 3, rows=4 if step == 0 else 1,
            profile_first=(first_order == 'profile-first') ^ ((sample + step) % 2 == 1),
            committed_after=4 + step, pending_after=0, history_after=4 + step,
            selected=contract['teacher_tokens'][step], all_logits_exact=True,
            logits_sha256=contract['expected_logits_sha256'][step])
        need(exact({k: row[k] for k in expected}, expected), 'Profile work/state/parity differs from frozen capture')
        base, profiled = timing(row['baseline']), timing(row['instrumented'])
        observation = counters(row, profiled)
        need(row['clock_start']['cpu_nanoseconds'] >= previous_cpu
             and row['clock_start']['gpu_ticks'] >= previous_gpu,
             'Profile calibration origins overlap or reorder across serial phases')
        if first_cpu is None:
            first_cpu = row['clock_start']['cpu_nanoseconds']
        previous_cpu = row['clock_end']['cpu_nanoseconds']
        previous_gpu = row['clock_end']['gpu_ticks']
        order = ['instrumented', 'baseline'] if row['profile_first'] else ['baseline', 'instrumented']
        for execution in order:
            command = row[execution]['completed_timing']
            need(previous_command_gpu_end <= command['gpu_start_seconds'],
                 'Completed command timestamps overlap or reorder across serial phases')
            if first_command_gpu is None:
                first_command_gpu = command['gpu_start_seconds']
            previous_command_gpu_end = command['gpu_end_seconds']
        derived.append(dict(sample=sample, step=step, warmup=sample < 3, context=4 + step,
            baseline_wall_ns=base, instrumented_wall_ns=profiled, **observation))
        total += base + profiled
    need(exact(records[-1], dict(event='complete', phase_pairs=52, compared_logits_values=6789120,
         ordinary_telemetry_overhead_qualified=False, model_quality_adoption_qualified=False,
         engine_optimization_adopted=False)), 'Complete profile receipt missing or relabelled')
    validate_load_timing(header['load_timing'], elapsed, total)
    load_ns = duration_ns(header['load_timing']['load_seconds'])
    need(load_ns + previous_cpu - first_cpu <= elapsed + 1,
         'Disjoint load plus CPU calibration series exceeds owned lifetime')
    command_span_ns = duration_ns(previous_command_gpu_end - first_command_gpu)
    need(load_ns + command_span_ns <= elapsed + timestamp_grain_ns(first_command_gpu, previous_command_gpu_end),
         'Disjoint load plus completed GPU command series exceeds owned lifetime')
    measured = [row for row in derived if not row['warmup']]
    depths = []
    def stats(values):
        q = statistics.quantiles(values, n=4)
        return dict(median=statistics.median(values), iqr=q[2] - q[0], stdev=statistics.pstdev(values))
    for step in range(4):
        rows = [row for row in measured if row['step'] == step]
        roles = {role: stats([row['role_gpu_seconds'][role] for row in rows])
                 for role in rows[0]['role_gpu_seconds']}
        base = stats([row['baseline_wall_ns'] for row in rows])
        profiled = stats([row['instrumented_wall_ns'] for row in rows])
        depths.append(dict(context=step + 4, rows=4 if step == 0 else 1, baseline_wall_ns=base,
            instrumented_wall_ns=profiled, diagnostic_phase_inflation_percent=100 * (profiled['median'] / base['median'] - 1),
            roles_by_instrumented_gpu_cost=sorted(roles.items(), key=lambda item: item[1]['median'], reverse=True)))
    return dict(passed=True, phase_pairs=52, measured_phase_pairs=40, warmup_phase_pairs=12,
        events=37336, compared_logits_values=6789120, completed_disjoint_wall_ns=total,
        depths=depths, all_samples=derived,
        attribution_scope='sampled stage intervals in instrumented execution; gaps not attributed; no ordinary operator-cost extrapolation',
        environment_evaluated=False, ordinary_request_ttft_available=False,
        one_percent_telemetry_overhead_qualified=False, model_quality_adoption_qualified=False,
        engine_optimization_adopted=False)


def read(path, owned):
    records, total, digest_state = [], 0, hashlib.sha256()
    with Path(path).open('rb') as stream:
        while raw := stream.readline(2 * 1024 ** 2 + 1):
            total += len(raw)
            need(len(raw) <= 2 * 1024 ** 2 and total <= 64 * 1024 ** 2
                 and len(records) < 54 and raw.endswith(b'\n'), 'Profile byte/record bounds exceeded or line incomplete')
            digest_state.update(raw)
            records.append(json.loads(raw, object_pairs_hook=unique))
    need(digest(owned.get('profile_report_sha256')) and digest_state.hexdigest() == owned['profile_report_sha256'],
         'Profile bytes differ from completed owned receipt')
    return records


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['report', 'owned', 'contract', 'output']:
        parser.add_argument(name, type=Path)
    parser.add_argument('first_order', choices=['baseline-first', 'profile-first'])
    args = parser.parse_args()
    def metadata(path):
        with path.open('rb') as stream:
            raw = stream.read(1024 ** 2 + 1)
        need(len(raw) <= 1024 ** 2, 'Profile metadata too large')
        return json.loads(raw, object_pairs_hook=unique)
    owner = metadata(args.owned)
    result = validate(read(args.report, owner), owner, metadata(args.contract), args.first_order)
    with args.output.open('x') as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + '\n')
    for depth in result['depths']:
        role, value = depth['roles_by_instrumented_gpu_cost'][0]
        print(f"Context {depth['context']}: largest instrumented role {role} {1000*value['median']:.3f} ms; "
              f"diagnostic phase inflation {depth['diagnostic_phase_inflation_percent']:.2f}%")

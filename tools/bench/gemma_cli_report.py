#!/usr/bin/env python3
"""Validate one explicitly SHA-pinned ordinary Gemma CLI summary export.

This schema is separate from Swift summary bundles. The report validates retained
recorded facts, not model identity, numerical parity, runtime enforcement or an
overhead/adoption gate. No inference libraries or producer code are imported.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import statistics
import sys

LINE_LIMIT = 4096
INPUT_LIMIT = 1024 * 1024 + 2 * LINE_LIMIT
MEMORY_LIMIT = 6 * 1024**3
U64 = 2**64 - 1
COLLECTOR_FIXED_LIMIT = 16384
RETAINED_RING_CAPACITY = 1032192
COLLECTOR_OWNED_LIMIT = 1048576
TELEMETRY_ADMITTED_BYTES = 1064960
LOGICAL_SCOPE = ('Numerical component sum equals logical_memory.admitted_bytes. Whole session '
                 'admission is numerical plus additional owner/external admission; do not add '
                 'logical_memory twice. Admission is not measured process footprint.')
HEADER_FIXED = dict(
    collector_memory_scope='Logical Collector value plus retained ring capacity after construction, fixed until drop. Excludes allocator overhead, temporary stack/copy/export storage and native/OS memory; not RSS or process peak.',
    cancel_to_terminal_availability_reason='signal_timestamp_not_observed; cancellation_outcome_and_state_do_not_measure_signal_to_ack_latency',
    model_profile='gemma4-bf16', backend='metal', model_repository='google/gemma-4-E2B-it',
    model_revision='70af34e20bd4b7a91f0de6b22675850c43922a03', weight_dtype='bf16',
    weight_dtype_scope='Original BF16 matrix coefficients and file-backed PLE; resident learned vectors are FP32.',
    kv_dtype='fp32', sampling='greedy', thinking_enabled=False, instrumentation='summary',
    engine_revision=None, build_flags=None, device_identity=None, device_power=None,
    model_content_sha256=None, tokenizer_content_sha256=None,
    external_provenance_reason='external_capture_manifest_required',
    conversion_ms=None, compile_ms=None, process_start_to_ready_ms=None,
    conversion_availability_reason='not_separately_timed_within_owner_load',
    compile_availability_reason='context_compilation_overlaps_owner_load; phase_not_separately_timed',
    process_start_to_ready_availability_reason='process_start_not_observed; route_start_to_ready_ms_is_handled_route_clock',
    route_start_to_ready_scope='Handled Gemma route entry to ready; process startup and stdout/UI receipt excluded.',
    load_timing_scope='Completed OfficialBf16::load owner construction; context/pipeline and authenticated asset preparation overlap; conversion/compile components unavailable.')
OBSERVATIONS = set('gpu_ms host_submit_ms host_wait_ms command_count dispatch_count wait_count '
                   'readback_bytes buffer_growth_count pipeline_compile_count allocation_count '
                   'process_footprint_bytes ui_first_visible_ms energy_joules'.split())
AVAILABILITY = dict(ordinary_observations='not_observed_in_ordinary_request',
                    process_footprint_bytes='admission_is_not_sampled_process_footprint',
                    ui_first_visible_ms='stdout_publication_is_not_ui_visibility',
                    energy_joules='not_measured')
TIMES = set('request_total_ms native_total_ms tokenization_ms prefill_ms decode_ms sampling_ms '
            'ttft_event_ms first_text_publication_ms last_token_event_ms token_interval_count '
            'token_interval_min_ms token_interval_max_ms token_interval_sum_ms output_decode_ms '
            'output_publication_ms cancel_to_terminal_ms'.split())
WORK = set('prompt_tokens reused_prefix_tokens prefill_rows decode_evaluations sampled_tokens '
           'emitted_tokens callback_tokens replayed_rows'.split())
MEMORY = set('weights_owned_bytes kv_allocated_bytes scratch_bytes kv_valid_positions '
             'owner_reserved_bytes load_admitted_bytes session_admitted_bytes app_allowance_bytes '
             'stream_allowance_bytes telemetry_allowance_bytes helper_reserve_bytes'.split())
CORE_COUNTS = 'reused_prefix_tokens prefill_rows decode_evaluations sampled_tokens emitted_tokens'.split()
CORE_TIMES = 'native_total_ms prefill_ms decode_ms sampling_ms'.split()
DERIVED = 'prefill_tokens_per_s decode_tokens_per_s decode_evaluations_per_s token_interval_mean_ms'.split()
OUTCOMES = set('success stop cancelled rejected execution_error output_error'.split())
RECORD = set(('event schema_version session_id request_id attempt_id owner_generation request_kind '
              'terminal_status failed_phase error_category cache_action replay_required before after '
              'work timing memory ordinary_observations availability').split()) | set(DERIVED)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def keys(value, expected, name):
    require(type(value) is dict and set(value) == set(expected), name + ': fields differ')


def integer(value, name, maximum=U64):
    require(type(value) is int and 0 <= value <= maximum, name + ': unsigned integer required')
    return value


def number(value, name, optional=False):
    if value is None and optional:
        return None
    require(type(value) in (int, float), name + ': number required')
    try:
        good = math.isfinite(value) and value >= 0
    except OverflowError:
        good = False
    require(good, name + ': finite nonnegative number required')
    return value


def close(actual, expected, name):
    if expected is None:
        require(actual is None, name + ': unavailable rate must be null')
    else:
        number(actual, name)
        # Decimal JSON arithmetic only; this is not a model numerical tolerance.
        require(math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-9), name + ': arithmetic differs')


def unique(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, 'duplicate JSON key: ' + key)
        result[key] = value
    return result


def parse(raw):
    return json.loads(raw, object_pairs_hook=unique,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError('nonfinite JSON')))


def identity(value):
    require(type(value) is str and re.fullmatch(r'[A-Za-z0-9_.:-]{1,64}', value) is not None,
            'invalid session identity')


def endpoint(st):
    return st.st_dev, st.st_ino, st.st_mode, st.st_size, st.st_mtime_ns, st.st_ctime_ns


def read_pinned(path, expected_sha256):
    require(type(expected_sha256) is str and re.fullmatch('[0-9a-f]{64}', expected_sha256) is not None,
            'explicit lowercase input SHA256 required')
    require(hasattr(os, 'O_NOFOLLOW') and hasattr(os, 'O_NONBLOCK'), 'secure file admission unsupported')
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(descriptor, 'rb') as source:
        before = os.fstat(source.fileno())
        require(stat.S_ISREG(before.st_mode) and before.st_size <= INPUT_LIMIT,
                'input must be a bounded regular file')
        raw = source.read(INPUT_LIMIT + 1)
        require(len(raw) == before.st_size and len(raw) <= INPUT_LIMIT, 'input EOF/size differs')
        require(endpoint(before) == endpoint(os.fstat(source.fileno())) ==
                endpoint(os.stat(path, follow_symlinks=False)), 'input identity changed')
    require(hashlib.sha256(raw).hexdigest() == expected_sha256, 'input SHA256 differs')
    require(raw.endswith(b'\n'), 'missing terminal newline')
    return raw


def state(value, capacity):
    keys(value, ('history_tokens', 'committed_tokens', 'pending_tokens'), 'state')
    for key, count in value.items():
        integer(count, key, capacity)
    require(value['history_tokens'] == value['committed_tokens'] + value['pending_tokens'],
            'history must equal committed plus pending')


def manifest(value):
    fields = set(('event', 'schema_version', 'session_id', 'capacity', 'prefill_chunk_size',
                 'submission', 'load_ms', 'route_start_to_ready_ms', 'load_admitted_bytes',
                 'session_admitted_bytes', 'logical_memory', 'numerical_session_admitted_bytes',
                 'additional_owner_and_external_admitted_bytes', 'logical_memory_scope',
                 'collector_fixed_bytes', 'retained_ring_capacity_bytes',
                 'collector_peak_bytes', 'telemetry_admitted_bytes'))
    keys(value, fields | set(HEADER_FIXED) | {'engine_crate_version'}, 'manifest')
    require(value['event'] == 'request_metrics_manifest' and type(value['schema_version']) is int
            and value['schema_version'] == 1, 'unsupported manifest schema')
    for key, expected in HEADER_FIXED.items():
        require(type(value[key]) is type(expected) and value[key] == expected,
                key + ': fixed route,clock or unavailable provenance differs')
    version = value['engine_crate_version']
    require(type(version) is str and len(version) <= 64 and
            re.fullmatch(r'(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)'
                         r'(?:-[0-9A-Za-z.-]+)?(?:\+[0-9A-Za-z.-]+)?', version) is not None,
            'invalid engine crate version')
    fixed = integer(value['collector_fixed_bytes'], 'collector fixed bytes', COLLECTOR_FIXED_LIMIT)
    require(fixed > 0, 'collector fixed value is empty')
    ring = integer(value['retained_ring_capacity_bytes'], 'retained ring capacity')
    require(ring in (0, RETAINED_RING_CAPACITY), 'retained ring capacity differs')
    peak = integer(value['collector_peak_bytes'], 'collector owned peak', COLLECTOR_OWNED_LIMIT)
    admitted = integer(value['telemetry_admitted_bytes'], 'telemetry admission')
    require(peak == fixed + ring, 'collector fixed/ring/peak sum differs')
    require(admitted == TELEMETRY_ADMITTED_BYTES and peak <= admitted,
            'collector storage exceeds or changes telemetry admission')
    identity(value['session_id'])
    require(1 <= integer(value['capacity'], 'capacity', 512), 'capacity is zero')
    require(1 <= integer(value['prefill_chunk_size'], 'chunk', 32) <= value['capacity'], 'chunk differs')
    require(value['submission'] in ('per-tile', 'batched-single-row'), 'unknown submission')
    require(number(value['load_ms'], 'load') <= number(value['route_start_to_ready_ms'], 'ready'),
            'load exceeds route readiness')
    for key in ('load_admitted_bytes', 'session_admitted_bytes'):
        integer(value[key], key, MEMORY_LIMIT)
    logical = value['logical_memory']
    required = set(('weights_bytes kv_bytes scratch_bytes history_bytes logits_bytes metadata_bytes '
                    'metadata_allowance_bytes admitted_bytes').split())
    require(type(logical) is dict and required <= set(logical) <= required | {'positional_bytes'},
            'logical memory owner inventory differs')
    for key, amount in logical.items():
        integer(amount, key, MEMORY_LIMIT)
    numerical = integer(value['numerical_session_admitted_bytes'], 'numerical admission', MEMORY_LIMIT)
    additional = integer(value['additional_owner_and_external_admitted_bytes'], 'additional admission', MEMORY_LIMIT)
    require(sum(amount for key, amount in logical.items() if key != 'admitted_bytes') ==
            logical['admitted_bytes'] == numerical, 'logical memory sum differs')
    require(numerical + additional == value['session_admitted_bytes'], 'whole session sum differs')
    require(value['logical_memory_scope'] == LOGICAL_SCOPE, 'logical memory scope differs')


def rate(count, duration):
    if count is None or duration is None or count <= 0 or duration <= 0:
        return None
    value = 1000 * count / duration
    return value if math.isfinite(value) else None


def record(value, header):
    keys(value, RECORD, 'record')
    require(value['event'] == 'request_summary' and type(value['schema_version']) is int
            and value['schema_version'] == 1, 'unsupported record schema')
    require(value['session_id'] == header['session_id'], 'record belongs to another session')
    for key in ('request_id', 'attempt_id', 'owner_generation'):
        require(integer(value[key], key) > 0, key + ': zero identity')
    require(value['owner_generation'] == 1, 'immutable owner generation differs')
    require(value['request_kind'] in ('generate', 'append', 'continue', 'reset', 'invalid'), 'request kind')
    outcome = value['terminal_status']
    require(outcome in OUTCOMES, 'terminal status')
    require(type(value['replay_required']) is bool, 'replay flag must be boolean')
    require(value['cache_action'] in ('fresh', 'exact_prefix', 'append', 'fully_cached', 'pending',
                                      'replay', 'reset', 'unchanged'), 'cache action')
    if outcome in ('success', 'stop', 'cancelled'):
        require(value['failed_phase'] is None and value['error_category'] is None, 'success has error fields')
    else:
        require(value['failed_phase'] in ('parse', 'tokenization', 'admission', 'execution', 'output')
                and value['error_category'] in ('invalid_input', 'session', 'sampling', 'output'),
                'missing typed failure fields')
    if outcome == 'output_error':
        require(value['failed_phase'] == 'output' and value['error_category'] == 'output', 'output error category')
    for key in ('before', 'after'):
        state(value[key], header['capacity'])
    if outcome == 'rejected':
        require(value['before'] == value['after'], 'rejection mutates session state')
    work, timing, memory = value['work'], value['timing'], value['memory']
    keys(work, WORK, 'work')
    keys(timing, TIMES, 'timing')
    keys(memory, MEMORY, 'memory')
    for key, amount in work.items():
        if amount is not None:
            integer(amount, key)
    callbacks = integer(work['callback_tokens'], 'callbacks', header['capacity'])
    for key, amount in timing.items():
        if key == 'token_interval_count':
            integer(amount, key, header['capacity'])
        else:
            number(amount, key, optional=key not in ('request_total_ms', 'token_interval_sum_ms',
                                                   'output_decode_ms', 'output_publication_ms'))
    require(timing['cancel_to_terminal_ms'] is None,
            'signal timestamp is not observed; cancellation latency must remain null')
    known = timing['native_total_ms'] is not None
    require(all((work[key] is not None) == known for key in CORE_COUNTS) and
            all((timing[key] is not None) == known for key in CORE_TIMES), 'partial core report bundle')
    if known:
        require(work['prompt_tokens'] is not None, 'completed core report lacks logical prompt')
        if value['request_kind'] == 'continue':
            require(work['prompt_tokens'] == value['before']['history_tokens'],
                    'Continue logical prompt differs from existing history')
        if value['request_kind'] == 'append':
            require(work['prompt_tokens'] >= value['before']['history_tokens'],
                    'Append logical prompt omits existing history')
        evaluated_prompt = work['reused_prefix_tokens'] + work['prefill_rows']
        require(evaluated_prompt <= work['prompt_tokens'], 'prompt work exceeds logical input history')
        require(work['emitted_tokens'] == callbacks <= work['sampled_tokens'] <= callbacks + 1,
                'sampled/emitted/callback accounting differs')
        require(work['decode_evaluations'] <= work['sampled_tokens'], 'decode work exceeds sampled work')
        if outcome in ('success', 'stop'):
            require(evaluated_prompt == work['prompt_tokens'], 'completed prompt coverage differs')
            require(value['after']['history_tokens'] == work['prompt_tokens'] + work['sampled_tokens'],
                    'completed history differs from logical prompt plus sampled tokens')
            require(value['after']['pending_tokens'] == int(work['sampled_tokens'] > 0),
                    'completed pending sampled-token state differs')
            require(work['decode_evaluations'] == max(work['sampled_tokens'] - 1, 0),
                    'completed decode evaluation count differs')
        require(sum(timing[key] for key in ('prefill_ms', 'decode_ms', 'sampling_ms')) <=
                timing['native_total_ms'] + 1e-6, 'core phases exceed completed native total')
        require(timing['native_total_ms'] <= timing['request_total_ms'] + 1e-6,
                'completed native duration exceeds enclosing request')
    if work['replayed_rows'] is not None:
        require(known and work['replayed_rows'] <= work['prefill_rows'], 'replay work exceeds completed prefill')
    first, last = timing['ttft_event_ms'], timing['last_token_event_ms']
    count = timing['token_interval_count']
    require(count == max(callbacks - 1, 0), 'interval count differs from observed callbacks')
    if callbacks:
        require(first is not None and last is not None and first <= last <= timing['request_total_ms'],
                'callback clock order/enclosure differs')
    else:
        require(first is None and last is None, 'no callbacks have callback clocks')
    if count:
        minimum, maximum = timing['token_interval_min_ms'], timing['token_interval_max_ms']
        require(minimum is not None and maximum is not None and minimum <= maximum,
                'interval extrema unavailable or reversed')
        total = timing['token_interval_sum_ms']
        require(minimum * count <= total + 1e-6 and total <= maximum * count + 1e-6,
                'interval sum outside extrema bounds')
        close(total, last - first, 'interval sum')
    else:
        require(timing['token_interval_min_ms'] is None and timing['token_interval_max_ms'] is None
                and timing['token_interval_sum_ms'] == 0, 'no intervals must remain unavailable')
    for key in ('first_text_publication_ms', 'cancel_to_terminal_ms'):
        require(timing[key] is None or timing[key] <= timing['request_total_ms'], key + ': outside request')
    close(value['prefill_tokens_per_s'], rate(work['prefill_rows'], timing['prefill_ms']), 'prefill rate')
    close(value['decode_tokens_per_s'], rate(callbacks - 1, None if first is None else last - first),
          'callback decode rate')
    close(value['decode_evaluations_per_s'], rate(work['decode_evaluations'], timing['decode_ms']),
          'numerical decode rate')
    close(value['token_interval_mean_ms'], timing['token_interval_sum_ms'] / count if count else None,
          'interval mean')
    for key, amount in memory.items():
        if amount is not None:
            integer(amount, key, MEMORY_LIMIT)
    require(memory['kv_valid_positions'] == value['after']['committed_tokens'], 'KV-valid scope differs')
    for key in ('load_admitted_bytes', 'session_admitted_bytes'):
        require(memory[key] == header[key], key + ': manifest differs')
    for key, logical in (('weights_owned_bytes', 'weights_bytes'), ('kv_allocated_bytes', 'kv_bytes'),
                         ('scratch_bytes', 'scratch_bytes')):
        require(memory[key] is None or memory[key] == header['logical_memory'][logical], 'owner bytes differ')
    additional = sum(memory[k] for k in ('owner_reserved_bytes', 'app_allowance_bytes',
                                        'stream_allowance_bytes', 'telemetry_allowance_bytes',
                                        'helper_reserve_bytes'))
    require(additional == header['additional_owner_and_external_admitted_bytes'],
            'additional owner/external partition differs')
    require(memory['telemetry_allowance_bytes'] == header['telemetry_admitted_bytes'],
            'record telemetry allowance differs from manifest admission')
    for key, expected in (('app_allowance_bytes', 128 * 1024**2),
                          ('stream_allowance_bytes', 4 * 1024**2),
                          ('telemetry_allowance_bytes', 16 * 1024 + 1024**2),
                          ('helper_reserve_bytes', 32 * 1024**2)):
        require(memory[key] == expected, key + ': frozen allowance differs')
    require(max(memory['load_admitted_bytes'], memory['session_admitted_bytes']) <= MEMORY_LIMIT,
            'whole owner/session admission exceeds6GiB')
    keys(value['ordinary_observations'], OBSERVATIONS, 'ordinary observations')
    require(all(v is None for v in value['ordinary_observations'].values()), 'invented ordinary observation')
    require(value['availability'] == AVAILABILITY, 'unavailable reasons differ')
    return known


def report(path, *, expected_sha256):
    raw = read_pinned(Path(path), expected_sha256)
    lines = raw.splitlines()
    require(len(lines) >= 2 and all(0 < len(line) + 1 <= LINE_LIMIT for line in lines),
            'missing framing,blank or oversized line')
    data = [parse(line) for line in lines]
    header, footer = data[0], data[-1]
    manifest(header)
    keys(footer, ('event', 'schema_version', 'complete', 'retained', 'telemetry_dropped',
                  'export_errors', 'invalid_records', 'counter_overflow'), 'footer')
    require(footer['event'] == 'request_metrics_export' and type(footer['schema_version']) is int
            and footer['schema_version'] == 1 and footer['complete'] is True, 'incomplete export footer')
    require(type(footer['counter_overflow']) is bool, 'overflow flag must be boolean')
    for key in ('retained', 'telemetry_dropped', 'export_errors', 'invalid_records'):
        integer(footer[key], key)
    records = data[1:-1]
    require(footer['retained'] == len(records), 'footer retained count differs')
    previous = None
    known = 0
    for row in records:
        known += record(row, header)
        ids = row['request_id'], row['attempt_id'], row['owner_generation']
        if previous is not None:
            require(ids[0] > previous[0] and ids[1] > previous[1] and ids[2] >= previous[2],
                    'duplicate/reordered request,attempt or owner generation')
        previous = ids
    complete = not any(footer[key] for key in ('telemetry_dropped', 'invalid_records',
                                              'export_errors', 'counter_overflow'))
    if complete:
        require(all(row['request_id'] == index and row['attempt_id'] == index
                    for index, row in enumerate(records, 1)),
                'loss-free export has an inconsistent attempted-request inventory')
    metrics = {}
    for key in ('request_total_ms', 'prefill_ms', 'decode_ms', 'sampling_ms', 'ttft_event_ms',
                'first_text_publication_ms'):
        values = [row['timing'][key] for row in records if row['timing'][key] is not None]
        metrics[key] = dict(retained_observations=len(values), p50=statistics.median(values) if values else None)
    for key in DERIVED:
        values = [row[key] for row in records if row[key] is not None]
        metrics[key] = dict(retained_observations=len(values), p50=statistics.median(values) if values else None)
    outcomes = {key: sum(row['terminal_status'] == key for row in records) for key in sorted(OUTCOMES)}
    attempted = len(records) if complete else None
    attempted_rates = {key: count / attempted if attempted else None for key, count in outcomes.items()}
    rate_reason = ('telemetry_incomplete' if not complete else
                   'no_attempted_requests' if not records else None)
    return dict(format='toklet-gemma-cli-retained-facts-v1', input_sha256=expected_sha256,
                input_bytes=len(raw), validated_complete_export=True, telemetry_complete=complete,
                manifest=header, export_status=footer, retained_records=len(records),
                completed_core_reports=known, unavailable_core_reports=len(records) - known,
                retained_outcomes=outcomes, attempted_requests=attempted,
                attempted_outcome_rates=attempted_rates,
                attempted_outcome_rate_unit='fraction_of_attempted_requests',
                attempted_outcome_rates_unavailable_reason=rate_reason,
                outcome_denominator='Exact attempted inventory only with loss-free contiguous IDs from1; '
                                    'otherwise rates are unavailable and retained counts are descriptive.',
                metrics=metrics, ordinary_unavailable=AVAILABILITY,
                source_model_runtime_independently_verified=False, numerical_quality_qualified=False,
                observer_overhead_qualified=False, adoption_qualified=False,
                scope='Recorded ordinary CLI facts only. Milliseconds; p50 with retained observation counts. '
                      'No UI/render,tail,energy,DRAM,model parity or performance-comparison claim.')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('--input-sha256', required=True)
    parser.add_argument('--output', required=True, type=Path, help='new JSON report file')
    args = parser.parse_args(argv)
    try:
        result = report(args.input, expected_sha256=args.input_sha256)
        raw = (json.dumps(result, indent=2, allow_nan=False) + '\n').encode()
        require(len(raw) <= 64 * 1024, 'report exceeds bound')
        with args.output.open('xb') as sink:
            sink.write(raw)
        return 0
    except (ValueError, OSError, KeyError, TypeError, RecursionError) as error:
        print('gemma_cli_report: ' + str(error)[:256], file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())

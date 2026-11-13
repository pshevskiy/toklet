#!/usr/bin/env python3
"""Explicit local Gemma O1 checks and paired overhead trials. No downloads."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import statistics
import subprocess
import time
import re
import uuid
from m7_swift_sources import ROOT, worker_sources

WARMUP_PAIRS = 3
CAPTURE_BYTE_LIMIT = 512 * 1024
WORKLOAD_BYTE_LIMIT = 16 * 1024
WORKLOAD_KEYS = {'schema_version', 'workload_id', 'conversation_json', 'conversation_sha256',
                 'input_token_count', 'input_tokens_sha256', 'maximum_output_tokens',
                 'capacity', 'chunk', 'cache_policy', 'expected_output'}
OUTPUT_KEYS = {'tokens', 'history', 'committed', 'pending', 'reason', 'stop_token'}
ADMISSION_KEYS = {'operation', 'base_history', 'input', 'history_after_input', 'output', 'required', 'remaining'}


def _strict_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, 'Duplicate JSON key: ' + key)
        result[key] = value
    return result


def _strict_json(raw):
    def nonfinite(value):
        raise ValueError('Nonfinite JSON number: ' + value)
    return json.loads(raw, object_pairs_hook=_strict_object, parse_constant=nonfinite)


def _uint(value, maximum=0xFFFFFFFF):
    return type(value) is int and 0 <= value <= maximum


def _digest(value):
    return type(value) is str and re.fullmatch('[0-9a-f]{64}', value) is not None


def _positive_clock(value):
    try:
        return type(value) in (int, float) and math.isfinite(value) and value > 0
    except OverflowError:
        return False


def _validate_output(value, workload):
    require(type(value) is dict and set(value) == OUTPUT_KEYS, 'Invalid expected output fields')
    tokens, cap = value['tokens'], workload['maximum_output_tokens']
    require(type(tokens) is list and 0 < len(tokens) <= cap and all(_uint(t) for t in tokens), 'Invalid expected output tokens')
    require(all(_uint(value[k]) for k in OUTPUT_KEYS - {'tokens'}), 'Invalid expected output counts')
    reason = value['reason']
    require((reason == 1 and len(tokens) == cap and value['stop_token'] == 0xFFFFFFFF)
            or (reason == 2 and len(tokens) < cap and value['stop_token'] in (1, 106, 50)), 'Invalid output-limit/stop semantics')
    history = workload['input_token_count'] + len(tokens) + (reason == 2)
    require(value['history'] == history <= workload['capacity'] and value['pending'] == 1
            and value['committed'] + 1 == history, 'Expected output state differs')


def _validate_workload(value, preflight_only):
    require(type(value) is dict and set(value) in (WORKLOAD_KEYS, WORKLOAD_KEYS | {'_sha256'}), 'Invalid workload fields')
    require(type(value['schema_version']) is int and value['schema_version'] == 1, 'Unsupported workload version')
    require(type(value['workload_id']) is str and re.fullmatch('[a-z0-9][a-z0-9_-]{0,63}', value['workload_id']), 'Invalid workload ID')
    conversation = value['conversation_json']
    require(type(conversation) is str and 0 < len(conversation.encode('utf-8')) <= 8192, 'Conversation exceeds byte bound')
    require(type(_strict_json(conversation)) is list, 'Conversation must be a JSON array')
    require(_digest(value['conversation_sha256']) and hashlib.sha256(conversation.encode('utf-8')).hexdigest()
            == value['conversation_sha256'], 'Conversation identity differs')
    require(_uint(value['input_token_count'], 512) and value['input_token_count'] > 0, 'Invalid rendered input count')
    require(_digest(value['input_tokens_sha256']), 'Invalid rendered input hash')
    require(_uint(value['maximum_output_tokens'], 256) and value['maximum_output_tokens'] > 0, 'Invalid output cap')
    require(type(value['capacity']) is int and value['capacity'] == 512
            and type(value['chunk']) is int and value['chunk'] == 32, 'Workload configuration differs')
    require(value['input_token_count'] + value['maximum_output_tokens'] <= value['capacity'], 'Workload exceeds shared window')
    require(value['cache_policy'] == 'reset_chat_each_request', 'Unsupported cache policy')
    if value['expected_output'] is None:
        require(preflight_only, 'Input-only workload cannot enter timing')
    else:
        _validate_output(value['expected_output'], value)


def load_workload(path, *, preflight_only=False):
    with Path(path).open('rb') as stream:
        raw = stream.read(WORKLOAD_BYTE_LIMIT + 1)
    require(0 < len(raw) <= WORKLOAD_BYTE_LIMIT, 'Workload exceeds byte bound')
    value = _strict_json(raw.decode('utf-8'))
    require(type(value) is dict and set(value) == WORKLOAD_KEYS, 'Invalid workload fields')
    _validate_workload(value, preflight_only)
    value['_sha256'] = hashlib.sha256(raw).hexdigest()
    return value


def freeze_workload(path, workload, expected):
    frozen = {key: value for key, value in workload.items() if key != '_sha256'}
    frozen['expected_output'] = expected
    _validate_workload(frozen, False)
    raw = (json.dumps(frozen, indent=2, ensure_ascii=False, allow_nan=False, sort_keys=True)+'\n').encode('utf-8')
    require(len(raw) <= WORKLOAD_BYTE_LIMIT, 'Frozen workload exceeds byte bound')
    with Path(path).open('xb') as stream:
        stream.write(raw)
    return load_workload(path)


def validate_workload_rows(rows, workload, *, preflight_only=False):
    _validate_workload(workload, preflight_only)
    require(_digest(workload.get('_sha256')), 'Workload file identity missing')
    if preflight_only:
        require(type(rows) is list and len(rows) == 2, 'Incomplete preflight pair')
        order = [(0, 'disabled'), (0, 'summary')]
    else:
        require(type(rows) is list and len(rows) % 2 == 0, 'Incomplete workload rows')
        count = len(rows) // 2 - WARMUP_PAIRS
        _validated_rows(rows, count)
        order = [(t, mode) for t in range(WARMUP_PAIRS + count)
                 for mode in (('disabled', 'summary') if t % 2 == 0 else ('summary', 'disabled'))]
    expected = workload['expected_output']
    ids, owners = set(), set()
    for row, (trial, mode) in zip(rows, order, strict=True):
        require(type(row) is dict and type(row.get('trial')) is int and row['trial'] == trial
                and row.get('mode') == mode and row.get('warmup') is (not preflight_only and trial < WARMUP_PAIRS), 'Workload pair order differs')
        request_id = row.get('request_id')
        require(type(request_id) is str and str(uuid.UUID(request_id)).lower() == request_id.lower()
                and request_id.lower() not in ids, 'Invalid or duplicate request ID')
        ids.add(request_id.lower())
        for key in ('submitted_uptime_ns', 'completed_uptime_ns', 'model_instance', 'session_instance'):
            require(_uint(row.get(key), 0xFFFFFFFFFFFFFFFF) and row[key] > 0, 'Invalid host clock/owner: ' + key)
        owners.add((row['model_instance'], row['session_instance']))
        duration = row['completed_uptime_ns'] - row['submitted_uptime_ns']
        require(duration > 0, 'Invalid completed host interval')
        for key in ('ttft_ms', 'total_ms'):
            value = row.get(key)
            require(_positive_clock(value), 'Invalid workload clock: ' + key)
        require(row['ttft_ms'] <= row['total_ms'] and math.isclose(row['total_ms'], duration / 1e6, rel_tol=1e-12, abs_tol=1e-9), 'Workload host clocks differ')
        for key, value in [('workload_sha256', workload['_sha256']), ('conversation_sha256', workload['conversation_sha256']),
                           ('input_tokens', workload['input_token_count']), ('input_tokens_sha256', workload['input_tokens_sha256']),
                           ('maximum_output_tokens', workload['maximum_output_tokens']), ('capacity', 512), ('chunk', 32),
                           ('cache_policy', workload['cache_policy']), ('status', 0)]:
            require(type(row.get(key)) is type(value) and row[key] == value, 'Workload observation differs: ' + key)
        admission = row.get('admission')
        require(type(admission) is dict and set(admission) == ADMISSION_KEYS, 'Missing or invalid admission')
        count, cap = workload['input_token_count'], workload['maximum_output_tokens']
        wanted = dict(operation=3, base_history=0, input=count, history_after_input=count,
                      output=cap, required=count + cap, remaining=512 - count - cap)
        require(all(type(admission[k]) is int and admission[k] == value for k, value in wanted.items()), 'Workload admission differs')
        observed = {k: row.get(k) for k in OUTPUT_KEYS}
        _validate_output(observed, workload)
        if expected is None:
            expected = observed
        require(observed == expected, 'Workload output/state differs')
    require(len(owners) == 1, 'Workload model/session owner changed')
    return expected


def validate_native_work_capture(raw, records, *, required=False):
    """Read legacy work as unavailable; require explicit V2 for new captures.

    This checks exported counters, never derives a general decode count from
    emitted tokens. Cancellation can complete evaluation before the next sample.
    """
    require(type(required) is bool and type(raw) is bytes and len(raw) <= CAPTURE_BYTE_LIMIT,
            'Invalid native work capture bound/policy')
    lines = raw.splitlines()
    require(len(lines) >= 2 and all(lines), 'Incomplete native work capture')
    header, *captured = [_strict_json(line) for line in lines]
    require(type(header) is dict and type(records) is list and captured == records,
            'Native work records differ from capture')
    manifest = header.get('nativeManifest') if header.get('kind') == 'benchmark_request_summary_capture' else header
    require(type(manifest) is dict and manifest.get('kind') == 'request_summary_manifest',
            'Native work manifest missing')
    version = manifest.get('nativeGenerationMetricsVersion', 1)
    require(type(version) is int and version in (1, 2) and (not required or version == 2),
            'Native work version unavailable or unsupported')
    available = 0
    # Validate wire types, even when Python value equality would equate a bool
    # with an integer in a separately supplied decoded list.
    for record in captured:
        require(type(record) is dict, 'Invalid native work record')
        native = record.get('nativeMetrics')
        if version == 1:
            require(native is None or type(native) is dict and 'decodeEvaluations' not in native,
                    'Legacy capture claims an unversioned decode count')
            continue
        status = record.get('nativeMetricStatus')
        if status != 'completed':
            require(status in ('notStarted', 'generationFailed', 'invalidRecord') and native is None,
                    'Unavailable native work retained stale counters')
            continue
        require(type(native) is dict and _uint(native.get('sampledTokens'), 32768)
                and _uint(native.get('emittedTokens'), native['sampledTokens'])
                and _uint(native.get('decodeEvaluations'), native['sampledTokens']),
                'Invalid completed decode-evaluation count')
        available += 1
    return dict(version=version, available=available, unavailable=len(records) - available)


def validate_workload_capture(raw, rows, workload, *, preflight_only=False, measured_pairs=10, require_native_work=False):
    validate_workload_rows(rows, workload, preflight_only=preflight_only)
    if preflight_only:
        require(type(raw) is bytes and len(raw) <= CAPTURE_BYTE_LIMIT, 'Preflight capture exceeds byte bound')
        lines = raw.splitlines()
        require(len(lines) == 2 and all(lines) and len(lines[1]) <= 4096, 'Incomplete preflight capture')
        header, record = [_strict_json(line) for line in lines]
        require(type(header) is dict and header.get('kind') == 'request_summary_manifest'
                and type(header.get('droppedRecords')) is int and header['droppedRecords'] == 0, 'Invalid preflight manifest')
        records = [record]
    else:
        records = validate_summary_capture(raw, rows, measured_pairs)
    validate_native_work_capture(raw, records, required=require_native_work)
    enabled = {row['request_id']: row for row in rows if row['mode'] == 'summary'}
    require(len(records) == len(enabled), 'Incomplete workload summaries')
    seen = set()
    for record in records:
        require(type(record) is dict and record.get('requestID') in enabled and record['requestID'] not in seen, 'Summary ID differs or repeats')
        seen.add(record['requestID']); row = enabled[record['requestID']]
        for key, value in [('schemaVersion', 2), ('status', 0), ('promptTokens', workload['input_token_count']),
                           ('requestedOutputTokens', workload['maximum_output_tokens']), ('callbackTokens', len(row['tokens'])),
                           ('capacity', 512), ('chunk', 32), ('nativeEmittedTokens', len(row['tokens']))]:
            require(type(record.get(key)) is int and record[key] == value, 'Workload summary differs: ' + key)
        require(record.get('outcome') == ('outputLimit' if row['reason'] == 1 else 'stopToken')
                and record.get('nativeMetricStatus') == 'completed', 'Workload summary outcome/metric availability differs')
        for key in ('processLifetimePeakBytes', 'processPhysicalBytes', 'sessionAdmittedBytes'):
            require(_uint(record.get(key), 6 * 1024**3) and record[key] > 0, 'Workload memory missing or exceeds budget')
        after = record.get('after')
        require(type(after) is dict and all(type(after.get(k)) is int and after[k] == row[v] for k, v in
                [('modelInstance', 'model_instance'), ('sessionInstance', 'session_instance'), ('history', 'history'), ('committed', 'committed'), ('pending', 'pending')]), 'Workload summary state/owner differs')
        before = record.get('before')
        require(type(before) is dict and all(type(before.get(k)) is int and before[k] == row[v] for k, v in
                [('modelInstance', 'model_instance'), ('sessionInstance', 'session_instance')])
                and all(_uint(before.get(k), 512) for k in ('history', 'committed', 'pending'))
                and before['pending'] <= 1 and before['history'] == before['committed'] + before['pending'], 'Workload summary prior state/owner differs')
        native = record.get('nativeMetrics')
        require(type(native) is dict and all(type(native.get(k)) is int and native[k] == value for k, value in
                [('sessionInstance', row['session_instance']), ('reusedTokens', 0), ('prefillTokens', workload['input_token_count']),
                 ('sampledTokens', row['history'] - workload['input_token_count']), ('emittedTokens', len(row['tokens']))]), 'Workload native work differs')
        if 'decodeEvaluations' in native:
            # These exact reset/output-limit-or-stop cases have one final pending
            # sample. This expectation is specific to this validated workload.
            require(native['decodeEvaluations'] == row['committed'] - workload['input_token_count'],
                    'Workload completed decode rows differ')
        start, first, end = (record.get(k) for k in ('submittedUptimeNanoseconds', 'firstCallbackNanoseconds', 'completedUptimeNanoseconds'))
        require(all(_uint(x, 0xFFFFFFFFFFFFFFFF) for x in (start, first, end))
                and row['submitted_uptime_ns'] <= start <= first <= end <= row['completed_uptime_ns'], 'Workload summary clocks escape host interval')
        require(math.isclose((first - row['submitted_uptime_ns']) / 1e6, row['ttft_ms'], rel_tol=1e-12, abs_tol=1e-9), 'Workload first callback differs')
    return records


def require(value, message):
    if not value:
        raise ValueError(message)


def parse_measured_pairs(text):
    if not isinstance(text, str) or not text or not text.isascii() or not text.isdecimal():
        raise argparse.ArgumentTypeError('measured pairs must be ASCII digits from 10 through 100')
    normalized = text.lstrip('0') or '0'
    if len(normalized) > 3 or not 10 <= int(normalized) <= 100:
        raise argparse.ArgumentTypeError('measured pairs must be from 10 through 100')
    return int(normalized)


def _validated_rows(rows, measured_pairs):
    require(type(measured_pairs) is int and 10 <= measured_pairs <= 100, 'Measured pair count outside 10..100')
    require(type(rows) is list and len(rows) == 2 * (WARMUP_PAIRS + measured_pairs), 'Incomplete cohort rows')
    indexed = {}
    for row in rows:
        require(type(row) is dict, 'Invalid sample row')
        trial, mode = row.get('trial'), row.get('mode')
        require(type(trial) is int and 0 <= trial < WARMUP_PAIRS + measured_pairs
                and mode in ('disabled', 'summary'), 'Invalid trial or mode')
        require(row.get('warmup') is (trial < WARMUP_PAIRS), 'Incorrect warmup flag')
        require((trial, mode) not in indexed, 'Duplicate trial/mode pair')
        for metric in ('ttft_ms', 'total_ms'):
            value = row.get(metric)
            try:
                valid = type(value) in (int, float) and math.isfinite(value) and value > 0
            except OverflowError:
                valid = False
            require(valid, 'Metric must be finite and positive: ' + metric)
        require(row['total_ms'] >= row['ttft_ms'], 'Total duration precedes first token')
        require(type(row.get('tokens')) is list and all(type(t) is int and t >= 0 for t in row['tokens']), 'Invalid output tokens')
        require(all(type(row.get(k)) is int and row[k] >= 0 for k in ('history', 'committed', 'pending')), 'Invalid session state')
        require(row['history'] == row['committed'] + row['pending'], 'Session state counts do not reconcile')
        indexed[trial, mode] = row
    for trial in range(WARMUP_PAIRS + measured_pairs):
        pair = {mode: indexed[trial, mode] for mode in ('disabled', 'summary')}
        require(pair['disabled']['tokens'] == pair['summary']['tokens'], 'Token output changed')
        require(all(pair['disabled'][k] == pair['summary'][k] for k in ('history', 'committed', 'pending')), 'Session state changed')
    return indexed


def validate_summary_capture(raw, rows, measured_pairs=10):
    indexed = _validated_rows(rows, measured_pairs)
    require(type(raw) is bytes and len(raw) <= CAPTURE_BYTE_LIMIT, 'Summary capture exceeds byte bound')
    lines = raw.splitlines()
    require(len(lines) == WARMUP_PAIRS + measured_pairs + 1 and all(lines), 'Missing summary observations')
    parsed = [json.loads(line) for line in lines]
    require(all(type(row) is dict for row in parsed), 'Invalid summary row')
    header, records = parsed[0], parsed[1:]
    if measured_pairs > 10:
        require(set(header) == {'schemaVersion', 'kind', 'nativeManifest', 'recordCount', 'ringCapacity', 'captureByteLimit', 'outsideTimedPairs'}
                and header['kind'] == 'benchmark_request_summary_capture'
                and type(header['schemaVersion']) is int and header['schemaVersion'] == 1, 'Invalid external capture header')
        for key, value in [('recordCount', WARMUP_PAIRS + measured_pairs), ('ringCapacity', 32), ('captureByteLimit', CAPTURE_BYTE_LIMIT)]:
            require(type(header[key]) is int and header[key] == value, 'Capture bound differs: ' + key)
        require(header['outsideTimedPairs'] is True, 'Capture must occur outside timed pairs')
        manifest = header['nativeManifest']
    else:
        manifest = header
    require(type(manifest) is dict and manifest.get('kind') == 'request_summary_manifest'
            and type(manifest.get('droppedRecords')) is int
            and manifest['droppedRecords'] == max(0, WARMUP_PAIRS + measured_pairs - 32), 'Native ring-drop accounting differs')
    expected = [indexed[t, 'summary'].get('request_id') for t in range(WARMUP_PAIRS + measured_pairs)]
    require(all(type(value) is str and value for value in expected) and len(set(expected)) == len(expected), 'Summary request IDs missing or duplicated')
    observed = []
    for line, record in zip(lines[1:], records, strict=True):
        require(len(line) <= 4096, 'Native summary exceeds 4KiB')
        value = record.get('processLifetimePeakBytes')
        require(type(value) is int and 0 <= value <= 6 * 1024**3, 'Process peak exceeded budget or is invalid')
        observed.append(record.get('requestID'))
    require(all(type(value) is str for value in observed) and len(set(observed)) == len(observed)
            and set(observed) == set(expected), 'Summary IDs do not match every enabled request')
    return records


def comparison(rows, measured_pairs=10):
    indexed = _validated_rows(rows, measured_pairs)
    measured = [r for r in rows if not r['warmup']]
    results = {}
    for metric in ('ttft_ms', 'total_ms'):
        pairs = []
        for trial in range(WARMUP_PAIRS, WARMUP_PAIRS + measured_pairs):
            pair = {mode: indexed[trial, mode] for mode in ('disabled', 'summary')}
            pairs.append(100 * (pair['summary'][metric] / pair['disabled'][metric] - 1))
        off = [r[metric] for r in measured if r['mode'] == 'disabled']
        on = [r[metric] for r in measured if r['mode'] == 'summary']
        rng = random.Random(1)
        bootstrap = sorted(statistics.median(rng.choices(pairs, k=len(pairs))) for _ in range(10000))
        results[metric] = dict(disabled_median=statistics.median(off), summary_median=statistics.median(on),
            median_ratio_overhead_pct=100 * (statistics.median(on) / statistics.median(off) - 1),
            paired_overhead_pct=pairs, paired_median_pct=statistics.median(pairs),
            paired_median_bootstrap_95_pct=[bootstrap[250], bootstrap[9749]],
            disabled_range=[min(off), max(off)], summary_range=[min(on), max(on)])
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model', type=Path)
    parser.add_argument('archive', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--measured-pairs', type=parse_measured_pairs, default=10)
    parser.add_argument('--workload', type=Path)
    parser.add_argument('--preflight-only', action='store_true')
    args = parser.parse_args()
    if args.preflight_only and args.workload is None:
        parser.error('--preflight-only requires --workload')
    workload = load_workload(args.workload, preflight_only=args.preflight_only) if args.workload else None
    out = args.output.resolve(); out.mkdir(parents=True, exist_ok=False)
    sources = [*worker_sources(), ROOT / 'apple/tests/swift/RequestSummaryMain.swift', ROOT / 'apple/include/toklet.h', Path(__file__).resolve()]
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    pins = {str(p.relative_to(ROOT)): sha(p) for p in sources}
    pins[str(args.archive)] = sha(args.archive)
    if workload is not None:
        require(sha(args.workload) == workload['_sha256'], 'Workload changed after validation')
        pins[str(args.workload.resolve())] = workload['_sha256']
    build_id = hashlib.sha256(json.dumps(pins, sort_keys=True).encode()).hexdigest()
    env = dict(os.environ, TOKLET_BUILD_ID=build_id)
    commands = []

    def run(label, command, timeout=1800):
        start = time.monotonic_ns()
        with (out / (label + '.log')).open('w') as log:
            result = subprocess.run(list(map(str, command)), cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=timeout)
        commands.append(dict(label=label, command=list(map(str, command)), exit_code=result.returncode,
                             start_monotonic_ns=start, end_monotonic_ns=time.monotonic_ns()))
        (out / 'commands.json').write_text(json.dumps(commands, indent=2)+'\n')
        if result.returncode:
            raise RuntimeError(label + ' failed; see log')

    binary = out / 'summary-check'
    run('compile', ['xcrun', 'swiftc', '-swift-version', '6', '-strict-concurrency=complete', '-warnings-as-errors', '-O',
        '-I', ROOT/'apple/include', '-module-cache-path', out/'ModuleCache', *sources[:-2],
        args.archive, '-framework', 'Metal', '-framework', 'MetalPerformanceShaders', '-framework', 'Foundation',
        '-framework', 'Security', '-lobjc', '-liconv', '-lresolv', '-lc++', '-o', binary])
    run('conformance', [binary, args.model, '--conformance-output', out/'conformance.jsonl'])
    if args.preflight_only:
        sample = out / 'preflight.json'
        run('preflight', [binary, args.model, sample, '--workload', args.workload.resolve(), '--preflight-only'])
        rows = _strict_json(sample.read_bytes())
        expected = validate_workload_rows(rows, workload, preflight_only=True)
        with Path(str(sample)+'.summaries.jsonl').open('rb') as stream:
            raw = stream.read(CAPTURE_BYTE_LIMIT + 1)
        validate_workload_capture(raw, rows, workload, preflight_only=True, require_native_work=True)
        frozen_path = out / 'frozen-workload.json'
        freeze_workload(frozen_path, workload, expected)
        require(all(sha(ROOT/p) == pin for p, pin in pins.items()), 'Source/archive/workload changed')
        record = dict(format='toklet-o1-gemma-workload-preflight-v1', passed=True,
            source_sha256=pins, build_id=build_id, native_archive_sha256=sha(args.archive),
            workload_sha256=workload['_sha256'], frozen_workload_sha256=sha(frozen_path),
            frozen_workload=str(frozen_path), expected_output=expected, requests=2,
            measured_requests=0, overhead_qualified=False, default_mode='disabled',
            native_generation_metrics_version=2,
            scope='Untimed exact disabled/summary managed Chat pair; input identity/admission/output-state proof only; no overhead qualification')
        (out/'result.json').write_text(json.dumps(record, indent=2)+'\n')
        print(json.dumps(dict(passed=True, preflight_only=True, result=str(out/'result.json'))))
        return
    results = []
    request_ids = set()
    for repeat in (1, 2):
        sample = out / f'samples-{repeat}.json'
        command = [binary, args.model, sample]
        if args.measured_pairs != 10:
            command += ['--measured-pairs', str(args.measured_pairs)]
        if workload is not None:
            command += ['--workload', args.workload.resolve()]
        run(f'overhead-{repeat}', command)
        rows = _strict_json(sample.read_bytes()) if workload is not None else json.loads(sample.read_text())
        if workload is not None:
            validate_workload_rows(rows, workload)
            current = {row['request_id'].lower() for row in rows}
            require(request_ids.isdisjoint(current), 'Request IDs reused across workload cohorts')
            request_ids.update(current)
        results.append(comparison(rows, args.measured_pairs))
        with Path(str(sample)+'.summaries.jsonl').open('rb') as stream:
            raw = stream.read(CAPTURE_BYTE_LIMIT + 1)
        if workload is None:
            records = validate_summary_capture(raw, rows, args.measured_pairs)
            validate_native_work_capture(raw, records, required=True)
        else:
            validate_workload_capture(raw, rows, workload, measured_pairs=args.measured_pairs, require_native_work=True)
    require(all(sha(ROOT/p) == pin for p, pin in pins.items()), 'Source/archive changed')
    qualified = all(r[m]['median_ratio_overhead_pct'] <= 1 and r[m]['paired_median_bootstrap_95_pct'][1] <= 1
                    for r in results for m in r)
    record = dict(format='toklet-o1-gemma-summary-v1', source_sha256=pins, build_id=build_id,
        native_archive_sha256=sha(args.archive), results=results, overhead_qualified=qualified,
        warmup_pairs=WARMUP_PAIRS, measured_pairs=args.measured_pairs, cohorts=2,
        native_generation_metrics_version=2,
        summary_capture='native_ring' if args.measured_pairs == 10 else 'external_after_pairs',
        default_mode='disabled', scope='managed official Gemma BF16 Mac Chat; full caller submit-to-idle acknowledgment; callback TTFT, not render latency',
        hardware=subprocess.check_output(['sysctl', '-n', 'machdep.cpu.brand_string'], text=True).strip(),
        os=subprocess.check_output(['sw_vers'], text=True).strip(),
        power=subprocess.check_output(['pmset', '-g', 'batt'], text=True).strip(),
        tail_claims=False, passed=True)
    if workload is not None:
        record.update(format='toklet-o1-gemma-workload-summary-v1',
            workload_sha256=workload['_sha256'], workload={k:v for k,v in workload.items() if k != '_sha256'},
            scope='Exact descriptor-bound managed official Gemma BF16 Mac Chat; reset each request, full caller submit-to-idle, callback TTFT; capacity512/chunk32, not numerical512-input or UI/render qualification')
    (out/'result.json').write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps(dict(passed=True, overhead_qualified=qualified, result=str(out/'result.json'))))


if __name__ == '__main__':
    main()

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
from m7_swift_sources import ROOT, worker_sources

WARMUP_PAIRS = 3
CAPTURE_BYTE_LIMIT = 512 * 1024


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
    args = parser.parse_args()
    out = args.output.resolve(); out.mkdir(parents=True, exist_ok=False)
    sources = [*worker_sources(), ROOT / 'apple/tests/swift/RequestSummaryMain.swift', ROOT / 'apple/include/toklet.h', Path(__file__).resolve()]
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    pins = {str(p.relative_to(ROOT)): sha(p) for p in sources}
    pins[str(args.archive)] = sha(args.archive)
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
    results = []
    for repeat in (1, 2):
        sample = out / f'samples-{repeat}.json'
        command = [binary, args.model, sample]
        if args.measured_pairs != 10:
            command += ['--measured-pairs', str(args.measured_pairs)]
        run(f'overhead-{repeat}', command)
        rows = json.loads(sample.read_text())
        results.append(comparison(rows, args.measured_pairs))
        with Path(str(sample)+'.summaries.jsonl').open('rb') as stream:
            raw = stream.read(CAPTURE_BYTE_LIMIT + 1)
        validate_summary_capture(raw, rows, args.measured_pairs)
    require(all(sha(ROOT/p) == pin for p, pin in pins.items()), 'Source/archive changed')
    qualified = all(r[m]['median_ratio_overhead_pct'] <= 1 and r[m]['paired_median_bootstrap_95_pct'][1] <= 1
                    for r in results for m in r)
    record = dict(format='toklet-o1-gemma-summary-v1', source_sha256=pins, build_id=build_id,
        native_archive_sha256=sha(args.archive), results=results, overhead_qualified=qualified,
        warmup_pairs=WARMUP_PAIRS, measured_pairs=args.measured_pairs, cohorts=2,
        summary_capture='native_ring' if args.measured_pairs == 10 else 'external_after_pairs',
        default_mode='disabled', scope='managed official Gemma BF16 Mac Chat; full caller submit-to-idle acknowledgment; callback TTFT, not render latency',
        hardware=subprocess.check_output(['sysctl', '-n', 'machdep.cpu.brand_string'], text=True).strip(),
        os=subprocess.check_output(['sw_vers'], text=True).strip(),
        power=subprocess.check_output(['pmset', '-g', 'batt'], text=True).strip(),
        tail_claims=False, passed=True)
    (out/'result.json').write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps(dict(passed=True, overhead_qualified=qualified, result=str(out/'result.json'))))


if __name__ == '__main__':
    main()

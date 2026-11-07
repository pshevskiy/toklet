#!/usr/bin/env python3
"""Offline O2 cost report from a pinned Gemma request-summary measurement bundle.

Uses instrumented native phases for attribution and raw caller clocks for paired
latency. Never infers UI/render timing or measured bandwidth from these records.
"""
import argparse
import json
import math
import re
from pathlib import Path
import statistics

LIMIT = 135201
# Host durations use integer DispatchTime nanoseconds; allow one nanosecond
# when comparing their decimal millisecond serialization. Native Instant and
# Swift enclosing wall intervals allow 1 microsecond clock/conversion grain.
HOST_MS_ABSOLUTE_TOLERANCE = 1e-6
NATIVE_ENCLOSURE_SECONDS_TOLERANCE = 1e-6


def require(value, message):
    if not value:
        raise ValueError(message)


def unique(pairs):
    value = {}
    for key, item in pairs:
        require(key not in value, 'Duplicate JSON field: ' + key)
        value[key] = item
    return value


def parse(data):
    return json.loads(data, object_pairs_hook=unique,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError('Non-finite JSON: ' + value)))


def number(value, name, integer=False):
    require(type(value) in ((int,) if integer else (int, float)), 'Invalid number: ' + name)
    require(math.isfinite(value) and value >= 0, 'Invalid range: ' + name)
    return value


def read_bounded(path, limit):
    with path.open('rb') as stream:
        data = stream.read(limit + 1)
    require(len(data) <= limit, 'Oversized input: ' + str(path))
    return data


def digest(value, length):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{'+str(length)+'}', value) is not None


def summaries(path, build_id):
    data = read_bounded(path, LIMIT)
    rows = [parse(line) for line in data.splitlines()]
    require(1 <= len(rows) <= 33, 'Invalid summary inventory')
    manifest = rows[0]
    require(manifest['kind'] == 'request_summary_manifest' and manifest['schemaVersion'] == 2, 'Unsupported summary schema')
    require(manifest['buildIdentity'] == build_id, 'Summary/build identity mismatch')
    require(digest(manifest['revision'], 40) and digest(manifest['modelAndCodecManifestSHA256'], 64), 'Invalid model identity')
    require(manifest['droppedRecords'] == 0 and manifest['budgetSkippedRecords'] == 0 and manifest['exportErrors'] == 0,
            'Measurement summary coverage incomplete')
    records = {}
    for row in rows[1:]:
        require(row['schemaVersion'] == 2 and row['requestID'] not in records, 'Duplicate request or schema mismatch')
        require(row['outcome'] == 'stopToken' and row['status'] == 0, 'Unexpected benchmark outcome')
        start = number(row['submittedUptimeNanoseconds'], 'submit', True)
        end = number(row['completedUptimeNanoseconds'], 'complete', True)
        worker_start = number(row['startedUptimeNanoseconds'], 'worker start', True)
        callback = number(row['firstCallbackNanoseconds'], 'first callback', True)
        require(start <= worker_start <= callback <= end, 'Request clock ordering invalid')
        for key in ('requestMilliseconds', 'callbackTTFTMilliseconds'):
            number(row[key], key)
        for key in ('encodingNanoseconds', 'evaluationNanoseconds', 'decodingNanoseconds'):
            number(row[key], key, True)
        require(math.isclose(row['requestMilliseconds'], (end-start)/1e6, rel_tol=1e-12,
                             abs_tol=HOST_MS_ABSOLUTE_TOLERANCE), 'Summary request duration inconsistent')
        require(math.isclose(row['callbackTTFTMilliseconds'], (callback-start)/1e6, rel_tol=1e-12,
                             abs_tol=HOST_MS_ABSOLUTE_TOLERANCE), 'Summary callback TTFT inconsistent')
        wrapper_phases = sum(row[k] for k in ('encodingNanoseconds','evaluationNanoseconds','decodingNanoseconds'))
        require(wrapper_phases <= end-start, 'Wrapper phases exceed request duration')
        require(row['nativeMetricStatus'] == 'completed', 'Missing native phase report')
        native = row['nativeMetrics']
        for key in ('prefillSeconds', 'decodeSeconds', 'samplingSeconds', 'totalSeconds', 'firstTokenSeconds'):
            number(native[key], key)
        for key in ('reusedTokens', 'prefillTokens', 'sampledTokens', 'emittedTokens'):
            number(native[key], key, True)
        require(number(native['sessionInstance'], 'session instance', True) > 0 and native['sessionInstance'] == row['after']['sessionInstance'], 'Native phase owner mismatch')
        require(native['emittedTokens'] == row['callbackTokens'] == row['nativeEmittedTokens'], 'Emission counts differ')
        require(native['sampledTokens'] == native['emittedTokens'] + 1, 'Hidden stop not accounted for')
        require(native['prefillTokens'] == row['promptTokens'] and native['reusedTokens'] == 0, 'Unexpected reset/prefill work')
        total = native['totalSeconds']
        require(total > 0 and native['firstTokenSeconds'] <= total, 'Invalid native duration')
        require(total <= row['evaluationNanoseconds']/1e9 + NATIVE_ENCLOSURE_SECONDS_TOLERANCE,
                'Native total exceeds enclosing evaluation')
        phases = sum(native[k] for k in ('prefillSeconds', 'decodeSeconds', 'samplingSeconds'))
        require(phases <= total + 1e-9, 'Overlapping phases misreported as exclusive')
        expected = native['prefillTokens'] / native['prefillSeconds'] if native['prefillTokens'] and native['prefillSeconds'] else None
        require(expected is not None and math.isclose(number(native['prefillTokensPerSecond'], 'prefill rate'), expected, rel_tol=1e-12), 'Prefill rate inconsistent')
        records[row['requestID']] = row
    return manifest, records


def report_bundle(folder):
    result_data = read_bounded(folder/'result.json', 1024 * 1024)
    result = parse(result_data)
    require(result['format'] == 'toklet-o1-gemma-summary-v1' and result['passed'] is True, 'Successful pinned runner result required')
    require(digest(result['build_id'], 64) and digest(result['native_archive_sha256'], 64), 'Invalid build/archive identity')
    cohorts = []
    identity = None
    for repeat in (1, 2):
        data = read_bounded(folder/f'samples-{repeat}.json', 1024 * 1024)
        samples = parse(data)
        require(len(samples) == 26, 'Expected three warmup and ten measured pairs')
        manifest, records = summaries(folder/f'samples-{repeat}.json.summaries.jsonl', result['build_id'])
        model = tuple(manifest[k] for k in ('profile', 'revision', 'modelAndCodecManifestSHA256', 'clock', 'metricScope'))
        require(identity is None or identity == model, 'Model/clock scope changed between batches')
        identity = model
        require(len(records) == 13, 'Missing instrumented requests')
        phases, paired = [], {m: [] for m in ('ttft_ms', 'total_ms')}
        seen = set()
        for trial in range(13):
            group = [r for r in samples if r['trial'] == trial]
            require(len(group) == 2 and {r['mode'] for r in group} == {'disabled', 'summary'}, 'Invalid pairing')
            pair = {r['mode']: r for r in group}
            off, on = pair['disabled'], pair['summary']
            require(all(r['warmup'] is (trial < 3) for r in group), 'Warmup labels differ')
            require(off['tokens'] == on['tokens'] and all(off[k] == on[k] for k in ('history', 'committed', 'pending')), 'Compared work/state differs')
            require(on['request_id'] in records and on['request_id'] not in seen, 'Missing or repeated request correlation')
            seen.add(on['request_id']); row = records[on['request_id']]
            for sample in group:
                caller_start = number(sample['submitted_uptime_ns'], 'caller submit', True)
                caller_end = number(sample['completed_uptime_ns'], 'caller completion', True)
                number(sample['total_ms'], 'caller duration')
                require(caller_end >= caller_start and math.isclose((caller_end-caller_start)/1e6,
                        sample['total_ms'], rel_tol=1e-12, abs_tol=HOST_MS_ABSOLUTE_TOLERANCE),
                        'Caller duration inconsistent')
            caller_start, caller_end = on['submitted_uptime_ns'], on['completed_uptime_ns']
            require(caller_start <= row['submittedUptimeNanoseconds'] <= row['completedUptimeNanoseconds'] <= caller_end,
                    'Caller clock does not enclose collection')
            require(math.isclose((row['firstCallbackNanoseconds']-caller_start)/1e6,
                    number(on['ttft_ms'], 'caller TTFT'), rel_tol=1e-12,
                    abs_tol=HOST_MS_ABSOLUTE_TOLERANCE), 'Caller callback TTFT inconsistent')
            require(len(on['tokens']) == row['callbackTokens'], 'Raw token count differs')
            require((row['capacity'], row['chunk'], row['requestedOutputTokens']) == (512, 32, 12), 'Workload configuration changed')
            require((row['after']['history'], row['after']['committed'], row['after']['pending']) == (on['history'], on['committed'], on['pending']), 'Raw session state differs')
            for sample in group:
                for metric in paired:
                    number(sample[metric], metric)
                require(sample['total_ms'] >= sample['ttft_ms'] > 0, 'Invalid caller latency')
            if trial < 3:
                continue
            for metric in paired:
                paired[metric].append(100 * (on[metric] / off[metric] - 1))
            phases.append(row)
        require(set(records) == seen, 'Unmatched summaries')
        median = lambda values: statistics.median(values)
        native_phase = {key: median([r['nativeMetrics'][key] for r in phases]) for key in ('prefillSeconds', 'decodeSeconds', 'samplingSeconds', 'totalSeconds')}
        remainder = median([max(0, r['nativeMetrics']['totalSeconds'] - sum(r['nativeMetrics'][k] for k in ('prefillSeconds', 'decodeSeconds', 'samplingSeconds'))) for r in phases])
        cohorts.append(dict(batch=repeat, measured_requests_per_mode=10, native_phase_median_seconds=native_phase,
            native_remainder_median_seconds=remainder,
            prefill_tokens_per_second_median=median([r['nativeMetrics']['prefillTokensPerSecond'] for r in phases]),
            caller_median_ms={mode: {metric: median([r[metric] for r in samples if not r['warmup'] and r['mode'] == mode]) for metric in paired} for mode in ('disabled', 'summary')},
            paired_overhead_median_pct={metric: median(values) for metric, values in paired.items()},
            slowest_instrumented_requests=[dict(request_id=r['requestID'], milliseconds=r['requestMilliseconds']) for r in sorted(phases,key=lambda r:r['requestMilliseconds'],reverse=True)[:3]],
            missing_observations=manifest['unavailable']))
    dominant = max(('prefillSeconds', 'decodeSeconds', 'samplingSeconds'), key=lambda k: statistics.median([r['native_phase_median_seconds'][k] for r in cohorts]))
    return dict(format='toklet-request-cost-report-v1', build_id=result['build_id'], model=dict(zip(('profile','revision','manifest_sha256','clock','scope'), identity)),
        native_archive_sha256=result['native_archive_sha256'], hardware=result['hardware'], cohorts=cohorts,
        dominant_observed_native_phase=dominant,
        next_investigation='Attribute concrete operations within the dominant phase before selecting a kernel/caller change.',
        scope='Warm reset+prompt short Chat; native phases instrumented; caller clocks compared; no UI, cold-storage, tail, sustained or measured-DRAM claim.')


def markdown(report):
    lines = ['# Gemma request cost report', '', report['scope'], '',
             '| Batch | Prefill ms | Decode ms | Sampling ms | Native total ms | Prefill tokens/s |',
             '| --- | ---: | ---: | ---: | ---: | ---: |']
    for row in report['cohorts']:
        p = row['native_phase_median_seconds']
        lines.append(f"| {row['batch']} | {1000*p['prefillSeconds']:.3f} | {1000*p['decodeSeconds']:.3f} | {1000*p['samplingSeconds']:.3f} | {1000*p['totalSeconds']:.3f} | {row['prefill_tokens_per_second_median']:.2f} |")
    lines += ['', 'Dominant observed phase: ' + report['dominant_observed_native_phase'] + '.', report['next_investigation'], '',
              'Native total includes callbacks and unclassified request work. Medians of separate phases need not sum to the median total.', '',
              'Unavailable observations:', *['- '+item for item in report['cohorts'][0]['missing_observations']], '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('bundle', type=Path)
    parser.add_argument('output', type=Path, help='new report directory')
    args = parser.parse_args()
    report = report_bundle(args.bundle)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    (args.output/'report.md').write_text(markdown(report))
    print(json.dumps(dict(passed=True, dominant_phase=report['dominant_observed_native_phase'])))


if __name__ == '__main__':
    main()

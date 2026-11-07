#!/usr/bin/env python3
"""Explicit local Gemma O1 checks and paired overhead trials. No downloads."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import statistics
import subprocess
import time
from m7_swift_sources import ROOT, worker_sources


def require(value, message):
    if not value:
        raise ValueError(message)


def comparison(rows):
    measured = [r for r in rows if not r['warmup']]
    require(len(measured) == 20, 'Expected 20 measured requests')
    results = {}
    for metric in ('ttft_ms', 'total_ms'):
        pairs = []
        for trial in range(3, 13):
            pair = {r['mode']: r for r in measured if r['trial'] == trial}
            require(set(pair) == {'disabled', 'summary'}, 'Missing paired modes')
            require(pair['disabled']['tokens'] == pair['summary']['tokens'], 'Token output changed')
            require(all(pair['disabled'][k] == pair['summary'][k] for k in ('history', 'committed', 'pending')), 'Session state changed')
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
        run(f'overhead-{repeat}', [binary, args.model, sample])
        rows = json.loads(sample.read_text())
        results.append(comparison(rows))
        summaries = [json.loads(line) for line in Path(str(sample)+'.summaries.jsonl').read_text().splitlines()]
        require(len(summaries) == 14, 'Missing summary observations')
        require(all(r['processLifetimePeakBytes'] <= 6 * 1024**3 for r in summaries[1:]), 'Process peak exceeded budget')
    require(all(sha(ROOT/p) == pin for p, pin in pins.items()), 'Source/archive changed')
    qualified = all(r[m]['median_ratio_overhead_pct'] <= 1 and r[m]['paired_median_bootstrap_95_pct'][1] <= 1
                    for r in results for m in r)
    record = dict(format='toklet-o1-gemma-summary-v1', source_sha256=pins, build_id=build_id,
        native_archive_sha256=sha(args.archive), results=results, overhead_qualified=qualified,
        default_mode='disabled', scope='managed official Gemma BF16 Mac Chat; full caller submit-to-idle acknowledgment; callback TTFT, not render latency',
        hardware=subprocess.check_output(['sysctl', '-n', 'machdep.cpu.brand_string'], text=True).strip(),
        os=subprocess.check_output(['sw_vers'], text=True).strip(),
        power=subprocess.check_output(['pmset', '-g', 'batt'], text=True).strip(),
        tail_claims=False, passed=True)
    (out/'result.json').write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps(dict(passed=True, overhead_qualified=qualified, result=str(out/'result.json'))))


if __name__ == '__main__':
    main()

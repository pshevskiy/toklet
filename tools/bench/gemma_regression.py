#!/usr/bin/env python3
"""Offline regression gate for two independently collected Gemma Chat bundles.

Consumes the O1/O2 pinned measurement format; never fetches models or promotes
an optimization. Disabled-mode caller latency is the acceptance measurement.
"""
import argparse
import hashlib
import json
from pathlib import Path
import random
import statistics

from request_cost_report import digest, parse, read_bounded, report_bundle, require

DRIVERS = ('tools/experiments/o1_gemma_summary.py','apple/tests/swift/RequestSummaryMain.swift')
FILES = ('result.json', 'samples-1.json', 'samples-1.json.summaries.jsonl',
         'samples-2.json', 'samples-2.json.summaries.jsonl')


def bundle(folder):
    report = report_bundle(folder)  # Also validates observed phases, work and clocks.
    result = parse(read_bounded(folder/'result.json',1024*1024))
    require(type(result.get('overhead_qualified')) is bool, 'Missing overhead qualification state')
    pins = result['source_sha256']
    require(type(pins) is dict, 'Invalid measurement source identities')
    require(all(digest(pins.get(path),64) for path in DRIVERS), 'Missing measurement driver identities')
    require(isinstance(result['os'],str) and result['os'], 'Missing OS identity')
    require(isinstance(result['hardware'],str) and result['hardware'], 'Missing hardware identity')
    require(isinstance(result['power'],str) and bool(result['power'].splitlines())
            and result['power'].splitlines()[0] == "Now drawing from 'AC Power'",
            'Comparable AC power receipt required')
    samples = [parse(read_bounded(folder/f'samples-{batch}.json',1024*1024)) for batch in (1,2)]
    expected = [818,5279,529,7001,563,5213,50429,84750]
    require(all(row['tokens'] == expected and (row['history'],row['committed'],row['pending']) == (24,23,1)
                for rows in samples for row in rows), 'Frozen Gemma Chat work differs')
    ids = [row['request_id'] for rows in samples for row in rows]
    require(all(isinstance(identity,str) and 0 < len(identity) <= 128 for identity in ids)
            and len(set(ids)) == 52, 'Repeated or invalid request identities')
    require(report['model']['profile'] == 'google/gemma-4-E2B-it BF16 managed Mac G2'
            and report['model']['revision'] == '70af34e20bd4b7a91f0de6b22675850c43922a03'
            and report['model']['manifest_sha256'] == '654dc6dfd66fd893a24f0faa39355b71d5e7ffc3f53c2dac394dfcfd9ed95298',
            'Unsupported regression route')
    fingerprint = hashlib.sha256()
    for path in FILES:
        fingerprint.update(path.encode()+b'\0'); fingerprint.update(read_bounded(folder/path,1024*1024))
    # Ignore metadata, whitespace and request-ID relabeling when detecting reuse
    # of the same raw clock/work observations. Both checks must pass.
    def observation(row):
        value = {key:int(row[key]) for key in ('trial','submitted_uptime_ns','completed_uptime_ns','history','committed','pending')}
        value.update(mode=row['mode'],warmup=row['warmup'],tokens=list(map(int,row['tokens'])))
        return value
    data = [[observation(row) for row in sorted(rows,key=lambda r:(r['trial'],r['mode']))] for rows in samples]
    measured_hash = hashlib.sha256(json.dumps(data,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
    return report, result, samples, fingerprint.hexdigest(), measured_hash, set(ids)


def bootstrap_difference(baseline, candidate):
    # Independent request samples, not falsely paired across different runs.
    rng = random.Random(1)
    differences = sorted(100*(statistics.median(rng.choices(candidate,k=len(candidate))) /
                             statistics.median(rng.choices(baseline,k=len(baseline)))-1)
                         for _ in range(10000))
    return [differences[250],differences[9749]]


def compare(baseline_folder, candidate_folder, baseline_commit, candidate_commit):
    require(digest(baseline_commit,40) and digest(candidate_commit,40), 'Exact baseline/candidate revision references required')
    old, new = bundle(baseline_folder), bundle(candidate_folder)
    br, bm, bs, bh, bmh, bids = old; cr, cm, cs, ch, cmh, cids = new
    require(bh != ch and bmh != cmh and bids.isdisjoint(cids), 'The same or overlapping measurement collection cannot confirm itself')
    require(br['model'] == cr['model'] and bm['hardware'] == cm['hardware'] and bm['os'] == cm['os'],
            'Model/hardware/OS scope mismatch')
    require(all(bm['source_sha256'][path] == cm['source_sha256'][path] for path in DRIVERS), 'Measurement boundaries changed')
    batches = []
    for batch in range(2):
        metrics = {}
        for metric in ('ttft_ms','total_ms'):
            before = [r[metric] for r in bs[batch] if not r['warmup'] and r['mode'] == 'disabled']
            after = [r[metric] for r in cs[batch] if not r['warmup'] and r['mode'] == 'disabled']
            require(len(before) == len(after) == 10 and min(before+after) > 0, 'Incomplete acceptance sample')
            median_change = 100*(statistics.median(after)/statistics.median(before)-1)
            interval = bootstrap_difference(before,after)
            # Five percent is the existing screening threshold. Ten samples can
            # only screen this fixed request; they do not establish tail or 1% gates.
            state = 'regression' if median_change >= 5 and interval[0] > 0 else (
                'speed_signal' if median_change <= -5 and interval[1] < 0 else (
                    'noisy' if abs(median_change) >= 5 or (interval[0] <= -5 and interval[1] >= 5) else 'within_threshold'))
            metrics[metric] = dict(baseline_median=statistics.median(before),candidate_median=statistics.median(after),
                change_pct=median_change,independent_bootstrap_95_pct=interval,status=state,
                baseline_range=[min(before),max(before)],candidate_range=[min(after),max(after)])
        batches.append(dict(batch=batch+1,metrics=metrics))
    states = [m['status'] for b in batches for m in b['metrics'].values()]
    # A regression in either confirmation batch is actionable; inconsistent
    # direction is noisy, never an accepted improvement.
    total_states = [b['metrics']['total_ms']['status'] for b in batches]
    if 'regression' in states:
        status = 'regression'
    elif 'noisy' in states or ('speed_signal' in total_states and len(set(total_states)) != 1):
        status = 'noisy'
    else:
        status = 'pass'
    return dict(format='toklet-gemma-regression-v1',status=status,primary_metric='disabled_mode_caller_total_ms',
        screening_threshold_pct=5,baseline_commit=baseline_commit,candidate_commit=candidate_commit,
        baseline_bundle_sha256=bh,candidate_bundle_sha256=ch,
        baseline_measurements_sha256=bmh,candidate_measurements_sha256=cmh,
        baseline_build=bm['build_id'],candidate_build=cm['build_id'],model=br['model'],hardware=bm['hardware'],os=bm['os'],
        batches=batches,repeatable_primary_speed_signal=all(s == 'speed_signal' for s in total_states),
        accepted_optimization=False,
        revision_reference_scope='Caller-supplied Git references; exact measured inputs are identified by bundle build/archive/source hashes, not inferred from these references',
        missing_acceptance=['Broader admitted workloads, independent numerical/task quality, memory and UI acceptance remain separate required gates',
                            'No diagnostic GPU profile is used in acceptance timing',
                            'No thermal equivalence or p95/p99 qualification from this legacy bundle format'],
        summary_overhead_qualified=bool(cm.get('overhead_qualified',False)),
        evidence=dict(baseline=str(baseline_folder.resolve()),candidate=str(candidate_folder.resolve())))


def markdown(report):
    lines = ['# Gemma regression screening','', 'Status: '+report['status']+'.',
             'Primary metric: disabled-mode caller request latency. Negative change is faster.','',
             '| Batch | Metric | Baseline ms | Candidate ms | Change % | 95% bootstrap % | Result |',
             '| --- | --- | ---: | ---: | ---: | --- | --- |']
    for batch in report['batches']:
        for metric,value in batch['metrics'].items():
            interval = ', '.join(f'{x:.2f}' for x in value['independent_bootstrap_95_pct'])
            lines.append(f"| {batch['batch']} | {metric} | {value['baseline_median']:.3f} | {value['candidate_median']:.3f} | {value['change_pct']:.2f} | [{interval}] | {value['status']} |")
    lines += ['', 'Accepted optimization: false. This job screens one frozen workload; it does not replace the remaining quality, memory, workload and UI gates.', '',
              *['- '+item for item in report['missing_acceptance']], '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('baseline',type=Path)
    parser.add_argument('candidate',type=Path)
    parser.add_argument('output',type=Path,help='new output directory')
    parser.add_argument('--baseline-commit',required=True)
    parser.add_argument('--candidate-commit',required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    try:
        report = compare(args.baseline,args.candidate,args.baseline_commit,args.candidate_commit)
        serialized = json.dumps(report,indent=2,allow_nan=False)+'\n'
        rendered = markdown(report)
    except (ValueError,KeyError,TypeError,OSError,ArithmeticError,RecursionError) as error:
        (args.output/'failure.json').write_text(json.dumps(dict(status='incomparable',reason=str(error)[:2048]),indent=2)+'\n')
        raise SystemExit(1) from error
    (args.output/'regression.json').write_text(serialized)
    (args.output/'regression.md').write_text(rendered)
    print(json.dumps(dict(status=report['status'],accepted_optimization=False)))
    raise SystemExit({'pass':0,'regression':2,'noisy':3}[report['status']])


if __name__ == '__main__':
    main()

"""Summarize the four completed frozen short records; no numerical execution."""
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
ROOT = Path.cwd()
METRICS = ('prefill_seconds', 'ttft_seconds', 'decode_seconds', 'total_seconds')
rows = []
for model in ('small', 'larger'):
    records = {plan: json.loads((ROOT/f'artifacts/m6-context-{model}-{plan}-v1.json').read_text())
               for plan in ('matrix', 'confirmation')}
    assert all(r['passed'] is True and r['collection_completed'] is True for r in records.values())
    primary = {plan: next(w for w in r['workloads'] if w['name'] == 'medium') for plan, r in records.items()}
    assert all(w['warmups'] == 3 and w['measured_pairs'] == 10 for w in primary.values())
    assert records['matrix']['worker']['ready'] == records['confirmation']['worker']['ready'] or all(
        records['matrix']['worker']['ready'][k] == records['confirmation']['worker']['ready'][k]
        for k in records['matrix']['worker']['ready'] if k != 'load_seconds')
    metrics = {}
    for key in METRICS:
        runs = {}
        for plan, w in primary.items():
            summary = w['summary']
            ref, online = summary['reference'][key], summary['online'][key]
            assert ref['count'] == online['count'] == 10
            gain = (ref['median']-online['median'])/ref['median']
            measured = [s for s in w['samples'] if s['phase'] == 'sample']
            by_index = {}
            for sample in measured:
                by_index.setdefault(sample['index'], {})[sample['mode']] = sample['sample'][key]
            assert len(by_index) == 10 and all(set(pair) == {'reference', 'online'} for pair in by_index.values())
            paired = [(pair['reference']-pair['online'])/pair['reference'] for pair in by_index.values()]
            runs[plan] = dict(reference=ref, online=online, median_reduction_fraction=gain,
                              at_least_five_percent=gain >= .05,
                              median_gap_exceeds_sum_mad=ref['median']-online['median'] > ref['mad']+online['mad'],
                              paired_reduction_minimum=min(paired), paired_reduction_maximum=max(paired))
        separated = min(run['reference']['minimum'] for run in runs.values()) > max(run['online']['maximum'] for run in runs.values())
        metrics[key] = dict(runs=runs, combined_observed_ranges_separated=separated,
                            both_collections_at_least_five_percent=all(run['at_least_five_percent'] for run in runs.values()))
    screen = []
    for workload in records['matrix']['workloads']:
        ref = workload['summary']['reference']['total_seconds']['median']
        online = workload['summary']['online']['total_seconds']['median']
        screen.append(dict(name=workload['name'], warmups=workload['warmups'], measured_pairs=workload['measured_pairs'],
                           reference_seconds=ref, online_seconds=online, median_reduction_fraction=(ref-online)/ref))
    rows.append(dict(model=model, metrics=metrics, regression_screen=screen,
                     max_admission_bytes=max(r['worker']['ready']['unique_memory_admitted_bytes'] for r in records.values()),
                     max_joint_native_bytes=max(u['joint_native_lifetime_peak_bytes'] for r in records.values() for u in r['worker']['usage']),
                     raw_events=sum(1+sum(len(p['events']) for p in r['worker']['phases']) for r in records.values()),
                     device=records['matrix']['worker']['ready']['device'],
                     platform=records['matrix']['platform'], compiler=records['matrix']['compiler'],
                     collector_revisions={p:r['revision'] for p,r in records.items()}))
result = dict(format='toklet-m6-short-summary-v1', models=rows,
              interpretation='Frozen512/128 workload, reported primary complete numerical request. Show unchanged5% useful-gain check for all four recorded metrics, avoiding favorable metric selection; no tail/application/external ranking.',
              all_reported_metrics_confirmed_useful=all(m['both_collections_at_least_five_percent'] and m['combined_observed_ranges_separated'] and all(r['median_gap_exceeds_sum_mad'] for r in m['runs'].values()) for row in rows for m in row['metrics'].values()),
              m6_complete=False)
result['confirmed_complete_request_useful'] = all(row['metrics']['total_seconds']['both_collections_at_least_five_percent'] and row['metrics']['total_seconds']['combined_observed_ranges_separated'] and all(run['median_gap_exceeds_sum_mad'] for run in row['metrics']['total_seconds']['runs'].values()) for row in rows)
result['all_median_metrics_exceed_five_percent_and_sum_mad'] = all(metric['both_collections_at_least_five_percent'] and all(run['median_gap_exceeds_sum_mad'] for run in metric['runs'].values()) for row in rows for metric in row['metrics'].values())
result['strict_flag_definition'] = 'all_reported_metrics_confirmed_useful additionally requires complete observed range separation for EVERY metric. This stricter diagnostic is false for small prefill/TTFT; it is not the frozen contract acceptance gate.'
with args.output.open('x') as stream:
    json.dump(result, stream, indent=2, allow_nan=False); stream.write('\n')
print('Confirmed useful across all recorded primary metrics:', result['all_reported_metrics_confirmed_useful'])
for row in rows:
    print(row['model'], row['raw_events'], row['max_admission_bytes'], row['max_joint_native_bytes'])
    for key, metric in row['metrics'].items():
        print(key, {run: round(s['median_reduction_fraction']*100, 6) for run,s in metric['runs'].items()}, 'ranges_separated', metric['combined_observed_ranges_separated'])

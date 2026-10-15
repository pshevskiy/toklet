"""Require the complete frozen M6 context matrix before reporting its coverage.

Replay all saved native collections first. This checks the 32 long-context
records, not application/task quality, KV adoption or overall M6 completion.
Long timings remain descriptive regardless of their observed reductions.
"""
import argparse
import json
from pathlib import Path
import statistics

import m6_context_audit as audit
import m6_long_quality_audit as quality

PLANS = ('quality', 'profile', 'long', 'long-confirmation')
KEYS = {(model, plan, depth) for model in quality.MODELS for plan in PLANS for depth in quality.DEPTHS}
THRESHOLD = .20  # Exact unchanged frozen M6 profile threshold, not a DRAM measurement.


def summarize(records):
    """Accept only records already verified by the exact native-event replay."""
    indexed = {}
    for record in records:
        key = record.get('model'), record.get('plan'), record.get('depth')
        if type(key[2]) is not int or key not in KEYS or key in indexed:
            raise ValueError('Unknown or duplicate long-context role')
        indexed[key] = record
    if set(indexed) != KEYS:
        raise ValueError('Require all32 long-context model/plan/depth records')
    q = quality.summarize([indexed[model, 'quality', depth]
                           for model in quality.MODELS for depth in quality.DEPTHS])
    models = {}
    for model in quality.MODELS:
        depths = []
        for depth in quality.DEPTHS:
            profile = indexed[model, 'profile', depth]
            medians = {}
            timings = {}
            for mode in audit.base.MODES:
                rows = [r for r in profile['profiles'] if r['mode'] == mode]
                if (len(rows) != 3 or any(type(r['index']) is not int for r in rows)
                        or [r['index'] for r in rows] != [0, 1, 2]):
                    raise ValueError('Exactly three ordered profiles per mode required')
                fractions = [quality.number(r['attention_fraction']) for r in rows]
                if any(not 0 <= f <= 1 for f in fractions):
                    raise ValueError('Attention fraction outside matching completed GPU command')
                medians[mode] = statistics.median(fractions)
                if quality.number(profile['profile_medians'][mode]) != medians[mode]:
                    raise ValueError('Profile median differs')
            if len(profile['profiles']) != 6:
                raise ValueError('Profile mode coverage differs')
            for plan, warmups, measured in (('long', 1, 2), ('long-confirmation', 0, 1)):
                workloads = indexed[model, plan, depth]['workloads']
                if len(workloads) != 1 or workloads[0]['name'] != 'boundary':
                    raise ValueError('Long boundary workload coverage differs')
                workload = workloads[0]
                if (type(workload['warmups']) is not int or workload['warmups'] != warmups
                        or type(workload['measured_pairs']) is not int or workload['measured_pairs'] != measured):
                    raise ValueError('Long warmup/measurement count differs')
                # Raw phase/state/capacity semantics are checked by audit.replay;
                # preserve all per-mode summaries without selecting a winner.
                timings[plan] = workload['summary']
            depths.append(dict(depth=depth, attention_fraction_medians=medians,
                               descriptive_timings=timings, performance_ranking_qualified=False))
        maximum = depths[-1]['attention_fraction_medians']['online']
        models[model] = dict(depths=depths, maximum_depth=32768,
                            maximum_depth_online_attention_fraction=maximum,
                            frozen_profile_threshold=THRESHOLD,
                            conditional_kv_investigation_required=maximum > THRESHOLD)
    return dict(format='toklet-m6-context-scope-audit-v1', passed=q['passed'],
                complete_frozen_context_coverage=True, record_count=32, long_quality=q, models=models,
                kv_retention_qualified=False, m6_complete=False,
                scope='Replayed long numerical context evidence only; timings descriptive; profile fraction is not DRAM traffic',
                final_synthesis_evidence_outside_this_audit=['actual32K tokenizer-owning application evidence',
                    'already accepted short-performance/full-loss/full-task evidence; no rerun implied',
                    'reviewed KV tradeoff decision and overall M6 acceptance'])


def run(paths):
    if len(paths) != 32:
        raise ValueError('Require exactly32 long-context collections')
    source = Path(__file__).resolve()
    _, source_sha = audit.read_bytes(source)
    quality_source = Path(quality.__file__).resolve()
    _, quality_sha = audit.read_bytes(quality_source)
    loaded = [(path.resolve(), *audit.read_json(path)) for path in paths]
    proof = audit.audit(paths)
    for path, _, sha in loaded:
        if proof['bindings'].get(str(path.relative_to(audit.ROOT))) != sha:
            raise ValueError('Parsed context collection differs from replayed bytes')
    result = summarize([record for _, record, _ in loaded])
    pins = dict(proof['bindings'])
    pins[str(source.relative_to(audit.ROOT))] = source_sha
    pins[str(quality_source.relative_to(audit.ROOT))] = quality_sha
    audit.verify_bindings(pins)
    result.update(bindings=pins, replay=proof, audit_source_sha256=source_sha)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('records', type=Path, nargs='+')
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Exclusive context-scope audit evidence exists')
    result = run(args.records)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    if not result['passed']:
        raise SystemExit('Long quality gate failed or winner investigations remain open; see saved audit')

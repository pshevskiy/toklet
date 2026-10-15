"""Check complete M6 long-loss coverage and explain saved winner reversals.

Replay every collection with the qualified auditor first. This additional check
requires four frozen depths and 512 observations per model. A changed winner
still requires review: two-candidate arithmetic does not identify an internal
cause, establish task quality, or replace a missing full-distribution capture.
"""
import argparse
import json
import math
from pathlib import Path

import m6_context_audit as audit

DEPTHS = (4096, 8192, 16384, 32768)
MODELS = ('small', 'larger')
TARGETS = 128
LOSS_LIMIT = .05  # Unchanged frozen M6 contract.


def number(value):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError('Invalid finite comparison value')
    return value


def winner_accounting(row):
    """Validate saved winners and account for a reversal without causal claims."""
    a, b = row['reference_greedy'], row['online_greedy']
    if any(type(v) is not int or not 0 <= v < 151936 for v in (a, b)):
        raise ValueError('Invalid winner ID')
    ar, br = row['reference_winner_logits'], row['online_winner_logits']
    if any(type(v) is not list or len(v) != 2 for v in (ar, br)):
        raise ValueError('Invalid winner logit pairs')
    for value in ar + br:
        number(value)
    margin = number(row['reference_margin'])
    error = number(row['max_absolute_logit_delta'])
    rms = number(row['rms_logit_delta'])
    if margin < 0 or error < 0 or not 0 <= rms <= error:
        raise ValueError('Invalid margin/error ordering')
    ref_gap, online_gap = ar[0] - br[0], br[1] - ar[1]
    if ref_gap < 0 or online_gap < 0:
        raise ValueError('Saved winner loses to the other candidate')
    shifts = [ar[1] - ar[0], br[1] - br[0]]
    if any(abs(v) > error + 1e-12 for v in shifts):
        raise ValueError('Winner shift exceeds full-vocabulary maximum')
    if a == b:
        if ar != br:
            raise ValueError('Matching winner IDs have different saved logits')
        return None
    if margin > ref_gap:
        raise ValueError('Top-two margin exceeds the gap to another candidate')
    if (ref_gap == 0 and a > b) or (online_gap == 0 and b > a):
        raise ValueError('Winner tie violates lowest-ID ordering')
    differential = shifts[1] - shifts[0]
    if not math.isclose(differential, ref_gap + online_gap, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError('Saved reversal arithmetic is inconsistent')
    return dict(target_index=row['target_index'], target_token=row['target_token'],
                prefix_tokens=row['prefix_tokens'], reference_winner=a, online_winner=b,
                reference_margin=margin, reference_candidate_gap=ref_gap,
                online_reversed_gap=online_gap, reference_winner_logit_shift=shifts[0],
                online_winner_logit_shift=shifts[1], differential_shift=differential,
                max_absolute_logit_delta=error, rms_logit_delta=rms,
                nll_change=number(row['online_nll'])-number(row['reference_nll']),
                numerical_reversal_accounted=True, investigation_complete=False,
                scope='Two saved candidates only; no internal-cause or full-distribution reconstruction claim')


def summarize(records):
    """Summarize records already replayed against exact frozen inputs."""
    indexed = {}
    for record in records:
        key = record.get('model'), record.get('depth')
        if (key[0] not in MODELS or type(key[1]) is not int or key[1] not in DEPTHS
                or record.get('plan') != 'quality' or key in indexed):
            raise ValueError('Unknown, duplicate or non-quality coverage')
        indexed[key] = record
    if set(indexed) != {(model, depth) for model in MODELS for depth in DEPTHS}:
        raise ValueError('Require all four frozen depths for both models')
    models = {}
    for model in MODELS:
        depths, pooled, observations, changes = [], [], set(), []
        for depth in DEPTHS:
            record = indexed[model, depth]
            phases = record['worker']['phases']
            if len(phases) != 1 or phases[0]['request']['action'] != 'compare':
                raise ValueError('Expected exactly one paired comparison phase')
            events = phases[0]['events']
            if (len(events) != TARGETS+2 or events[0].get('event') != 'comparison_begin'
                    or events[-1].get('event') != 'comparison_done'):
                raise ValueError('Comparison event coverage differs')
            deltas, depth_changes = [], []
            for index, row in enumerate(events[1:-1]):
                if (row.get('event') != 'comparison_row' or type(row.get('target_index')) is not int
                        or row['target_index'] != index or row.get('prefix_tokens') != depth-TARGETS+index):
                    raise ValueError('Depth/target observation differs')
                observations.add((depth, index))
                deltas.append(number(row['online_nll'])-number(row['reference_nll']))
                change = winner_accounting(row)
                if change is not None:
                    change['depth'] = depth
                    depth_changes.append(change)
            mean = math.fsum(deltas)/TARGETS
            depths.append(dict(depth=depth, scored_targets=TARGETS, mean_nll_increase=mean,
                               gate_passed=mean <= LOSS_LIMIT, argmax_changes=depth_changes,
                               max_absolute_logit_delta=max(row['max_absolute_logit_delta'] for row in events[1:-1])))
            pooled.extend(deltas)
            changes.extend(depth_changes)
        if len(observations) != 512 or len(pooled) != 512:
            raise ValueError('Pooled depth/target coverage differs')
        mean = math.fsum(pooled)/512
        models[model] = dict(depths=depths, scored_targets=512, pooled_mean_nll_increase=mean,
                             pooled_gate_passed=mean <= LOSS_LIMIT,
                             each_depth_gate_passed=all(row['gate_passed'] for row in depths),
                             argmax_change_count=len(changes), argmax_investigations_complete=not changes)
    gates = all(row['pooled_gate_passed'] and row['each_depth_gate_passed'] for row in models.values())
    investigations = all(row['argmax_investigations_complete'] for row in models.values())
    return dict(format='toklet-m6-long-quality-audit-v1', models=models,
                complete_depth_target_coverage=True, frozen_nll_gates_passed=gates,
                argmax_investigations_complete=investigations, passed=gates and investigations,
                m6_complete=False,
                scope='Long numerical quality only; any winner changes require review of the reported observations')


def run(paths):
    if len(paths) != len(MODELS)*len(DEPTHS):
        raise ValueError('Require exactly eight long-quality collections')
    source = Path(__file__).resolve()
    _, source_sha = audit.read_bytes(source)
    loaded = [(path.resolve(), *audit.read_json(path)) for path in paths]
    proof = audit.audit(paths)
    for path, _, sha in loaded:
        if proof['bindings'].get(str(path.relative_to(audit.ROOT))) != sha:
            raise ValueError('Parsed collection differs from replayed bytes')
    result = summarize([record for _, record, _ in loaded])
    pins = dict(proof['bindings'])
    pins[str(source.relative_to(audit.ROOT))] = source_sha
    audit.verify_bindings(pins)
    result.update(bindings=pins, replay=proof, audit_source_sha256=source_sha)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('records', type=Path, nargs='+')
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Exclusive long-quality audit evidence exists')
    result = run(args.records)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    if not result['passed']:
        raise SystemExit('Long-quality gates failed or winner investigations remain open; see saved audit')

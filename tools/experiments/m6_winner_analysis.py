"""Independent bounded FP64 analysis of exact captured FP32 distributions.

Only captures matching the original full-vocabulary SHA are admissible. This
explains an observed winner reversal without identifying a hidden-layer cause,
changing numerical gates, or making a model/task-quality acceptance decision.
"""
import hashlib
import math
import struct

from m6_fp32_context_audit import fp32
from m6_long_quality_audit import winner_accounting

VOCABULARY = 151936


def distribution(path, expected_sha, vocabulary=VOCABULARY):
    if type(vocabulary) is not int or not 2 <= vocabulary <= VOCABULARY:
        raise ValueError('Invalid bounded capture vocabulary')
    with path.open('rb') as stream:
        data = stream.read(vocabulary*4+1)
    if len(data) != vocabulary*4 or hashlib.sha256(data).hexdigest() != expected_sha:
        raise ValueError('Capture differs from original full-distribution identity')
    values = list(struct.unpack(f'<{vocabulary}f', data))
    if any(not math.isfinite(v) for v in values):
        raise ValueError('Nonfinite captured distribution')
    return values


def top(values):
    # Independent complete ordering; descending score, lowest ID on ties.
    ordered = sorted(range(len(values)), key=lambda i: (-values[i], i))
    winner, runner_up = ordered[:2]
    ties = [i for i in ordered if values[i] == values[winner]]
    return dict(winner=winner, runner_up=runner_up,
                winner_value=values[winner], runner_up_value=values[runner_up],
                margin=values[winner]-values[runner_up], tie_count=len(ties),
                first_tied_ids=ties[:16], top_five_ids=ordered[:5])


def nll(values, target):
    maximum = max(values)
    return maximum-values[target]+math.log(math.fsum(math.exp(v-maximum) for v in values))


def same(actual, expected, label, *, reduction=False):
    # Only independent FP64 sum/exp/sqrt replay receives a 1e-12 rounding
    # comparison. Exact FP32 values, maxima, margins, IDs and hashes stay exact.
    good = (math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12)
            if reduction else type(actual) is type(expected) and actual == expected)
    if not good:
        raise ValueError(f'Captured full-distribution {label} differs')


def analyze(reference, online, row, *, vocabulary=VOCABULARY):
    a = distribution(reference, row['reference_logits_sha256'], vocabulary)
    b = distribution(online, row['online_logits_sha256'], vocabulary)
    target = row['target_token']
    if type(target) is not int or not 0 <= target < vocabulary:
        raise ValueError('Capture target outside vocabulary')
    ta, tb = top(a), top(b)
    same(ta['winner'], row['reference_greedy'], 'Reference argmax')
    same(tb['winner'], row['online_greedy'], 'Online argmax')
    same(ta['margin'], row['reference_margin'], 'Reference top-two margin')
    restored = dict(row)
    bit_pairs = {}
    for key, token in (('reference_winner_logits', ta['winner']), ('online_winner_logits', tb['winner'])):
        pair = row[key]
        if type(pair) is not list or len(pair) != 2:
            raise ValueError('Malformed captured winner pair')
        reconstructed = [fp32(value) for value in pair]
        exact = [v for v, _ in reconstructed]
        captured_bits = [f'{struct.unpack("<I", struct.pack("<f", value))[0]:08x}' for value in (a[token], b[token])]
        if [bits for _, bits in reconstructed] != captured_bits:
            raise ValueError('Captured winner pair bits differ from producer-typed values')
        restored[key] = exact
        bit_pairs[key] = [bits for _, bits in reconstructed]
    deltas = [abs(x-y) for x, y in zip(a, b, strict=True)]
    maximum = max(deltas)
    rms = math.sqrt(math.fsum(d*d for d in deltas)/vocabulary)
    na, nb = nll(a, target), nll(b, target)
    same(maximum, row['max_absolute_logit_delta'], 'maximum error')
    same(rms, row['rms_logit_delta'], 'RMS error', reduction=True)
    same(na, row['reference_nll'], 'Reference NLL', reduction=True)
    same(nb, row['online_nll'], 'Online NLL', reduction=True)
    reversal = winner_accounting(restored)
    if reversal is None:
        raise ValueError('Diagnostic requires an original changed winner')
    return dict(target_index=row['target_index'], target_token=target, prefix_tokens=row['prefix_tokens'],
                reference_logits_sha256=row['reference_logits_sha256'], online_logits_sha256=row['online_logits_sha256'],
                full_vocabulary_values_verified=vocabulary, original_distributions_reproduced_exactly=True,
                reference_top=ta, online_top=tb, winner_bits_hex=bit_pairs,
                independent_max_absolute_logit_delta=maximum, independent_rms_logit_delta=rms,
                independent_reference_nll=na, independent_online_nll=nb,
                numerical_reversal=reversal, investigation_complete=False,
                scope='Full distributions reproduce the original hashes and verify both winners, top-two margin, candidate bits and metrics; internal numerical cause and task quality not inferred')

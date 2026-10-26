"""Independently replay bounded-chunk logits and memory against frozen fixtures."""
from pathlib import Path
import argparse
import json
import math
import re
import check_gemma_hybrid_tiny as fixture

ROOT = Path(__file__).resolve().parents[2]
TEST = 'model::gemma_packed::tests::chunk::bounded_chunk_memory_full_prefix_oracle_and_transaction_recovery'


def records(raw):
    fixture.require(len(raw) <= 1024**2, 'bounded chunk transcript required')
    result = []; wrapper = []
    for line in raw.decode().splitlines():
        if 'GEMMA_CHUNK_SCRATCH_' in line:
            before, body = line.split('GEMMA_CHUNK_SCRATCH_', 1)
            fixture.require(before in ('', 'test ' + TEST + ' ... '), 'exact chunk test prefix required')
            label, body = body.split(' ', 1); result.append((label, fixture.document(body)))
        elif line.strip(): wrapper.append(line.strip())
    fixture.require(len(wrapper) == 3 and wrapper[:2] == ['running 1 test', 'ok'] and
                    re.fullmatch(r'test result: ok\. 1 passed; 0 failed; 0 ignored; 0 measured; \d+ filtered out; finished in \d+\.\d+s', wrapper[2]),
                    'exact successful chunk wrapper required')
    return result


def verify(raw, *, owned_context=False):
    data, binary = fixture.fixture(ROOT / 'tests/fixtures/gemma-packed-graph-v1')
    events = iter(records(raw)); summaries = []; compared = rows_checked = 0; maximum = 0.
    def take(label):
        item = next(events, None)
        fixture.require(item is not None and item[0] == label, 'complete ordered chunk records required')
        return item[1]
    for chunk in (1, 3, 8, 32):
        m = take('MEMORY'); fixture.same(m['chunk_rows'], chunk); fixture.same(m['context_positions'], 32)
        fields = {'chunk_rows', 'context_positions', 'model_bytes', 'budget', 'budget_minus_one_rejected',
                             'planned_session_bytes', 'actual_session_bytes', 'session_header_bytes',
                             'cache_owner_capacity', 'cache_owner_bytes', 'staging_owner_capacity', 'staging_owner_bytes',
                             'cache_payload_bytes', 'staging_payload_bytes', 'scratch_elements', 'budget_minus_one_allocations'}
        if owned_context: fields |= {'route_heap_bytes', 'history_bytes'}
        fixture.same(set(m), fields)
        expected = ([chunk * 64] * 4 + [chunk * 2 * 128] * 3 + [chunk * 128] * 2
                    + [chunk * 4 * 64] * 4 + [chunk * 64] + [chunk * 64] * 2
                    + [chunk * 128] * 3 + [2 * chunk * 32] * 2 + [chunk * 17] * 2)
        fixture.require(m['scratch_elements'] == expected and all(type(v) is int for v in m['scratch_elements']), 'independent chunk scratch shape differs')
        fixture.same(m['cache_payload_bytes'], 34816)
        fixture.same(m['staging_payload_bytes'], 2 * chunk * (64 + 128) * 4)
        for key in ('session_header_bytes', 'cache_owner_bytes', 'staging_owner_bytes', 'model_bytes'):
            fixture.require(type(m[key]) is int and 0 < m[key] <= 2 * 1024**2, 'bounded chunk metadata required')
        for key in ('cache_owner_capacity', 'staging_owner_capacity'): fixture.same(m[key], 2)
        total = (m['session_header_bytes'] + m['cache_owner_capacity'] * m['cache_owner_bytes']
                 + m['staging_owner_capacity'] * m['staging_owner_bytes'] + m['cache_payload_bytes']
                 + m['staging_payload_bytes'] + 4 * sum(expected))
        if owned_context:
            fixture.same(m['history_bytes'], 32 * 4)
            fixture.require(type(m['route_heap_bytes']) is int and 0 < m['route_heap_bytes'] <= 4096, 'bounded owned route metadata required')
            total += m['route_heap_bytes'] + m['history_bytes']
        fixture.same(m['planned_session_bytes'], total); fixture.same(m['actual_session_bytes'], total)
        fixture.same(m['budget'], m['model_bytes'] + total)
        fixture.same(m['budget_minus_one_rejected'], True); fixture.same(m['budget_minus_one_allocations'], 0)
        before = 0
        while before < 32:
            row = take('ROWS'); count = min(chunk, 32 - before)
            fixture.same(set(row), {'chunk_rows', 'before', 'rows', 'values', 'forward_allocations'})
            fixture.same(row['chunk_rows'], chunk); fixture.same(row['before'], before); fixture.same(row['rows'], count)
            fixture.same(row['forward_allocations'], 0)
            fixture.require(type(row['values']) is list and len(row['values']) == count * 17 and
                            all(type(v) in (int, float) and math.isfinite(v) for v in row['values']), 'finite complete chunk values required')
            for contract in ('math', 'staged_fp32'):
                expected_values = fixture.values(data, binary, contract, 'logits', before, count)
                for actual, target in zip(row['values'], expected_values, strict=True):
                    fraction = abs(actual - target) / (1e-5 + 1e-4 * abs(target)); maximum = max(maximum, fraction)
                    fixture.require(fraction <= 1., 'independent chunk numerical gate failed'); compared += 1
            rows_checked += 1; before += count
        recovered = take('RECOVERY')
        fixture.same(recovered, dict(chunk_rows=chunk, context_overflow_rejected=True, oversized_chunk_rejected=True,
                                    prior_prefix_preserved=True, final_failure_retry_passed=True, reset_allocations=0, failure_allocations=0))
        summaries.append(dict(chunk_rows=chunk, context_positions=32, session_bytes=total,
                              logits_bytes=2 * chunk * 17 * 4, attention_scratch_bytes=2 * 2 * chunk * 32 * 4))
    fixture.require(next(events, None) is None, 'extra chunk records rejected')
    return dict(passed=True, schedules=4, row_records=rows_checked, compared_values=compared,
                maximum_gate_fraction=maximum, memory=summaries, larger_context_qualified=False,
                real_e2b_qualified=False, app_eligibility_changed=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('capture', type=Path); parser.add_argument('--owned-context', action='store_true'); args = parser.parse_args()
    print(json.dumps(verify(fixture.read(args.capture, 1024**2), owned_context=args.owned_context)))

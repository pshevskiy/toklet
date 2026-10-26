"""Independent native capacity/state replay; no engine or numerical imports."""
from pathlib import Path
import argparse
import hashlib
import json
import math
import re
import struct
import check_gemma_session_capacity_reference as oracle
import check_gemma_hybrid_tiny as packed

ROOT = Path(__file__).resolve().parents[2]
SHA = '96f1aba1335e3efd7b72f4b36fa05b7925970ef480d5e4a373b8e90787e25dfc'
TESTS = dict(kv='model::shared_kv::tests::capacity::narrowed_lifetimes_preserve_wide_logical_heads_routes_and_ownership',
             raw='model::gemma_packed::tests::capacity::admitted_contexts_match_frozen_prefixes_budgets_and_late_failure_recovery',
             generation='model::gemma_packed::generation::tests::capacity::context_sized_histories_preserve_final_pending_failure_and_replay',
             e2b='model::gemma_packed::e2b::tests::real_plan_uses_named_geometry_without_declared_context_allocations')


def records(raw, mode):
    packed.require(len(raw) <= 16 * 1024**2, 'bounded session-capacity transcript required')
    events = []; wrapper = []
    for line in raw.decode().splitlines():
        packed.require(len(line.encode()) <= 16384, 'bounded session-capacity record required')
        if 'GEMMA_SESSION_CAPACITY_' in line:
            before, body = line.split('GEMMA_SESSION_CAPACITY_', 1)
            packed.require(before in ('', 'test ' + TESTS[mode] + ' ... '), 'exact session-capacity test required')
            name, body = body.split(' ', 1); events.append((name, packed.document(body)))
        elif line.strip(): wrapper.append(line.strip())
    packed.require(len(wrapper) == 3 and wrapper[:2] == ['running 1 test', 'ok'] and
                   re.fullmatch(r'test result: ok\. 1 passed; 0 failed; 0 ignored; 0 measured; \d+ filtered out; finished in \d+\.\d+s', wrapper[2]),
                   'exact successful session-capacity wrapper required')
    return events


def counts(capacity, chunk, *, e2b=False):
    h, heads, dim, layers, ple, ff, vocab = (1536, 8, 512, 35, 256, 12288, 262144) if e2b else (64, 2, 128, 4, 64, 128, 17)
    return ([chunk*h]*4 + [chunk*heads*dim]*3 + [chunk*dim]*2 + [chunk*layers*ple]*4
            + [chunk*ple] + [chunk*h]*2 + [chunk*ff]*3 + [heads*chunk*capacity]*2 + [chunk*vocab]*2)


def verify(raw, mode):
    data, binary = oracle.frozen(ROOT / 'tests/fixtures/gemma-session-capacity-v1', SHA)
    events = iter(records(raw, mode)); observations = []
    def take(label):
        event = next(events, None)
        packed.require(event is not None and event[0] == label, 'complete ordered session-capacity records required')
        return event[1]
    if mode == 'kv':
        viewed = words = 0
        for index, case in enumerate(data['cases']):
            capacity, chunk = case['capacity'], case['chunk_rows']
            for before in range(0, capacity, chunk):
                count = min(chunk, capacity - before); end = before + count
                for variant in ('wide', 'narrow'):
                    for layer, producer in enumerate(data['routes']):
                        entry = data['producers'][producer]; dims = entry['head_dim']
                        expected = dict(case=index, capacity=capacity, chunk_rows=chunk, variant=variant,
                                        before=before, rows=count, layer=layer, producer=producer, start=0,
                                        end=end, heads=2, head_dim=dims)
                        for kind in ('k', 'v'):
                            offset = entry[kind]['offset']
                            expected[kind + '_bits'] = [struct.unpack_from('<I', binary, offset + ((head * 32 + position) * dims + feature) * 4)[0]
                                                       for head in range(2) for position in range(end) for feature in range(dims)]
                        segments = []
                        for start, keys, stride in [(0, before, (512 if entry['kind'] == 'sliding' else 32)*dims if variant == 'wide' else capacity*dims),
                                                    (before, 0, 0), (before, count, count*dims)]:
                            if not keys: start, stride = before, 0
                            segment = dict(key_start=start, keys=keys, head_stride=stride)
                            for kind in ('k', 'v'):
                                offset = entry[kind]['offset']
                                body = b''.join(binary[offset+(head*32+position)*dims*4:offset+(head*32+position+1)*dims*4]
                                                for head in range(2) for position in range(start, start+keys))
                                segment[kind+'_sha256'] = hashlib.sha256(body).hexdigest()
                            segments.append(segment)
                        expected['segments'] = segments
                        expected['visible_ranges'] = [[0, query+1] for query in range(before, end)]
                        actual = take('VIEW')
                        packed.require(actual == expected, 'independent logical KV view differs')
                        packed.same(actual, expected); viewed += 1; words += len(expected['k_bits']) + len(expected['v_bits'])
            expected = dict(case=index, capacity=capacity, chunk_rows=chunk, cache_payload_bytes=528*capacity,
                            logical_prefix_preserved=True, overflow_rejected=True, foreign_rejected=True, stale_rejected=True)
            packed.same(take('KV_COMPLETE'), expected)
        result = dict(passed=True, cases=19, view_records=viewed, compared_words=words)
        packed.same(viewed, 6860)
    elif mode == 'raw':
        old, old_binary = packed.fixture(ROOT / 'tests/fixtures/gemma-packed-graph-v1'); compared = rows_checked = 0; maximum = 0.
        for index, case in enumerate(data['cases']):
            capacity, chunk = case['capacity'], case['chunk_rows']; m = take('MEMORY')
            packed.same(set(m), {'case', 'capacity', 'chunk_rows', 'model_bytes', 'budget', 'planned_session_bytes', 'actual_session_bytes',
                                 'session_header_bytes', 'route_heap_bytes', 'history_bytes', 'cache_owner_capacity', 'cache_owner_bytes',
                                 'staging_owner_capacity', 'staging_owner_bytes', 'cache_payload_bytes', 'staging_payload_bytes',
                                 'scratch_elements', 'budget_minus_one_allocations'})
            packed.same((m['case'], m['capacity'], m['chunk_rows']), (index, capacity, chunk))
            expected = counts(capacity, chunk); packed.require(m['scratch_elements'] == expected, 'independent context scratch differs')
            packed.same(m['scratch_elements'], expected)
            cache = min(4, capacity)*64*2*4 + capacity*128*2*4; stage = chunk*(64+128)*2*4
            packed.same(m['cache_payload_bytes'], cache); packed.same(m['staging_payload_bytes'], stage); packed.same(m['history_bytes'], capacity*4)
            for key in ('session_header_bytes', 'route_heap_bytes', 'cache_owner_bytes', 'staging_owner_bytes', 'model_bytes'):
                packed.require(type(m[key]) is int and 0 < m[key] <= 2*1024**2, 'bounded actual session metadata required')
            packed.same(m['cache_owner_capacity'], 2); packed.same(m['staging_owner_capacity'], 2)
            total = m['session_header_bytes'] + m['route_heap_bytes'] + m['history_bytes'] + 2*m['cache_owner_bytes'] + 2*m['staging_owner_bytes'] + cache + stage + 4*sum(expected)
            packed.same(m['planned_session_bytes'], total); packed.same(m['actual_session_bytes'], total)
            packed.same(m['budget'], m['model_bytes'] + total); packed.same(m['budget_minus_one_allocations'], 0)
            for before in range(0, capacity, chunk):
                count = min(chunk, capacity-before); row = take('ROWS')
                packed.same(set(row), {'case', 'capacity', 'chunk_rows', 'before', 'rows', 'values', 'forward_allocations'})
                packed.same({k: row[k] for k in ('case', 'capacity', 'chunk_rows', 'before', 'rows', 'forward_allocations')},
                            dict(case=index, capacity=capacity, chunk_rows=chunk, before=before, rows=count, forward_allocations=0))
                packed.require(type(row['values']) is list and len(row['values']) == count*17 and
                               all(type(v) in (int,float) and math.isfinite(v) for v in row['values']), 'finite complete capacity logits required')
                for contract in ('math', 'staged_fp32'):
                    target = packed.values(old, old_binary, contract, 'logits', before, count)
                    for a, b in zip(row['values'], target, strict=True):
                        fraction = abs(a-b)/(1e-5+1e-4*abs(b)); maximum=max(maximum,fraction)
                        packed.require(fraction <= 1., 'independent capacity numerical gate failed'); compared += 1
                rows_checked += 1
            packed.same(take('RAW_COMPLETE'), dict(case=index, capacity=capacity, chunk_rows=chunk, overflow_rejected=True,
                                                 late_failure_prefix_preserved=True, retry_passed=True, reset_allocations=0, late_failure_allocations=0))
            observations.append(dict(capacity=capacity, chunk_rows=chunk, session_bytes=total, cache_payload_bytes=cache, history_bytes=capacity*4))
        packed.same((rows_checked, compared), (98,7956))
        result=dict(passed=True,cases=19,row_records=rows_checked,compared_values=compared,maximum_gate_fraction=maximum,memory=observations)
    elif mode == 'generation':
        old, old_binary = packed.fixture(ROOT / 'tests/fixtures/gemma-packed-graph-v1')
        for capacity in (1,2,3,4,5,17,32):
            item=take('GENERATION'); chunk=min(3,capacity); m=item['memory']
            packed.same(set(item), {'capacity','chunk_rows','memory','sampled_token','pending_history','evaluated_before_retry','pending_before_retry',
                                   'final_evaluated','single_position_shared_budget_rejected','final_pending_preserved','late_failure_retry_passed',
                                   'full_replay_agreed','frozen_prompt_restored','budget_minus_one_allocations'})
            packed.same((item['capacity'],item['chunk_rows']),(capacity,chunk))
            categories=('weights_bytes','kv_bytes','scratch_bytes','history_bytes','logits_bytes','metadata_bytes','metadata_allowance_bytes')
            packed.same(set(m),set(categories)|{'admitted_bytes'})
            packed.require(all(type(v)is int and v>=0 for v in m.values()),'bounded generation ownership required')
            packed.same(sum(m[k]for k in categories),m['admitted_bytes'])
            packed.same(m['history_bytes'],8*capacity);packed.same(m['logits_bytes'],2*chunk*17*4)
            packed.same(m['kv_bytes'],min(4,capacity)*512+capacity*1024+chunk*1536)
            packed.same(m['scratch_bytes'],4*sum(counts(capacity,chunk))-m['logits_bytes'])
            expected_history=old['token_ids'][:capacity]
            if capacity==1:packed.same(item['sampled_token'],None)
            else:
                winners=[]
                for contract in ('math','staged_fp32'):
                    values=packed.values(old,old_binary,contract,'logits',capacity-2,1)
                    winners.append(max(range(17),key=lambda i:values[i]))
                packed.same(winners[0],winners[1]);packed.same(item['sampled_token'],winners[0]);expected_history[-1]=winners[0]
            packed.same(item['pending_history'],expected_history)
            packed.same((item['evaluated_before_retry'],item['pending_before_retry'],item['final_evaluated']),(capacity-1,1,capacity))
            packed.same(item['single_position_shared_budget_rejected'],capacity==1)
            for key in ('final_pending_preserved','late_failure_retry_passed','full_replay_agreed','frozen_prompt_restored'):packed.same(item[key],True)
            packed.same(item['budget_minus_one_allocations'],0);observations.append(dict(capacity=capacity,chunk_rows=chunk,memory=m))
        result=dict(passed=True,cases=7,memory=observations)
    else:
        scratch=take('E2B_SCRATCH'); expected=counts(32,32,e2b=True)
        packed.same(scratch,dict(positions=32,chunk_rows=32,declared_positions=131072,scratch_elements=expected,scratch_bytes=4*sum(expected)))
        for capacity in (1,5,32):
            item=take('E2B_ALLOCATION')
            packed.same(item,dict(capacity=capacity,cache_payload_bytes=capacity*36864,producer_count=15,
                                  whole_model_executed=False,real_numerical_graph_qualified=False));observations.append(item)
        result=dict(passed=True,geometry_allocations=observations)
    packed.require(next(events,None)is None,'extra session-capacity records rejected')
    return dict(**result,larger_context_qualified=False,real_e2b_full32_qualified=False,app_eligibility_changed=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=TESTS);parser.add_argument('capture',type=Path);args=parser.parse_args()
    print(json.dumps(verify(packed.read(args.capture,16*1024**2),args.mode)))

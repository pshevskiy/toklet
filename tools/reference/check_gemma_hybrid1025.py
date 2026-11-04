"""Check retained physical KV, schedule consistency and independent numerics."""
from pathlib import Path
import argparse
import hashlib
import json
import re
import struct
import gemma4_e2b_wrap_reference as oracle
import check_gemma_hybrid32 as prior

ROOT = oracle.ROOT
TEST = 'model::gemma_packed::tests::hybrid::real::wrap::diagnostic1025_wrap_schedules_overflow_and_reset'
POSITIONS = [0, 31, 255, 511, 512, 767, 1023, 1024]
MILESTONES = [512, 513, 1024, 1025]
PARTITIONS = [[32]*16+[1]+[32]*15+[31, 1], [32]*8+[31]*8+[32]+[32]*15+[9]]
CONTRACTS = ['math_fp64', 'staged_fp32']


def qualified_prefix():
    proof = prior.bounded.document(prior.bounded.read(
        ROOT / 'benchmarks/results/2026-11-04-gemma-hybrid512-v1.json', 65536))
    if not (proof['passed'] and proof['state_passed'] and proof['numerical_passed']
            and proof['astra_review'].startswith('approved')):
        raise ValueError('Qualified 512-prefix evidence required')
    path = 'artifacts/gemma-hybrid512-2026-11-04/native.stdout'
    raw = prior.bounded.read(ROOT / path, 1024**2)
    if hashlib.sha256(raw).hexdigest() != proof['receipts_sha256'][path]:
        raise ValueError('Qualified prefix transcript differs')
    rows = [prior.bounded.document(line.split(b' ', 1)[1])
            for line in raw.splitlines() if line.startswith(b'GEMMA_HYBRID_ARRAY ')]
    selected = [v for v in rows if v['name'] == 'context512.initial_logits']
    if len(selected) != 1 or selected[0]['before'] != 0 or selected[0]['count'] != 32:
        raise ValueError('Qualified prefix record differs')
    return selected[0]['sha256']


def references(folder):
    record = prior.bounded.document(prior.bounded.read(folder / 'result.json', 2*1024**2))
    ids, positions = oracle.inputs('full1025')
    if (not record['completed'] or record['phase'] != 'full1025' or record['tokens'] != ids
            or record['selected_positions'] != positions
            or record['source_pins'] != oracle.source_graph.source.PINS
            or record['input_contract_sha256'] != hashlib.sha256((oracle.CONTRACT / 'manifest.json').read_bytes()).hexdigest()):
        raise ValueError('Independent wrap reference identity differs')
    raw = prior.bounded.read(folder / 'rows.bin', 160*1024**2)
    if hashlib.sha256(raw).hexdigest() != record['rows_sha256']:
        raise ValueError('Reference body digest differs')
    expected = [(c, kind, owner) for c in CONTRACTS for owner in range(15) for kind in ['v', 'k']]
    expected = [key for c in CONTRACTS for key in
                ([v for v in expected if v[0] == c] + [(c, 'logits', p) for p in POSITIONS])]
    rows = {}; offset = 0
    if len(record['arrays']) != len(expected):
        raise ValueError('Reference inventory length differs')
    for item, key in zip(record['arrays'], expected, strict=True):
        c, kind, number = key
        width = 8 if c == 'math_fp64' else 4
        if kind == 'logits':
            fields = dict(contract=c, kind=kind, position=number)
            count = 262144
        else:
            dim = 512 if number % 5 == 4 else 256
            fields = dict(contract=c, kind=kind, owner=number, tokens=1025, head_dim=dim,
                          retention='global' if number % 5 == 4 else 'local512')
            count = 1025 * dim
        fields.update(offset=offset, bytes=count*width)
        block = memoryview(raw)[offset:offset+count*width]
        if ({k: v for k, v in item.items() if k != 'sha256'} != fields
                or len(block) != count*width or hashlib.sha256(block).hexdigest() != item['sha256']):
            raise ValueError('Reference array identity differs')
        rows[key] = block; offset += len(block)
    if offset != len(raw):
        raise ValueError('Trailing reference bytes')
    return rows


def native_capture(transcript, body):
    spec = prior.bounded.document(prior.bounded.read(ROOT / 'benchmarks/gemma-hybrid1025-v1/manifest.json', 65536))
    if (spec['partitions'] != PARTITIONS or spec['selected_positions'] != POSITIONS
            or spec['retained_milestones'] != MILESTONES or spec['cross_schedule_endpoints'] != [256, 1025]):
        raise ValueError('Frozen native wrap contract differs')
    if hashlib.sha256((ROOT / spec['oracle_contract']['path']).read_bytes()).hexdigest() != spec['oracle_contract']['sha256']:
        raise ValueError('Frozen oracle contract differs')
    events = []; wrapper = []
    for line in transcript.decode().splitlines():
        if 'GEMMA_HYBRID_' in line:
            prefix, rest = line.split('GEMMA_HYBRID_', 1)
            if prefix not in ['', f'test {TEST} ... ']:
                raise ValueError('Unexpected test wrapper prefix')
            name, raw = rest.split(' ', 1)
            events.append((name, prior.bounded.document(raw.encode())))
        elif line.strip():
            wrapper.append(line)
    if (len(wrapper) != 3 or wrapper[:2] != ['running 1 test', 'ok']
            or not re.fullmatch(r'test result: ok\. 1 passed; 0 failed; 0 ignored; 0 measured; \d+ filtered out; finished in \d+\.\d+s', wrapper[2])):
        raise ValueError('Complete successful native wrapper required')
    cursor = 0; offset = 0; number = 0; main = {}; kv = {}; states = []

    def take(name):
        nonlocal cursor
        if cursor >= len(events) or events[cursor][0] != name:
            raise ValueError('Native event order differs: ' + name)
        value = events[cursor][1]; cursor += 1
        return value

    def array(name, before, count, values):
        nonlocal offset, number
        fields = dict(record=number, name=name, offset=offset, bytes=values*4,
                      before=before, count=count, values=values)
        if take('ARRAY_ATTEMPT') != fields:
            raise ValueError('Native array attempt differs')
        item = take('ARRAY'); block = memoryview(body)[offset:offset+values*4]
        if ({k: v for k, v in item.items() if k != 'sha256'} != fields
                or len(block) != values*4 or hashlib.sha256(block).hexdigest() != item['sha256']):
            raise ValueError('Native array identity differs')
        offset += len(block); number += 1
        return block

    prefix_sha = qualified_prefix()
    for index, partition in enumerate(PARTITIONS):
        if take('REAL_WRAP_SCHEDULE') != dict(index=index, chunks=partition):
            raise ValueError('Native partition differs')
        committed = 0
        for count in partition:
            if committed == 0 and take('REAL_WRAP_PREFIX') != dict(schedule=index, sha256=prefix_sha):
                raise ValueError('Qualified initial prefix differs')
            for position in POSITIONS:
                if committed <= position < committed+count:
                    block = array('wrap.logits', position, 1, 262144)
                    if index == 0:
                        main[position] = block
                    elif block != main[position]:
                        raise ValueError('Cross-schedule logit bytes differ')
            committed += count
            if index == 0 and committed in MILESTONES:
                for owner in range(15):
                    capacity = 1025 if owner % 5 == 4 else 512
                    start = max(0, committed-capacity)
                    slots = [None] * capacity
                    for absolute in range(start, committed):
                        slots[absolute % capacity] = absolute
                    if take('REAL_WRAP_SLOTS') != dict(committed=committed, owner=owner, slots=slots):
                        raise ValueError('Independent retained absolute-slot map differs')
                    dim = 512 if owner % 5 == 4 else 256
                    for kind in ['k', 'v']:
                        kv[committed, owner, kind] = array(f'wrap.layer.{owner}.{kind}', committed,
                                                         committed-start, (committed-start)*dim)
        expected = dict(schedule=index, committed=1025, overflow_preserved=True,
                        reset_all32_rows_exact=True, reset_all15_valid_kv_prefixes_exact=True,
                        allocated_bytes=271117548, cross_schedule_endpoints=[256, 1025])
        if take('REAL_WRAP_STATE') != expected:
            raise ValueError('Overflow/reset/state completion differs')
        states.append(expected)
    if (take('CAPTURE') != dict(records=136, bytes=len(body), sha256=hashlib.sha256(body).hexdigest(), whole_e2b_qualified=False)
            or number != 136 or offset != len(body)):
        raise ValueError('Complete capture identity differs')
    if take('REAL_WRAP_COMPLETE') != dict(state_passed=True, numerical_decision='separate reference replay',
            positions=1025, selected_positions=POSITIONS, retained_milestones=MILESTONES,
            prior_precision_failures_waived=False, app_qualified=False, phone_qualified=False) or cursor != len(events):
        raise ValueError('Complete native wrap verdict differs')
    return main, kv, states


def replay(transcript, body, reference_folder):
    main, kv, states = native_capture(transcript, body)
    reference = references(reference_folder)
    reports = []
    for position, block in main.items():
        reports.append(dict(kind='logits', position=position, contracts={
            c: prior.fractions(struct.unpack('<262144f', block), struct.unpack('<262144' + ('d' if c == 'math_fp64' else 'f'), reference[c, 'logits', position]))
            for c in CONTRACTS}))
    for (committed, owner, kind), block in kv.items():
        dim = 512 if owner % 5 == 4 else 256
        start = 0 if owner % 5 == 4 else max(0, committed-512)
        count = (committed-start)*dim; contracts = {}
        for c in CONTRACTS:
            width = 8 if c == 'math_fp64' else 4
            selected = reference[c, kind, owner][start*dim*width:committed*dim*width]
            stats = prior.fractions(struct.unpack('<'+str(count)+'f', block),
                                   struct.unpack('<'+str(count)+('d' if width == 8 else 'f'), selected))
            for example in stats['examples']:
                example.update(absolute_position=start+example['index']//dim, component=example['index']%dim)
            contracts[c] = stats
        reports.append(dict(kind=kind, owner=owner, committed=committed, first_absolute=start, contracts=contracts))
    numerical = {c: dict(compared_values=sum(v['contracts'][c]['values'] for v in reports),
                        failed_values=sum(v['contracts'][c]['failed_values'] for v in reports),
                        maximum_gate_ratio=max(v['contracts'][c]['maximum_gate_ratio'] for v in reports)) for c in CONTRACTS}
    return dict(state_passed=True, overflow_reset_passed=True,
                numerical_passed=all(v['failed_values'] == 0 for v in numerical.values()),
                numerical=numerical, rows=reports, state=states,
                prior_precision_failures_waived=False, app_qualified=False, phone_qualified=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('transcript', type=Path)
    parser.add_argument('arrays', type=Path)
    parser.add_argument('reference', type=Path, nargs='?')
    parser.add_argument('--state-only', action='store_true')
    args = parser.parse_args()
    transcript = prior.bounded.read(args.transcript, 2*1024**2)
    body = prior.bounded.read(args.arrays, 128*1024**2)
    if args.state_only:
        if args.reference is not None:
            parser.error('State-only replay does not consume numerical references')
        main, kv, states = native_capture(transcript, body)
        result = dict(state_passed=True, overflow_reset_passed=True, state=states,
                      selected_logit_positions=len(main), retained_kv_records=len(kv),
                      numerical_passed=None, numerical_decision='pending independent reference')
    else:
        if args.reference is None:
            parser.error('Independent numerical reference required')
        result = replay(transcript, body, args.reference)
    print(json.dumps(result, indent=2, allow_nan=False))

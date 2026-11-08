"""Offline full selected-logit/retained-KV comparison; owned receipts stay separate."""
import argparse
import json
from pathlib import Path
import re
import check_gemma4_heretic_reference as safe
import gemma4_heretic_quality_reference as reference

VOCAB = 262144
POSITIONS = [0, 31, 255, 511, 512, 767, 1023, 1024]
MILESTONES = [512, 513, 1024, 1025]
CONTRACTS = ['math_fp64', 'staged_fp32']
PARTITIONS = [[32]*16+[1]+[32]*15+[31, 1], [32]*8+[31]*8+[32]+[32]*15+[9]]
need = safe.need


def exact(got, expected):
    """JSON literals retain their types, including booleans versus zero/one."""
    if type(got) is not type(expected):
        return False
    if type(expected) is dict:
        return got.keys() == expected.keys() and all(exact(got[k],v) for k,v in expected.items())
    if type(expected) is list:
        return len(got) == len(expected) and all(exact(a,b) for a,b in zip(got,expected,strict=True))
    return got == expected


def memory(value):
    need(type(value) is int and 0 < value <= 6*1024**3, 'Typed memory guard differs')


def statistics(got, expected):
    import numpy as np
    need(got.shape == expected.shape and got.size > 0 and np.isfinite(got).all()
         and np.isfinite(expected).all(), 'Finite equal complete arrays required')
    error = np.abs(got.astype(np.float64)-expected.astype(np.float64))
    ratio = error/(1e-5+1e-4*np.abs(expected.astype(np.float64)))
    return dict(values=int(got.size), failed_values=int(np.count_nonzero(ratio > 1)),
                maximum_gate_ratio=float(ratio.max()), maximum_absolute_error=float(error.max()))


def native(folder, source):
    receipt = safe.document(folder/'checks.json')
    need(receipt['passed'] is True and type(receipt['exit_code']) is int and receipt['exit_code'] == 0 and receipt['group_absent'] is True
         and receipt['changed_inputs'] == [] and receipt['source'] == source and receipt['job'] == 'wrap',
         'Completed owned native wrap required')
    memory(receipt['lifetime_peak_physical_bytes'])
    transcript = safe.read(folder/'console.log', 4*1024**2)
    need(safe.digest(transcript) == receipt['files_sha256']['console.log'], 'Native transcript differs')
    events = {}
    for line in transcript.splitlines():
        if line.startswith(b'GEMMA_HYBRID_'):
            tag, raw = line.split(b' ', 1)
            events.setdefault(tag.decode(), []).append(safe.prior.bounded.document(raw))
    def rows(tag):
        return events['GEMMA_HYBRID_'+tag]
    need(set(events) == {'GEMMA_HYBRID_'+tag for tag in [
        'REAL_HERETIC_QUALITY_SOURCE','REAL_HERETIC_QUALITY_MEMORY','REAL_HERETIC_QUALITY_ADMISSION',
        'REAL_WRAP_COMPLETE','REAL_WRAP_SCHEDULE','REAL_WRAP_PREFIX','REAL_WRAP_STATE','REAL_WRAP_SLOTS',
        'CAPTURE','ARRAY','ARRAY_ATTEMPT']}, 'Unexpected native event inventory')
    adapter = reference.SOURCES[source]
    identity, = rows('REAL_HERETIC_QUALITY_SOURCE')
    need(identity == receipt['source_identity'][0] and identity['kind'] == source
         and identity['repository'] == adapter.REPOSITORY and identity['revision'] == adapter.REVISION
         and identity['sha256'] == adapter.FILE_SHA256 and identity['quality_qualified'] is False,
         'Native source identity differs')
    memory(identity['load_admission_bytes'])
    need(identity['load_admission_bytes']+128*1024**2 <= 6*1024**3, 'Native load admission differs')
    memories = rows('REAL_HERETIC_QUALITY_MEMORY')
    need(len(memories) == 2 and memories == receipt['native_memory'], 'Native memory coverage differs')
    for item in memories:
        memory(item['physical_bytes']); memory(item['lifetime_peak_bytes'])
        need(item['physical_bytes'] <= item['lifetime_peak_bytes'] and item['limit_bytes'] == 6*1024**3,
             'Native memory identity differs')
    admission, = rows('REAL_HERETIC_QUALITY_ADMISSION')
    memory(admission['required_bytes']); memory(admission['session_bytes'])
    need(admission['required_bytes'] > admission['session_bytes']
         and admission['required_bytes']+admission['diagnostic_bytes'] <= 6*1024**3
         and admission['diagnostic_bytes'] == 128*1024**2 and admission['capacity'] == 1025
         and admission['chunk_capacity'] == 32 and admission['minus_one_rejected'] is True,
         'Native admission differs')
    complete, = rows('REAL_WRAP_COMPLETE')
    need(exact(complete, dict(state_passed=True, numerical_decision='separate reference replay',
         positions=1025, selected_positions=POSITIONS, retained_milestones=MILESTONES,
         prior_precision_failures_waived=False, app_qualified=False, phone_qualified=False)), 'Wrap completion differs')
    need(exact(rows('REAL_WRAP_SCHEDULE'), [dict(index=i, chunks=p) for i,p in enumerate(PARTITIONS)]), 'Wrap partitions differ')
    prefixes = rows('REAL_WRAP_PREFIX')
    need(len(prefixes) == 2 and exact([p['schedule'] for p in prefixes],[0,1])
         and prefixes[0]['sha256'] == prefixes[1]['sha256'], 'Cross-schedule initial rows differ')
    need(exact(rows('REAL_WRAP_STATE'), [dict(schedule=i, committed=1025, overflow_preserved=True,
         reset_all32_rows_exact=True, reset_all15_valid_kv_prefixes_exact=True,
         allocated_bytes=admission['session_bytes'], cross_schedule_endpoints=[256,1025]) for i in range(2)]),
         'Overflow/reset/cross-schedule state differs')
    slots = rows('REAL_WRAP_SLOTS')
    need(len(slots) == 60, 'Retained slot coverage differs')
    for item, (committed, owner) in zip(slots, [(p,o) for p in MILESTONES for o in range(15)], strict=True):
        capacity = 1025 if owner%5 == 4 else 512
        expected = [None]*capacity
        for absolute in range(max(0,committed-capacity),committed):
            expected[absolute%capacity] = absolute
        need(exact(item, dict(committed=committed, owner=owner, slots=expected)), 'Absolute retained slot map differs')
    body = safe.read(folder/'arrays.bin', 128*1024**2)
    footer, = rows('CAPTURE')
    need(exact(footer, dict(records=136, bytes=len(body), sha256=safe.digest(body), whole_e2b_qualified=False))
         and safe.digest(body) == receipt['files_sha256']['arrays.bin'], 'Native capture footer differs')
    entries = rows('ARRAY'); attempts = rows('ARRAY_ATTEMPT')
    need(len(entries) == len(attempts) == 136, 'Native array coverage differs')
    result = {}; offset = 0
    for number, (item, attempt) in enumerate(zip(entries, attempts, strict=True)):
        need(all(type(item[k]) is int for k in ['record','offset','bytes','before','count','values'])
             and exact({k:v for k,v in item.items() if k != 'sha256'}, attempt) and item['record'] == number
             and item['offset'] == offset and item['bytes'] == item['values']*4, 'Native array descriptor differs')
        block = memoryview(body)[offset:offset+item['bytes']]
        need(len(block) == item['bytes'] and safe.digest(block) == item['sha256'], 'Native array body differs')
        result.setdefault((item['name'],item['before']), []).append((item['count'],block))
        offset += item['bytes']
    need(offset == len(body), 'Trailing native bytes')
    expected_keys = {('wrap.logits',p) for p in POSITIONS} | {
        (f'wrap.layer.{o}.{k}',p) for o in range(15) for k in ('k','v') for p in MILESTONES}
    need(set(result) == expected_keys, 'Native selected-state inventory differs')
    for key, copies in result.items():
        if key[0] == 'wrap.logits':
            need(len(copies) == 2 and copies[0] == copies[1] and copies[0][0] == 1
                 and len(copies[0][1]) == VOCAB*4, 'Cross-schedule full logits differ')
        else:
            owner = int(key[0].split('.')[2]); count = min(key[1],1025 if owner%5 == 4 else 512)
            need(len(copies) == 1 and copies[0][0] == count
                 and len(copies[0][1]) == count*(512 if owner%5 == 4 else 256)*4, 'Native retained geometry differs')
    return result


def references(folder, source):
    receipt = safe.document(folder/'checks.json'); data = safe.document(folder/'capture/result.json')
    ids, positions, planned = reference.plan('full1025'); adapter = reference.SOURCES[source]
    need(receipt['collection_passed'] is True and type(receipt['exit_code']) is int and receipt['exit_code'] == 0
         and receipt['changed_sources'] == [] and receipt['error'] is None
         and receipt['source'] == source and receipt['phase'] == 'full1025', 'Completed independent collection required')
    memory(receipt['lifetime_peak_physical_bytes'])
    need(data['completed'] is True and data['format'] == 'toklet-heretic-quality-reference-v1'
         and data['source_kind'] == source and data['repository'] == adapter.REPOSITORY
         and data['revision'] == adapter.REVISION and data['source_sha256'] == adapter.FILE_SHA256
         and data['phase'] == 'full1025' and data['tokens'] == ids and data['selected_positions'] == positions
         and all(type(v) is int for v in data['tokens']+data['selected_positions'])
         and data['gate'] == dict(atol=1e-5, rtol=1e-4), 'Independent source/frozen contract differs')
    need(data['source']['closed'] is True and data['source']['poisoned'] is False
         and data['source']['bytes'] == adapter.FILE_BYTES
         and data['source']['sha256'] == data['source_endpoint_sha256'] == adapter.FILE_SHA256,
         'Independent source endpoint differs')
    need(data['quality_scored'] is False and data['managed_adoption_qualified'] is False
         and data['engine_executed'] is False, 'Independent scope differs')
    body = safe.read(folder/'capture/rows.bin', reference.MAX_CAPTURE)
    need(len(body) == data['planned_rows_bytes'] == planned and safe.digest(body) == data['rows_sha256'], 'Reference body differs')
    result = {}; offset = 0
    keys = [(c,k,o) for c in CONTRACTS for o in range(15) for k in ('v','k')]
    keys = [x for c in CONTRACTS for x in [*[v for v in keys if v[0] == c], *[(c,'logits',p) for p in POSITIONS]]]
    need(len(data['arrays']) == len(keys), 'Reference array coverage differs')
    for item, (contract, kind, index) in zip(data['arrays'], keys, strict=True):
        width = 8 if contract == 'math_fp64' else 4
        if kind == 'logits':
            fields = dict(contract=contract, kind=kind, position=index); count = VOCAB
        else:
            dim = 512 if index%5 == 4 else 256; count = 1025*dim
            fields = dict(contract=contract, kind=kind, owner=index, tokens=1025, head_dim=dim,
                          retention='global' if index%5 == 4 else 'local512')
        fields.update(offset=offset, bytes=count*width)
        block = memoryview(body)[offset:offset+count*width]
        need(exact({k:v for k,v in item.items() if k != 'sha256'}, fields) and len(block) == count*width
             and safe.digest(block) == item['sha256'], 'Reference array identity differs')
        result[contract,kind,index] = block; offset += len(block)
    need(offset == len(body), 'Trailing reference bytes')
    return result


def check(native_folder, reference_folder, source):
    import numpy as np
    actual = native(native_folder, source); expected = references(reference_folder, source)
    reports = []; greedy = []
    for (name, committed), copies in actual.items():
        got = np.frombuffer(copies[0][1], dtype='<f4')
        for contract in CONTRACTS:
            dtype = '<f8' if contract == 'math_fp64' else '<f4'
            if name == 'wrap.logits':
                want = np.frombuffer(expected[contract,'logits',committed], dtype=dtype)
                greedy.append(dict(contract=contract, position=committed,
                                   native_id=int(got.argmax()), reference_id=int(want.argmax())))
            else:
                match = re.fullmatch(r'wrap\.layer\.(\d+)\.([kv])', name)
                owner, kind = int(match[1]), match[2]; dim = 512 if owner%5 == 4 else 256
                start = 0 if owner%5 == 4 else max(0,committed-512)
                want = np.frombuffer(expected[contract,kind,owner], dtype=dtype)[start*dim:committed*dim]
            reports.append(dict(name=name, committed=committed, contract=contract, **statistics(got,want)))
    mismatches = [v for v in greedy if v['native_id'] != v['reference_id']]
    return dict(selected_wrap_fidelity_passed=all(v['failed_values'] == 0 for v in reports) and not mismatches,
                rows=reports, compared_values=sum(v['values'] for v in reports), greedy=greedy,
                unresolved_greedy_mismatches=mismatches, source=source, gate=dict(atol=1e-5,rtol=1e-4),
                scope='Selected logits and all captured retained KV through1025; owned execution receipts remain separate. No full held-out quality or adoption verdict.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('native', type=Path); parser.add_argument('reference', type=Path)
    parser.add_argument('--source', choices=reference.SOURCES, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); result = check(args.native,args.reference,args.source)
    with args.output.open('x') as file:
        json.dump(result,file,indent=2,allow_nan=False); file.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('rows','greedy')}))
    raise SystemExit(0 if result['selected_wrap_fidelity_passed'] else 1)

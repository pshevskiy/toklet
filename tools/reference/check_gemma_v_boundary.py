"""Replay V projection/normalization from actual native inputs, without feedback."""
from pathlib import Path
import argparse
import hashlib
import json
import math
import re
import struct
import check_gemma_hybrid1025 as prior
import check_gemma_hybrid32 as policy

ROOT = prior.ROOT
TEST = 'model::gemma_packed::tests::hybrid::real::v_boundary::native_v_projection_norm_and_retained_rows'
CASES = [(13, 433, 256), (12, 650, 256), (14, 650, 512)]


def normalized(raw, staged):
    import numpy as np
    values = np.asarray(raw, dtype=np.float32)
    if values.ndim != 1 or len(values) == 0 or not np.isfinite(values).all():
        raise ValueError('Finite native head row required')
    epsilon = float(np.float32(1e-6))  # Same stored epsilon as the frozen graph.
    denominator = math.sqrt(math.fsum(float(v)*float(v) for v in values)/len(values)+epsilon)
    return np.asarray([float(v)/denominator for v in values], dtype=np.float32 if staged else np.float64)


def native_rows(transcript, body):
    events = []; wrapper = []
    for line in transcript.decode().splitlines():
        if 'GEMMA_HYBRID_' in line:
            prefix, rest = line.split('GEMMA_HYBRID_', 1)
            if prefix not in ['', f'test {TEST} ... ']:
                raise ValueError('Unexpected native wrapper prefix')
            name, raw = rest.split(' ', 1)
            events.append((name, policy.bounded.document(raw.encode())))
        elif line.strip():
            wrapper.append(line)
    if (len(wrapper) != 3 or wrapper[:2] != ['running 1 test', 'ok']
            or not re.fullmatch(r'test result: ok\. 1 passed; 0 failed; 0 ignored; 0 measured; \d+ filtered out; finished in \d+\.\d+s', wrapper[2])):
        raise ValueError('Successful complete witness wrapper required')
    cursor = 0; offset = 0; number = 0; rows = {}
    def take(name):
        nonlocal cursor
        if cursor >= len(events) or events[cursor][0] != name:
            raise ValueError('Witness event order differs: '+name)
        value = events[cursor][1]; cursor += 1
        return value
    def array(owner, position, name, count):
        nonlocal offset, number
        spec = dict(record=number, name=f'vboundary.{owner}.{name}', offset=offset,
                    bytes=count*4, before=position, count=1, values=count)
        if take('ARRAY_ATTEMPT') != spec:
            raise ValueError('Witness attempt differs')
        item = take('ARRAY'); block = memoryview(body)[offset:offset+count*4]
        if ({k:v for k,v in item.items() if k!='sha256'} != spec or len(block)!=count*4
                or hashlib.sha256(block).hexdigest()!=item['sha256']):
            raise ValueError('Witness array identity differs')
        values = struct.unpack('<'+str(count)+'f', block)
        if not all(math.isfinite(v) for v in values):
            raise ValueError('Nonfinite witness row')
        rows[owner, name] = block
        offset += len(block); number += 1
    if take('REAL_VBOUNDARY_CONTRACT') != dict(cases=[[a,b] for a,b,_ in CASES], input_tokens=651, capacity=1025, chunk_rows=32):
        raise ValueError('Witness contract differs')
    for owner, position, dim in CASES:
        before = position//32*32
        if take('REAL_VBOUNDARY_CASE') != dict(owner=owner, position=position, before=before,
                count=min(32, 651-before), head_dim=dim, fault_preserved_prefix=True):
            raise ValueError('Witness case differs')
        for name, count in [('input',1536), ('raw',dim), ('normalized',dim)]:
            array(owner, position, name, count)
    for owner, position, dim in CASES:
        array(owner, position, 'retained', dim)
        if rows[owner,'retained'] != rows[owner,'normalized']:
            raise ValueError('Normalized-to-retained transfer differs')
    if (take('CAPTURE') != dict(records=12, bytes=30720, sha256=hashlib.sha256(body).hexdigest(), whole_e2b_qualified=False)
            or offset != len(body) or number != 12):
        raise ValueError('Complete witness capture differs')
    if take('REAL_VBOUNDARY_COMPLETE') != dict(committed=651, all3_stage_retained_exact=True,
            allocated_bytes=271117548, numerical_decision='separate same-input replay',
            earlier_failures_waived=False, app_qualified=False, phone_qualified=False) or cursor!=len(events):
        raise ValueError('Witness completion differs')
    return rows


def compare(rows, weights):
    import numpy as np
    reports = []
    for owner, position, dim in CASES:
        input_row = np.frombuffer(rows[owner,'input'], dtype='<f4').reshape(1,1536)
        raw = np.frombuffer(rows[owner,'raw'], dtype='<f4')
        output = np.frombuffer(rows[owner,'normalized'], dtype='<f4')
        for name, actual in [('projection',raw), ('normalization',output)]:
            contracts = {}
            for c, staged in [('math_fp64',False), ('staged_fp32',True)]:
                reference = (weights.linear(f'layers.{owner}.self_attn.v_proj.weight', input_row,
                             np.float32 if staged else np.float64).reshape(-1)
                             if name=='projection' else normalized(raw, staged))
                if reference.shape != (dim,):
                    raise ValueError('Same-input reference shape differs')
                contracts[c] = policy.fractions(actual, reference)
            reports.append(dict(owner=owner, absolute_position=position, operation=name, contracts=contracts))
    return dict(local_operations_passed=all(s['failed_values']==0 for r in reports for s in r['contracts'].values()), rows=reports)


def earlier_rows(rows, transcript, body):
    _, kv, _ = prior.native_capture(transcript, body)
    for owner, position, dim in CASES:
        committed = 512 if position==433 else 1024
        first = 0 if owner%5==4 else max(0, committed-512)
        start = (position-first)*dim*4
        expected = kv[committed,owner,'v'][start:start+dim*4]
        if rows[owner,'normalized'] != expected:
            raise ValueError('Previously captured full V row differs')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['transcript','arrays','prior_transcript','prior_arrays']:
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    read = policy.bounded.read
    rows = native_rows(read(args.transcript, 1024**2), read(args.arrays, 1024**2))
    earlier_rows(rows, read(args.prior_transcript, 2*1024**2), read(args.prior_arrays, 128*1024**2))
    source = prior.oracle.source_graph
    for name, sha in source.source.PINS.items():
        if source.source.sha(source.source.MODEL/name) != sha:
            raise ValueError('Original packed source differs')
    with (source.source.MODEL/'model.safetensors').open('rb') as stream:
        result = compare(rows, source.weights(stream))
    for name, sha in source.source.PINS.items():
        if source.source.sha(source.source.MODEL/name) != sha:
            raise ValueError('Original packed source changed')
    result.update(cache_transfer_passed=True, prior_full_rows_exact=True,
                  earlier_failures_waived=False, whole_graph_qualified=False)
    print(json.dumps(result, indent=2, allow_nan=False))

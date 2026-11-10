"""Audit native output representability under both unchanged CPU profile gates."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import struct
import time

import check_minicpm5_context as reader

producer = reader.producer
need = reader.need
CONTRACT_SHA = 'ea652001de705b2081eb9873c379afd1e49f96c1272688ae2f565bde0f8688a0'
FORMATS = {'<f2': (16, 'e', 'H', 0x7bff), '<f4': (32, 'f', 'I', 0x7f7fffff)}
MAXIMUM_BLOCK = 65536


def fresh_report(path):
    try:
        path.lstat()
    except FileNotFoundError:
        return
    need(False, 'Fresh absent report path required')


def sources():
    return [*reader.sources(), Path(__file__).resolve(),
            Path(__file__).with_name('test_check_minicpm5_precision.py').resolve()]


def policy(a, b, gate, dtype):
    need(dtype in FORMATS and type(a) in (int, float) and type(b) in (int, float)
         and math.isfinite(a) and math.isfinite(b), 'Finite scalar native-format inputs required')
    need(type(gate) is dict and all(type(gate.get(k)) in (int, float)
         and math.isfinite(gate[k]) for k in ['atol', 'rtol'])
         and gate['atol'] > 0 and gate['rtol'] >= 0, 'Original finite positive gate required')
    return gate['atol'] + gate['rtol'] * abs(a), gate['atol'] + gate['rtol'] * abs(b)


def decode(bits, dtype):
    _, floating, integer, _ = FORMATS[dtype]
    return struct.unpack('<' + floating, struct.pack('<' + integer, bits))[0]


def scalar_first(a, b, gate, dtype):
    """Independent ordered-bit binary search; no NumPy or endpoint rounding.

    The lower predicate is monotone over finite native values. Its first true
    value decides feasibility because the upper predicate is nonincreasing.
    Signed zeros have equal numerical membership and may share that boundary.
    """
    ta, tb = policy(a, b, gate, dtype)
    width, _, _, maximum = FORMATS[dtype]
    sign = 1 << (width - 1)
    mask = (1 << width) - 1

    def value(order):
        return decode((order ^ sign) if order & sign else (~order & mask), dtype)

    def lower(x):
        return (x >= a or abs(x - a) <= ta) and (x >= b or abs(x - b) <= tb)

    first, last = (~(maximum | sign) & mask), maximum | sign
    lo, hi = first, last + 1
    while lo < hi:
        middle = (lo + hi) // 2
        if lower(value(middle)):
            hi = middle
        else:
            lo = middle + 1
    need(lo <= last, 'No finite native value satisfies the scalar lower bounds')
    result = value(lo)
    need(lower(result) and (lo == first or not lower(value(lo - 1))),
         'Independent scalar lower bracket differs')
    return result, abs(result - a) <= ta and abs(result - b) <= tb


def first_native(a, b, gate, dtype, np):
    """Return the first lower-feasible native value and original-gate verdict.

    Endpoint casting is only an estimate. Actual original floating predicates at
    the candidate and its immediate numerical predecessor certify the exact
    discrete bracket. Any unestablished bracket rejects the audit itself.
    """
    policy(0., 0., gate, dtype)
    need(a.ndim == b.ndim == 1 and a.shape == b.shape and 0 < a.size <= MAXIMUM_BLOCK
         and a.dtype == b.dtype == np.dtype('<f8')
         and np.isfinite(a).all() and np.isfinite(b).all(), 'Bounded finite binary64 block required')
    native = np.dtype(dtype)
    need(np.array_equal(a.astype(native).astype(np.float64), a)
         and np.array_equal(b.astype(native).astype(np.float64), b),
         'Source values must preserve the pinned native stage storage domain')
    ta = gate['atol'] + gate['rtol'] * np.abs(a)
    tb = gate['atol'] + gate['rtol'] * np.abs(b)
    low = np.maximum(a - ta, b - tb)
    high = np.minimum(a + ta, b + tb)
    need(np.isfinite(ta).all() and np.isfinite(tb).all()
         and np.isfinite(low).all() and np.isfinite(high).all(), 'Finite original bounds required')

    def lower(x):
        x = x.astype(np.float64)
        return (np.isfinite(x) & ((x >= a) | (np.abs(x - a) <= ta))
                & ((x >= b) | (np.abs(x - b) <= tb)))

    with np.errstate(over='ignore', under='ignore', invalid='raise'):
        positive, negative = np.asarray(math.inf, dtype=native), np.asarray(-math.inf, dtype=native)
        candidate = low.astype(native)
        candidate = np.where(candidate.astype(np.float64) < low,
                             np.nextafter(candidate, positive), candidate)
        previous = np.nextafter(candidate, negative)
        candidate = np.where(lower(previous), previous, candidate)
        candidate = np.where(lower(candidate), candidate, np.nextafter(candidate, positive))
        previous = np.nextafter(candidate, negative)
        need(lower(candidate).all() and not lower(previous).any(),
             'Exact discrete native lower bracket not established')
    values = candidate.astype(np.float64)
    # The authoritative verdict is the original reader's exact comparison body.
    _, _, bad_a = reader.compare(values, a, gate, np)
    _, _, bad_b = reader.compare(values, b, gate, np)
    return candidate, previous, ~(bad_a | bad_b), low > high


def literal_controls(np):
    """Exhaust every finite Half input and cross-check independent scalar search."""
    bits = np.arange(65536, dtype='<u2')
    half = bits.view('<f2')
    values = half[np.isfinite(half)].astype(np.float64)
    need(len(values) == 63488, 'Complete finite Half bit inventory differs')
    with np.errstate(over='ignore', under='ignore'):
        peer = np.nextafter(values.astype('<f2'), np.asarray(math.inf, dtype='<f2')).astype(np.float64)
    peer = np.where(np.isfinite(peer), peer, values)
    gate = dict(atol=.0001, rtol=.001)
    first, _, feasible, _ = first_native(values, peer, gate, '<f2', np)
    need(feasible.all(), 'Adjacent original-domain literal pairs must be feasible')
    checked = 0
    for i in sorted({0, 1, len(values)-1, *range(0, len(values), 251)}):
        value, verdict = scalar_first(float(values[i]), float(peer[i]), gate, '<f2')
        need(value == float(first[i]) and verdict == bool(feasible[i]), 'Independent Half scalar oracle differs')
        checked += 1
    for dtype, exponent in [('<f2', -10), ('<f4', -23)]:
        a, b = 1., 1. + 3 * 2.**exponent
        midpoint = dict(atol=1.5 * 2.**exponent, rtol=0.)
        first, _, feasible, disjoint = first_native(np.array([a]), np.array([b]), midpoint, dtype, np)
        value, verdict = scalar_first(a, b, midpoint, dtype)
        need(not feasible[0] and not disjoint[0] and not verdict and value == float(first[0]),
             'Touching continuous intervals without a native value must reject')
        checked += 1
    float_bits = [0, 1, 0x007fffff, 0x00800000, 0x3f800000, 0x3f800001,
                  0x3f800003, 0x7f7fffff, 0x80000000, 0x80000001, 0x807fffff,
                  0x80800000, 0xbf800000, 0xff7fffff]
    for bits in float_bits:
        a = decode(bits, '<f4')
        with np.errstate(over='ignore', under='ignore'):
            b = float(np.nextafter(np.asarray(a, dtype='<f4'), np.asarray(math.inf, dtype='<f4')))
        if not math.isfinite(b):
            b = a
        first, _, feasible, _ = first_native(np.array([a]), np.array([b]), gate, '<f4', np)
        value, verdict = scalar_first(a, b, gate, '<f4')
        need(value == float(first[0]) and verdict == bool(feasible[0]), 'Independent Float32 scalar oracle differs')
        checked += 1
    return dict(passed=True,finite_half_inputs=63488,independent_scalar_pairs=checked,
                touching_intersection_without_native_value_controls=2,signed_zero_numeric_equal=True)


def validate_contract(contract):
    need(contract['format'] == 'toklet-minicpm5-native-representability-contract-v1'
         and contract['prospective_before_representability_outcomes'] is True
         and contract['gates_recomputed_or_relaxed'] is False
         and contract['native_precision_changed'] is False and contract['native_execution'] is False
         and contract['original_contract_sha256'] == producer.CONTRACT_SHA
         and reader.same(contract['counts'], reader.COUNTS)
         and reader.same(contract['native_output_domains'], dict(non_logit_stages='<f2', logits='<f4',
             half_stage_count=50, float_stage_count=1, signed_zero_numeric_equal=True))
         and reader.same(contract['paired_coordinates'], 442863360)
         and reader.same(contract['stages'], 51) and reader.same(contract['source_arrays'], 306)
         and reader.same(contract['expected_native_dtype_coordinates'], {'<f2': 225480960, '<f4': 217382400})
         and reader.same(contract['maximum_block_coordinates'], MAXIMUM_BLOCK)
         and reader.same(contract['audit_guard_bytes'], 512*1024**2)
         and reader.same(contract['engine_working_set_limit_bytes'], 6*1024**3)
         and reader.same(contract['disk_floor_bytes'], 2*1024**3)
         and reader.same(contract['deadline_seconds'], 1800)
         and reader.same(contract['full_quality_targets_unchanged'], 10000)
         and reader.same(contract['full_quality_tasks_unchanged'], 100)
         and all(contract[k] is False for k in ['quality_scored', 'model_adoption_qualified',
                                              'performance_or_overhead_qualified']),
         'Exact prospective native-domain policy required')
    need(set(contract['native_schema_source_sha256']) == {'src/metal/diagnostic.rs',
         'src/metal/mod.rs', 'tools/bench/minicpm5_depth.rs'}, 'Complete actual native snapshot schema required')
    root = Path(__file__).resolve().parents[2]
    for rel, pin in contract['native_schema_source_sha256'].items():
        need(producer.model.sha(root/rel) == pin, 'Pinned actual native snapshot/source schema changed')


def reference_evidence(contract, reference):
    records = {}
    need(set(contract['reference_evidence']) == {'receipt', 'owner', 'samples', 'freeze'},
         'Complete original CPU owner evidence required')
    for role, pin in contract['reference_evidence'].items():
        need(type(pin) is dict and set(pin) == {'path', 'bytes', 'sha256'}
             and type(pin['path']) is str and Path(pin['path']).is_absolute()
             and reader.integer(pin['bytes'], 1) and pin['bytes'] <= 16*1024**2,
             'Bounded absolute CPU owner evidence required')
        raw = producer.runtime.common.captured(Path(pin['path']), pin['bytes'], pin['sha256'])
        need(len(raw) == pin['bytes'], 'CPU evidence extent differs')
        records[role] = producer.parsed(raw)
    reader.validate_owner(records['receipt'], records['freeze'], native=False)
    reader.validate_samples(records['receipt'], records['owner'], records['samples'], records['freeze'])
    command = records['freeze']['command']
    need(Path(command[5]).name == Path(producer.__file__).name, 'Original CPU source entry differs')
    origin, local = Path(command[5]).resolve().parents[2], Path(producer.__file__).resolve().parents[2]
    for p in producer.sources():
        need(records['freeze']['source_sha256'].get(str(origin/p.relative_to(local))) == producer.model.sha(p),
             'Original CPU source closure differs across worktrees')
    need(reference['elapsed_ns'] <= records['receipt']['completed_ns'] - records['receipt']['started_ns'],
         'Original CPU reference clock exceeds closed owner')
    return dict(closed_owned_reference=True, source_files=8, actual_samples_reconciled=True)


def audit(reference, raw, original, np):
    groups = {}
    for count in reader.COUNTS:
        for stage, columns in original['columns'].items():
            name = f'prefix{count}.{stage}'
            items = [reference['arrays'][p+'.'+name] for p in ['fp32', 'fp64']]
            dtype = '<f4' if stage == 'logits' else '<f2'
            need(all(v['dtype'] == dtype for v in items), 'Original/native stage storage domain differs')
            group = dict(native_dtype=dtype, coordinates=0, infeasible_coordinates=0,
                         continuous_disjoint_coordinates=0, exact_lower_brackets_verified=True, witnesses=[])
            digests = [hashlib.sha256(), hashlib.sha256()]
            for first in range(0, count*columns, MAXIMUM_BLOCK):
                n = min(MAXIMUM_BLOCK, count*columns-first)
                pair = []
                for profile, item, digest in zip(['fp32', 'fp64'], items, digests):
                    width = 4 if dtype == '<f4' else 2
                    consumed = raw[profile].read(item['offset']+first*width, n*width)
                    digest.update(consumed)
                    pair.append(np.frombuffer(consumed, dtype=dtype).astype(np.float64))
                a, b = pair
                candidate, previous, feasible, disjoint = first_native(a, b, original['gates'][stage], dtype, np)
                group['coordinates'] += n
                group['infeasible_coordinates'] += int(np.count_nonzero(~feasible))
                group['continuous_disjoint_coordinates'] += int(np.count_nonzero(disjoint))
                for index in np.flatnonzero(~feasible)[:max(0, 4-len(group['witnesses']))]:
                    i = int(index)
                    value, verdict = scalar_first(float(a[i]), float(b[i]), original['gates'][stage], dtype)
                    need(not verdict and value == float(candidate[i]), 'Actual infeasible witness fails independent scalar search')
                    group['witnesses'].append(dict(row=(first+i)//columns, column=(first+i)%columns,
                        fp32=float(a[i]), fp64=float(b[i]), first_lower_feasible_native=float(candidate[i]),
                        predecessor=float(previous[i]), independent_scalar_search_infeasible=True,
                        original_gate=original['gates'][stage]))
            for item, digest in zip(items, digests):
                need(digest.hexdigest() == item['sha256'], 'Every consumed CPU array must match original digest')
            groups[name] = group
        print(json.dumps(dict(event='case-complete', tokens=count,
            coordinates=sum(g['coordinates'] for name, g in groups.items() if name.startswith(f'prefix{count}.')),
            infeasible_coordinates=sum(g['infeasible_coordinates'] for name, g in groups.items()
                                       if name.startswith(f'prefix{count}.')))), flush=True)
    return dict(completed=True, paired_coordinates=sum(g['coordinates'] for g in groups.values()),
        infeasible_coordinates=sum(g['infeasible_coordinates'] for g in groups.values()),
        continuous_disjoint_coordinates=sum(g['continuous_disjoint_coordinates'] for g in groups.values()),
        native_dtype_coordinates={dtype: sum(g['coordinates'] for g in groups.values() if g['native_dtype'] == dtype)
                                 for dtype in FORMATS}, groups=groups)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['environment_root', 'contract', 'contract_sha256', 'original_contract', 'reference',
                 'reference_sha256', 'old_reference', 'quality_inputs', 'runtime_manifest',
                 'runtime_sha256', 'fresh_report']:
        parser.add_argument('--'+name.replace('_', '-'), required=True)
    args = parser.parse_args()
    fresh_report(Path(args.fresh_report))
    producer.runtime.bind_environment(args.environment_root)
    need(args.contract_sha256 == CONTRACT_SHA, 'Exact prospective precision contract required')
    capture = producer.runtime.common.captured
    parsed = producer.parsed
    contract = parsed(capture(Path(args.contract), 2*1024**2, CONTRACT_SHA))
    validate_contract(contract)
    original = parsed(capture(Path(args.original_contract), 2*1024**2, producer.CONTRACT_SHA))
    old = parsed(capture(Path(args.old_reference)/'manifest.json', 2*1024**2, producer.OLD_SHA))
    quality = parsed(capture(Path(args.quality_inputs)/'manifest.json', 2*1024**2, producer.QUALITY_SHA))
    tokens = capture(Path(args.quality_inputs)/'held_out_loss.u32le', 40004, producer.TOKENS_SHA)
    need(args.reference_sha256 == contract['reference_sha256'], 'Original CPU reference association differs')
    reference = reader.load_reference(Path(args.reference), args.reference_sha256, original, old, quality, tokens)
    ownership = reference_evidence(contract, reference)
    selected = sources()
    admitted, endpoint = producer.runtime.admit(Path(args.runtime_manifest), args.runtime_sha256, selected)
    import numpy as np
    controls = literal_controls(np)
    print(json.dumps(dict(event='literal-controls', **controls)), flush=True)
    started = time.monotonic_ns()
    raw = {}
    try:
        for profile in ['fp32', 'fp64']:
            raw[profile] = reader.Raw(Path(args.reference)/(profile+'.bin'), reference['files'][profile+'.bin'], 3*1024**3)
        result = audit(reference, raw, original, np)
        for stream in raw.values():
            stream.verify()
    finally:
        for stream in raw.values():
            stream.close()
    need(result['paired_coordinates'] == contract['paired_coordinates'] and len(result['groups']) == 153
         and reader.same(result['native_dtype_coordinates'], contract['expected_native_dtype_coordinates']),
         'Complete original native-domain coordinate coverage differs')
    need(sources() == selected, 'Precision audit source inventory changed')
    result.update(format='toklet-minicpm5-native-representability-v1', elapsed_ns=time.monotonic_ns()-started,
        contract_sha256=CONTRACT_SHA, original_contract_sha256=producer.CONTRACT_SHA,
        reference_sha256=args.reference_sha256, admitted_runtime=admitted, endpoint_origins=endpoint(),
        literal_controls=controls, original_reference_ownership=ownership,
        native_domain_feasible=result['infeasible_coordinates'] == 0,
        numerical_graph_qualified=False, quality_scored=False, model_adoption_qualified=False,
        performance_or_overhead_qualified=False, native_execution=False,
        scope='Existence of a finite native Half/F32 value within both unchanged original CPU predicates at every captured coordinate. Feasibility does not qualify any graph or quality score; no precision/tolerance change.')
    with Path(args.fresh_report).open('x') as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k: result[k] for k in ['completed', 'paired_coordinates', 'infeasible_coordinates',
        'continuous_disjoint_coordinates', 'native_domain_feasible', 'numerical_graph_qualified']}))


if __name__ == '__main__':
    main()

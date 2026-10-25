#!/usr/bin/env python3
"""Pre-target L1 admission witnesses, with separate ordered/exact references.

The existing max proof and engine are unchanged. Real products remain bounded
envelopes. This module imports only the previously audited reference utilities.
"""
from fractions import Fraction
from pathlib import Path
import hashlib
import json
import math
import struct
import sys

import gemma_affine_admission as prior

COORDINATE_SHA = '2d082715177522d4ea353d699a8415b792fa52ffa943b9d083d8aee832c7d821'
BASE_SHA = '2a8770eda606a659c7e8e78fed95b77d0508efe485733d5416083a1761c82597'
LIMIT = 1e-5/16


def upper_sum(values):
    total = 0.
    for value in values:
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError('finite nonnegative magnitudes required')
        if value == 0:
            continue
        total = float(value) if total == 0 else math.nextafter(total+value, math.inf)
        if not math.isfinite(total):
            raise ValueError('bounded FP64 sum required')
    return total


def l1_bound(width, maximum_weight, magnitude_sum):
    if (type(width) is not int or not 64 <= width <= 12288 or width % 64
            or any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in (maximum_weight, magnitude_sum))
            or maximum_weight > 2.**32):
        raise ValueError('fixed admitted L1 domain required')
    def up(a, b):
        return 0. if a == 0 or b == 0 else math.nextafter(a*b, math.inf)
    s = up(maximum_weight, magnitude_sum)
    u = up(up(up(2., float(width)), 2.**-53), s)
    reason = 'absolute-products' if s > prior.MAXIMUM/4 else 'ordered-error' if u > LIMIT else None
    return dict(admitted=reason is None, reason=reason, absolute_products=s, ordered_error=u)


def cases():
    result = []
    def add(name, width, rows, weights=(1., .5, -.75)):
        if any(len(row) != width for row in rows):
            raise ValueError('complete synthetic rows required')
        result.append(dict(name=name, width=width, inputs=rows, weights=list(weights)))
    width = 12288
    first = [0.]*width
    first[0] = 2.**16
    add('sparse-positive', width, [first])
    for name, positions in [('within-lane', (0, 32, 64)), ('across-lanes', (0, 1, 31)),
                            ('cross-group', (0, 65, 127)), ('deep', (0, 8193, 12287))]:
        row = [0.]*width
        row[positions[0]], row[positions[1]], row[positions[2]] = 2.**16, 2.**-8, -2.**16
        add('sparse-cancel-'+name, width, [row])
    dense = [16.]*6144+[-16.]*6144
    dense[67] = prior.f32(16.+2.**-14)
    add('independent-token-rows', width, [first, dense])
    add('dense-cancel', width, [dense], (1.,))
    add('zero-weights-maximum-input', 64, [[prior.MAXIMUM, -prior.MAXIMUM]+[0.]*62], (0.,))
    tiny = [0.]*width
    tiny[0] = 2.**-149
    add('subnormal-input', width, [tiny])
    estimate = prior.f32(LIMIT/(2*width*2.**-53))
    word = struct.unpack('<I', struct.pack('<f', estimate))[0]
    choices = [struct.unpack('<f', struct.pack('<I', word+i))[0] for i in range(-4, 5)]
    lower = max(v for v in choices if l1_bound(width, 1., v)['admitted'])
    upper = min(v for v in choices if not l1_bound(width, 1., v)['admitted'])
    if struct.unpack('<I', struct.pack('<f', upper))[0] != struct.unpack('<I', struct.pack('<f', lower))[0]+1:
        raise ValueError('adjacent L1 cutoff required')
    for name, value in [('cutoff-below', lower), ('cutoff-above', upper)]:
        row = [0.]*width
        row[67] = value
        add(name, width, [row], (1.,))
    add('ordered-exact-conflict', 64, [[2.**100, 1., -2.**100]+[0.]*61], (1.,))
    add('finite-product-overflow', 64, [[prior.MAXIMUM, -prior.MAXIMUM]+[0.]*62], (2.,))
    add('positive-small-term-and-signed-zeros', 64,
        [[2.**40, 2.**-20, -0., 0.]+[0.]*60, [-0.]*64], (1.,))
    for width in (64, 1536, 4096, 6144, 8960):
        row = [0.]*width
        row[-1] = 1e7 if width == 64 else 4096.
        add('shape-sparse-'+str(width), width, [row])
    if len(result) != 19:
        raise ValueError('fixed witness count required')
    return result


def synthetic():
    raw = bytearray()
    descriptions = []
    for case in cases():
        width, rows, weights = case['width'], case['inputs'], case['weights']
        maximum = max(map(abs, weights))
        sums = [upper_sum(abs(v) for v in row) for row in rows]
        l1 = l1_bound(width, maximum, max(sums))
        old = prior.bound(width, maximum, max(abs(v) for row in rows for v in row))
        ordered, mathematical, rational, exact_l1 = [], [], [], []
        for row in rows:
            exact_sum = sum((Fraction(v) for v in row), Fraction())
            exact_abs = sum((Fraction(abs(v)) for v in row), Fraction())
            exact_l1.append([str(exact_abs.numerator), str(exact_abs.denominator)])
            for weight in weights:
                value = 0.
                for x in row:
                    value += weight*x
                exact = exact_sum*Fraction(weight)
                ordered.append(value)
                mathematical.append(float(exact))
                rational.append([str(exact.numerator), str(exact.denominator)])
        payload = struct.pack('<'+'f'*(width*len(rows)), *(v for row in rows for v in row))
        entry = dict(name=case['name'], width=width, batch=len(rows), weights=weights,
                     arrays=dict(input=dict(dtype='F32', shape=[len(rows), width], offset=len(raw), bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest())),
                     per_token_l1=sums, exact_l1=exact_l1, l1=l1, legacy_max=old,
                     combined_admitted=old['admitted'] or l1['admitted'],
                     ordered_fp64=ordered, mathematical_fp64=mathematical, exact_dot=rational)
        raw.extend(payload)
        descriptions.append(entry)
    return dict(format='toklet-gemma-affine-l1-math-v1', producer_sha256=prior.sha(Path(__file__)),
                matrix_representation='packed4/group64/BF16; all codes0, scales0, each output row bias equals weights[row]',
                gate=dict(absolute=1e-5, relative=1e-4), ordered_error_limit=LIMIT,
                cases=descriptions, binary=dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest()),
                engine_admission_changed=False, native_target_executed=False, gpu_qualified=False), bytes(raw)


def input_rows(kind, names, read):
    if kind == 'direct':
        values, shape = read(names[0])
        if names[0].endswith('context'):
            return [[values[(head*4+token)*shape[-1]+c] for head in range(8) for c in range(shape[-1])] for token in range(4)]
        return [values[t*shape[-1]:(t+1)*shape[-1]] for t in range(4)]
    if kind == 'exact-residual':
        data = [read(name)[0] for name in names]
        combined = [prior.f32(prior.f32(a+b)+c) for a, b, c in zip(*data, strict=True)]
        return [combined[t*1536:(t+1)*1536] for t in range(4)]
    if kind != 'product-envelope':
        raise ValueError('known input recipe required')
    a, shape = read(names[0])
    b, other = read(names[1])
    if (type(shape) is not list or type(other) is not list or len(shape) != 3
            or shape != other or any(type(v) is not int for v in shape+other)
            or shape[:2] != [1, 4] or not 1 <= shape[-1] <= 12288
            or len(a) != 4*shape[-1] or len(b) != len(a)):
        raise ValueError('paired product shape differs')
    if any(type(v) not in (int, float) for v in a+b):
        raise ValueError('typed product coordinates required')
    values = [prior.product_ceiling(abs(x), abs(y)) for x, y in zip(a, b, strict=True)]
    return [values[t*shape[-1]:(t+1)*shape[-1]] for t in range(4)]


def trace_report():
    return dict(format='toklet-gemma-affine-l1-trace-v1', passed=False, records=[], statuses={},
                coordinate_receipt_sha256=COORDINATE_SHA, codebook_receipt_sha256=BASE_SHA,
                engine_admission_changed=False, products_are_envelopes=True,
                actual_gpu_executed=False, native_activation_parity=False, whole_graph_qualified=False,
                app_or_phone_qualified=False, source_failures_waived=False, primary_error=None)


def actual(coordinate, base, trace, arrays, compact, report):
    for path, wanted in [(coordinate, COORDINATE_SHA), (base, BASE_SHA), (trace, prior.TRACE_SHA), (arrays, prior.ARRAY_SHA), (compact, prior.COMPACT_SHA)]:
        if prior.sha(path) != wanted:
            raise ValueError('authenticated L1 prerequisite differs')
    observed = prior.decode(prior.bounded(coordinate, 2*1024**2))
    baseline = prior.decode(prior.bounded(base, 2*1024**2))
    if (observed['passed'] is not True or baseline['passed'] is not True
            or observed['base_receipt_sha256'] != BASE_SHA or observed['matrices'] != baseline['matrices']
            or observed['metadata_sha256'] != '5bbbe6bdcbee8135e271926da1ae7a254dbdb620c9a6ac2c1c25aeeaa01d3947'):
        raise ValueError('unchanged admitted codebooks required')
    metadata = prior.decode(prior.bounded(trace, 4*1024**2))
    frozen = prior.decode(prior.bounded(compact, 2*1024**2))
    matrices = {v['name']: v for v in observed['matrices']}
    records = report['records']
    with arrays.open('rb') as stream:
        ledger = prior.trace_ledger(metadata, frozen, stream)
        if len(ledger) != 1426:
            raise ValueError('complete trace required')
        def read(name):
            entry = ledger[('staged_fp32', name)]
            if entry['bytes'] > 196608:
                raise ValueError('bounded activation required')
            stream.seek(entry['offset'])
            raw = stream.read(entry['bytes'])
            if len(raw) != entry['bytes'] or hashlib.sha256(raw).hexdigest() != entry['sha256']:
                raise ValueError('raw activation changed')
            values = [v for (v,) in struct.iter_unpack('<f', raw)]
            if any(not math.isfinite(v) for v in values):
                raise ValueError('finite activation required')
            return values, entry['shape']
        for previous in observed['invocations']:
            name = previous['name']
            record = dict(name=name, input_kind=previous['input_kind'], sources=previous['sources'],
                          legacy_status=previous['status'], completed=False)
            records.append(record)
            matrix = matrices[name]
            names = [v['name'] for v in previous['sources']]
            rows = input_rows(previous['input_kind'], names, read)
            if len(rows) != 4 or any(len(row) != matrix['columns'] for row in rows):
                raise ValueError('complete per-token row required')
            sums = [None if None in row else upper_sum(abs(v) for v in row) for row in rows]
            bound = None if None in sums else l1_bound(matrix['columns'], matrix['maximum_weight'], max(sums))
            status = ('geometry-rejected' if not matrix['geometry_admitted'] else
                      'proved-legacy' if previous['status'] == 'proved' else
                      'proved-l1' if bound is not None and bound['admitted'] else 'not-proved')
            record.update(per_token_l1=sums, l1=bound,
                          per_token_bounds=None if None in sums else [l1_bound(matrix['columns'], matrix['maximum_weight'], value) for value in sums],
                          proposed_status=status, completed=True)
            report['statuses'][status] = report['statuses'].get(status, 0)+1
    if len(records) != 277:
        raise ValueError('complete invocation inventory required')
    report['passed'] = True


def write(coordinate, base, trace, arrays, compact, output):
    output.mkdir()
    report, primary = trace_report(), None
    try:
        data, raw = synthetic()
        for name, body in [('math.bin', raw), ('math.json', (json.dumps(data, indent=2, allow_nan=False)+'\n').encode())]:
            with (output/name).open('xb') as stream:
                if stream.write(body) != len(body):raise OSError('short witness write')
        manifest = dict(format='toklet-gemma-affine-l1-manifest-v1', files={name: dict(bytes=(output/name).stat().st_size, sha256=prior.sha(output/name)) for name in ('math.json', 'math.bin')},
                        cases=19, producer_sha256=data['producer_sha256'], native_target_executed=False)
        with (output/'manifest.json').open('x') as stream:json.dump(manifest, stream, indent=2);stream.write('\n')
        actual(coordinate, base, trace, arrays, compact, report)
    except BaseException as error:
        primary = error
        report['passed'] = False
        report['primary_error'] = type(error).__name__+': '+str(error)
        raise
    finally:
        try:
            with (output/'trace.json').open('x') as stream:
                json.dump(report, stream, indent=2, allow_nan=False);stream.write('\n')
        except BaseException as error:
            if primary is None:
                raise
            primary.add_note('Partial trace publication failed: '+type(error).__name__+': '+str(error))
    return dict(passed=True, cases=19, statuses=report['statuses'])


if __name__ == '__main__':
    if len(sys.argv) != 7:
        raise SystemExit('coordinate-receipt base-receipt trace-json trace-bin compact-json new-output')
    print(json.dumps(write(*(Path(v) for v in sys.argv[1:]))))

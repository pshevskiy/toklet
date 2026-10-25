#!/usr/bin/env python3
"""Independent rational L1 bounds and Decimal400 synthetic-dot replay."""
from decimal import Decimal, localcontext
from fractions import Fraction
from pathlib import Path
import hashlib
import json
import math
import struct
import sys

import check_gemma_affine_admission as previous


def accumulated(values):
    total = 0.
    for value in values:
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError('finite nonnegative terms required')
        if not value:
            continue
        old = total
        total = float(value) if not total else math.nextafter(total+value, math.inf)
        if not math.isfinite(total) or Fraction(total) < Fraction(old)+Fraction(value):
            raise ValueError('directed addition failed')
    return total


def bound(k, w, a):
    if type(k) is not int or not 64 <= k <= 12288 or k % 64 or any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in (w, a)) or w > 2.**32:
        raise ValueError('bounded L1 proof required')
    def product(x, y):
        result = 0. if not x or not y else math.nextafter(x*y, math.inf)
        if Fraction(result) < Fraction(x)*Fraction(y):
            raise ValueError('directed product failed')
        return result
    absolute = product(w, a)
    error = product(product(product(2., float(k)), 2.**-53), absolute)
    if Fraction(absolute) < Fraction(w)*Fraction(a) or Fraction(error) < 2*k*Fraction(2.**-53)*Fraction(absolute):
        raise ValueError('summation enclosure failed')
    why = 'absolute-products' if absolute > previous.FMAX/4 else 'ordered-error' if error > 1e-5/16 else None
    return dict(admitted=why is None, reason=why, absolute_products=absolute, ordered_error=error)


def replay_synthetic(directory):
    document = previous.load(directory/'math.json', 2*1024**2)
    manifest = previous.load(directory/'manifest.json', 65536)
    with (directory/'math.bin').open('rb') as stream:
        raw = stream.read(8*1024**2+1)
    if len(raw) > 8*1024**2:
        raise ValueError('bounded fixture required')
    previous.same(document['format'], 'toklet-gemma-affine-l1-math-v1')
    previous.same(manifest['format'], 'toklet-gemma-affine-l1-manifest-v1')
    previous.same(document['gate'], dict(absolute=1e-5, relative=1e-4))
    previous.same(document['ordered_error_limit'], 1e-5/16)
    previous.same(document['matrix_representation'], 'packed4/group64/BF16; all codes0, scales0, each output row bias equals weights[row]')
    for field in ('engine_admission_changed', 'native_target_executed', 'gpu_qualified'):
        previous.same(document[field], False)
    previous.same(manifest['native_target_executed'], False)
    producer = previous.digest(Path(__file__).with_name('gemma_affine_l1_math.py'))
    previous.same(document['producer_sha256'], producer)
    previous.same(manifest['producer_sha256'], producer)
    if set(manifest['files']) != {'math.json', 'math.bin'}:
        raise ValueError('exact fixture files required')
    for name in ('math.json', 'math.bin'):
        previous.same(manifest['files'][name], dict(bytes=(directory/name).stat().st_size, sha256=previous.digest(directory/name)))
    previous.same(document['binary'], manifest['files']['math.bin'])
    if len(document['cases']) != 19 or manifest['cases'] != 19 or len({c['name'] for c in document['cases']}) != 19:
        raise ValueError('fixed unique case inventory required')
    position, outputs, newly_admitted = 0, 0, 0
    named = {}
    with localcontext() as context:
        context.prec = 400
        for case in document['cases']:
            k, batch, weights = case['width'], case['batch'], case['weights']
            if type(k) is not int or not 64 <= k <= 12288 or k % 64 or type(batch) is not int or not 1 <= batch <= 32 or not 1 <= len(weights) <= 3:
                raise ValueError('bounded witness geometry required')
            if any(type(w) is not float or not math.isfinite(w) or w and not 2.**-126 <= abs(w) <= 2.**32 or previous.rounded(Fraction(w)) != w or struct.unpack('<I', struct.pack('<f', w))[0] & 65535 for w in weights):
                raise ValueError('original normal BF16 bias weights required')
            if set(case['arrays']) != {'input'}:
                raise ValueError('exact input array required')
            item = case['arrays']['input']
            body = raw[position:position+batch*k*4]
            previous.same(item, dict(dtype='F32', shape=[batch, k], offset=position, bytes=batch*k*4, sha256=hashlib.sha256(body).hexdigest()))
            if len(body) != batch*k*4:
                raise ValueError('short input')
            position += len(body)
            values = [v for (v,) in struct.iter_unpack('<f', body)]
            if any(not math.isfinite(v) for v in values):
                raise ValueError('finite inputs required')
            rows = [values[t*k:(t+1)*k] for t in range(batch)]
            sums = [accumulated(abs(v) for v in row) for row in rows]
            exact_l1, ordered, ideal, exact_dot = [], [], [], []
            for row, upper in zip(rows, sums, strict=True):
                exact_abs = sum((Fraction(abs(v)) for v in row), Fraction())
                if Fraction(upper) < exact_abs:
                    raise ValueError('L1 sum rounded inward')
                exact_l1.append([str(exact_abs.numerator), str(exact_abs.denominator)])
                for weight in weights:
                    reference = sum((Decimal.from_float(v)*Decimal.from_float(weight) for v in row), Decimal())
                    rational = Fraction(reference)
                    total = 0.
                    for v in row:total = total+float(weight)*float(v)
                    ordered.append(total)
                    ideal.append(float(reference))
                    exact_dot.append([str(rational.numerator), str(rational.denominator)])
            maximum = max(map(abs, weights))
            l1 = bound(k, maximum, max(sums))
            old = previous.bounds(k, maximum, max(map(abs, values)))
            for field, expected in [('per_token_l1', sums), ('exact_l1', exact_l1), ('l1', l1), ('legacy_max', old),
                                    ('combined_admitted', old['admitted'] or l1['admitted']), ('ordered_fp64', ordered),
                                    ('mathematical_fp64', ideal), ('exact_dot', exact_dot)]:
                previous.same(case[field], expected)
            if l1['admitted']:
                for total, rational in zip(ordered, exact_dot, strict=True):
                    if abs(Fraction(total)-Fraction(int(rational[0]), int(rational[1]))) > Fraction(l1['ordered_error']):
                        raise ValueError('observed ordered/exact discrepancy exceeds proof')
            outputs += len(ordered)
            newly_admitted += l1['admitted'] and not old['admitted']
            named[case['name']] = case
    if position != len(raw):
        raise ValueError('trailing fixture bytes')
    conflict = named['ordered-exact-conflict']
    if conflict['ordered_fp64'] != [0.] or conflict['mathematical_fp64'] != [1.] or conflict['l1']['admitted'] or conflict['legacy_max']['admitted']:
        raise ValueError('historical conflict must reject')
    if named['finite-product-overflow']['l1']['reason'] != 'absolute-products':
        raise ValueError('product overflow must reject')
    lower, upper = named['cutoff-below'], named['cutoff-above']
    if not lower['l1']['admitted'] or upper['l1']['admitted']:
        raise ValueError('cutoff classifications differ')
    lower_value, upper_value = lower['per_token_l1'][0], upper['per_token_l1'][0]
    if struct.unpack('<I', struct.pack('<f', upper_value))[0] != struct.unpack('<I', struct.pack('<f', lower_value))[0]+1:
        raise ValueError('cutoff values are not adjacent')
    return dict(cases=19, outputs=outputs, newly_admitted_cases=newly_admitted, ordered_exact_conflict_rejected=True)


def real_rows(kind, source, reader):
    if kind == 'direct':
        value, shape = reader(source[0])
        if source[0].endswith('.context'):
            return [[value[h*4*shape[-1]+t*shape[-1]+d] for h in range(8) for d in range(shape[-1])] for t in range(4)]
        return [value[t*shape[-1]:(t+1)*shape[-1]] for t in range(4)]
    if kind == 'exact-residual':
        a, b, c = [reader(n)[0] for n in source]
        return [[previous.rounded(Fraction(previous.rounded(Fraction(a[t*1536+d])+Fraction(b[t*1536+d])))+Fraction(c[t*1536+d])) for d in range(1536)] for t in range(4)]
    if kind != 'product-envelope':
        raise ValueError('known profile recipe required')
    a, shape = reader(source[0])
    b, other = reader(source[1])
    if (not isinstance(shape, list) or not isinstance(other, list) or len(shape) != 3
            or shape != other or any(type(v) is not int for v in shape+other)
            or shape[:2] != [1, 4] or not 0 < shape[-1] <= 12288
            or len(a) != 4*shape[-1] or len(a) != len(b)):
        raise ValueError('paired shape differs')
    if any(type(v) not in (int, float) for v in a+b):
        raise ValueError('typed product coordinates required')
    return [[previous.envelope(abs(a[t*shape[-1]+d]), abs(b[t*shape[-1]+d])) for d in range(shape[-1])] for t in range(4)]


def replay_actual(coordinate, base, trace, arrays, compact, path):
    fixed = [(coordinate, '2d082715177522d4ea353d699a8415b792fa52ffa943b9d083d8aee832c7d821'),
             (base, '2a8770eda606a659c7e8e78fed95b77d0508efe485733d5416083a1761c82597'),
             (trace, 'ca86bd9836c88d5084f770b9ae84ecdb6e1e5a91d775e7cda315702450e918e9'),
             (arrays, '352385699238b422b430b6f535c2eac99b6e90543becb4c46caa0c59f930ed93'),
             (compact, 'f60c9f355b9008cfba6acd5d39c89c108de2d1ed5a3c8013a3a595576d8f910b')]
    for file, digest in fixed:
        if previous.digest(file) != digest:
            raise ValueError('authenticated prerequisite differs')
    old = previous.load(coordinate, 2*1024**2)
    earlier = previous.load(base, 2*1024**2)
    if old['passed'] is not True or earlier['passed'] is not True or old['matrices'] != earlier['matrices']:
        raise ValueError('admitted matrix maxima differ')
    metadata = previous.load(trace, 4*1024**2)
    sample = previous.load(compact, 2*1024**2)
    expected = {(v['contract'], v['name']): v for v in sample['arrays']}
    ledger = {(v['contract'], v['name']): v for v in metadata['arrays']}
    if len(ledger) != len(metadata['arrays']) or len(ledger) != len(expected) or len(ledger) != 1426:
        raise ValueError('complete accepted trace required')
    matrices = {v['name']: v for v in old['matrices']}
    records = []
    with arrays.open('rb') as stream:
        def reader(name):
            key = ('staged_fp32', name)
            e = ledger[key]
            previous.same(e['shape'], expected[key]['source_shape'])
            previous.same(e['dtype'], '<f4')
            if not 0 < e['bytes'] <= 196608 or e['bytes'] != math.prod(e['shape'])*4:
                raise ValueError('bounded state required')
            stream.seek(e['offset'])
            raw = stream.read(e['bytes'])
            if len(raw) != e['bytes'] or hashlib.sha256(raw).hexdigest() != e['sha256'] or e['sha256'] != expected[key]['source_array_sha256']:
                raise ValueError('raw state differs')
            result = [v for (v,) in struct.iter_unpack('<f', raw)]
            if any(not math.isfinite(v) for v in result):
                raise ValueError('finite captured input required')
            return result, e['shape']
        for item in old['invocations']:
            name = item['name']
            matrix = matrices[name]
            rows = real_rows(item['input_kind'], [s['name'] for s in item['sources']], reader)
            if len(rows) != 4 or any(len(row) != matrix['columns'] for row in rows):
                raise ValueError('complete row geometry required')
            totals = [None if None in row else accumulated(abs(v) for v in row) for row in rows]
            policy = None if None in totals else bound(matrix['columns'], matrix['maximum_weight'], max(totals))
            status = 'geometry-rejected' if not matrix['geometry_admitted'] else 'proved-legacy' if item['status'] == 'proved' else 'proved-l1' if policy is not None and policy['admitted'] else 'not-proved'
            records.append(dict(name=name, input_kind=item['input_kind'], sources=item['sources'], legacy_status=item['status'],
                                per_token_l1=totals, l1=policy, per_token_bounds=None if None in totals else [bound(matrix['columns'], matrix['maximum_weight'], a) for a in totals], proposed_status=status, completed=True))
    result = dict(format='toklet-gemma-affine-l1-trace-v1', passed=True, records=records,
                  statuses={s: sum(v['proposed_status'] == s for v in records) for s in sorted({v['proposed_status'] for v in records})},
                  coordinate_receipt_sha256=fixed[0][1], codebook_receipt_sha256=fixed[1][1],
                  engine_admission_changed=False, products_are_envelopes=True, actual_gpu_executed=False,
                  native_activation_parity=False, whole_graph_qualified=False, app_or_phone_qualified=False, source_failures_waived=False, primary_error=None)
    previous.same(previous.load(path, 2*1024**2), result)
    return result['statuses']


if __name__ == '__main__':
    if len(sys.argv) != 7:
        raise SystemExit('coordinate-receipt base-receipt trace-json trace-bin compact-json capture-directory')
    arguments = [Path(v) for v in sys.argv[1:]]
    result = replay_synthetic(arguments[-1])
    result.update(passed=True, proposed_statuses=replay_actual(*arguments[:-1], arguments[-1]/'trace.json'))
    print(json.dumps(result))

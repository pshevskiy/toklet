#!/usr/bin/env python3
"""Independent Decimal400 replay of captured packed projection bytes."""
from decimal import Decimal, localcontext
from pathlib import Path
import hashlib
import json
import math
import struct
import sys


def captured(path, limit):
    with path.open('rb') as stream:
        raw = stream.read(limit + 1)
    if len(raw) > limit:
        raise ValueError('bounded projection witness required')
    return raw


def unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate JSON key')
        result[key] = value
    return result


def invalid_number(value):
    raise ValueError('nonfinite JSON number: ' + value)


def verify(math_bytes, raw, manifest_bytes):
    decode = lambda body: json.loads(body, object_pairs_hook=unique, parse_constant=invalid_number)
    data, manifest = decode(math_bytes), decode(manifest_bytes)
    assert data['format'] == 'toklet-gemma-metal-affine-math-v1'
    assert manifest['format'] == 'toklet-gemma-metal-affine-manifest-v1'
    assert set(manifest['files']) == {'math.json', 'math.bin'}
    for name, body in [('math.json', math_bytes), ('math.bin', raw)]:
        assert manifest['files'][name] == dict(bytes=len(body), sha256=hashlib.sha256(body).hexdigest())
    assert data['binary'] == dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
    assert manifest['producer_sha256'] == data['producer_sha256'] and len(data['producer_sha256']) == 64
    assert data['bits'] == 4 and data['group_size'] == 64
    assert data['gate'] == dict(absolute=1e-5, relative=1e-4)
    assert data['admission'] == dict(maximum_weight=2. ** 32, minimum_normal=2. ** -126,
                                      ordered_unit=2. ** -53, ordered_error_limit=1e-5 / 16,
                                      upward_positive_products=True)
    assert manifest['native_target_executed'] is False
    for name in ('source_or_device_qualified', 'whole_graph_qualified', 'performance_qualified'):
        assert data[name] is False
    cases = data['cases']
    assert len(cases) == manifest['cases'] == 27 and len({c['name'] for c in cases}) == 27
    position, outputs, admitted, constructor_rejected, input_rejected = 0, 0, 0, 0, 0
    observed = {}
    maximum_ordered_difference = 0.0
    f32 = lambda value: struct.unpack('<f', struct.pack('<f', value))[0]
    maximum = struct.unpack('<f', bytes.fromhex('ffff7f7f'))[0]
    with localcontext() as context:
        context.prec = 400
        for case in cases:
            width, rows, batch = (case[k] for k in ('width', 'outputs', 'batch'))
            assert all(type(v) is int for v in (width, rows, batch))
            assert 64 <= width <= 12288 and width % 64 == 0 and 1 <= rows <= 16384 and 1 <= batch <= 32
            assert set(case['arrays']) == {'weight', 'scales', 'biases', 'input'}
            arrays = {}
            for name, dtype, shape, size in [('weight', 'U32', [rows, width // 8], 4),
                                            ('scales', 'BF16', [rows, width // 64], 2),
                                            ('biases', 'BF16', [rows, width // 64], 2),
                                            ('input', 'F32', [batch, width], 4)]:
                spec = case['arrays'][name]
                assert type(spec['offset']) is int and type(spec['bytes']) is int
                assert type(spec['shape']) is list and all(type(v) is int for v in spec['shape'])
                assert spec['dtype'] == dtype and spec['shape'] == shape and spec['offset'] == position
                assert spec['bytes'] == math.prod(shape) * size
                part = raw[position:position + spec['bytes']]
                assert len(part) == spec['bytes'] and hashlib.sha256(part).hexdigest() == spec['sha256']
                arrays[name] = part
                position += spec['bytes']
            coefficients = lambda part: [struct.unpack('<f', struct.pack('<I', v << 16))[0]
                                         for (v,) in struct.iter_unpack('<H', part)]
            scales, biases = coefficients(arrays['scales']), coefficients(arrays['biases'])
            inputs = [v for (v,) in struct.iter_unpack('<f', arrays['input'])]
            assert all(math.isfinite(v) for v in scales + biases + inputs)
            reason, max_weight = None, 0.0
            for scale, bias in zip(scales, biases):
                if any(v != 0 and abs(v) < 2. ** -126 for v in (scale, bias)):
                    reason = 'metadata-domain'
                    break
                for code in range(16):
                    try:
                        value = f32(float(Decimal.from_float(scale) * code + Decimal.from_float(bias)))
                    except OverflowError:
                        reason = 'reconstruction-finite'
                        break
                    if not math.isfinite(value):
                        reason = 'reconstruction-finite'
                        break
                    if value != 0 and (abs(value) < 2. ** -126 or abs(value) > 2. ** 32):
                        reason = 'weight-domain'
                        break
                    max_weight = max(max_weight, abs(value))
                if reason is not None:
                    break
            wanted_constructor = dict(admitted=reason is None, reason=reason,
                                      max_weight=max_weight if reason is None else None)
            assert case['constructor'] == wanted_constructor and type(case['constructor']['admitted']) is bool
            assert case['constructor']['max_weight'] is None or type(case['constructor']['max_weight']) in (int, float)
            weights = []
            for row in range(rows):
                for col in range(width):
                    byte = arrays['weight'][(row * width + col) // 2]
                    code = (byte >> (4 * (col % 2))) & 15
                    group = (row * width + col) // 64
                    weights.append(f32(float(Decimal.from_float(scales[group]) * code + Decimal.from_float(biases[group]))))
            assert hashlib.sha256(struct.pack('<' + 'f' * len(weights), *weights)).hexdigest() == case['reconstructed_sha256']
            if reason is None:
                def up(a, b):
                    return 0. if a == 0 or b == 0 else math.nextafter(a * b, math.inf)
                max_input = max(abs(v) for v in inputs)
                bound = up(up(float(width), max_weight), max_input)
                error = up(up(up(2., float(width)), 2. ** -53), bound)
                assert Decimal.from_float(bound) >= width * Decimal.from_float(max_weight) * Decimal.from_float(max_input)
                assert Decimal.from_float(error) >= 2 * width * Decimal.from_float(2. ** -53) * Decimal.from_float(bound)
                why = ('input-absolute-products' if bound > maximum / 4 else
                       'input-summation' if error > 1e-5 / 16 else None)
                wanted_input = dict(admitted=why is None, reason=why, absolute_products_bound=bound, ordered_error_bound=error)
                assert case['input_admission'] == wanted_input and type(case['input_admission']['admitted']) is bool
                assert all(type(case['input_admission'][key]) in (int, float) for key in ('absolute_products_bound', 'ordered_error_bound'))
                if why is None:
                    admitted += 1
                else:
                    input_rejected += 1
            else:
                assert case['input_admission'] is None
                constructor_rejected += 1
            size = rows * batch
            assert len(case['ordered_fp64']) == len(case['math_fp64']) == len(case['exact_dot_rationals']) == size
            for token in range(batch):
                for row in range(rows):
                    total = Decimal(0)
                    ordered = 0.0
                    for col in range(width):
                        weight, value = weights[row * width + col], inputs[token * width + col]
                        total += Decimal.from_float(weight) * Decimal.from_float(value)
                        ordered += weight * value
                    index = token * rows + row
                    pair = case['exact_dot_rationals'][index]
                    assert type(pair) is list and len(pair) == 2 and all(type(v) is str for v in pair)
                    numerator, denominator = map(int, pair)
                    assert [str(numerator), str(denominator)] == pair
                    assert denominator > 0 and Decimal(numerator) / Decimal(denominator) == total
                    assert type(case['math_fp64'][index]) in (float, int) and math.isfinite(case['math_fp64'][index])
                    assert float(total) == case['math_fp64'][index]
                    assert type(case['ordered_fp64'][index]) in (int, float) and math.isfinite(case['ordered_fp64'][index])
                    assert struct.pack('<d', ordered) == struct.pack('<d', case['ordered_fp64'][index])
                    difference = abs(ordered - float(total))
                    if case['input_admission'] is not None and case['input_admission']['admitted']:
                        assert difference <= case['input_admission']['ordered_error_bound']
                        maximum_ordered_difference = max(maximum_ordered_difference, difference)
                    outputs += 1
            observed[case['name']] = case
    assert position == len(raw) and (admitted, input_rejected, constructor_rejected) == (21, 3, 3)
    conflict = observed['ordered-exact-conflict']
    assert conflict['ordered_fp64'] == [0.] and conflict['math_fp64'] == [1.]
    assert conflict['input_admission']['reason'] == 'input-summation'
    assert conflict['input_admission']['absolute_products_bound'] < maximum / 4
    overflow = observed['finite-product-overflow']
    assert overflow['ordered_fp64'] == overflow['math_fp64'] == [0.]
    assert overflow['input_admission']['reason'] == 'input-absolute-products' and 2 * maximum > maximum
    below, above = observed['boundary-below'], observed['boundary-above']
    assert below['input_admission']['admitted'] is True and above['input_admission']['admitted'] is False
    a = struct.unpack('<I', raw[below['arrays']['input']['offset']:below['arrays']['input']['offset'] + 4])[0]
    b = struct.unpack('<I', raw[above['arrays']['input']['offset']:above['arrays']['input']['offset'] + 4])[0]
    assert b == a + 1
    return dict(passed=True, cases=27, outputs=outputs, admitted_cases=admitted,
                input_rejected=input_rejected, constructor_rejected=constructor_rejected,
                exact_dot_decimal_precision=400, maximum_admitted_ordered_difference=maximum_ordered_difference,
                scalar_exact_conflict_preserved=True, adjacent_boundaries=True)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise SystemExit('usage: check_gemma_affine_math.py CAPTURE_DIRECTORY')
    directory = Path(sys.argv[1])
    print(json.dumps(verify(captured(directory / 'math.json', 1024 ** 2),
                            captured(directory / 'math.bin', 8 * 1024 ** 2),
                            captured(directory / 'manifest.json', 65536)), allow_nan=False))

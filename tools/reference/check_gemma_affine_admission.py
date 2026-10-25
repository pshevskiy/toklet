#!/usr/bin/env python3
"""Independent rational replay of the real Gemma admission audit.

Imports no producer or engine. Metadata and original trace remain streamed;
only a bounded coefficient-pair LRU and one small layer's inputs are retained.
"""
from fractions import Fraction
from functools import lru_cache
from pathlib import Path
import hashlib
import json
import math
import struct
import sys

CHUNK = 65536
FMAX = float.fromhex('0x1.fffffep+127')
ERROR_LIMIT = 1e-5 / 16
PREFIX = 'language_model.model.'


def digest(path):
    state = hashlib.sha256()
    with path.open('rb') as stream:
        for part in iter(lambda: stream.read(CHUNK), b''):
            state.update(part)
    return state.hexdigest()


def load(path, limit):
    def pairs(values):
        result = {}
        for name, value in values:
            if name in result:
                raise ValueError('duplicate field')
            result[name] = value
        return result
    def invalid(value):
        raise ValueError('invalid number '+value)
    with path.open('rb') as stream:
        raw = stream.read(limit+1)
    if len(raw) > limit:
        raise ValueError('bounded replay input required')
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=invalid)


def same(actual, expected):
    if type(actual) is not type(expected):
        raise ValueError('typed result differs')
    if isinstance(expected, dict):
        if set(actual) != set(expected):
            raise ValueError('result fields differ')
        for key in expected:
            same(actual[key], expected[key])
    elif isinstance(expected, list):
        if len(actual) != len(expected):
            raise ValueError('result count differs')
        for a, b in zip(actual, expected, strict=True):
            same(a, b)
    elif actual != expected or isinstance(actual, float) and not math.isfinite(actual):
        raise ValueError('result value differs')


def rounded(value):
    """Round an exact Fraction to binary32, including ties and subnormals."""
    if not isinstance(value, Fraction):
        raise ValueError('exact rational required')
    if value == 0:
        return 0.
    negative = value < 0
    n, d = abs(value.numerator), value.denominator
    e = n.bit_length()-d.bit_length()
    if (n < d << e) if e >= 0 else (n << -e < d):
        e -= 1
    shift = 149 if e < -126 else 23-e
    numerator, denominator = (n << shift, d) if shift >= 0 else (n, d << -shift)
    q, r = divmod(numerator, denominator)
    if 2*r > denominator or 2*r == denominator and q % 2:
        q += 1
    if e < -126:
        bits = q
    else:
        if q == 1 << 24:
            q >>= 1
            e += 1
        bits = 0x7f800000 if e > 127 else (e+127) << 23 | (q-(1 << 23))
    if negative:
        bits |= 1 << 31
    return struct.unpack('<f', struct.pack('<I', bits))[0]


def bf16(bits):
    exponent, mantissa = (bits >> 7) & 255, bits & 127
    if exponent == 255 or exponent == 0 and mantissa:
        return None
    if exponent == 0:
        return Fraction(0)
    power = exponent-134
    value = Fraction((128+mantissa) << power) if power >= 0 else Fraction(128+mantissa, 1 << -power)
    return -value if bits & 0x8000 else value


@lru_cache(maxsize=262144)
def coefficients(key):
    a, b = bf16(key & 65535), bf16(key >> 16)
    if a is None or b is None:
        return None, 'metadata-domain'
    maximum = 0.
    for number in range(16):
        value = rounded(number*a+b)
        if not math.isfinite(value):
            return None, 'reconstruction-finite'
        if value and not 2. ** -126 <= abs(value) <= 2. ** 32:
            return None, 'weight-domain'
        maximum = max(maximum, abs(value))
    return maximum, None


def envelope(a, b):
    if not all(type(v) in (int, float) and math.isfinite(v) and v >= 0 and rounded(Fraction(v)) == v for v in (a, b)):
        raise ValueError('finite FP32 maxima required')
    exact = Fraction(a)*Fraction(b)
    if exact > Fraction(FMAX):
        return None
    value = rounded(exact)
    if Fraction(value) < exact:
        word = struct.unpack('<I', struct.pack('<f', value))[0]+1
        value = struct.unpack('<f', struct.pack('<I', word))[0]
    return value


def coordinate_envelope(a, a_shape, b, b_shape):
    if (not isinstance(a_shape, list) or not isinstance(b_shape, list)
            or len(a_shape) != 3 or len(b_shape) != 3
            or any(type(n) is not int for n in a_shape+b_shape)
            or a_shape != b_shape or a_shape[0] != 1 or a_shape[1] != 4
            or not 0 < a_shape[-1] <= 12288
            or len(a) != 4*a_shape[-1] or len(a) != len(b)):
        raise ValueError('complete equal four-token shapes required')
    result = []
    for start in range(0, len(a), a_shape[-1]):
        upper, unproved = 0., False
        for index in range(start, start+a_shape[-1]):
            if type(a[index]) not in (int, float) or type(b[index]) not in (int, float):
                raise ValueError('typed numerical coordinates required')
            value = envelope(abs(a[index]), abs(b[index]))
            if value is None:
                unproved = True
            elif value > upper:
                upper = value
        result.append(None if unproved else upper)
    return result


def mode_identity(format_name, recipe, coordinate_products):
    if type(coordinate_products) is not bool:
        raise ValueError('external Boolean mode required')
    same(format_name, 'toklet-gemma-affine-admission-v2' if coordinate_products else 'toklet-gemma-affine-admission-v1')
    same(recipe, 'coordinate-correlated' if coordinate_products else None)


def bounds(k, w, x):
    if type(k) is not int or not 64 <= k <= 12288 or k % 64:
        raise ValueError('width differs')
    if not all(type(v) in (int, float) and math.isfinite(v) and v >= 0 for v in (w, x)) or w > 2. ** 32 or rounded(Fraction(x)) != x:
        raise ValueError('admitted finite bounds required')
    def multiply(a, b):
        result = 0. if a == 0 or b == 0 else math.nextafter(a*b, math.inf)
        if Fraction(result) < Fraction(a)*Fraction(b):
            raise ValueError('positive product rounded inward')
        return result
    total = multiply(multiply(float(k), w), x)
    error = multiply(multiply(multiply(2., float(k)), 2. ** -53), total)
    if Fraction(total) < k*Fraction(w)*Fraction(x) or Fraction(error) < 2*k*Fraction(2. ** -53)*Fraction(total):
        raise ValueError('rational envelope differs')
    reason = 'absolute-products' if total > FMAX/4 else 'ordered-error' if error > ERROR_LIMIT else None
    return dict(admitted=reason is None, reason=reason, absolute_products=total, ordered_error=error)


def inventory():
    matrices = {'embed_tokens.weight': (262144, 1536),
                'embed_tokens_per_layer.weight': (262144, 8960),
                'per_layer_model_projection.weight': (8960, 1536)}
    for layer in range(35):
        head = 512 if layer % 5 == 4 else 256
        middle = 6144 if layer <= 14 else 12288
        entries = {'self_attn.q_proj': (8*head, 1536), 'self_attn.o_proj': (1536, 8*head),
                   'mlp.gate_proj': (middle, 1536), 'mlp.up_proj': (middle, 1536),
                   'mlp.down_proj': (1536, middle), 'per_layer_input_gate': (256, 1536),
                   'per_layer_projection': (1536, 256)}
        if layer <= 14:
            entries.update({'self_attn.k_proj': (head, 1536), 'self_attn.v_proj': (head, 1536)})
        for suffix, geometry in entries.items():
            matrices[f'layers.{layer}.{suffix}.weight'] = geometry
    return matrices


def invocation_inputs(reader, coordinate_products=False):
    if type(coordinate_products) is not bool:
        raise ValueError('external Boolean mode required')
    def maximum(name, head_major=False):
        values, shape = reader(name)
        if head_major:
            return [max(abs(values[(h*4+t)*shape[-1]+d]) for h in range(8) for d in range(shape[-1])) for t in range(4)]
        return [max(map(abs, values[t*shape[-1]:(t+1)*shape[-1]])) for t in range(4)]
    yield 'per_layer_model_projection.weight', 'direct', ['scaled_embedding'], maximum('scaled_embedding')
    for layer in range(35):
        stem = f'layer.{layer}.'
        mapping = [('self_attn.q_proj', 'input_layernorm')]
        if layer < 15:
            mapping += [('self_attn.k_proj', 'input_layernorm'), ('self_attn.v_proj', 'input_layernorm')]
        mapping += [('self_attn.o_proj', 'context'), ('mlp.gate_proj', 'pre_feedforward_layernorm'), ('mlp.up_proj', 'pre_feedforward_layernorm')]
        for suffix, state in mapping:
            yield f'layers.{layer}.{suffix}.weight', 'direct', [stem+state], maximum(stem+state, state == 'context')
        for suffix, a, b in [('mlp.down_proj', 'mlp.gate_proj', 'mlp.up_proj'), ('per_layer_projection', 'per_layer_input_gate', 'ple_input')]:
            source = [stem+a, stem+b]
            if coordinate_products:
                a_values, a_shape = reader(source[0])
                b_values, b_shape = reader(source[1])
                values = coordinate_envelope(a_values, a_shape, b_values, b_shape)
            else:
                values = [envelope(x, y) for x, y in zip(maximum(source[0]), maximum(source[1]), strict=True)]
            yield f'layers.{layer}.{suffix}.weight', 'product-envelope', source, values
        source = [stem+'input', stem+'post_attention_layernorm', stem+'post_feedforward_layernorm']
        data = [reader(n)[0] for n in source]
        per_token = []
        for t in range(4):
            maximum_value = 0.
            for column in range(1536):
                p = t*1536+column
                first = rounded(Fraction(data[0][p])+Fraction(data[1][p]))
                result = rounded(Fraction(first)+Fraction(data[2][p]))
                maximum_value = max(maximum_value, abs(result))
            per_token.append(maximum_value)
        yield f'layers.{layer}.per_layer_input_gate.weight', 'exact-residual', source, per_token
    yield 'embed_tokens.weight', 'direct', ['final_norm'], maximum('final_norm')


def verify(model, trace, arrays, compact, report_path, metadata, coordinate_products=False):
    report = load(report_path, 8*1024**2)
    mode_identity(report.get('format'), report.get('product_envelope'), coordinate_products)
    fixed = [(model, '038e39a37a7667373d2c3991375446b10c96ae1d717a68674870343db376b76e'),
             (trace, 'ca86bd9836c88d5084f770b9ae84ecdb6e1e5a91d775e7cda315702450e918e9'),
             (arrays, '352385699238b422b430b6f535c2eac99b6e90543becb4c46caa0c59f930ed93'),
             (compact, 'f60c9f355b9008cfba6acd5d39c89c108de2d1ed5a3c8013a3a595576d8f910b')]
    for path, wanted in fixed:
        if path.is_symlink() or not path.is_file() or digest(path) != wanted:
            raise ValueError('authenticated source differs')
    required = {'format', 'passed', 'matrices', 'invocations', 'tokens', 'metadata_bytes', 'metadata_pairs', 'source_sha256', 'primary_error', 'operator_sha256', 'ordered_error_limit', 'process_allowance_bytes', 'actual_gpu_executed', 'native_activation_parity', 'whole_graph_qualified', 'app_or_phone_qualified', 'source_or_precision_failures_waived', 'long_output_qualified', 'metadata_sha256'}
    if coordinate_products:
        required.add('product_envelope')
    if set(report) != required:
        raise ValueError('exact audit fields required')
    for key, value in dict(format='toklet-gemma-affine-admission-v2' if coordinate_products else 'toklet-gemma-affine-admission-v1', passed=True, tokens=[2, 100, 257, 100], metadata_bytes=289267712,
                           metadata_pairs=72316928, source_sha256={str(p): h for p, h in fixed}, primary_error=None,
                           operator_sha256='14ac5d6f5df9db2ade7a833c422178d3c4ae889db560bcf7cdaab177c367ac79',
                           ordered_error_limit=ERROR_LIMIT, process_allowance_bytes=512*1024**2,
                           actual_gpu_executed=False, native_activation_parity=False, whole_graph_qualified=False,
                           app_or_phone_qualified=False, source_or_precision_failures_waived=False, long_output_qualified=False).items():
        same(report[key], value)
    if metadata.stat().st_size != 289267712 or digest(metadata) != report['metadata_sha256']:
        raise ValueError('metadata capture differs')
    expected_matrices = inventory()
    if len(report['matrices']) != 278 or [v['name'] for v in report['matrices']] != list(expected_matrices):
        raise ValueError('matrix inventory differs')
    cursor, count = 0, 0
    with model.open('rb') as checkpoint, metadata.open('rb') as saved:
        prefix = checkpoint.read(8)
        if len(prefix) != 8:
            raise ValueError('complete tensor header prefix required')
        length = struct.unpack('<Q', prefix)[0]
        if not 0 < length <= 1024**2:
            raise ValueError('bounded tensor header required')
        raw = checkpoint.read(length)
        if len(raw) != length or hashlib.sha256(raw).hexdigest() != '4d1819ededded19812059691c0f77cfec0fe10c9c89c720950f919bfc26544a7':
            raise ValueError('fixed header differs')
        header = json.loads(raw)
        for row in report['matrices']:
            name = row['name']
            outputs, columns = expected_matrices[name]
            spec = header[PREFIX+name]
            same(spec['shape'], [outputs, columns//8])
            same(spec['dtype'], 'U32')
            size = outputs*columns//64*2
            wanted = dict(name=name, rows=outputs, columns=columns, geometry_admitted=outputs <= 16384,
                          metadata_pairs=size//2, chunks=[], maximum_weight=0., reason=None)
            for start in range(0, size, CHUNK):
                parts = []
                for role in ('scales', 'biases'):
                    coefficient = header[PREFIX+name[:-7]+'.'+role]
                    same(coefficient['shape'], [outputs, columns//64])
                    same(coefficient['dtype'], 'BF16')
                    same(coefficient['data_offsets'][1]-coefficient['data_offsets'][0], size)
                    absolute = 8+length+coefficient['data_offsets'][0]+start
                    n = min(CHUNK, size-start)
                    part = saved.read(n)
                    checkpoint.seek(absolute)
                    if len(part) != n or checkpoint.read(n) != part:
                        raise ValueError('original coefficient bytes differ')
                    wanted['chunks'].append(dict(role=role, source_offset=absolute, offset=cursor, bytes=n, sha256=hashlib.sha256(part).hexdigest()))
                    cursor += n
                    parts.append(part)
                for offset in range(0, len(parts[0]), 2):
                    scale = int.from_bytes(parts[0][offset:offset+2], 'little')
                    bias = int.from_bytes(parts[1][offset:offset+2], 'little')
                    maximum, why = coefficients(scale+(bias << 16))
                    if why is not None and wanted['reason'] is None:
                        wanted['reason'] = why
                    if maximum is not None:
                        wanted['maximum_weight'] = max(wanted['maximum_weight'], maximum)
                    count += 1
            if wanted['reason'] is not None:
                wanted['maximum_weight'] = None
            same(row, wanted)
        if saved.read(1) or cursor != 289267712 or count != 72316928:
            raise ValueError('full coefficient accounting differs')
    old = load(trace, 4*1024**2)
    sample = load(compact, 2*1024**2)
    expected = {(v['contract'], v['name']): v for v in sample['arrays']}
    if len(expected) != 1426 or len(sample['arrays']) != 1426:
        raise ValueError('compact inventory differs')
    ledger, offset = {}, 0
    with arrays.open('rb') as stream:
        for entry in old['arrays']:
            key = (entry['contract'], entry['name'])
            if key not in expected or key in ledger:
                raise ValueError('trace inventory differs')
            e = expected[key]
            dtype = '<f4' if key[0] == 'staged_fp32' else '<f8'
            same(entry['shape'], e['source_shape'])
            same(entry['dtype'], dtype)
            same(entry['completed'], True)
            same(entry['offset'], offset)
            same(entry['bytes'], math.prod(e['source_shape'])*(4 if dtype == '<f4' else 8))
            same(entry['sha256'], e['source_array_sha256'])
            h = hashlib.sha256()
            left = entry['bytes']
            while left:
                part = stream.read(min(CHUNK, left))
                if not part:
                    raise ValueError('short trace')
                h.update(part)
                left -= len(part)
            if h.hexdigest() != entry['sha256']:
                raise ValueError('trace bytes differ')
            offset += entry['bytes']
            ledger[key] = entry
        if stream.read(1) or set(ledger) != set(expected):
            raise ValueError('incomplete/trailing trace')
    matrices = {v['name']: v for v in report['matrices']}
    with arrays.open('rb') as stream:
        def read(name):
            e = ledger[('staged_fp32', name)]
            if e['bytes'] > 196608:
                raise ValueError('bounded activation read required')
            stream.seek(e['offset'])
            raw = stream.read(e['bytes'])
            if hashlib.sha256(raw).hexdigest() != e['sha256']:
                raise ValueError('activation bytes differ')
            values = list(struct.unpack('<'+'f'*(len(raw)//4), raw))
            if any(not math.isfinite(v) for v in values):
                raise ValueError('nonfinite activation')
            return values, e['shape']
        invocations = []
        for name, kind, names, maxima in invocation_inputs(read, coordinate_products):
            matrix = matrices[name]
            row = dict(name=name, input_kind=kind, sources=[dict(name=n, sha256=ledger[('staged_fp32', n)]['sha256']) for n in names],
                       per_token_maximum=maxima, geometry_admitted=matrix['geometry_admitted'])
            if matrix['reason'] is not None:
                row.update(status='constructor-rejected', reason=matrix['reason'], batch=None, per_token=None)
            elif None in maxima:
                row.update(status='not-proved-envelope', reason='product-envelope-overflow', batch=None, per_token=None)
            else:
                total = bounds(matrix['columns'], matrix['maximum_weight'], max(maxima))
                row.update(batch=total, per_token=[bounds(matrix['columns'], matrix['maximum_weight'], x) for x in maxima],
                           status='proved' if total['admitted'] else 'not-proved-envelope' if kind == 'product-envelope' else 'input-rejected', reason=total['reason'])
            if not matrix['geometry_admitted']:
                row.update(status='geometry-rejected', reason='whole-owner-output-rows')
            invocations.append(row)
    if len(invocations) != 277:
        raise ValueError('invocation count differs')
    same(report['invocations'], invocations)
    return dict(passed=True, matrices=278, coefficient_pairs=count, invocations=277,
                statuses={s: sum(v['status'] == s for v in invocations) for s in sorted({v['status'] for v in invocations})})


if __name__ == '__main__':
    if len(sys.argv) not in (7, 8) or len(sys.argv) == 8 and sys.argv[-1] != 'coordinate-products':
        raise SystemExit('model trace-json trace-bin compact-json observation-json metadata-bin [coordinate-products]')
    print(json.dumps(verify(*(Path(v) for v in sys.argv[1:7]), coordinate_products=len(sys.argv) == 8)))

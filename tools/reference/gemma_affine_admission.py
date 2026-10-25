#!/usr/bin/env python3
"""Stream the pinned Gemma checkpoint and frozen FP32 trace admission domains.

No engine, NumPy, MLX, network, projection execution or capability changes.
Product envelopes prove sufficient admission only; their failure is not an
observed rejection of an uncaptured activation.
"""
from functools import lru_cache
from pathlib import Path
import hashlib
import json
import math
import struct
import sys

PREFIX = 'language_model.model.'
CHUNK = 65536
CACHE = 262144
META_LIMIT = 320 * 1024 ** 2
MAXIMUM = struct.unpack('<f', bytes.fromhex('ffff7f7f'))[0]
LIMIT = 1e-5 / 16
MODEL_SHA = '038e39a37a7667373d2c3991375446b10c96ae1d717a68674870343db376b76e'
TRACE_SHA = 'ca86bd9836c88d5084f770b9ae84ecdb6e1e5a91d775e7cda315702450e918e9'
ARRAY_SHA = '352385699238b422b430b6f535c2eac99b6e90543becb4c46caa0c59f930ed93'
COMPACT_SHA = 'f60c9f355b9008cfba6acd5d39c89c108de2d1ed5a3c8013a3a595576d8f910b'
HEADER_SHA = '4d1819ededded19812059691c0f77cfec0fe10c9c89c720950f919bfc26544a7'


def sha(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for raw in iter(lambda: stream.read(CHUNK), b''):
            value.update(raw)
    return value.hexdigest()


def unique(pairs):
    out = {}
    for key, value in pairs:
        if key in out:
            raise ValueError('duplicate JSON key')
        out[key] = value
    return out


def decode(raw):
    def invalid(value):
        raise ValueError('nonfinite JSON: ' + value)
    return json.loads(raw, object_pairs_hook=unique, parse_constant=invalid)


def bounded(path, limit):
    with path.open('rb') as stream:
        raw = stream.read(limit + 1)
    if len(raw) > limit:
        raise ValueError('bounded JSON required')
    return raw


def f32(value):
    return struct.unpack('<f', struct.pack('<f', value))[0]


def product_ceiling(a, b):
    """Finite outward FP32 product envelope, or explicit unproved overflow."""
    if not all(type(v) in (int, float) and math.isfinite(v) and v >= 0 and f32(v) == v for v in (a, b)):
        raise ValueError('nonnegative finite FP32 maxima required')
    exact = a * b  # A product of FP32 numbers is exact in finite FP64.
    if exact > MAXIMUM:
        return None
    rounded = f32(exact)
    if rounded < exact:
        bits = struct.unpack('<I', struct.pack('<f', rounded))[0] + 1
        rounded = struct.unpack('<f', struct.pack('<I', bits))[0]
    return rounded


def coordinate_maxima(first, first_shape, second, second_shape):
    """Preserve operand correlation; outward round each product before max."""
    if (type(first_shape) is not list or type(second_shape) is not list
            or len(first_shape) != 3 or any(type(v) is not int for v in first_shape)
            or any(type(v) is not int for v in second_shape)
            or first_shape != second_shape or first_shape[:2] != [1, 4]
            or not 1 <= first_shape[2] <= 12288
            or len(first) != math.prod(first_shape) or len(second) != len(first)):
        raise ValueError('matching complete [1,4,width] operands required')
    width = first_shape[2]
    result = []
    for token in range(4):
        maximum, overflow = 0., False
        for column in range(width):
            offset = token*width+column
            if any(type(v) not in (int, float) for v in (first[offset], second[offset])):
                raise ValueError('typed FP32 coordinates required')
            value = product_ceiling(abs(first[offset]), abs(second[offset]))
            if value is None:
                overflow = True
            else:
                maximum = max(maximum, value)
        result.append(None if overflow else maximum)
    return result


def bound(columns, weight, maximum):
    if type(columns) is not int or not 64 <= columns <= 12288 or columns % 64:
        raise ValueError('fixed projection width required')
    if not all(type(v) in (int, float) and math.isfinite(v) and v >= 0 for v in (weight, maximum)):
        raise ValueError('finite nonnegative bounds required')
    if weight > 2 ** 32 or f32(maximum) != maximum:
        raise ValueError('admitted weight and FP32 maximum required')
    def up(a, b):
        return 0. if a == 0 or b == 0 else math.nextafter(a * b, math.inf)
    s = up(up(float(columns), weight), maximum)
    u = up(up(up(2., float(columns)), 2. ** -53), s)
    why = 'absolute-products' if s > MAXIMUM / 4 else 'ordered-error' if u > LIMIT else None
    return dict(admitted=why is None, reason=why, absolute_products=s, ordered_error=u)


@lru_cache(maxsize=CACHE)
def codebook(key):
    scale, bias = (struct.unpack('<f', struct.pack('<I', v << 16))[0]
                   for v in (key & 65535, key >> 16))
    if any(not math.isfinite(v) or v != 0 and abs(v) < 2. ** -126 for v in (scale, bias)):
        return None, 'metadata-domain'
    maximum = 0.
    for code in range(16):
        try:
            value = f32(float(scale) * code + float(bias))
        except OverflowError:
            return None, 'reconstruction-finite'
        if not math.isfinite(value):
            return None, 'reconstruction-finite'
        if value != 0 and (abs(value) < 2. ** -126 or abs(value) > 2. ** 32):
            return None, 'weight-domain'
        maximum = max(maximum, abs(value))
    return maximum, None


def matrix_shapes():
    out = {'embed_tokens.weight': (262144, 1536),
           'embed_tokens_per_layer.weight': (262144, 8960),
           'per_layer_model_projection.weight': (8960, 1536)}
    for i in range(35):
        d = 512 if i % 5 == 4 else 256
        ff = 6144 if i < 15 else 12288
        for suffix, shape in [('self_attn.q_proj', (8*d, 1536)),
                              ('self_attn.o_proj', (1536, 8*d)),
                              ('mlp.gate_proj', (ff, 1536)), ('mlp.up_proj', (ff, 1536)),
                              ('mlp.down_proj', (1536, ff)),
                              ('per_layer_input_gate', (256, 1536)),
                              ('per_layer_projection', (1536, 256))]:
            out[f'layers.{i}.{suffix}.weight'] = shape
        if i < 15:
            for suffix in ('k_proj', 'v_proj'):
                out[f'layers.{i}.self_attn.{suffix}.weight'] = (d, 1536)
    return out


def trace_ledger(observation, compact, stream):
    expected = {(v['contract'], v['name']): v for v in compact['arrays']}
    if len(expected) != len(compact['arrays']):
        raise ValueError('duplicate compact state')
    ledger, position = {}, 0
    for entry in observation['arrays']:
        key = (entry['contract'], entry['name'])
        if key in ledger or key not in expected:
            raise ValueError('unexpected or duplicate trace state')
        wanted = expected[key]
        dtype = '<f4' if key[0] == 'staged_fp32' else '<f8' if key[0] == 'math_fp64' else None
        shape = entry['shape']
        if (dtype is None or entry['dtype'] != dtype or entry['completed'] is not True
                or type(shape) is not list or any(type(v) is not int or v <= 0 for v in shape)
                or shape != wanted['source_shape'] or type(entry['offset']) is not int
                or entry['offset'] != position or type(entry['bytes']) is not int
                or entry['bytes'] != math.prod(shape) * (4 if dtype == '<f4' else 8)
                or entry['sha256'] != wanted['source_array_sha256']):
            raise ValueError('trace shape/type/range/source identity differs')
        digest = hashlib.sha256()
        remaining = entry['bytes']
        while remaining:
            raw = stream.read(min(CHUNK, remaining))
            if not raw:
                raise ValueError('short captured state')
            digest.update(raw)
            remaining -= len(raw)
        if digest.hexdigest() != entry['sha256']:
            raise ValueError('captured state digest differs')
        position += entry['bytes']
        ledger[key] = entry
    if set(ledger) != set(expected) or stream.read(1):
        raise ValueError('incomplete or trailing trace')
    return ledger


def inputs(read, coordinate_products=False):
    """Yield named per-token maxima and explicit input proof recipes."""
    if type(coordinate_products) is not bool:
        raise ValueError('explicit Boolean envelope mode required')
    def maxima(name, context=False):
        values, shape = read(name)
        if context:
            _, heads, tokens, width = shape
            return [max(abs(values[(h*tokens+t)*width+c])
                        for h in range(heads) for c in range(width)) for t in range(tokens)]
        _, tokens, width = shape
        return [max(abs(v) for v in values[t*width:(t+1)*width]) for t in range(tokens)]
    yield 'per_layer_model_projection.weight', 'direct', ['scaled_embedding'], maxima('scaled_embedding')
    for i in range(35):
        base = f'layer.{i}.'
        def direct(suffix, source, context=False):
            return f'layers.{i}.{suffix}.weight', 'direct', [base+source], maxima(base+source, context)
        yield direct('self_attn.q_proj', 'input_layernorm')
        if i < 15:
            yield direct('self_attn.k_proj', 'input_layernorm')
            yield direct('self_attn.v_proj', 'input_layernorm')
        yield direct('self_attn.o_proj', 'context', True)
        yield direct('mlp.gate_proj', 'pre_feedforward_layernorm')
        yield direct('mlp.up_proj', 'pre_feedforward_layernorm')
        for suffix, first, second in [('mlp.down_proj', 'mlp.gate_proj', 'mlp.up_proj'),
                                      ('per_layer_projection', 'per_layer_input_gate', 'ple_input')]:
            names = [base+first, base+second]
            if coordinate_products:
                a, a_shape = read(names[0])
                b, b_shape = read(names[1])
                product_maximum = coordinate_maxima(a, a_shape, b, b_shape)
            else:
                product_maximum = [product_ceiling(a, b) for a, b in zip(maxima(names[0]), maxima(names[1]), strict=True)]
            yield f'layers.{i}.{suffix}.weight', 'product-envelope', names, product_maximum
        names = [base+'input', base+'post_attention_layernorm', base+'post_feedforward_layernorm']
        arrays = [read(name)[0] for name in names]
        combined = [f32(f32(a+b)+c) for a, b, c in zip(*arrays, strict=True)]
        yield f'layers.{i}.per_layer_input_gate.weight', 'exact-residual', names, [
            max(abs(v) for v in combined[t*1536:(t+1)*1536]) for t in range(4)]
    yield 'embed_tokens.weight', 'direct', ['final_norm'], maxima('final_norm')


def produce(model, trace, arrays, compact, output, coordinate_products=False):
    if type(coordinate_products) is not bool:
        raise ValueError('explicit Boolean envelope mode required')
    paths = [(model, MODEL_SHA), (trace, TRACE_SHA), (arrays, ARRAY_SHA), (compact, COMPACT_SHA)]
    for path, wanted in paths:
        if path.is_symlink() or not path.is_file() or sha(path) != wanted:
            raise ValueError('fixed input digest differs: ' + str(path))
    observation = decode(bounded(trace, 4*1024**2))
    frozen = decode(bounded(compact, 2*1024**2))
    if (observation['passed'] is not True or observation['primary_error'] is not None
            or observation['endpoint_errors'] or observation['tokens'] != [2, 100, 257, 100]):
        raise ValueError('accepted four-token observation required')
    with arrays.open('rb') as stream:
        ledger = trace_ledger(observation, frozen, stream)
    if len(ledger) != 1426:
        raise ValueError('complete 713-state two-contract trace required')
    report = dict(format='toklet-gemma-affine-admission-v1', passed=False, matrices=[], invocations=[],
                  tokens=[2, 100, 257, 100], metadata_bytes=0, metadata_pairs=0,
                  source_sha256={str(p): s for p, s in paths}, primary_error=None,
                  operator_sha256='14ac5d6f5df9db2ade7a833c422178d3c4ae889db560bcf7cdaab177c367ac79',
                  ordered_error_limit=LIMIT, process_allowance_bytes=512*1024**2,
                  actual_gpu_executed=False, native_activation_parity=False,
                  whole_graph_qualified=False, app_or_phone_qualified=False,
                  source_or_precision_failures_waived=False, long_output_qualified=False)
    if coordinate_products:
        report.update(format='toklet-gemma-affine-admission-v2', product_envelope='coordinate-correlated')
    metadata_path = output/'metadata.bin'
    primary = None
    try:
        with model.open('rb') as source, metadata_path.open('xb') as saved:
            prefix = source.read(8)
            if len(prefix) != 8:
                raise ValueError('complete tensor header prefix required')
            length = struct.unpack('<Q', prefix)[0]
            if not 0 < length <= 1024**2:
                raise ValueError('bounded tensor header required')
            raw_header = source.read(length)
            if len(raw_header) != length or hashlib.sha256(raw_header).hexdigest() != HEADER_SHA:
                raise ValueError('checkpoint header differs')
            header = decode(raw_header)
            for name, (rows, columns) in matrix_shapes().items():
                physical = PREFIX+name
                packed = header[physical]
                scale = header[physical[:-7]+'.scales']
                bias = header[physical[:-7]+'.biases']
                for spec, shape, dtype, size in [(packed, [rows, columns//8], 'U32', 4),
                                                (scale, [rows, columns//64], 'BF16', 2),
                                                (bias, [rows, columns//64], 'BF16', 2)]:
                    if spec['shape'] != shape or spec['dtype'] != dtype or spec['data_offsets'][1]-spec['data_offsets'][0] != math.prod(shape)*size:
                        raise ValueError('named matrix storage differs')
                size = rows*columns//64*2
                record = dict(name=name, rows=rows, columns=columns, geometry_admitted=rows <= 16384,
                              metadata_pairs=size//2, chunks=[], maximum_weight=0., reason=None)
                report['matrices'].append(record)
                for start in range(0, size, CHUNK):
                    parts = []
                    for role, spec in [('scales', scale), ('biases', bias)]:
                        absolute = 8+length+spec['data_offsets'][0]+start
                        source.seek(absolute)
                        raw = source.read(min(CHUNK, size-start))
                        if len(raw) != min(CHUNK, size-start) or saved.tell()+len(raw) > META_LIMIT:
                            raise ValueError('metadata read/capture bound differs')
                        item = dict(role=role, source_offset=absolute, offset=saved.tell(), bytes=len(raw),
                                    sha256=hashlib.sha256(raw).hexdigest())
                        record['chunks'].append(item)
                        if saved.write(raw) != len(raw):
                            raise OSError('short metadata capture write')
                        saved.flush()
                        parts.append(raw)
                    for (s,), (b,) in zip(struct.iter_unpack('<H', parts[0]), struct.iter_unpack('<H', parts[1]), strict=True):
                        maximum, reason = codebook(s | (b << 16))
                        if reason and record['reason'] is None:
                            record['reason'] = reason
                        if maximum is not None:
                            record['maximum_weight'] = max(record['maximum_weight'], maximum)
                if record['reason'] is not None:
                    record['maximum_weight'] = None
                report['metadata_pairs'] += size//2
            report['metadata_bytes'] = saved.tell()
        by_name = {row['name']: row for row in report['matrices']}
        with arrays.open('rb') as source:
            def read(name):
                item = ledger[('staged_fp32', name)]
                if item['bytes'] > 4*12288*4:
                    raise ValueError('bounded activation state required')
                source.seek(item['offset'])
                raw = source.read(item['bytes'])
                if len(raw) != item['bytes'] or hashlib.sha256(raw).hexdigest() != item['sha256']:
                    raise ValueError('activation source changed')
                values = [v for (v,) in struct.iter_unpack('<f', raw)]
                if not all(math.isfinite(v) for v in values):
                    raise ValueError('finite FP32 activation required')
                return values, item['shape']
            for name, kind, names, maxima in inputs(read, coordinate_products):
                matrix = by_name[name]
                result = dict(name=name, input_kind=kind, sources=[dict(name=n, sha256=ledger[('staged_fp32', n)]['sha256']) for n in names],
                              per_token_maximum=maxima, geometry_admitted=matrix['geometry_admitted'])
                if matrix['reason']:
                    result.update(status='constructor-rejected', reason=matrix['reason'], batch=None, per_token=None)
                elif any(v is None for v in maxima):
                    result.update(status='not-proved-envelope', reason='product-envelope-overflow', batch=None, per_token=None)
                else:
                    batch = bound(matrix['columns'], matrix['maximum_weight'], max(maxima))
                    result.update(batch=batch, per_token=[bound(matrix['columns'], matrix['maximum_weight'], v) for v in maxima],
                                  status='proved' if batch['admitted'] else 'not-proved-envelope' if kind == 'product-envelope' else 'input-rejected', reason=batch['reason'])
                if not matrix['geometry_admitted']:
                    result.update(status='geometry-rejected', reason='whole-owner-output-rows')
                report['invocations'].append(result)
        report['metadata_sha256'] = sha(metadata_path)
        if (len(report['matrices']) != 278 or len(report['invocations']) != 277
                or report['metadata_bytes'] != 289267712 or report['metadata_pairs'] != 72316928
                or {kind: sum(v['input_kind'] == kind for v in report['invocations']) for kind in ('direct', 'exact-residual', 'product-envelope')}
                != {'direct': 172, 'exact-residual': 35, 'product-envelope': 70}):
            raise ValueError('fixed inventory/accounting differs')
        for path, wanted in paths:
            if sha(path) != wanted:
                raise ValueError('input changed at endpoint: ' + str(path))
        report['passed'] = True
    except BaseException as error:
        primary = error
        report['primary_error'] = type(error).__name__+': '+str(error)
    finally:
        try:
            with (output/'observation.json').open('x') as stream:
                json.dump(report, stream, indent=2, allow_nan=False)
                stream.write('\n')
        except BaseException as capture_error:
            if primary is None:
                raise
            primary.add_note('Observation retention failed: '+repr(capture_error))
    if primary is not None:
        raise primary
    return report


if __name__ == '__main__':
    if len(sys.argv) not in (6, 7) or len(sys.argv) == 7 and sys.argv[-1] != 'coordinate-products':
        raise SystemExit('model.safetensors trace.json trace.bin compact.json output-directory [coordinate-products]')
    args = [Path(v) for v in sys.argv[1:6]]
    args[-1].mkdir()
    result = produce(*args, coordinate_products=len(sys.argv) == 7)
    print(json.dumps(dict(passed=True, matrices=len(result['matrices']), invocations=len(result['invocations']),
                          statuses={s: sum(v['status'] == s for v in result['invocations']) for s in sorted({v['status'] for v in result['invocations']})})))

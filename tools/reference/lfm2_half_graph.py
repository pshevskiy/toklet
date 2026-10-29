"""Independent tiny LFM2 Half graph over an authenticated original Q4 container.

Only qualified RMS/Conv leaf oracles and generic word serializers are reused.
No scalar graph, Rust/Metal implementation, NumPy, MLX or Torch is imported.
Affine sums use exact integers and round directly to the declared destination.
"""
import argparse
from decimal import Decimal, localcontext
import hashlib
import json
import math
from pathlib import Path
import struct
import sys

import gemma3_half_rms as words
import lfm2_half_rms as rms_leaf
import causal_conv3_half as conv_leaf

FORMAT = 'toklet-lfm2-native-half-tiny-v1'
PROTOCOL_SHA = '068322ec60f6e0e78f4ce89c8a80b01a38f1f6da2fc5369101e3eae78bace927'
ARTIFACT_SHA = '5965adf8a9663fed07fd2b293e717d72f907ddae931649e3189d161d58257487'
CONFIG_SHA = 'd99adc70a5df647c9356ba284a219843a4b31a3b0646fc7d63d5780097db9853'
H, I, V, Q, KV, D, CAPACITY = 256, 256, 64, 4, 2, 64, 12
MAX_BINARY, MAX_MANIFEST, MAX_COMPONENTS = 4*1024**2, 2*1024**2, 256*1024


def identity(raw):
    return dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def encoded(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False)+'\n').encode()


def strict_json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('Duplicate JSON key')
            result[key] = value
        return result
    return json.loads(raw, object_pairs_hook=pairs,
                      parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))


def bounded_bytes(path, maximum):
    with path.open('rb') as source:
        raw = source.read(maximum+1)
    if len(raw) > maximum:
        raise ValueError('Source input byte bound')
    return raw


def f32(value):
    value = words.f32(value)
    if not math.isfinite(value):
        raise ValueError('Nonfinite FP32 store')
    return value


def half(value):
    value = words.half_value(words.half_word(value))
    if not math.isfinite(value):
        raise ValueError('Nonfinite Half store')
    return value


def flat(value):
    return [v for row in value for v in flat(row)] if isinstance(value, (list, tuple)) else [value]


def finite(value):
    if any(not math.isfinite(v) for v in flat(value)):
        raise ValueError('Nonfinite active value')


def rows(values, width):
    if width <= 0 or len(values) % width:
        raise ValueError('Row extent')
    return [values[start:start+width] for start in range(0, len(values), width)]


def scaled_integer(value, kind):
    """Exact finite value on the Half2^-24 or FP32 2^-149 grid."""
    if kind == 'half':
        bits = words.half_word(value)
        if words.half_value(bits) != value:
            raise ValueError('Input is not stored Half')
        fraction, exponent, sign = bits & 1023, (bits >> 10) & 31, bits >> 15
        if exponent == 31:
            raise ValueError('Nonfinite Half input')
        integer = fraction if exponent == 0 else (1024+fraction) << (exponent-1)
    elif kind == 'f32':
        bits = words.float_word(value)
        if words.float_value(bits) != value:
            raise ValueError('Input is not stored FP32')
        fraction, exponent, sign = bits & 0x7fffff, (bits >> 23) & 255, bits >> 31
        if exponent == 255:
            raise ValueError('Nonfinite FP32 input')
        integer = fraction if exponent == 0 else (0x800000+fraction) << (exponent-1)
    else:
        raise ValueError('Integer grid kind')
    return -integer if sign else integer


def round_dyadic(integer, exponent, kind):
    """Round exact integer*2^exponent directly, including subnormals and signs."""
    if type(integer) is not int or type(exponent) is not int or kind not in ('half', 'f32'):
        raise ValueError('Exact dyadic arguments')
    precision, minimum, maximum, bias, sign_shift = ((11, -14, 15, 15, 15) if kind == 'half'
                                                   else (24, -126, 127, 127, 31))
    sign = (1 << sign_shift) if integer < 0 else 0
    value = abs(integer)
    if value == 0:
        return 0  # Exact cancellation has the prospective +0 convention.
    high = value.bit_length()-1+exponent
    quantum = max(high, minimum)-(precision-1)
    shift = quantum-exponent
    if shift > 0:
        quotient, remainder = divmod(value, 1 << shift)
        midpoint = 1 << (shift-1)
        rounded = quotient + (remainder > midpoint or (remainder == midpoint and quotient & 1))
    else:
        rounded = value << -shift
    if rounded == 0:
        return sign
    actual_high = rounded.bit_length()-1+quantum
    if actual_high > maximum:
        raise ValueError('Exact destination overflow')
    if actual_high < minimum:
        return sign | rounded
    adjustment = quantum-(actual_high-(precision-1))
    significand = rounded << adjustment if adjustment >= 0 else rounded >> -adjustment
    return sign | ((actual_high+bias) << (precision-1)) | (significand-(1 << (precision-1)))


def stored_dyadic(integer, exponent, kind):
    bits = round_dyadic(integer, exponent, kind)
    return words.half_value(bits) if kind == 'half' else words.float_value(bits)


class Matrix:
    """Cold exact row-major reconstruction; an exact common factor reduces source cost."""
    def __init__(self, values):
        if not values or not values[0] or any(len(row) != len(values[0]) for row in values):
            raise ValueError('Matrix shape')
        finite(values)
        self.values = values
        integers = [[scaled_integer(v, 'f32') for v in row] for row in values]
        shift = min(((abs(v) & -abs(v)).bit_length()-1 for row in integers for v in row if v), default=0)
        self.integers = [[v >> shift for v in row] for row in integers]
        self.exponent = -149+shift

    def apply(self, inputs, destination='half'):
        if not inputs or any(len(row) != len(self.values[0]) for row in inputs):
            raise ValueError('Affine input shape')
        result = []
        for row in inputs:
            integers = [scaled_integer(v, 'half') for v in row]
            result.append([stored_dyadic(sum(a*b for a, b in zip(integers, weight, strict=True)),
                                         self.exponent-24, destination) for weight in self.integers])
        return result


def original_shapes():
    result = {'model.embed_tokens.weight': [V, H], 'model.embedding_norm.weight': [H]}
    for layer in range(3):
        base = f'model.layers.{layer}.'
        for name in ('operator_norm', 'ffn_norm'):
            result[base+name+'.weight'] = [H]
        for name in ('w1', 'w3', 'w2'):
            result[base+'feed_forward.'+name+'.weight'] = [I, H]
        if layer == 1:
            for name in ('q', 'k'):
                result[base+'self_attn.'+name+'_layernorm.weight'] = [D]
            for name, width in [('q_proj', Q*D), ('k_proj', KV*D), ('v_proj', KV*D), ('out_proj', H)]:
                result[base+'self_attn.'+name+'.weight'] = [width, H]
        else:
            result[base+'conv.in_proj.weight'] = [3*H, H]
            result[base+'conv.out_proj.weight'] = [H, H]
            result[base+'conv.conv.weight'] = [H, 3, 1]
    return result


def decode(raw):
    if identity(raw) != dict(bytes=760078, sha256=ARTIFACT_SHA):
        raise ValueError('Original artifact identity')
    header_bytes = struct.unpack_from('<Q', raw)[0]
    if header_bytes != 6662:
        raise ValueError('Original unaligned header')
    header = strict_json(raw[8:8+header_bytes])
    expected = {}
    for name, shape in original_shapes().items():
        if len(shape) == 2:
            out, inner = shape
            expected[name] = ('U32', [out, inner//8])
            for suffix in ('scales', 'biases'):
                expected[name.removesuffix('.weight')+'.'+suffix] = ('BF16', [out, inner//64])
        else:
            expected[name] = ('BF16', shape)
    if set(header) != set(expected) | {'__metadata__'} or header['__metadata__'] != {'format': 'mlx'}:
        raise ValueError('Original physical inventory')
    spans = []
    for name, (dtype, shape) in expected.items():
        spec = header[name]
        if set(spec) != {'dtype', 'shape', 'data_offsets'} or spec['dtype'] != dtype or spec['shape'] != shape:
            raise ValueError('Original descriptor')
        start, end = spec['data_offsets']
        if any(type(v) is not int for v in [start, end]+shape) or start < 0 or end < start:
            raise ValueError('Original integral extent')
        if end-start != math.prod(shape)*(4 if dtype == 'U32' else 2) or 6670+end > len(raw):
            raise ValueError('Original byte extent')
        spans.append((start, end, name))
    physical, cursor = {}, 0
    for start, end, name in sorted(spans):
        if start != cursor:
            raise ValueError('Original overlap/gap')
        physical[name] = raw[6670+start:6670+end]
        cursor = end
    if cursor != 753408:
        raise ValueError('Original payload extent')
    def bf16(name):
        values = [words.float_value(v << 16) for (v,) in struct.iter_unpack('<H', physical[name])]
        finite(values)
        return values
    matrices, norms, convs = {}, {}, {}
    for name, shape in original_shapes().items():
        if len(shape) != 2:
            values = bf16(name)
            (norms if len(shape) == 1 else convs)[name] = values
            continue
        out, inner = shape
        codes = [v for (v,) in struct.iter_unpack('<I', physical[name])]
        base = name.removesuffix('.weight')
        scales, biases = bf16(base+'.scales'), bf16(base+'.biases')
        reconstructed = []
        for row in range(out):
            values = []
            for column in range(inner):
                nibble = (codes[row*(inner//8)+column//8] >> (4*(column % 8))) & 15
                group = row*(inner//64)+column//64
                values.append(f32(f32(nibble*scales[group])+biases[group]))
            reconstructed.append(values)
        matrices[name] = Matrix(reconstructed)
    return matrices, norms, convs


def norm(inputs, gains, profile):
    if not inputs or any(len(row) != len(gains) for row in inputs):
        raise ValueError('RMS row shape')
    packed = [words.half_word(v) for v in flat(inputs)]
    output = rms_leaf.rms(packed, gains, len(inputs), len(gains), len(gains), profile)
    return rows([words.half_value(v) for v in output], len(gains))


def head_norm(inputs, gains, heads, profile):
    if type(heads) is not int or heads not in (Q, KV) or any(len(row) != heads*D for row in inputs) or len(gains) != D:
        raise ValueError('Head norm shape')
    normalized = norm(rows(flat(inputs), D), gains, profile)
    return rows(flat(normalized), heads*D)


def product(left, right):
    if not left or not left[0] or len(left) != len(right) or any(
            len(a) != len(left[0]) or len(b) != len(left[0]) for a, b in zip(left, right)):
        raise ValueError('Half product shape')
    finite(left)
    finite(right)
    return [[half(f32(a*b)) for a, b in zip(x, y, strict=True)] for x, y in zip(left, right, strict=True)]


def residual(left, right, profile):
    if profile not in ('main', 'wide') or not left or not left[0] or len(left) != len(right) or any(
            len(a) != len(left[0]) or len(b) != len(left[0]) for a, b in zip(left, right)):
        raise ValueError('Residual shape')
    finite(left)
    finite(right)
    return [[half(f32(a+b) if profile == 'main' else a+b)
             for a, b in zip(x, y, strict=True)] for x, y in zip(left, right, strict=True)]


def silu(inputs, profile):
    if profile not in ('main', 'wide'):
        raise ValueError('SiLU profile')
    finite(inputs)
    def one(value):
        exponential = math.exp(-abs(value))
        if profile == 'main':
            exponential = f32(exponential)
            denominator = f32(1+exponential)
            sigmoid = f32(1/denominator) if value >= 0 else f32(exponential/denominator)
            return half(f32(value*sigmoid))
        denominator = 1+exponential
        sigmoid = 1/denominator if value >= 0 else exponential/denominator
        return half(value*sigmoid)
    return [[one(value) for value in row] for row in inputs]


def inverse_frequencies():
    """80-digit Decimal construction; a120-digit stability pass must agree in FP64.

    These are two precisions of the same method, not certified correct rounding
    of the transcendental or an emulation of a provider's intrinsic.
    """
    candidates = []
    for precision in (80, 120):
        with localcontext() as context:
            context.prec = precision
            logarithm = Decimal(1000000).ln()
            candidates.append([float((logarithm*Decimal(-2*j)/D).exp()) for j in range(D//2)])
    if candidates[0] != candidates[1] or candidates[0][0] != 1.0 or candidates[0][16] != .001:
        raise ValueError('Inverse-frequency construction instability')
    return candidates[0]


def rotary(inputs, heads, offset, profile):
    if profile not in ('main', 'wide') or type(offset) is not int or offset < 0 or type(heads) is not int or heads not in (1, KV, Q):
        raise ValueError('RoPE profile/offset')
    if not inputs or offset+len(inputs) > CAPACITY or any(len(row) != heads*D for row in inputs):
        raise ValueError('RoPE shape/position')
    finite(inputs)
    frequencies = ([f32(math.pow(f32(1000000.), f32(-2*j/D))) for j in range(D//2)]
                   if profile == 'main' else inverse_frequencies())
    output = [row[:] for row in inputs]
    for token, row in enumerate(inputs):
        for feature, frequency in enumerate(frequencies):
            angle = (offset+token)*frequency
            if profile == 'main':
                angle = f32(angle)
                cosine, sine = f32(math.cos(angle)), f32(math.sin(angle))
            else:
                cosine, sine = math.cos(angle), math.sin(angle)
            for head in range(heads):
                a, b = head*D+feature, head*D+feature+D//2
                if profile == 'main':
                    output[token][a] = half(f32(f32(row[a]*cosine)-f32(row[b]*sine)))
                    output[token][b] = half(f32(f32(row[b]*cosine)+f32(row[a]*sine)))
                else:
                    output[token][a] = half(row[a]*cosine-row[b]*sine)
                    output[token][b] = half(row[b]*cosine+row[a]*sine)
    return output


def convolution(inputs, coefficients, profile):
    if not inputs or any(len(row) != H for row in inputs) or len(coefficients) != 3*H:
        raise ValueError('Conv3 shape')
    packed = [words.half_word(v) for v in flat(inputs)]
    output, candidate = conv_leaf.conv(packed, [0]*(2*H), coefficients, len(inputs), H, H, profile)
    joined = [[0.]*H, [0.]*H]+[row[:] for row in inputs]
    before = [[row[:] for row in joined[t:t+2]] for t in range(len(inputs))]
    after = [[row[:] for row in joined[t+1:t+3]] for t in range(len(inputs))]
    if candidate != [words.half_word(v) for v in flat(after[-1])]:
        raise ValueError('Raw-u candidate identity')
    return rows([words.half_value(v) for v in output], H), before, after


def attention(query, keys, values, profile):
    n = len(query)
    if profile not in ('main', 'wide') or not 1 <= n <= CAPACITY or len(keys) != n or len(values) != n:
        raise ValueError('Attention count/profile')
    if any(len(row) != Q*D for row in query) or any(len(row) != KV*D for row in keys+values):
        raise ValueError('Attention head geometry')
    finite(query)
    finite(keys)
    finite(values)
    scores, probabilities, output = [], [], [[0.]*(Q*D) for _ in range(n)]
    root = f32(math.sqrt(D)) if profile == 'main' else math.sqrt(D)
    for head in range(Q):
        qbase, kvbase = head*D, (head//(Q//KV))*D
        for token, row in enumerate(query):
            row_scores = []
            for key in keys:
                if profile == 'main':
                    total = 0.
                    for feature in range(D):
                        total = f32(total+f32(row[qbase+feature]*key[kvbase+feature]))
                else:
                    total = math.fsum(row[qbase+feature]*key[kvbase+feature] for feature in range(D))
                row_scores.append(f32(total/root))
            maximum = max(row_scores[:token+1])
            exponentials, mass = [], 0.
            for position, score in enumerate(row_scores):
                value = 0.
                if position <= token:
                    difference = f32(score-maximum) if profile == 'main' else score-maximum
                    value = math.exp(difference)
                    if profile == 'main':
                        value = f32(value)
                exponentials.append(value)
                mass = f32(mass+value) if profile == 'main' else mass+value
            if not math.isfinite(mass) or mass <= 0:
                raise ValueError('Attention softmax mass')
            row_probabilities = [f32(value/mass) for value in exponentials]
            for feature in range(D):
                if profile == 'main':
                    total = 0.
                    for position in range(token+1):
                        total = f32(total+f32(row_probabilities[position]*values[position][kvbase+feature]))
                else:
                    total = math.fsum(row_probabilities[position]*values[position][kvbase+feature]
                                      for position in range(token+1))
                output[token][qbase+feature] = half(total)
            scores.extend(row_scores)
            probabilities.extend(row_probabilities)
    finite(scores)
    finite(probabilities)
    return scores, probabilities, output


def snapshots(values):
    return [[[[values[position][head*D+feature] if position <= token else 0.
               for feature in range(D)] for position in range(CAPACITY)]
             for head in range(KV)] for token in range(len(values))]


def forward(matrices, norms, convs, tokens, profile):
    if profile not in ('main', 'wide') or not tokens or len(tokens) > CAPACITY:
        raise ValueError('Tiny graph request/profile')
    if any(type(token) is not int or not 0 <= token < V for token in tokens):
        raise ValueError('Tiny graph token')
    captures = {}
    def save(name, value):
        if name in captures:
            raise ValueError('Duplicate stage')
        finite(value)
        captures[name] = flat(value)
        return value
    table = matrices['model.embed_tokens.weight']
    hidden = save('embedding_lookup', [[half(v) for v in table.values[token]] for token in tokens])
    for layer in range(3):
        base, stage = f'model.layers.{layer}.', f'layers.{layer}.'
        normalized = save(stage+'operator_norm', norm(hidden, norms[base+'operator_norm.weight'], profile))
        if layer != 1:
            bcx = save(stage+'bcx_projection', matrices[base+'conv.in_proj.weight'].apply(normalized))
            b = save(stage+'b', [row[:H] for row in bcx])
            c = save(stage+'c', [row[H:2*H] for row in bcx])
            x = save(stage+'x', [row[2*H:] for row in bcx])
            u = save(stage+'u', product(b, x))
            z, before, after = convolution(u, convs[base+'conv.conv.weight'], profile)
            save(stage+'history_before', before)
            save(stage+'history_after', after)
            z = save(stage+'convolution', z)
            gated = save(stage+'gated_conv', product(c, z))
            projection = save(stage+'operator_projection', matrices[base+'conv.out_proj.weight'].apply(gated))
        else:
            q = save(stage+'q_projection', matrices[base+'self_attn.q_proj.weight'].apply(normalized))
            k = save(stage+'k_projection', matrices[base+'self_attn.k_proj.weight'].apply(normalized))
            v = save(stage+'v_projection', matrices[base+'self_attn.v_proj.weight'].apply(normalized))
            q = save(stage+'q_norm', head_norm(q, norms[base+'self_attn.q_layernorm.weight'], Q, profile))
            k = save(stage+'k_norm', head_norm(k, norms[base+'self_attn.k_layernorm.weight'], KV, profile))
            q = save(stage+'q_rotary', rotary(q, Q, 0, profile))
            k = save(stage+'k_rotary', rotary(k, KV, 0, profile))
            scores, probabilities, context = attention(q, k, v, profile)
            save(stage+'attention_scores', scores)
            save(stage+'attention_probabilities', probabilities)
            save(stage+'kv_keys_after', snapshots(k))
            save(stage+'kv_values_after', snapshots(v))
            context = save(stage+'attention_context', context)
            projection = save(stage+'operator_projection', matrices[base+'self_attn.out_proj.weight'].apply(context))
        hidden = save(stage+'operator_residual', residual(hidden, projection, profile))
        normalized = save(stage+'ffn_norm', norm(hidden, norms[base+'ffn_norm.weight'], profile))
        gate = save(stage+'gate_projection', matrices[base+'feed_forward.w1.weight'].apply(normalized))
        up = save(stage+'up_projection', matrices[base+'feed_forward.w3.weight'].apply(normalized))
        active = save(stage+'gate_silu', silu(gate, profile))
        gated = save(stage+'ffn_gated', product(active, up))
        down = save(stage+'down_projection', matrices[base+'feed_forward.w2.weight'].apply(gated))
        hidden = save(stage+'output', residual(hidden, down, profile))
    final = save('final_norm', norm(hidden, norms['model.embedding_norm.weight'], profile))
    save('logits', table.apply(final, 'f32'))
    return captures


def causal_equal(first, changed, stages, count):
    """Raw future scores are observable, but only their visible columns are causal."""
    if set(first) != set(stages) or set(changed) != set(stages):
        raise ValueError('Causal stage inventory')
    for name, spec in stages.items():
        left, right = first[name], changed[name]
        if name.endswith('attention_scores'):
            for head in range(Q):
                for token in range(count-1):
                    start = (head*count+token)*count
                    if left[start:start+token+1] != right[start:start+token+1]:
                        raise ValueError('Causal visible scores')
        elif name.endswith('attention_probabilities'):
            for head in range(Q):
                start = head*count*count
                if left[start:start+(count-1)*count] != right[start:start+(count-1)*count]:
                    raise ValueError('Causal probability rows')
        else:
            active = (count-1)*math.prod(spec['shape'][1:])
            if left[:active] != right[:active]:
                raise ValueError('Causal earlier stage '+name)


def components():
    exact = [
        ('half_even', (1 << 11)+1, -11, 'half', 0x3c00),
        ('half_odd', (1 << 11)+3, -11, 'half', 0x3c02),
        ('half_minsub', 1, -24, 'half', 1),
        ('half_underflow_even', 1, -25, 'half', 0),
        ('half_underflow_odd', 3, -25, 'half', 2),
        ('negative_underflow', -1, -25, 'half', 0x8000),
        ('f32_minsub', 1, -149, 'f32', 1),
        ('f32_underflow_even', 1, -150, 'f32', 0),
        ('f32_underflow_odd', 3, -150, 'f32', 2),
    ]
    rounding = []
    for name, integer, exponent, kind, expected in exact:
        actual = round_dyadic(integer, exponent, kind)
        if actual != expected:
            raise ValueError('Exact rounding witness '+name)
        rounding.append(dict(name=name, integer=integer, exponent=exponent, kind=kind, bits=actual))
    # FP64 fsum must itself round; rounding that intermediate to Half/FP32 can
    # miss the side of an exact midpoint. The source center never does that.
    half_matrix = Matrix([[1., 2.**-11, 2.**-60]])
    half_direct = words.half_word(half_matrix.apply([[1., 1., 1.]])[0][0])
    half_double = words.half_word(math.fsum([1., 2.**-11, 2.**-60]))
    head_matrix = Matrix([[1., 2.**-24, 2.**-60]])
    head_direct = words.float_word(head_matrix.apply([[1., 1., 1.]], 'f32')[0][0])
    head_double = words.float_word(math.fsum([1., 2.**-24, 2.**-60]))
    if (half_direct, half_double, head_direct, head_double) != (0x3c01, 0x3c00, 0x3f800001, 0x3f800000):
        raise ValueError('Direct-store double-rounding witness')
    cancellation = Matrix([[2.**100, 1., -2.**100]]).apply([[1., 1., 1.]])[0][0]
    ordered_fp64 = 0.
    for term in [2.**100, 1., -2.**100]:
        ordered_fp64 = ordered_fp64+term
    if cancellation != 1. or ordered_fp64 != 0.:
        raise ValueError('Exact cancellation witness')
    frequencies = inverse_frequencies()
    rotary_components = []
    for offset, count in [(0, 1), (1, 1), (5, 1), (11, 1), (9, 3)]:
        inputs = [[half(((11*r+7*c) % 31-15)/16) for c in range(D)] for r in range(count)]
        rotary_components.append(dict(offset=offset, heads=1, dim=D,
            input_words=[words.half_word(v) for v in flat(inputs)],
            main_words=[words.half_word(v) for v in flat(rotary(inputs, 1, offset, 'main'))],
            wide_words=[words.half_word(v) for v in flat(rotary(inputs, 1, offset, 'wide'))]))
    gains = [0., -.5, 1.5]+[1.]*(D-3)
    norm_input = [[half((i % 9-4)/4) for i in range(D)]]
    norm_main, norm_wide = norm(norm_input, gains, 'main'), norm(norm_input, gains, 'wide')
    u = [[0.]*H for _ in range(3)]
    for row, a, b in zip(u, [2., 6., 12.], [-3., -2., 4.], strict=True):
        row[:2] = [a, b]
    coefficients = [0.]*(3*H)
    coefficients[:6] = [1., 10., 100., 3., -2., 5.]
    z, before, after = convolution(u, coefficients, 'main')
    if [row[:2] for row in z] != [[200., -15.], [620., -4.], [1262., 15.]]:
        raise ValueError('Asymmetric Conv tap witness')
    _, probabilities, context = attention([[0.]*(Q*D)], [[0.]*(KV*D)],
        [[half((head+1)/4) for head in range(KV) for _ in range(D)]], 'main')
    if probabilities != [1.]*Q or [context[0][head*D] for head in range(Q)] != [.25, .25, .5, .5]:
        raise ValueError('One-key GQA head witness')
    operand = words.half_value(0x3c01)
    stored_product = product([[operand]], [[operand]])[0][0]
    if words.half_word(stored_product) != 0x3c02 or stored_product == operand*operand:
        raise ValueError('Separate product-store witness')
    combined = separate_store_witness()
    return dict(rounding=rounding,
        direct_store=dict(half_direct=half_direct, half_via_fp64=half_double,
                          head_direct=head_direct, head_via_fp64=head_double, cancellation=cancellation,
                          cancellation_ordered_fp64=ordered_fp64,
                          product_operand_word=0x3c01, product_store_word=0x3c02,
                          product_before_store=operand*operand, product_after_store=stored_product),
        separate_stores=combined,
        inverse_frequencies=dict(method='Decimal ln/exp80 with120-digit stability check of the same method; then FP64 angle/trig',
                                 decimal_precisions=[80, 120], f64_stable=True, values=frequencies,
                                 certified_rounding_proof=False,
                                 not_a_certified_native_intrinsic_emulation=True),
        rotary=rotary_components,
        direct_rms=dict(input_words=[words.half_word(v) for v in flat(norm_input)], gains=gains,
                        main_words=[words.half_word(v) for v in flat(norm_main)],
                        wide_words=[words.half_word(v) for v in flat(norm_wide)]),
        convolution=dict(u_first_two=[row[:2] for row in u], taps_first_two=coefficients[:6],
                         z_first_two=[row[:2] for row in z],
                         before_first_two=[[row[:2] for row in state] for state in before],
                         after_first_two=[[row[:2] for row in state] for state in after]),
        one_key_gqa=dict(probabilities=probabilities, first_feature_per_query_head=[context[0][head*D] for head in range(Q)]))


def separate_store_witness():
    """Each omitted intermediate store changes a later exact Half word.

    a=1025/1024, current tap5/4 and final affine coefficient3/2 are dyadic.
    The counterexamples below have <=38 significant bits, so their FP64
    products are exact. Expected words are fixed analytically, not target-fit.
    """
    operand = words.half_value(0x3c01)
    row = [[operand]+[0.]*(H-1)]
    u = product(row, row)
    taps = [0.]*(3*H)
    taps[2] = 1.25
    z, _, _ = convolution(u, taps, 'main')
    gated = product(row, z)
    projected = Matrix([[1.5]+[0.]*(H-1)]).apply(gated)
    actual = [words.half_word(v) for v in [u[0][0], z[0][0], gated[0][0], projected[0][0]]]
    skipped = [words.half_word(operand*operand*1.25),
               words.half_word(operand*(u[0][0]*1.25)),
               words.half_word((operand*z[0][0])*1.5)]
    if actual != [0x3c02, 0x3d02, 0x3d03, 0x3f84] or skipped != [0x3d03, 0x3d04, 0x3f85]:
        raise ValueError('Bx/Conv/Cz separate-store witness')
    return dict(operand_word=0x3c01, current_tap=1.25, final_affine_weight=1.5,
                u_conv_cz_affine_words=actual,
                missing_u_store_conv_word=skipped[0], missing_conv_store_cz_word=skipped[1],
                missing_cz_store_affine_word=skipped[2])


def load_inputs(directory, protocol_path):
    protocol_bytes = bounded_bytes(protocol_path, 2*1024**2)
    if identity(protocol_bytes)['sha256'] != PROTOCOL_SHA:
        raise ValueError('Frozen graph protocol')
    protocol = strict_json(protocol_bytes)
    config_bytes = bounded_bytes(directory/'config.json', 499)
    if identity(config_bytes) != dict(bytes=499, sha256=CONFIG_SHA):
        raise ValueError('Original configuration identity')
    matrices, norms, convs = decode(bounded_bytes(directory/'model.safetensors', 760078))
    return protocol, matrices, norms, convs


def export(directory, artifact_dir, protocol_path):
    protocol, matrices, norms, convs = load_inputs(artifact_dir, protocol_path)
    paths = [Path(__file__).resolve(), Path(words.__file__).resolve(), Path(rms_leaf.__file__).resolve(),
             Path(conv_leaf.__file__).resolve(), protocol_path.resolve(),
             (artifact_dir/'config.json').resolve(), (artifact_dir/'model.safetensors').resolve()]
    dependencies = {str(path): identity(path.read_bytes()) for path in paths}
    directory.mkdir(parents=False, exist_ok=False)
    try:
        component_bytes = encoded(components())
        if len(component_bytes) > MAX_COMPONENTS:
            raise ValueError('Components byte bound')
        (directory/'components.json').write_bytes(component_bytes)
        exported, previous, descriptors = [], {}, 0
        with (directory/'arrays.bin').open('xb') as binary, (directory/'progress.jsonl').open('x') as progress:
            for case in protocol['cases']:
                name, tokens, specs = case['name'], case['tokens'], case['stages']
                profiles = {}
                for profile in ('main', 'wide'):
                    progress.write(json.dumps(dict(case=name, profile=profile, started=True, offset=binary.tell()))+'\n')
                    progress.flush()
                    captures = forward(matrices, norms, convs, tokens, profile)
                    if set(captures) != set(specs) or len(captures) != 63:
                        raise ValueError('Graph stage inventory')
                    if captures['layers.0.history_after'] == captures['layers.2.history_after']:
                        raise ValueError('Distinct Conv owners')
                    # Raw future scores may change; visible scores and earlier
                    # causal stage rows must remain exact in this source profile.
                    if name == 'future_changed12':
                        causal_equal(previous[profile], captures, specs, len(tokens))
                    if name == 'context12':
                        previous[profile] = captures
                    profiles[profile] = captures
                    progress.write(json.dumps(dict(case=name, profile=profile, complete=True, groups=len(captures)))+'\n')
                    progress.flush()
                groups = {}
                if list(profiles['main']) != list(profiles['wide']):
                    raise ValueError('Profile semantic capture order')
                for name, spec in specs.items():
                    main, wide = profiles['main'][name], profiles['wide'][name]
                    if len(main) != spec['elements'] or len(wide) != spec['elements'] or math.prod(spec['shape']) != spec['elements']:
                        raise ValueError('Graph stage shape')
                    envelope = max(abs(a-b) for a, b in zip(main, wide, strict=True))
                    stage = dict(shape=spec['shape'], envelope=envelope,
                        gate=dict(atol=max(1e-4, 2*envelope), rtol=1e-3))
                    for profile in ('main', 'wide'):
                        values = profiles[profile][name]
                        dtype = spec['dtype']
                        if dtype not in ('f16le', 'f32le'):
                            raise ValueError('Stage storage type')
                        raw = b''.join(struct.pack('<e' if dtype == 'f16le' else '<f', value) for value in values)
                        start = binary.tell()
                        if start+len(raw) > MAX_BINARY:
                            raise ValueError('Graph binary bound')
                        binary.write(raw)
                        stage[profile] = dict(file='arrays.bin', offset=start, dtype=dtype,
                                              shape=spec['shape'], **identity(raw))
                        descriptors += 1
                    groups[name] = stage
                binary.flush()
                exported.append(dict(name=case['name'], tokens=tokens, offset=0, partitions=case['partitions'],
                                     stage_capture_order=list(profiles['main']), stages=groups))
        if descriptors != 504 or (directory/'arrays.bin').stat().st_size != 2212480:
            raise ValueError('Prospective graph byte/descriptor counts')
        for path, expected in dependencies.items():
            if identity(Path(path).read_bytes()) != expected:
                raise ValueError('Source dependency changed')
        files = {name: identity((directory/name).read_bytes()) for name in ('arrays.bin', 'components.json', 'progress.jsonl')}
        manifest = dict(format=FORMAT, protocol_sha256=PROTOCOL_SHA, policy=protocol['policy'],
            geometry=protocol['geometry'], counts=protocol['counts'], cases=exported, files=files,
            dependencies=dependencies, original_artifact=dict(path=str((artifact_dir/'model.safetensors').resolve()),
                **identity((artifact_dir/'model.safetensors').read_bytes())),
            math=dict(affine='Exact integer-dyadic original FP32 weights times stored Half; direct RNE to destination, both profiles',
                exact_integer_common_factor='A matrix-wide power of two is factored exactly from the2^-149 weight grid; no information is lost',
                rms='Qualified D1 direct-gain epsilon1e-5 MAIN/diagnostic WIDE',
                conv='Qualified D2 explicit FP32 tap order/ordered FP64 WIDE with raw Half u history',
                attention='D3 serial Reference FP32 stores/ordered FP64 WIDE; all query/head/column cells',
                rotary='MAIN FP32 exponent/pow/angle/trig and explicit products; WIDE Decimal80/120 stable frequencies then FP64 angle/trig',
                pointwise='Separate Half stores for Bx, Conv, Cz, SiLU, gating and residual'),
            physical_tensors=65, logical_owners=29, affine_owners=18, direct_norm_owners=9, depthwise3_owners=2,
            source_graph_executed=True, target_executed=False, gpu_executed=False, real_model_executed=False,
            mathematical_source_not_native_bit_emulation=True,
            target_state_scope='Compare observed pre/post chunk histories and active committed KV prefixes; inactive source+0 is not native tail clearing',
            actual_helper_calls=['RMS.rms', 'Conv.conv', 'generic word serializers only'],
            runtime=dict(python=sys.version, executable=str(Path(sys.executable).resolve())))
        raw = encoded(manifest)
        if len(raw) > MAX_MANIFEST:
            raise ValueError('Graph manifest bound')
        (directory/'manifest.json').write_bytes(raw)
        return identity(raw)
    except BaseException as error:
        (directory/'failure.json').write_bytes(encoded(dict(passed=False, error=repr(error), target_executed=False)))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--artifact-dir', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export(args.output_dir, args.artifact_dir, args.protocol)))

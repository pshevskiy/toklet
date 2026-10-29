"""Bounded actual-input LFM2 contracts, never a graph/trajectory oracle.

Inputs are authenticated observed FP32 words, with Half-valued operands checked
before use. The caller authenticates original weights, record identities and
literal source gates. No Gemma wrapper, source producer, model or codec import.
"""
from functools import lru_cache
import math
import numpy as np
from gemma3_conformance_math import (
    ContractError, D, F, H, require, finite, half, half_input, bits, dec,
    decimal_context, point, mul, trans, provider, exp_contract, sincos_true,
    rne_interval, f32_members, sqrt32,
)
import lfm2_exact_affine as affine

PROFILE = 'lfm2-native-half-actual-input-contract-v1'
PROTOCOL_SHA256 = '49005e6e0342adc1e05177063623d149c2319b1966b52d0174ea00d46e6ce59f'
PROVIDER_ULP = (('exp', 4), ('pow', 16), ('sin', 4), ('cos', 4))
PROJECTION_ZERO_POLICY = 'exact-zero:+0-singleton; nonzero-underflow:sign-correct-bracket-words'
MAX_VALUES = 4 * 4608
EPSILON = F(1e-5)


def array(value, maximum=MAX_VALUES):
    require(isinstance(value, np.ndarray) and value.dtype == np.dtype('<f4'), 'stored FP32 array required')
    require(0 < value.size <= maximum, 'bounded nonempty operand')
    finite(value)
    return value


def stored(value, maximum=MAX_VALUES):
    return half_input(array(value, maximum))


def direct_coefficients(value, shape):
    value = array(value)
    require(value.shape == shape, 'direct BF16 coefficient shape')
    require(np.all((value.view(np.uint32) & 65535) == 0), 'coefficient is not widened original BF16')
    return value


def numeric_gate(actual, center, gate):
    """Caller binds these literal coefficients to the authenticated source group."""
    actual = array(actual); center = array(center)
    require(actual.shape == center.shape, 'local gate shape')
    require(isinstance(gate, dict) and set(gate) == {'atol', 'rtol'}, 'literal gate schema')
    require(type(gate['atol']) in (int, float) and math.isfinite(gate['atol']) and gate['atol'] >= 1e-4
            and type(gate['rtol']) in (int, float) and gate['rtol'] == .001, 'literal coefficient policy')
    error = np.abs(actual.astype(np.float64) - center.astype(np.float64))
    limit = gate['atol'] + gate['rtol'] * np.abs(center.astype(np.float64))
    return error <= limit, float(error.max(initial=0)), float((error / limit).max(initial=0))


def conjoined(actual, center, semantic, gate):
    """Semantic membership and the old local numeric guard are independently fatal."""
    actual = array(actual); center = array(center)
    require(isinstance(semantic, np.ndarray) and semantic.dtype == np.bool_
            and semantic.shape == actual.shape, 'semantic result shape/type')
    numeric, error, fraction = numeric_gate(actual, center, gate)
    return semantic & numeric, {'semantic_failures': int(np.count_nonzero(~semantic)),
                               'numeric_failures': int(np.count_nonzero(~numeric)),
                               'maximum_error': error, 'maximum_fraction': fraction}


def gather_row(original):
    original = array(original)
    require(original.shape == (1024,), 'selected original table row')
    return half(original)


def bcx_view(parent, part):
    parent = stored(parent)
    require(parent.shape == (3072,) and part in ('b', 'c', 'x'), 'fused BCx view')
    offset = {'b': 0, 'c': 1024, 'x': 2048}[part]
    return parent[offset:offset + 1024]  # Read-only use; no invented parent copy.


def binary(a, b, operation):
    a = stored(a); b = stored(b)
    require(a.shape == b.shape and operation in ('product', 'residual'), 'binary operation/shape')
    with np.errstate(over='ignore', invalid='ignore'):
        value = F(a * b) if operation == 'product' else F(a + b)
    return half(finite(value))  # Ordinary residual: no Gemma clamp.


def square_sum(row):
    """Ascending FP32 Neumaier, shared by the direct-gain RMS contract."""
    row = stored(row)
    require(row.ndim == 1 and row.size in (64, 1024), 'RMS reduction width')
    total = F(0); correction = F(0)
    for value in row:
        product = F(value * value); update = F(total + product)
        error = F(F(total - update) + product) if abs(total) >= abs(product) else F(F(product - update) + total)
        correction = F(correction + error); total = update
    return F(total + correction)


def rms(x, gains):
    x = stored(x); gains = array(gains)
    require(gains.ndim == 1 and gains.size in (64, 1024), 'direct RMS gain width')
    width = gains.size
    direct_coefficients(gains, (width,))
    require(x.ndim in (1, 2) and x.shape[-1] == width
            and x.size // width <= (64 if width == 64 else 4), 'bounded direct RMS rows')
    rows = x.reshape(-1, width); out = np.empty_like(rows)
    for i, row in enumerate(rows):
        variance = F(F(square_sum(row) / F(width)) + EPSILON)
        inverse = F(F(1) / sqrt32(variance))
        with np.errstate(over='ignore', invalid='ignore'):
            normalized = F(row * inverse); scaled = F(normalized * gains)
        finite(normalized); out[i] = half(scaled)
    return out.reshape(x.shape)


def conv_row(u, old, coefficients, row):
    """Actual current-chunk u plus actual old history, not source predecessors."""
    u = stored(u, 32 * 1024); old = stored(old)
    require(u.ndim == 2 and u.shape[1] == 1024 and 1 <= len(u) <= 32
            and old.shape == (2, 1024), 'observed Conv3 operands')
    require(type(row) is int and 0 <= row < len(u), 'actual Conv3 row')
    coefficients = direct_coefficients(coefficients, (1024, 3))
    a = old[row] if row < 2 else u[row - 2]
    b = old[1] if row == 0 else u[row - 1]
    with np.errstate(over='ignore', invalid='ignore'):
        p0 = F(a * coefficients[:, 0]); finite(p0)
        p1 = F(b * coefficients[:, 1]); finite(p1)
        p2 = F(u[row] * coefficients[:, 2]); finite(p2)
        s01 = F(p0 + p1); finite(s01)
        total = F(s01 + p2); finite(total)
    return half(total)


def history_suffix(old, u):
    old = stored(old); u = stored(u, 32 * 1024)
    require(old.shape == (2, 1024) and u.ndim == 2 and u.shape[1] == 1024
            and 1 <= len(u) <= 32, 'actual history suffix inputs')
    return np.stack((old[1], u[0])) if len(u) == 1 else u[-2:].copy()


def _dyadic_half(value):
    word = int(H(value).view(np.uint16)); field = (word >> 10) & 31
    mantissa = (word & 1023) | (1024 if field else 0)
    if word & 32768: mantissa = -mantissa
    return mantissa, field - 25 if field else -24


def _compare(a, ae, b, be):
    e = min(ae, be); left = a << (ae - e); right = b << (be - e)
    return (left > right) - (left < right)


def _exact_bracket(value, exponent, center):
    require(_compare(abs(value), exponent, 65504, 0) <= 0, 'unsupported finite faithful Half bracket')
    if value == 0: return F(0), F(0)
    c, ce = _dyadic_half(center); order = _compare(value, exponent, c, ce)
    if order == 0: return center, center
    h = H(center)
    lower = F(np.nextafter(h, H(-np.inf), dtype=H)) if order < 0 else center
    upper = F(np.nextafter(h, H(np.inf), dtype=H)) if order > 0 else center
    lo, le = _dyadic_half(lower); hi, he = _dyadic_half(upper)
    require(_compare(lo, le, value, exponent) <= 0 and _compare(value, exponent, hi, he) <= 0,
            'exact affine center/bracket mismatch')
    return lower, upper


def projection(inputs, weights, actual):
    """One <=128-row original stripe; <=4 selected actual Half input rows.

    Nonzero interior exact RNE equality is a faithful certificate. Zero and
    maxfinite centers require exact checking, including signed-zero words.
    Neighbor candidates require an exact dyadic bracket; no ±one-ULP shortcut.
    """
    inputs = stored(inputs, 4 * 4608); actual = stored(actual, 4 * 128)
    require(isinstance(weights, np.ndarray) and weights.dtype == np.dtype('<f4')
            and weights.ndim == 2 and weights.flags.c_contiguous, 'original contiguous FP32 stripe')
    require(inputs.ndim == 2 and 1 <= len(inputs) <= 4 and inputs.flags.c_contiguous
            and 0 < inputs.shape[1] <= 4608 and inputs.shape[1] % 64 == 0
            and weights.shape[1] == inputs.shape[1] and 1 <= len(weights) <= 128
            and actual.shape == (len(inputs), len(weights)), 'affine stripe/input/output shape')
    finite(weights)
    half_inputs = inputs.astype(H)
    words, center_stats = affine.apply(weights, half_inputs, 'half')
    center = words.view(H).astype(F); semantic = np.zeros(actual.shape, dtype=bool)
    bracket_stats = dict(fp64=0, int64=0, zero=0, blocked_outputs=0, certified_blocks=0, scalar_blocks=0)
    checks = 0; equal_certificates = 0
    for token in range(len(inputs)):
        for row in range(len(weights)):
            near = center[token, row]; observed = actual[token, row]
            if 0 < abs(near) < 65504 and observed == near:
                semantic[token, row] = True; equal_certificates += 1; continue
            if 0 < abs(near) < 65504:
                low = F(np.nextafter(H(near), H(-np.inf), dtype=H))
                high = F(np.nextafter(H(near), H(np.inf), dtype=H))
                if observed not in (low, high): continue
            exact, exponent = affine.exact_pair(weights[row], half_inputs[token], bracket_stats); checks += 1
            lo, hi = _exact_bracket(exact, exponent, near)
            observed_word = int(H(observed).view(np.uint16))
            semantic[token, row] = observed_word in (int(H(lo).view(np.uint16)), int(H(hi).view(np.uint16)))
    return center, semantic, {'center': center_stats, 'bracket': bracket_stats,
                              'exact_bracket_checks': checks, 'equal_center_certificates': equal_certificates}


@lru_cache(maxsize=65536)
def silu_words(half_word):
    require(type(half_word) is int and 0 <= half_word <= 65535, 'Half SiLU word')
    x = np.array([half_word], dtype=np.uint16).view(H).astype(F)[0]
    finite(np.array([x], dtype=F))
    lo, hi, point_exp = exp_contract(F(-abs(x)))
    e = f32_members(lo, hi, 16)
    denominator = F(F(1) + e)
    sigma = F(F(1) / denominator) if x >= 0 else F(e / denominator)
    outputs = np.unique(half(F(x * sigma)).view(np.uint32)).view(F)
    d = F(F(1) + point_exp)
    near = half(np.array([F(x * (F(F(1) / d) if x >= 0 else F(point_exp / d)))], dtype=F))[0]
    outputs.flags.writeable = False
    return outputs, near


def silu(x, actual):
    x = stored(x); actual = stored(actual)
    require(x.shape == actual.shape, 'SiLU output shape')
    center = np.empty_like(x); semantic = np.empty(x.shape, dtype=bool)
    for i, value in enumerate(x.flat):
        values, near = silu_words(int(H(value).view(np.uint16)))
        center.flat[i] = near; semantic.flat[i] = np.any(values.view(np.uint32) == bits(actual.flat[i]))
    return center, semantic


def signed_trig_members(lo, hi, cap):
    """Numeric sin/cos provider bounds do not select a zero sign.

    nextafter traverses only one zero representation. Preserve BOTH words when
    zero is inside the bound, then propagate their separately rounded stores.
    This does not apply to the positive-exp candidate contract.
    """
    values = f32_members(lo, hi, cap)
    if lo <= 0 <= hi:
        words = np.unique(np.concatenate((values.view(np.uint32),
                                         np.array([0, 0x80000000], dtype=np.uint32))))
        require(len(words) <= cap, 'signed intrinsic discrete-set bound')
        values = words.view(F)
    return values


@lru_cache(maxsize=256)
@decimal_context
def rotary_pairs(theta, dimension, position, feature, profile=PROFILE):
    require(profile == PROFILE and type(theta) is int and theta == 1000000 and type(dimension) is int and dimension == 64
            and type(position) is int and 0 <= position <= 63 and type(feature) is int
            and 0 <= feature < 32, 'LFM ordinary rotary domain')
    if position == 0:
        ranges = ((provider(point(D(1)), 4)[:2], provider(point(D(0)), 4)[:2]),)
        point_cos, point_sin = F(1), F(0)
    else:
        exponent = F(F(F(-2) * F(feature)) / F(dimension))
        truth = point(D(1)) if feature == 0 else trans('exp', mul(point(dec(exponent)), trans('ln', point(D(theta)))))
        power = provider(truth, 16)
        frequencies = f32_members(power[0], power[1], 80)
        ranges = []
        for frequency in frequencies:
            angle = F(F(position) * frequency)
            ranges.append((provider(sincos_true(angle, True), 4)[:2], provider(sincos_true(angle), 4)[:2]))
        angle = F(F(position) * power[2])
        point_cos = rne_interval(sincos_true(angle, True)); point_sin = rne_interval(sincos_true(angle))
    co = []; si = []
    for cr, sr in ranges:
        c = signed_trig_members(*cr, 16); s = signed_trig_members(*sr, 16)
        co.append(np.repeat(c, len(s))); si.append(np.tile(s, len(c)))
    c = np.concatenate(co); s = np.concatenate(si)
    require(c.size <= 4096, 'rotary candidate owner bound')
    c.flags.writeable = False; s.flags.writeable = False
    return c, s, point_cos, point_sin


def rotary(x, actual, position, theta=1000000, dimension=64):
    x = stored(x); actual = stored(actual)
    require(x.shape == actual.shape and x.ndim == 2 and x.shape in ((16, 64), (8, 64)), 'Q16/KV8 rotary rows')
    center = np.empty_like(x); semantic = np.empty(x.shape, dtype=bool)
    for feature in range(32):
        co, si, c, s = rotary_pairs(theta, dimension, position, feature, PROFILE)
        a, b = x[:, feature], x[:, feature + 32]
        center[:, feature] = half(F(F(a * c) - F(b * s)))
        center[:, feature + 32] = half(F(F(b * c) + F(a * s)))
        left = half(F(F(a[:, None] * co) - F(b[:, None] * si)))
        right = half(F(F(b[:, None] * co) + F(a[:, None] * si)))
        compatible = (left.view(np.uint32) == actual[:, feature, None].view(np.uint32)) & (right.view(np.uint32) == actual[:, feature + 32, None].view(np.uint32))
        good = np.any(compatible, axis=1)
        semantic[:, feature] = good; semantic[:, feature + 32] = good
    return center, semantic


def probabilities(scores, actual, query, key_start=0):
    scores = array(scores); actual = array(actual)
    require(scores.ndim == 3 and scores.shape == actual.shape and scores.shape[:2] == (16, 1)
            and 1 <= scores.shape[2] <= 64, 'Q16 materialized probability shape')
    require(type(query) is int and 0 <= query < scores.shape[2] and type(key_start) is int
            and key_start == 0, 'full-retention actual query/key range')
    require(np.all((actual >= 0) & (actual <= 1)), 'probability outside[0,1]')
    visible = np.arange(scores.shape[2]) <= query
    require(np.all(actual[:, :, ~visible].view(np.uint32) == 0), 'masked probability must be positive zero')
    center = np.zeros_like(scores); semantic = np.ones(scores.shape, dtype=bool)
    for head in range(16):
        row = scores[head, 0]; maximum = np.max(row[visible]); low = []; high = []; points = []
        for i in range(query + 1):
            lo, hi, near = exp_contract(F(row[i] - maximum))
            require(0 <= lo <= near <= hi, 'positive exp enclosure')
            low.append(lo); high.append(hi); points.append(near)
        if query == 0:
            center[head, 0, 0] = 1; semantic[head, 0, 0] = actual[head, 0, 0] == 1; continue
        ml = mh = mp = F(0)
        for lo, hi, near in zip(low, high, points, strict=True):
            ml = F(ml + lo); mh = F(mh + hi); mp = F(mp + near)
        require(np.isfinite(mh) and 0 < ml <= mp <= mh, 'serial probability mass enclosure')
        for i, (lo, hi, near) in enumerate(zip(low, high, points, strict=True)):
            # Correlated hidden exponentials are enclosed per value, not asserted
            # to have a single independently realizable joint endpoint vector.
            center[head, 0, i] = F(near / mp)
            semantic[head, 0, i] = F(lo / mh) <= actual[head, 0, i] <= F(hi / ml)
    return center, semantic

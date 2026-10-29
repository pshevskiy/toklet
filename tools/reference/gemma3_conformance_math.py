"""Independent bounded scalar contracts; no graph or source producer imports.

FP32 basic operations use NumPy scalar/array FP32 operations, no reassociation.
Decimal intervals enclose exact intrinsic results; native outputs never choose a
bound. All caches are bounded and contain only scalar contracts, not activations.
"""
from decimal import Decimal as D, localcontext, ROUND_FLOOR, ROUND_CEILING
from fractions import Fraction
from functools import lru_cache, wraps
import math
import struct
import numpy as np

F = np.float32
H = np.float16
PRECISION = 120
# Truncated decimal pi, independently documented mathematical constant; true pi
# is in this interval (100 digits after the decimal point).
PI_LO = D('3.1415926535897932384626433832795028841971693993751058209749445923078164062862089986280348253421170679')
PI_HI = D('3.1415926535897932384626433832795028841971693993751058209749445923078164062862089986280348253421170680')
MIN_NORMAL = D.from_float(2.0**-126)

def decimal_context(function):
    @wraps(function)
    def bounded(*args, **kwargs):
        with localcontext() as ctx:
            ctx.prec = PRECISION
            return function(*args, **kwargs)
    return bounded

class ContractError(ValueError):
    pass

def require(ok, message):
    if not ok:
        raise ContractError(message)

def bits(x):
    return struct.unpack('<I', struct.pack('<f', float(x)))[0]

def from_bits(x):
    return F(struct.unpack('<f', struct.pack('<I', int(x)))[0])

def dec(x):
    return D.from_float(float(x))

def finite(x):
    require(bool(np.isfinite(x).all()), 'nonfinite arithmetic operand/result')
    return x

def half(x):
    x = np.asarray(x, dtype=F)
    finite(x)
    with np.errstate(over='ignore', invalid='ignore'):
        h = x.astype(H).astype(F)
    finite(h)
    return h

def half_input(x):
    x = np.asarray(x, dtype=F)
    require(np.array_equal(x, half(x)), 'operand is not exactly finite Half')
    return x

def equal_numeric(a, b):
    return np.asarray(a) == np.asarray(b)  # +/-0 equivalent, nonfinite rejected separately

def _op(a, b, sign, rounding):
    with localcontext() as ctx:
        ctx.prec = PRECISION; ctx.rounding = rounding
        return a + b if sign == '+' else a - b if sign == '-' else a * b if sign == '*' else a / b

def add(a, b):
    return (_op(a[0], b[0], '+', ROUND_FLOOR), _op(a[1], b[1], '+', ROUND_CEILING))

def neg(a):
    return (-a[1], -a[0])

def mul(a, b):
    lo = [_op(x, y, '*', ROUND_FLOOR) for x in a for y in b]
    hi = [_op(x, y, '*', ROUND_CEILING) for x in a for y in b]
    return min(lo), max(hi)

def div(a, b):
    require(b[0] > 0 or b[1] < 0, 'interval division crosses zero')
    lo = [_op(x, y, '/', ROUND_FLOOR) for x in a for y in b]
    hi = [_op(x, y, '/', ROUND_CEILING) for x in a for y in b]
    return min(lo), max(hi)

def point(x):
    return x, x

def trans(name, a):
    """Decimal exp/ln are correctly rounded; adjacent Decimals enclose truth."""
    out = []
    for x in a:
        with localcontext() as ctx:
            ctx.prec = PRECISION
            y = x.exp() if name == 'exp' else x.ln()
            require(y.is_finite() and (name != 'exp' or y > 0), 'unsupported Decimal intrinsic domain')
            out.append((ctx.next_minus(y), ctx.next_plus(y)))
    return out[0][0], out[1][1]

@decimal_context
def f32_bracket(x):
    """Directed FP32 enclosure of an exact Decimal, independent of double seed."""
    require(x.is_finite(), 'nonfinite true result')
    with np.errstate(over='ignore'):
        seed = F(float(x))
    require(np.isfinite(seed), 'true result outside finite FP32 domain')
    lo = hi = seed
    for _ in range(4):
        if dec(lo) <= x: break
        lo = np.nextafter(lo, F(-np.inf), dtype=F)
    for _ in range(4):
        if dec(hi) >= x: break
        hi = np.nextafter(hi, F(np.inf), dtype=F)
    require(np.isfinite(lo) and np.isfinite(hi) and dec(lo) <= x <= dec(hi), 'FP32 enclosure inconclusive')
    return lo, hi

@decimal_context
def rne_decimal(x):
    lo, hi = f32_bracket(x)
    dl, dh = x - dec(lo), dec(hi) - x
    return lo if dl < dh or (dl == dh and bits(lo) % 2 == 0) else hi

@decimal_context
def rne_interval(a):
    lo, hi = rne_decimal(a[0]), rne_decimal(a[1])
    require(lo == hi, 'true-result point rounding inconclusive')
    return lo

@decimal_context
def ulp_interval(a):
    """Apple8.4 true-result ULP. Exact-grid uses the nearest unequal spacing."""
    if a[0] == a[1]:
        x = a[0]; lo, hi = f32_bracket(x)
        if lo == hi:
            prev = np.nextafter(lo, F(-np.inf), dtype=F)
            nxt = np.nextafter(lo, F(np.inf), dtype=F)
            gaps = [abs(x - dec(v)) for v in (prev, nxt) if np.isfinite(v)]
            return min(gaps)
    a0, a1 = f32_bracket(a[0]), f32_bracket(a[1])
    # A true interval may touch an exact-grid endpoint. Its spacing choice must
    # be unambiguous; retry/fallback is not inferred from the target word.
    require(a0[0] == a1[0] and a0[1] == a1[1] and a0[0] != a0[1], 'true ULP binade/grid unresolved')
    return dec(a0[1]) - dec(a0[0])

@decimal_context
def provider(a, count):
    u = ulp_interval(a)
    radius = point(u * count)
    lo = add(a, neg(radius))[0]; hi = add(a, radius)[1]
    # First/last representable FP32 words in the outward exact-result envelope.
    low = f32_bracket(lo)[1]; high = f32_bracket(hi)[0]
    require(low <= high, 'empty provider set')
    for candidate in (low,high):
        # Outer enumeration is not enough: reject a Decimal boundary whose
        # uncertainty could straddle the actual published admissibility edge.
        distance=max(_op(dec(candidate),a[0],'-',ROUND_CEILING),
                     _op(a[1],dec(candidate),'-',ROUND_CEILING))
        require(distance<=radius[0], 'provider candidate boundary unresolved')
    return low, high, rne_interval(a)

@decimal_context
def exp_contract(x):
    x = F(x); require(np.isfinite(x) and x <= 0, 'exp argument outside nonpositive finite domain')
    if x == 0: return provider(point(D(1)), 4)
    if x <= -200:
        # e>2 => exp(-200)<2^-200<.5*2^-149. True ULP below
        # minnormal is2^-149; precisely the finite words0..4*2^-149 fit.
        return F(0), F(4*2.0**-149), F(0)
    a = trans('exp', point(dec(x)))
    # No new FTZ contract. Tiny exp whose complete4ULP interval rounds to zero
    # is accepted only if inside4ULP; positive subnormal values farther from
    # zero cannot be silently flushed. The oracle preserves subnormal F32 words.
    lo, hi, near = provider(a, 4)
    return max(F(0),lo), hi, near

@lru_cache(maxsize=65536)
@decimal_context
def tanh_contract(word):
    x = from_bits(word); require(np.isfinite(x), 'nonfinite tanh argument')
    if x == 0: return provider(point(D(0)), 5)
    z = dec(abs(x))
    # Avoid exp overflow using exp(-2|z|); for huge inputs use a rigorously
    # bounded distance from1. A strict interval just below1 has lower-binadeULP.
    if z >= 64:
        with localcontext() as ctx:
            ctx.prec = PRECISION
            a = (D(1) - D('1e-54'), D(1) - D('1e-80'))
        # tanh(64) differs from1 by <6e-56; upper endpoint must not exclude
        # larger z. Use1 as inclusive upper, with known strict-below1 ULP.
        u = D(2) ** -24
        low = f32_bracket(D(1) - 5*u - D('1e-54'))[1]
        high = f32_bracket(D(1) + 5*u)[0]
        result = (low, high, F(1))
    else:
        e = trans('exp', point(-2*z))
        a = div(add(point(D(1)), neg(e)), add(point(D(1)), e))
        result = provider(a, 5)
    return tuple(-v for v in (result[1], result[0], result[2])) if x < 0 else result

@decimal_context
def sincos_true(x, cosine=False):
    """Directed Taylor enclosure at an exact FP32 angle in[0,544]."""
    x = F(x); require(np.isfinite(x) and 0 <= x <= 544, 'unsupported rotary angle')
    if x == 0: return point(D(1) if cosine else D(0))
    with localcontext() as ctx:
        ctx.prec = PRECISION
        turns = int((dec(x) / (2*PI_LO)).to_integral_value())
    y = add(point(dec(x)), neg(mul(point(D(2*turns)), (PI_LO, PI_HI))))
    require(max(abs(y[0]), abs(y[1])) < D(4), 'range reduction failed')
    square = mul(y, y)
    term = point(D(1)) if cosine else y
    total = term
    for k in range(1, 72):
        denominator = D((2*k-1)*(2*k) if cosine else (2*k)*(2*k+1))
        term = neg(div(mul(term, square), point(denominator)))
        total = add(total, term)
    # At |y|<4 the alternating remainder is no larger than next term after72.
    k = 72; denominator = D((2*k-1)*(2*k) if cosine else (2*k)*(2*k+1))
    nxt = div(mul(term, square), point(denominator))
    bound = max(abs(nxt[0]), abs(nxt[1]))
    return add(total, (-bound, bound))

@lru_cache(maxsize=8)
@decimal_context
def rotary_contract(theta, position, feature, dim):
    require(theta in (10000, 1000000) and type(position) is int and 0 <= position <= 543 and dim == 256 and 0 <= feature < 128, 'rotary domain')
    if position == 0:
        co=provider(point(D(1)),4); si=provider(point(D(0)),4)
        return ((co[:2],si[:2]),),F(1),F(0)
    exponent = F(F(F(-2)*F(feature))/F(dim))
    if feature == 0:
        power = provider(point(D(1)), 16)
    else:
        truth = trans('exp', mul(point(dec(exponent)), trans('ln', point(D(theta)))))
        power = provider(truth, 16)
    freq = power[0]; pairs = []; count = 0
    while freq <= power[1]:
        angle = F(F(position)*freq)
        co = provider(sincos_true(angle, True), 4)
        si = provider(sincos_true(angle, False), 4)
        pairs.append((co[:2], si[:2])); count += 1
        require(count <= 80, 'provider pow candidate bound')
        freq = np.nextafter(freq, F(np.inf), dtype=F)
    angle = F(F(position)*power[2])
    co = rne_interval(sincos_true(angle, True)); si = rne_interval(sincos_true(angle, False))
    return tuple(pairs), co, si

def interval_product(a, b):
    vals = [F(x*y) for x in a for y in b]; finite(np.array(vals))
    return min(vals), max(vals)

def interval_add(a, b):
    return finite(np.array([F(a[0]+b[0]), F(a[1]+b[1])], dtype=F))

def interval_sub(a, b):
    return finite(np.array([F(a[0]-b[1]), F(a[1]-b[0])], dtype=F))

def contains_half(native, lo, hi):
    lo, hi = half(np.array([lo, hi]))
    return lo <= native <= hi

def sqrt32(x):
    """Correct FP32 sqrt, certified by exact squared rounding midpoints."""
    x = F(x); require(np.isfinite(x) and x >= 0, 'sqrt domain')
    if x == 0: return x
    q = Fraction.from_float(float(x)); seed = F(math.sqrt(float(x)))
    for _ in range(4):
        before = np.nextafter(seed, F(-np.inf), dtype=F); after = np.nextafter(seed, F(np.inf), dtype=F)
        a = (Fraction.from_float(float(before))+Fraction.from_float(float(seed)))/2
        b = (Fraction.from_float(float(after))+Fraction.from_float(float(seed)))/2
        if q < a*a or (q == a*a and bits(seed)%2): seed = before; continue
        if q > b*b or (q == b*b and bits(seed)%2): seed = after; continue
        return seed
    raise ContractError('sqrt rounding certificate inconclusive')

def rms(x, scale):
    x = half_input(x); scale = np.asarray(scale, dtype=F); finite(scale)
    require(scale.ndim == 1 and x.size % len(scale) == 0, 'RMS shape')
    rows = x.reshape(-1, len(scale)); out = np.empty_like(rows)
    for index, row in enumerate(rows):
        s = F(0); correction = F(0)
        for v in row:
            p = F(v*v); t = F(s+p)
            e = F(F(s-t)+p) if abs(s) >= abs(p) else F(F(p-t)+s)
            correction = F(correction+e); s = t
        inv = F(F(1)/sqrt32(F(F(F(s+correction)/F(len(scale)))+F(1e-6))))
        out[index] = half(F(F(row*inv)*scale))
    return out.reshape(x.shape)

def half_bracket_fraction(q):
    require(abs(q) <= 65504, 'unsupported finite faithful Half bracket')
    seed = H(float(q)); v = Fraction.from_float(float(seed))
    if v == q: return F(seed), F(seed), F(seed)
    lo = seed if v < q else np.nextafter(seed, H(-np.inf), dtype=H)
    hi = seed if v > q else np.nextafter(seed, H(np.inf), dtype=H)
    a, b = Fraction.from_float(float(lo)), Fraction.from_float(float(hi))
    require(a <= q <= b, 'Half bracket seed failure')
    if q-a < b-q: nearest = lo
    elif q-a > b-q: nearest = hi
    else: nearest = lo if int(lo.view(np.uint16)) % 2 == 0 else hi
    return F(lo), F(hi), F(nearest)

def projection(x, weights, actual):
    """One row, at most2048x2048; <=64MiB numeric owners including cache."""
    x = half_input(x).reshape(-1); w = np.asarray(weights, dtype=F); actual = half_input(actual).reshape(-1)
    require(w.ndim == 2 and w.shape == (len(actual), len(x)) and max(w.shape) <= 2048 and w.size <= 1310720, 'projection shape/bound')
    finite(w)
    require(np.all((w == 0) | (np.abs(w) >= F(2**-79))), 'projection weight domain')
    wide = w.astype(np.float64); estimate = wide @ x.astype(np.float64)
    n = len(x); u = 2.0**-53
    gamma = np.nextafter((n*u)/(1-n*u), np.inf)
    # Half values have common2^-24 grid and <=2048 length; this sum is exactF64.
    magnitude = float(np.sum(np.abs(x.astype(np.float64))))
    bound = np.nextafter(np.nextafter(gamma*magnitude, np.inf)*np.max(np.abs(w), axis=1).astype(np.float64), np.inf)
    lower = np.nextafter(estimate-bound, -np.inf); upper = np.nextafter(estimate+bound, np.inf)
    out = np.empty(len(actual), dtype=F); valid = np.empty(len(actual), dtype=bool); fallbacks = 0
    # Certificates use exact rational endpoints to avoid FP64->Half double rounding.
    for j in range(len(actual)):
        if magnitude == 0 or not np.any(w[j]):
            out[j]=F(0);valid[j]=actual[j]==0;continue
        finite_interval = -65504 <= lower[j] <= upper[j] <= 65504
        a = half_bracket_fraction(Fraction.from_float(float(lower[j]))) if finite_interval else None
        b = half_bracket_fraction(Fraction.from_float(float(upper[j]))) if finite_interval else None
        if finite_interval and a == b:
            lo, hi, near = a
        else:
            exact = sum((Fraction.from_float(float(v))*Fraction.from_float(float(t)) for v,t in zip(x,w[j],strict=True)), Fraction())
            lo, hi, near = half_bracket_fraction(exact); fallbacks += 1
        out[j] = near; valid[j] = lo <= actual[j] <= hi
    return out.reshape(actual.shape), valid.reshape(actual.shape), fallbacks

def f32_members(lo, hi, cap):
    values = []; x = F(lo)
    while x <= hi:
        values.append(x)
        require(len(values) <= cap, 'intrinsic discrete-set bound')
        x = np.nextafter(x, F(np.inf), dtype=F)
    require(values, 'empty intrinsic values')
    return np.asarray(values, dtype=F)

@lru_cache(maxsize=4096)
def gelu_words(word):
    x = from_bits(word); half_input(np.array([x]))
    cube = F(F(x*x)*x); z = F(F(0.7978845608028654)*F(x+F(F(.044715)*cube)))
    finite(np.array([cube,z]))
    lo, hi, near = tanh_contract(bits(z))
    t = f32_members(lo, hi, 32)
    outputs = half(F(F(F(.5)*x)*F(F(1)+t)))
    main = half(np.array([F(F(F(.5)*x)*F(F(1)+near))]))[0]
    return np.unique(outputs), main

def gelu(x, actual):
    x = half_input(x); actual = half_input(actual)
    require(x.shape == actual.shape, 'GELU shape')
    main = np.empty_like(x); valid = np.empty(x.shape, dtype=bool)
    for i, value in enumerate(x.flat):
        candidates, near = gelu_words(bits(value))
        main.flat[i] = near; valid.flat[i] = np.any(candidates == actual.flat[i])
    return main, valid

@lru_cache(maxsize=1024)
def rotary_pairs(theta, position, feature):
    pairs, c, s = rotary_contract(theta, position, feature, 256)
    co = []; si = []
    for cr, sr in pairs:
        cv = f32_members(*cr, 16); sv = f32_members(*sr, 16)
        co.append(np.repeat(cv, len(sv))); si.append(np.tile(sv, len(cv)))
    ca, sa = np.concatenate(co), np.concatenate(si)
    require(ca.size <= 4096, 'rotary discrete product cap')
    # <=32KiB per entry, <=32MiB entire1024-entry numeric cache.
    return ca, sa, c, s

def rotary(x, actual, position, theta):
    x = half_input(x); actual = half_input(actual)
    require(x.shape == actual.shape and x.shape[-1] == 256, 'rotary shape')
    a = x.reshape(-1,256); n = actual.reshape(-1,256)
    main = np.empty_like(a); valid = np.empty(a.shape, dtype=bool)
    for feature in range(128):
        ca, sa, c, s = rotary_pairs(theta, position, feature)
        av, bv = a[:,feature], a[:,feature+128]
        main[:,feature] = half(F(F(av*c)-F(bv*s)))
        main[:,feature+128] = half(F(F(bv*c)+F(av*s)))
        left = half(F(F(av[:,None]*ca)-F(bv[:,None]*sa)))
        right = half(F(F(bv[:,None]*ca)+F(av[:,None]*sa)))
        # The SAME candidate cos/sin pair must account for BOTH stored outputs.
        compatible = (left == n[:,feature,None]) & (right == n[:,feature+128,None])
        matched = np.any(compatible, axis=1)
        valid[:,feature] = matched; valid[:,feature+128] = matched
    return main.reshape(x.shape), valid.reshape(x.shape)

def softmax(scores, actual, query, first, window):
    scores = np.asarray(scores,dtype=F); actual = np.asarray(actual,dtype=F)
    finite(scores); finite(actual)
    require(np.all((actual>=0)&(actual<=1)), 'probability outside[0,1]')
    require(scores.shape == actual.shape and scores.ndim == 3 and scores.shape[:2] == (4,1) and 1 <= scores.shape[2] <= 544, 'softmax shape')
    require(type(query) is int and type(first) is int and 0 <= first <= query <= 543 and first+scores.shape[2] <= 544 and window in (None,512), 'softmax positions')
    keys = np.arange(first,first+scores.shape[2]); visible = (keys <= query)
    if window is not None: visible &= keys >= query-window+1
    indices = np.flatnonzero(visible); require(indices.size > 0, 'empty visible softmax')
    require(np.all(actual[:,:,~visible].view(np.uint32) == 0), 'masked probability must be positive zero')
    main = np.zeros_like(scores); valid = np.ones(scores.shape,dtype=bool)
    for head in range(4):
        row = scores[head,0]; mx = np.max(row[visible]); points = []; lower = []; upper = []
        for i in indices:
            lo, hi, near = exp_contract(F(row[i]-mx))
            require(0 <= lo <= near <= hi, 'positive exp enclosure')
            lower.append(lo); upper.append(hi); points.append(near)
        if len(indices) == 1:
            main[head,0,indices[0]] = 1
            valid[head,0,indices[0]] = actual[head,0,indices[0]] == 1
            continue
        ml = mh = mp = F(0)
        # Interval induction: every legal exp vector lies in this box; monotone
        # RNE addition contains its correlated serial mass. No independence is
        # assumed. Division uses a superset, not a claim of joint realizability.
        for lo,hi,p in zip(lower,upper,points,strict=True):
            ml=F(ml+lo); mh=F(mh+hi); mp=F(mp+p)
        require(np.isfinite(mh) and 0 < ml <= mp <= mh, 'softmax mass enclosure')
        for i,lo,hi,p in zip(indices,lower,upper,points,strict=True):
            low=F(lo/mh); high=F(hi/ml)
            main[head,0,i]=F(p/mp)
            valid[head,0,i]=low <= actual[head,0,i] <= high
    return main,valid

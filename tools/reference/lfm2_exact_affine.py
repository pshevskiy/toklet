"""Bounded independent original-Q4 affine oracle; no model/graph dependencies.

All destination bits are direct RNE of the exact original-F32 × stored-Half
sum. Certificates select an implementation, never a different numerical center.
The streaming caller authenticates its complete artifact/header before use.
"""
from dataclasses import dataclass
import sys
import numpy as np

GROUP = 64
MAX_ROWS = 128
MAX_COLUMNS = 4608
MAX_TOKENS = 64
F53 = 1 << 53
I63 = (1 << 63) - 1
U64 = (1 << 64) - 1
MAX_OFFSET = (1 << 63) - 1
if sys.byteorder != 'little':
    raise RuntimeError('This original-word oracle requires little-endian storage')
# Fixed byte lookup tables avoid logarithms and per-weight Python integers.
_TRAILING = np.array([(i & -i).bit_length()-1 if i else 8 for i in range(256)], dtype=np.int16)
_LENGTH = np.array([i.bit_length() for i in range(256)], dtype=np.int16)


def integer(value, name, minimum=0, maximum=MAX_OFFSET):
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(name)
    return value


def round_dyadic(value, exponent, destination):
    """Direct integer*2**exponent -> destination bits; no double rounding."""
    integer(exponent, 'exponent', -1024, 1024)
    if type(value) is not int or destination not in ('half', 'f32'):
        raise ValueError('dyadic destination')
    p, lo, hi, bias, signbit = ((11, -14, 15, 15, 15) if destination == 'half'
                              else (24, -126, 127, 127, 31))
    sign = (1 << signbit) if value < 0 else 0
    magnitude = abs(value)
    if magnitude == 0:
        return 0  # Exact cancellation/zero is canonical +0, before any cast.
    high = magnitude.bit_length()-1+exponent
    quantum = max(high, lo)-(p-1)
    shift = quantum-exponent
    if shift > 0:
        q, remainder = divmod(magnitude, 1 << shift)
        midpoint = 1 << (shift-1)
        q += remainder > midpoint or (remainder == midpoint and q & 1)
    else:
        q = magnitude << -shift
    if q == 0:
        return sign  # Negative nonzero underflow remains negative zero.
    actual_high = q.bit_length()-1+quantum
    if actual_high > hi:
        raise ValueError('destination overflow')
    if actual_high < lo:
        return sign | q
    adjustment = quantum-(actual_high-(p-1))
    significand = q << adjustment if adjustment >= 0 else q >> -adjustment
    return sign | ((actual_high+bias) << (p-1)) | (significand-(1 << (p-1)))


def components(values):
    """Vectorized exact mantissa, exponent and sign; no float -> integer cast."""
    if values.dtype == np.dtype('<f4'):
        bits = values.view(np.uint32)
        field = ((bits >> 23) & 255).astype(np.int16)
        if np.any(field == 255):
            raise ValueError('nonfinite FP32')
        mantissa = (bits & 0x7fffff).astype(np.uint64)
        mantissa |= np.where(field != 0, 1 << 23, 0).astype(np.uint64)
        exponent = np.where(field != 0, field-150, -149)
        negative = (bits >> 31) != 0
    elif values.dtype == np.dtype('<f2'):
        bits = values.view(np.uint16)
        field = ((bits >> 10) & 31).astype(np.int16)
        if np.any(field == 31):
            raise ValueError('nonfinite Half')
        mantissa = (bits & 1023).astype(np.uint64)
        mantissa |= np.where(field != 0, 1 << 10, 0).astype(np.uint64)
        exponent = np.where(field != 0, field-25, -24)
        negative = (bits >> 15) != 0
    else:
        raise ValueError('stored operand dtype')
    return mantissa, exponent.astype(np.int16), negative


def grid(values):
    """Return exact signed-int64 row/common exponent, or None if too wide."""
    if values.ndim != 1 or not 0 < values.size <= MAX_COLUMNS:
        raise ValueError('grid shape')
    m, e, negative = components(values)
    active = m != 0
    if not np.any(active):
        return np.zeros(values.size, dtype=np.int64), 0
    low, middle, high = m & 255, (m >> 8) & 255, (m >> 16) & 255
    trailing = np.where(low != 0, _TRAILING[low],
                        np.where(middle != 0, 8+_TRAILING[middle], 16+_TRAILING[high]))
    exponent = int(np.min((e+trailing)[active]))
    length = np.where(high != 0, 16+_LENGTH[high],
                      np.where(middle != 0, 8+_LENGTH[middle], _LENGTH[low]))
    shifts = e-exponent
    if np.any((length+shifts)[active] > 63):
        return None
    # Left shifts are performed only after a <=63 significant-bit proof.
    left = np.maximum(shifts, 0).astype(np.uint64)
    right = np.maximum(-shifts, 0).astype(np.uint64)
    magnitude = (m >> right) << left
    signed = magnitude.astype(np.int64)
    signed = np.where(negative, -signed, signed)
    return signed, exponent


def unsigned_sum(values, limit=I63):
    """Exact nonnegative sum or None if above limit; no wrapped reduction."""
    if values.dtype != np.uint64 or values.ndim != 1:
        raise ValueError('unsigned bound dtype/shape')
    integer(limit, 'bound limit', 0, U64)
    maximum = int(np.max(values, initial=0))
    if maximum == 0:
        return 0
    size = max(1, U64 // maximum)
    total = 0
    for start in range(0, values.size, size):
        total += int(np.sum(values[start:start+size], dtype=np.uint64))
        if total > limit:
            return None
    return total


def product_bound(w, x):
    """Actual abs-product sum, checked before signed multiplication/dot."""
    if w.dtype != np.int64 or x.dtype != np.int64 or w.shape != x.shape or w.ndim != 1:
        raise ValueError('product-bound shape/dtype')
    if np.any(w == -(1 << 63)) or np.any(x == -(1 << 63)):
        raise ValueError('unsupported abs-int64 minimum')
    aw, ax = np.abs(w).astype(np.uint64), np.abs(x).astype(np.uint64)
    active = (aw != 0) & (ax != 0)
    aw, ax = aw[active], ax[active]
    if not aw.size:
        return 0
    if np.any(aw > np.uint64(I63) // ax):
        return None
    return unsigned_sum(aw*ax)


def certified_integer_dot(w, x, bound, stats):
    """Only internal callers may supply a freshly checked actual bound."""
    if bound is None:
        raise ValueError('missing certificate')
    integer(bound, 'certificate', 0, I63)
    # Revalidate at this entry point so a corrupted/stale bound cannot authorize
    # arithmetic. The striped fast path separately validates its conservative bound.
    actual = product_bound(w, x)
    if actual is None or actual != bound:
        raise ValueError('wrong certificate')
    if bound == 0:
        stats['zero'] += 1
        return 0
    if bound <= F53:
        active = (w != 0) & (x != 0)
        # Nonzero operands <=abs(product)<=bound, hence casts are exact.
        value = np.dot(w[active].astype(np.float64), x[active].astype(np.float64))
        if not np.isfinite(value) or value != np.floor(value) or abs(value) > bound:
            raise ValueError('binary64 backend violated certificate')
        stats['fp64'] += 1
        return int(value)
    stats['int64'] += 1
    return int(np.dot(w, x))  # Each product and every partial sum bounded by I63.


def scalar_block(weights, inputs, stats):
    """Rare <=64-element exact fallback, never a full-model integer owner."""
    if not 0 < weights.size <= GROUP or inputs.shape != weights.shape:
        raise ValueError('scalar fallback block')
    wm, we, wn = components(weights)
    xm, xe, xn = components(inputs)
    total = 0
    for i in range(weights.size):
        product = int(wm[i])*int(xm[i])
        if product:
            if bool(wn[i]) != bool(xn[i]):
                product = -product
            total += product << (int(we[i])+int(xe[i])+173)
    stats['scalar_blocks'] += 1
    return total, -173


def exact_pair(weights, inputs, stats):
    wg, xg = grid(weights), grid(inputs)
    if wg is not None and xg is not None:
        bound = product_bound(wg[0], xg[0])
        if bound is not None:
            return certified_integer_dot(wg[0], xg[0], bound, stats), wg[1]+xg[1]
    stats['blocked_outputs'] += 1
    total = 0
    for start in range(0, weights.size, GROUP):
        w, x = weights[start:start+GROUP], inputs[start:start+GROUP]
        a, b = grid(w), grid(x)
        bound = product_bound(a[0], b[0]) if a is not None and b is not None else None
        if bound is None:
            subtotal, exponent = scalar_block(w, x, stats)
        else:
            subtotal = certified_integer_dot(a[0], b[0], bound, stats)
            exponent = a[1]+b[1]
            stats['certified_blocks'] += 1
        total += subtotal << (exponent+173)
    return total, -173


def apply(weights, inputs, destination='half'):
    """One <=128-row stripe, <=64 tokens, exact output bits and path counters.

    Returned arrays own their storage. Caller discards the result on any error.
    No partial result is returned; inputs and original weights stay unchanged.
    """
    if not isinstance(weights, np.ndarray) or not isinstance(inputs, np.ndarray):
        raise ValueError('array operands')
    if weights.dtype != np.dtype('<f4') or inputs.dtype != np.dtype('<f2'):
        raise ValueError('exact stored operand dtype')
    if weights.ndim != 2 or inputs.ndim != 2 or not weights.flags.c_contiguous or not inputs.flags.c_contiguous:
        raise ValueError('contiguous rank2 operands')
    rows, columns = weights.shape
    if not 1 <= rows <= MAX_ROWS or not 1 <= inputs.shape[0] <= MAX_TOKENS or inputs.shape[1] != columns:
        raise ValueError('stripe/token shape')
    if not 0 < columns <= MAX_COLUMNS or columns % GROUP or destination not in ('half', 'f32'):
        raise ValueError('column/destination profile')
    if not np.isfinite(weights).all() or not np.isfinite(inputs).all():
        raise ValueError('nonfinite operand')
    stats = dict(fp64=0, int64=0, zero=0, blocked_outputs=0, certified_blocks=0,
                 scalar_blocks=0, stripe_fp32_bytes=weights.nbytes,
                 maximum_fast_dot_rows=0, outputs=rows*inputs.shape[0])
    wi = np.zeros(weights.shape, dtype=np.int64)
    exponents, l1 = [], []
    for row in weights:
        factored = grid(row)
        index = len(exponents)
        if factored is None:
            exponents.append(None);l1.append(None)
        else:
            wi[index] = factored[0];exponents.append(factored[1])
            l1.append(unsigned_sum(np.abs(factored[0]).astype(np.uint64)))
    output = np.empty((inputs.shape[0], rows), dtype=np.uint16 if destination == 'half' else np.uint32)
    for token, source in enumerate(inputs):
        xi, xe = grid(source)  # Every finite Half row fits a signed-int64 grid.
        maximum = int(np.max(np.abs(xi), initial=0))
        eligible = [i for i in range(rows) if l1[i] is not None and
                    (maximum == 0 or l1[i] <= F53 // maximum)]
        eligible_set = set(eligible)
        if eligible:
            # A bound is checked BEFORE either conversion or matrix arithmetic.
            # Zero columns are removed: irrelevant huge weights cannot spoil casts.
            active = xi != 0
            if not np.any(active):
                for row in eligible:
                    output[token, row] = 0;stats['zero'] += 1
            else:
                selected = wi[eligible][:, active]
                values = selected.astype(np.float64) @ xi[active].astype(np.float64)
                stats['maximum_fast_dot_rows'] = max(stats['maximum_fast_dot_rows'], len(eligible))
                for row, value in zip(eligible, values, strict=True):
                    bound = l1[row]*maximum
                    if not np.isfinite(value) or value != np.floor(value) or abs(value) > bound:
                        raise ValueError('binary64 backend violated stripe certificate')
                    output[token, row] = round_dyadic(int(value), exponents[row]+xe, destination)
                    stats['fp64'] += 1
        for row in range(rows):
            if row not in eligible_set:
                exact, exponent = exact_pair(weights[row], source, stats)
                output[token, row] = round_dyadic(exact, exponent, destination)
    return output, stats


@dataclass(frozen=True)
class Layout:
    rows: int
    columns: int
    code_offset: int
    scale_offset: int
    bias_offset: int
    file_bytes: int

    def validate(self):
        integer(self.rows, 'matrix rows', 1, 65536)
        integer(self.columns, 'matrix columns', GROUP, MAX_COLUMNS)
        if self.columns % GROUP:
            raise ValueError('group64 columns')
        integer(self.file_bytes, 'whole container bytes', 1)
        sizes = (self.rows*self.columns//2, self.rows*(self.columns//GROUP)*2,
                 self.rows*(self.columns//GROUP)*2)
        spans = []
        for offset, size in zip((self.code_offset, self.scale_offset, self.bias_offset), sizes, strict=True):
            integer(offset, 'absolute offset')
            if offset+size > self.file_bytes or offset+size > MAX_OFFSET:
                raise ValueError('tensor extent')
            spans.append((offset, offset+size))
        spans.sort()
        if any(a[1] > b[0] for a, b in zip(spans, spans[1:])):
            raise ValueError('tensor overlap')
        return self


def decode(codes, scales, biases, columns):
    integer(columns, 'columns', GROUP, MAX_COLUMNS)
    if columns % GROUP or not all(isinstance(a, np.ndarray)for a in (codes, scales, biases)):
        raise ValueError('packed arrays/groups')
    if codes.ndim != 2 or codes.dtype != np.uint8 or not 1 <= codes.shape[0] <= MAX_ROWS:
        raise ValueError('packed codes')
    rows = codes.shape[0]
    if codes.shape != (rows, columns//2) or any(a.dtype != np.dtype('<u2') or a.shape != (rows, columns//GROUP) for a in (scales, biases)):
        raise ValueError('coefficient dtype/shape')
    gain = (scales.astype(np.uint32) << 16).view(np.float32)
    bias = (biases.astype(np.uint32) << 16).view(np.float32)
    if not np.isfinite(gain).all() or not np.isfinite(bias).all():
        raise ValueError('nonfinite original coefficient')
    q = np.empty((rows, columns), dtype=np.uint8)
    q[:, 0::2] = codes & 15;q[:, 1::2] = codes >> 4
    with np.errstate(over='ignore', invalid='ignore'):
        product = np.multiply(q, np.repeat(gain, GROUP, axis=1), dtype=np.float32)
        if not np.isfinite(product).all():
            raise ValueError('unsupported nonfinite nibble product')
        # Finite BF16×integer[0,15] fits FP32 exactly, hence separately adding
        # the original bias agrees with original FMA reconstruction in this domain.
        result = np.add(product, np.repeat(bias, GROUP, axis=1), dtype=np.float32)
    if not np.isfinite(result).all():
        raise ValueError('nonfinite reconstructed weight')
    return result


def read_stripes(stream, layout, start=0, stop=None, stripe_rows=MAX_ROWS):
    """Stream owned original-F32 stripes; no mapping-backed views survive yield.

    Layout offsets are absolute file offsets, including header/alignment. The
    caller counts the WHOLE mapped container if using mmap, not payload alone.
    This helper does not authenticate a file or infer descriptors from names.
    """
    layout.validate()
    stop = layout.rows if stop is None else stop
    integer(start, 'start', 0, layout.rows)
    integer(stop, 'stop', 0, layout.rows)
    integer(stripe_rows, 'stripe rows', 1, MAX_ROWS)
    if start >= stop:
        raise ValueError('nonempty row range')
    for row in range(start, stop, stripe_rows):
        count = min(stripe_rows, stop-row)
        arrays = []
        for offset, row_bytes, dtype in ((layout.code_offset, layout.columns//2, np.uint8),
                                         (layout.scale_offset, layout.columns//GROUP*2, np.dtype('<u2')),
                                         (layout.bias_offset, layout.columns//GROUP*2, np.dtype('<u2'))):
            stream.seek(offset+row*row_bytes)
            raw = stream.read(count*row_bytes)
            if type(raw) is not bytes or len(raw) != count*row_bytes:
                raise ValueError('truncated original tensor range')
            arrays.append(np.frombuffer(raw, dtype=dtype).reshape(count, -1))
        result = decode(*arrays, layout.columns)
        del arrays, raw
        yield row, result

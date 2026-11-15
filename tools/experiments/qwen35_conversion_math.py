"""Bounded canonical MLX affine4/group64 correspondence arithmetic.

Source: MLX v0.31.2, commit68cf2fddd8de5edd8ab3d926391772b2e2cedad8;
see affine-oracle-design-2.json and its pinned primary-source notes. These are
two numerical witnesses, not an assertion about a historical converter/backend.
Float arithmetic uses NumPy float32 stages under the canonical RNE environment.
No input is silently converted to the required dtype, and no model is loaded.
"""

import numpy as np


MATH_VERSION = 'toklet-qwen35-conversion-math-v1'
PROFILES = ('cpu_rint', 'metal_round')
MAX_GROUPS = 256
MAX_ELEMENTS = 65536
GROUP_SIZE = 64


def _array(value, dtype):
    # Reject subclasses with overridden ufunc/allocation behavior as well as
    # implicit array/dtype conversions. Inspect the bound before any temporary.
    if type(value) is not np.ndarray or value.dtype != np.dtype(dtype):
        raise ValueError('An ndarray with the exact native dtype is required')
    if value.size > MAX_ELEMENTS:
        raise ValueError('Scalar conversion exceeds65536 elements')
    return value


def _finite(value, stage):
    if not bool(np.all(np.isfinite(value))):
        raise ValueError('Nonfinite ' + stage)


def bf16_to_f32(words):
    """Expand uint16 BF16 bit patterns exactly, including signed zero/NaNs.

    Shape, including scalar/empty/noncontiguous inputs, is preserved. The output
    owns a new uint32 backing buffer; neither input values nor bits are changed.
    """
    words = _array(words, np.uint16)
    expanded = np.empty(words.shape, dtype=np.uint32)
    np.left_shift(words, np.uint32(16), out=expanded, dtype=np.uint32)
    return expanded.view(np.float32)


def f32_to_bf16(values):
    """Software MLX RNE store, with every NaN canonicalized to0x7fc0.

    Finite rounding is integer bit arithmetic. Infinities and signed zeros are
    preserved. This NaN policy belongs to the software source implementation;
    it makes no claim about a Metal compiler's NaN payload convention.
    """
    values = _array(values, np.float32)
    bits = values.view(np.uint32)
    rounded = np.empty(values.shape, dtype=np.uint32)
    retained_lsb = np.bitwise_and(np.right_shift(bits, np.uint32(16)), np.uint32(1))
    np.add(bits, np.uint32(0x7FFF), out=rounded, dtype=np.uint32)
    np.add(rounded, retained_lsb, out=rounded, dtype=np.uint32)
    np.right_shift(rounded, np.uint32(16), out=rounded)
    result = rounded.astype(np.uint16)
    # Inspect bits instead of floating arithmetic, including signaling NaNs.
    is_nan = np.bitwise_and(bits, np.uint32(0x7FFFFFFF)) > np.uint32(0x7F800000)
    np.copyto(result, np.uint16(0x7FC0), where=is_nan)
    return result


def add_one_bf16(words):
    """Typed BF16+BF16(1.0), through FP32 addition and BF16 RNE storage.

    Nonfinite words follow the software arithmetic/store policy: NaN becomes
    canonical NaN, and infinities retain their sign. Actual tensor admission
    must reject nonfinite sources separately, as quantize_groups does.
    """
    words = _array(words, np.uint16)
    values = bf16_to_f32(words)
    with np.errstate(over='ignore', under='ignore', invalid='ignore'):
        np.add(values, np.float32(1.0), out=values, dtype=np.float32)
    return f32_to_bf16(values)


def _round_f32(values, profile):
    if profile == 'cpu_rint':
        return np.rint(values, dtype=np.float32)
    # Metal round is halfway-away, unlike its ordinary Round functor's rint.
    # floor(abs(x)+.5) is wrong for float32 just below a half: that addition can
    # round up first. Split the exact integral/fractional parts instead. Values
    # above the float32 integral-precision boundary already have zero fraction.
    magnitude = np.abs(values)
    integral = np.floor(magnitude, dtype=np.float32)
    fraction = np.subtract(magnitude, integral, dtype=np.float32)
    increment = fraction >= np.float32(0.5)
    rounded = np.add(integral, increment, dtype=np.float32)
    return np.copysign(rounded, values, dtype=np.float32)


def quantize_groups(words, profile):
    """Quantize1..256 BF16 groups: (uint32[N,8],uint16[N],uint16[N]).

    All groups are exactly64 contiguous logical input coefficients. Resource
    checks precede every temporary. Source and intermediate/metadata values
    must be finite; source scale/bias stay FP32 until AFTER code generation.
    Explicit arrays are bounded well below16MiB; process/runtime memory is a
    separate admission. Inputs may be strided and are never mutated.
    """
    words = _array(words, np.uint16)
    if words.ndim != 2 or words.shape[1] != GROUP_SIZE or not 1 <= words.shape[0] <= MAX_GROUPS:
        raise ValueError('Quantization requires uint16[N,64] with1<=N<=256')
    if not isinstance(profile, str) or profile not in PROFILES:
        raise ValueError('Profile must be cpu_rint or metal_round')

    weights = bf16_to_f32(words)
    _finite(weights, 'source coefficient')
    minimum = np.min(weights, axis=1)
    maximum = np.max(weights, axis=1)
    if profile == 'metal_round':
        # The pinned kernel initializes w_max to zero before its SIMD maximum.
        np.maximum(maximum, np.float32(0.0), out=maximum)
    side = np.abs(minimum) > np.abs(maximum)
    edge = np.where(side, minimum, maximum)

    # Nonfinite intermediate results are rejected explicitly, rather than
    # inheriting a NumPy warning policy or selecting another recipe silently.
    with np.errstate(over='ignore', under='ignore', divide='ignore', invalid='ignore'):
        scale = np.subtract(maximum, minimum, dtype=np.float32)
        _finite(scale, 'group range')
        np.divide(scale, np.float32(15.0), out=scale, dtype=np.float32)
        np.maximum(scale, np.float32(1e-7), out=scale)
        scale = np.where(side, scale, np.negative(scale))
        q0_input = np.divide(edge, scale, dtype=np.float32)
        _finite(q0_input, 'endpoint quotient')
        q0 = _round_f32(q0_input, profile)
        nonzero = q0 != np.float32(0.0)
        # where prevents evaluating division by zero for the q0==0 branch.
        np.divide(edge, q0, out=scale, where=nonzero, dtype=np.float32)
        bias = np.zeros(scale.shape, dtype=np.float32)
        np.copyto(bias, edge, where=nonzero)
        _finite(scale, 'adjusted scale')
        _finite(bias, 'bias')
        if bool(np.any(scale == np.float32(0.0))):
            raise ValueError('Adjusted scale must be nonzero')
        difference = np.subtract(weights, bias[:, None], dtype=np.float32)
        _finite(difference, 'coefficient difference')
        quotient = np.divide(difference, scale[:, None], dtype=np.float32)
        _finite(quotient, 'coefficient quotient')
        codes = _round_f32(quotient, profile)
        if profile == 'cpu_rint':
            np.maximum(codes, np.float32(0.0), out=codes)
        # The Metal source only applies this upper clamp. A negative code is an
        # invalid canonical result, not permission to insert a CPU lower clamp.
        np.minimum(codes, np.float32(15.0), out=codes)
    _finite(codes, 'rounded code')
    if bool(np.any(codes < np.float32(0.0))):
        raise ValueError('Canonical code is outside unsigned4bit range')

    integer_codes = codes.astype(np.uint32).reshape(words.shape[0], 8, 8)
    shifts = np.arange(0, 32, 4, dtype=np.uint32)
    packed = np.bitwise_or.reduce(np.left_shift(integer_codes, shifts), axis=2, dtype=np.uint32)

    # Quantization above deliberately used unrounded FP32 scale/bias. Stored
    # BF16 metadata is produced only now and checked independently for overflow.
    scales = f32_to_bf16(scale)
    biases = f32_to_bf16(bias)
    _finite(bf16_to_f32(scales), 'stored BF16 scale')
    _finite(bf16_to_f32(biases), 'stored BF16 bias')
    return packed, scales, biases

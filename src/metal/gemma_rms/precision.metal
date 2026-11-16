#include <metal_stdlib>
using namespace metal;
#pragma clang fp contract(off)

// Artifact experiment only. Status invalid-input1, NEEDS_REFERENCE2.
// A reference request discards the ENTIRE dependent transaction. No publication
// of these finite diagnostic placeholders is permitted. Outside the admitted
// arithmetic domain this experiment makes no GPU numerical claim.
struct Pair { float hi; float lo; };
int exponent_of(uint magnitude) { return int(magnitude >> 23) - 127; }
// Exact integer preflight. It proves zero or magnitude>=MIN_NORMAL;
// same-sign overflow is a separate unchanged finite/result check.
bool addition_has_no_ftz(uint a, uint b) {
    uint ma = a & 0x7fffffffu, mb = b & 0x7fffffffu;
    // Reject nonfinite and nonzero subnormal helper operands BEFORE zeros.
    if (ma >= 0x7f800000u || mb >= 0x7f800000u ||
        (ma != 0u && ma < 0x00800000u) ||
        (mb != 0u && mb < 0x00800000u)) return false;
    if (ma == 0u || mb == 0u) return true;
    if (((a ^ b) & 0x80000000u) == 0u) return true;
    // Sort by exact positive magnitude. Gap0/1 alignment fits25bits.
    if (ma < mb) { uint swap = ma; ma = mb; mb = swap; }
    int ea = exponent_of(ma), eb = exponent_of(mb);
    uint gap = uint(ea - eb);
    if (gap >= 2u) return true;
    uint sa = (ma & 0x007fffffu) | 0x00800000u;
    uint sb = (mb & 0x007fffffu) | 0x00800000u;
    uint difference = (sa << gap) - sb;
    if (difference == 0u) return true;
    int highest = int(31u - clz(difference));
    return highest + eb - 23 >= -126;
}
float checked_add(float a, float b, thread uint& uncertain) {
    // No floating result is consulted to establish cancellation safety.
    uncertain |= uint(!addition_has_no_ftz(as_type<uint>(a), as_type<uint>(b))) * 16u;
    return a + b;
}
float checked_sub(float a, float b, thread uint& uncertain) {
    return checked_add(a, -b, uncertain);
}
float checked_multiply(float a, float b, thread uint& uncertain) {
    uint ab = as_type<uint>(a) & 0x7fffffffu;
    uint bb = as_type<uint>(b) & 0x7fffffffu;
    if (ab != 0u && bb != 0u) {
        int exponent = exponent_of(ab) + exponent_of(bb);
        uncertain |= uint(ab < 0x00800000u || bb < 0x00800000u ||
                          ab >= 0x7f800000u || bb >= 0x7f800000u ||
                          exponent < -125 || exponent > 120) * 32u;
    }
    return a * b;
}
float checked_divide(float a, float b, thread uint& uncertain) {
    uint ab = as_type<uint>(a) & 0x7fffffffu;
    uint bb = as_type<uint>(b) & 0x7fffffffu;
    uncertain |= uint(bb < 0x00800000u || bb >= 0x7f800000u || ab >= 0x7f800000u) * 64u;
    if (ab != 0u && bb >= 0x00800000u && bb < 0x7f800000u) {
        int exponent = exponent_of(ab) - exponent_of(bb);
        // The quotient mantissa can be half the exponent estimate.
        uncertain |= uint(ab < 0x00800000u || exponent < -124 || exponent > 120) * 64u;
    }
    return a / b;
}
Pair pair_sum(float a, float b, thread uint& uncertain) {
    float s = checked_add(a, b, uncertain);
    float v = checked_sub(s, a, uncertain);
    return {s, checked_add(checked_sub(a, checked_sub(s, v, uncertain), uncertain),
                           checked_sub(b, v, uncertain), uncertain)};
}
Pair pair_add(Pair a, Pair b, thread uint& uncertain) {
    Pair s = pair_sum(a.hi, b.hi, uncertain);
    Pair t = pair_sum(a.lo, b.lo, uncertain);
    Pair u = pair_sum(s.lo, t.hi, uncertain);
    Pair v = pair_sum(s.hi, u.hi, uncertain);
    return pair_sum(v.hi, checked_add(checked_add(v.lo, u.lo, uncertain), t.lo, uncertain), uncertain);
}
Pair pair_negate(Pair a) { return {-a.hi, -a.lo}; }
Pair pair_product(float a, float b, thread uint& uncertain) {
    uint ab = as_type<uint>(a) & 0x7fffffffu;
    uint bb = as_type<uint>(b) & 0x7fffffffu;
    if (ab != 0u && bb != 0u) {
        // Exact product grid is2^(ea+eb-46). With ea+eb>=-80,
        // the FMA residual is exactly zero or >=2^-126, before execution.
        uncertain |= uint(exponent_of(ab) + exponent_of(bb) < -80) * 128u;
    }
    float p = checked_multiply(a, b, uncertain);
    return {p, fma(a, b, -p)};
}
Pair pair_multiply(Pair a, Pair b, thread uint& uncertain) {
    Pair p = pair_product(a.hi, b.hi, uncertain);
    // Omitted lo*lo is the explicitly bounded second-order truncation of
    // two-float arithmetic, not a silently flushed attempted operation.
    Pair cross = pair_sum(checked_multiply(a.hi, b.lo, uncertain),
                          checked_multiply(a.lo, b.hi, uncertain), uncertain);
    return pair_add(p, cross, uncertain);
}
Pair pair_divide(Pair a, Pair b, thread uint& uncertain) {
    float first = checked_divide(a.hi, b.hi, uncertain);
    Pair result = {first, 0.0f};
    Pair remainder = pair_add(a, pair_negate(pair_multiply(b, result, uncertain)), uncertain);
    result = pair_add(result, {checked_divide(checked_add(remainder.hi, remainder.lo, uncertain), b.hi, uncertain), 0.0f}, uncertain);
    remainder = pair_add(a, pair_negate(pair_multiply(b, result, uncertain)), uncertain);
    return pair_add(result, {checked_divide(checked_add(remainder.hi, remainder.lo, uncertain), b.hi, uncertain), 0.0f}, uncertain);
}
Pair pair_root(Pair a, thread uint& uncertain) {
    Pair root = {sqrt(a.hi), 0.0f};
    Pair remainder = pair_add(a, pair_negate(pair_multiply(root, root, uncertain)), uncertain);
    root = pair_add(root, pair_divide(remainder, pair_add(root, root, uncertain), uncertain), uncertain);
    remainder = pair_add(a, pair_negate(pair_multiply(root, root, uncertain)), uncertain);
    return pair_add(root, pair_divide(remainder, pair_add(root, root, uncertain), uncertain), uncertain);
}
bool pair_valid(Pair a) {
    uint h = as_type<uint>(a.hi) & 0x7fffffffu;
    uint l = as_type<uint>(a.lo) & 0x7fffffffu;
    return isfinite(a.hi) && isfinite(a.lo) &&
           (h == 0u || h >= 0x00800000u) &&
           (l == 0u || l >= 0x00800000u);
}
float normal_mantissa(uint magnitude) {
    return as_type<float>(0x3f800000u | (magnitude & 0x007fffffu));
}

template<bool Learned>
void precision_row(device const uint* input, device const uint* weights,
                   device uint* output, constant uint* shape,
                   device atomic_uint* status, device uint* row_reasons, uint row, uint lane, uint lanes) {
    if (row >= shape[0]) return;
    uint width = shape[1], start = row * width;
    uint invalid = 0u, uncertain = 0u, maximum = 0u;
    for (uint col = lane; col < width; col += lanes) {
        uint x = input[start + col] & 0x7fffffffu;
        uint w = Learned ? (weights[col] & 0x7fffffffu) : 0x3f800000u;
        invalid |= uint(x >= 0x7f800000u || w >= 0x7f800000u);
        uncertain |= uint((x != 0u && x < 0x00800000u) ||
                          (w != 0u && w < 0x00800000u));
        maximum = max(maximum, x);
    }
    invalid = simd_or(invalid);
    uncertain = simd_or(uncertain);
    maximum = simd_max(maximum);
    // Existing host ABI retains once-rounded sqrt(epsilon) as word2. Arithmetic
    // uses the ORIGINAL epsilon bits below, never squares that rounded root.
    uncertain |= uint(shape[2] != 0x3a83126fu) * 2u;
    int scale_exponent = maximum == 0u ? -10 : max(-10, exponent_of(maximum));
    uncertain |= uint(scale_exponent > 40) * 4u;
    // Nonzero scaled input >=2^-25 keeps square/FMA residuals away from FTZ.
    // Extreme ratios/exponents use exact scalar replay rather than truncating a
    // tiny contribution which a learned scale could later amplify.
    for (uint col = lane; col < width; col += lanes) {
        uint x = input[start + col] & 0x7fffffffu;
        if (x != 0u && x < 0x7f800000u)
            uncertain |= uint(exponent_of(x) - scale_exponent < -25) * 8u;
    }
    uncertain = simd_or(uncertain);
    if (invalid != 0u || uncertain != 0u) {
        if (lane == 0u) {
            row_reasons[row] = uncertain | (invalid ? 0x80000000u : 0u);
            atomic_fetch_or_explicit(status, invalid ? 1u : 2u, memory_order_relaxed);
        }
        for (uint col = lane; col < width; col += lanes)
            output[start + col] = input[start + col] & 0x80000000u;
        return;
    }
    Pair sum = {0.0f, 0.0f};
    for (uint col = lane; col < width; col += lanes) {
        float q = ldexp(as_type<float>(input[start + col]), -scale_exponent);
        Pair square = pair_product(q, q, uncertain);
        sum = pair_add(sum, square, uncertain);
        uncertain |= uint(!pair_valid(square) || !pair_valid(sum)) * 256u;
    }
    for (uint hop = 1u; hop < lanes; hop *= 2u) {
        Pair other = {simd_shuffle_down(sum.hi, hop), simd_shuffle_down(sum.lo, hop)};
        if (lane % (2u * hop) == 0u && lane + hop < lanes)
            sum = pair_add(sum, other, uncertain);
    }
    Pair total = {simd_broadcast(sum.hi, 0), simd_broadcast(sum.lo, 0)};
    Pair mean = pair_divide(total, {float(width), 0.0f}, uncertain);
    // Scaling by an integer power of two is exact throughout this admitted
    // exponent range. 0x358637bd is exactly 1e-6_f32, widened by the scalar CPU.
    float epsilon = ldexp(as_type<float>(0x358637bdu), -2 * scale_exponent);
    Pair variance = pair_add(mean, {epsilon, 0.0f}, uncertain);
    Pair denominator = pair_root(variance, uncertain);
    uncertain |= uint(!pair_valid(total) || !pair_valid(mean) ||
                      !pair_valid(variance) || !pair_valid(denominator) ||
                      denominator.hi <= 0.0f || !isfinite(epsilon) || epsilon <= 0.0f) * 512u;
    uncertain = simd_or(uncertain);
    // Output is staged privately. A later row/output reference request still
    // discards the entire transaction; it cannot publish earlier computed cells.
    for (uint col = lane; col < width; col += lanes) {
        uint xb = input[start + col], x = xb & 0x7fffffffu;
        uint wb = Learned ? weights[col] : 0x3f800000u;
        uint w = wb & 0x7fffffffu, sign = (xb ^ wb) & 0x80000000u;
        if (x == 0u || w == 0u) { output[start + col] = sign; continue; }
        float q = ldexp(as_type<float>(x), -scale_exponent);
        Pair normalized = pair_divide({q, 0.0f}, denominator, uncertain);
        Pair scaled = pair_multiply(normalized, {normal_mantissa(w), 0.0f}, uncertain);
        float rounded_mantissa = checked_add(scaled.hi, scaled.lo, uncertain);
        uint rounded_bits = as_type<uint>(rounded_mantissa) & 0x7fffffffu;
        int final_exponent = exponent_of(rounded_bits) + exponent_of(w);
        uncertain |= uint(rounded_bits < 0x00800000u || rounded_bits >= 0x7f800000u ||
                          final_exponent < -125 || final_exponent > 126) * 1024u;
        float value = ldexp(rounded_mantissa, exponent_of(w));
        uint magnitude = as_type<uint>(value) & 0x7fffffffu;
        uint local_uncertain = uint(!pair_valid(normalized) || !pair_valid(scaled) ||
                                    !isfinite(value) || magnitude < 0x00800000u);
        uncertain |= local_uncertain * 2048u;
        output[start + col] = (local_uncertain ? 0u : magnitude) | sign;
    }
    uncertain = simd_or(uncertain);
    if (lane == 0u) {
        row_reasons[row] = uncertain;
        if (uncertain != 0u) atomic_fetch_or_explicit(status, 2u, memory_order_relaxed);
    }
}
#define PRECISION_KERNEL(NAME, LEARNED) \
kernel void NAME(device const uint* input [[buffer(0)]], \
                 device const uint* weights [[buffer(1)]], \
                 device uint* output [[buffer(2)]], \
                 constant uint* shape [[buffer(3)]], \
                 device atomic_uint* status [[buffer(4)]], \
                 device uint* row_reasons [[buffer(5)]], \
                 uint row [[threadgroup_position_in_grid]], \
                 uint lane [[thread_index_in_simdgroup]], \
                 uint lanes [[threads_per_simdgroup]]) { \
    precision_row<LEARNED>(input, weights, output, shape, status, row_reasons, row, lane, lanes); \
}
PRECISION_KERNEL(gemma_rms_precision_unlearned, false)
PRECISION_KERNEL(gemma_rms_precision_learned, true)


// Bounded target qualification of the exact predicate itself. Rejected cases
// retain a sentinel; accepted overflow can be Inf because this predicate is
// solely a no-FTZ proof, not the unchanged RMS output-finiteness check.
kernel void precision_add_guard_controls(device const uint* pairs [[buffer(0)]],
                                         device uint* output [[buffer(1)]],
                                         constant uint& count [[buffer(2)]],
                                         uint item [[thread_position_in_grid]]) {
    if (item >= count) return;
    uint a = pairs[item * 2], b = pairs[item * 2 + 1];
    bool allowed = addition_has_no_ftz(a, b);
    output[item * 2] = uint(allowed);
    output[item * 2 + 1] = allowed ? as_type<uint>(as_type<float>(a) + as_type<float>(b)) : 0x5a170031u;
}

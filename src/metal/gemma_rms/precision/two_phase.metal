// Appended to the exact qualified guard2 source; scheduling variant only.
constant uint RMS_CTX_VERSION = 0x52500000u;
constant uint RMS_PENDING = 4u, RMS_STRUCTURAL = 8u;
uint rms_context_check(uint hi, uint lo, uint scale, uint reason, uint tagged_mode,
                       uint geometry, uint row, uint op_tag) {
    return hi ^ lo ^ scale ^ reason ^ tagged_mode ^ geometry ^ row ^ op_tag ^ 0x9e3779b9u;
}
bool rms_operation_header(constant uint* shape, device const uint* header,
                          uint lanes, bool first_phase) {
    if (shape[0] < 1u || shape[0] > 256u || shape[1] < 1u || shape[1] > 1536u ||
        lanes == 0u || lanes > 65535u || shape[3] == 0u || shape[3] > 0x3fffffffu)
        return false;
    uint groups = (shape[1] + lanes - 1u) / lanes;
    if (header[0] != shape[3] || header[1] != shape[4] || header[2] != shape[0] * groups)
        return false;
    // All A work precedes B; A never modifies this caller-initialized counter.
    return !first_phase || header[3] == 0u;
}
void rms_write_context(device uint* context, uint row, constant uint* shape,
                       Pair denominator, int scale, uint reason, uint mode) {
    uint hi = as_type<uint>(denominator.hi), lo = as_type<uint>(denominator.lo);
    uint exponent = as_type<uint>(scale), tag = (shape[3] << 2u) | mode;
    uint geometry = RMS_CTX_VERSION | shape[0] | (shape[1] << 9u), at = row * 8u;
    context[at] = hi; context[at + 1u] = lo; context[at + 2u] = exponent;
    context[at + 3u] = reason; context[at + 4u] = tag; context[at + 5u] = geometry;
    context[at + 6u] = rms_context_check(hi, lo, exponent, reason, tag, geometry, row, shape[4]);
    context[at + 7u] = 0u;
}
template<bool Learned>
void precision_phase_a(device const uint* input, device const uint* weights,
                       device uint* context, constant uint* shape,
                       device atomic_uint* status, device uint* row_reasons,
                       device const uint* header, uint row, uint lane, uint lanes) {
    if (row >= shape[0]) return;
    if (lane == 0u) atomic_fetch_or_explicit(status, RMS_PENDING, memory_order_relaxed);
    if (!rms_operation_header(shape, header, lanes, true)) {
        if (lane == 0u) atomic_fetch_or_explicit(status, RMS_STRUCTURAL, memory_order_relaxed);
        return;
    }
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
            uint reason = uncertain | (invalid ? 0x80000000u : 0u);
            row_reasons[row] = reason;
            atomic_fetch_or_explicit(status, invalid ? 1u : 2u, memory_order_relaxed);
            rms_write_context(context, row, shape, {0.0f, 0.0f}, scale_exponent,
                              reason, invalid ? 2u : 1u);
        }
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
    // Only uncertainty association changes: helpers never branch on its value.
    uncertain = simd_or(uncertain);
    if (lane == 0u) {
        Pair mean = pair_divide(total, {float(width), 0.0f}, uncertain);
        // Scaling by an integer power of two is exact throughout this admitted
        // exponent range. 0x358637bd is exactly 1e-6_f32, widened by the scalar CPU.
        float epsilon = ldexp(as_type<float>(0x358637bdu), -2 * scale_exponent);
        Pair variance = pair_add(mean, {epsilon, 0.0f}, uncertain);
        Pair denominator = pair_root(variance, uncertain);
        uncertain |= uint(!pair_valid(total) || !pair_valid(mean) ||
                          !pair_valid(variance) || !pair_valid(denominator) ||
                          denominator.hi <= 0.0f || !isfinite(epsilon) || epsilon <= 0.0f) * 512u;
        row_reasons[row] = uncertain;
        rms_write_context(context, row, shape, denominator, scale_exponent, uncertain, 0u);
    }
}

template<bool Learned>
void precision_phase_b(device const uint* input, device const uint* weights,
                       device uint* output, constant uint* shape,
                       device atomic_uint* status, device atomic_uint* row_reasons,
                       device uint* context, device uint* header,
                       uint group_x, uint row, uint lane, uint lanes) {
    if (row >= shape[0]) return;
    bool valid = rms_operation_header(shape, header, lanes, false);
    uint width = shape[1], start = row * width;
    if (width > 1536u || lanes == 0u || group_x >= (width + lanes - 1u) / lanes) return;
    uint at = row * 8u, hi = context[at], lo = context[at + 1u];
    uint exponent = context[at + 2u], pre = context[at + 3u];
    uint tagged_mode = context[at + 4u], mode = tagged_mode & 3u;
    uint geometry = context[at + 5u];
    int scale_exponent = as_type<int>(exponent);
    valid = valid && (tagged_mode >> 2u) == shape[3] && mode != 3u &&
        geometry == (RMS_CTX_VERSION | shape[0] | (width << 9u)) &&
        context[at + 6u] == rms_context_check(hi, lo, exponent, pre, tagged_mode,
                                             geometry, row, shape[4]);
    if (mode == 0u) {
        valid = valid && scale_exponent >= -10 && scale_exponent <= 40 && (pre & ~0x3f0u) == 0u;
        if ((pre & 512u) == 0u) {
            uint hm = hi & 0x7fffffffu, lm = lo & 0x7fffffffu;
            valid = valid && (hi & 0x80000000u) == 0u && hm >= 0x00800000u && hm < 0x7f800000u &&
                (lm == 0u || (lm >= 0x00800000u && lm < 0x7f800000u));
        }
    } else if (mode == 1u) {
        valid = valid && hi == 0u && lo == 0u && pre != 0u && (pre & ~15u) == 0u &&
            scale_exponent >= -10 && scale_exponent <= 127;
    } else if (mode == 2u) {
        valid = valid && hi == 0u && lo == 0u && (pre & 0x80000000u) != 0u &&
            (pre & ~0x8000000fu) == 0u && scale_exponent >= -10 && scale_exponent <= 128;
    }
    uint first_col = group_x * lanes + lane;
    if (!valid) {
        if (first_col < width) output[start + first_col] = input[start + first_col] & 0x80000000u;
        if (lane == 0u) atomic_fetch_or_explicit(status, RMS_STRUCTURAL, memory_order_relaxed);
        return;
    }
    uint uncertain = first_col < width ? pre : 0u;
    Pair denominator = {as_type<float>(hi), as_type<float>(lo)};
    if (mode != 0u) {
        if (first_col < width) output[start + first_col] = input[start + first_col] & 0x80000000u;
    } else {
        // Zero or one iteration preserves the original output body/continue.
        for (uint col = first_col; col < min(width, first_col + 1u); ++col) {
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
    }
    uncertain = simd_or(uncertain);
    if (lane == 0u) {
        // Fatal early reasons must never manufacture reference bit2.
        if (mode == 0u && uncertain != 0u) {
            atomic_fetch_or_explicit(row_reasons + row, uncertain, memory_order_relaxed);
            atomic_fetch_or_explicit(status, 2u, memory_order_relaxed);
        }
        uint expected_row = (width + lanes - 1u) / lanes;
        device atomic_uint* row_count = reinterpret_cast<device atomic_uint*>(context + at + 7u);
        uint old_row = atomic_fetch_add_explicit(row_count, 1u, memory_order_relaxed);
        if (old_row >= expected_row) atomic_fetch_or_explicit(status, RMS_STRUCTURAL, memory_order_relaxed);
        device atomic_uint* total_count = reinterpret_cast<device atomic_uint*>(header + 3u);
        uint old_total = atomic_fetch_add_explicit(total_count, 1u, memory_order_relaxed);
        if (old_total >= header[2]) atomic_fetch_or_explicit(status, RMS_STRUCTURAL, memory_order_relaxed);
        if (old_total == header[2] - 1u)
            atomic_fetch_and_explicit(status, ~RMS_PENDING, memory_order_relaxed);
    }
}
#define PHASE_A(NAME, LEARNED) \
kernel void NAME(device const uint* input [[buffer(0)]], device const uint* weights [[buffer(1)]], \
    device uint* context [[buffer(2)]], constant uint* shape [[buffer(3)]], \
    device atomic_uint* status [[buffer(4)]], device uint* reasons [[buffer(5)]], \
    device const uint* header [[buffer(6)]], uint row [[threadgroup_position_in_grid]], \
    uint lane [[thread_index_in_simdgroup]], uint lanes [[threads_per_simdgroup]]) { \
    precision_phase_a<LEARNED>(input, weights, context, shape, status, reasons, header, row, lane, lanes); \
}
#define PHASE_B(NAME, LEARNED) \
kernel void NAME(device const uint* input [[buffer(0)]], device const uint* weights [[buffer(1)]], \
    device uint* output [[buffer(2)]], constant uint* shape [[buffer(3)]], \
    device atomic_uint* status [[buffer(4)]], device atomic_uint* reasons [[buffer(5)]], \
    device uint* context [[buffer(6)]], device uint* header [[buffer(7)]], \
    uint3 group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]], \
    uint lanes [[threads_per_simdgroup]]) { \
    precision_phase_b<LEARNED>(input, weights, output, shape, status, reasons, context, header, group.x, group.y, lane, lanes); \
}
PHASE_A(gemma_rms_phase_a_unlearned, false)
PHASE_A(gemma_rms_phase_a_learned, true)
PHASE_B(gemma_rms_phase_b_unlearned, false)
PHASE_B(gemma_rms_phase_b_learned, true)

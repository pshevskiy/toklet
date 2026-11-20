#include <metal_stdlib>
using namespace metal;

// Host inserts the shared integer header here.
__LFM3_DYADIC__

kernel void lfm3_admit(
    const device ushort* input [[buffer(0)]],
    const device ushort* weight [[buffer(1)]],
    device atomic_uint* status [[buffer(2)]],
    constant uint4& dims [[buffer(3)]],
    uint index [[thread_position_in_grid]]
) {
    const uint input_count = dims.x * dims.y;
    const uint weight_count = dims.y * dims.z;
    if ((index < input_count && (input[index] & 0x7fff) >= 0x7f80) ||
        (index < weight_count && (weight[index] & 0x7fff) >= 0x7f80)) {
        atomic_fetch_or_explicit(status, 2u, memory_order_relaxed);
    }
}

static float widen(ushort bits) { return as_type<float>(uint(bits) << 16); }

static Word ordered_fma(ushort a, ushort b, uint c, thread uint& counts) {
    const uint abs_c = c & positive_mask;
    const int ca = int((a >> 7) & 255);
    const int cb = int((b >> 7) & 255);
    const int cc = int(abs_c >> 23);
    const int product_exponent = ca + cb - 254;
    // Even at the first normal word of a binade, the smaller halfway gap
    // around c is at least 2^(cc-127-25). Each BF16 magnitude, including
    // subnormals with field zero, is strictly below 2^(field-126).
    if (cc != 0 && product_exponent + 2 <= cc - 152) {
        counts += 1u << 12;
        return {c, 0};
    }
    // Both exact terms are on the 2^-126 grid and cannot overflow.
    // Opposite signs can cancel exactly to +0 under nearest-even, or
    // leave at least a minimum normal. The device witness tests zero.
    if (ca > 0 && cb > 0 && product_exponent >= -112 && product_exponent <= 124 &&
        (abs_c == 0 || (cc >= 24 && cc <= 252))) {
        return {as_type<uint>(fma(widen(a), widen(b), as_type<float>(c))), 0};
    }
    counts += 1;
    return exact_fma(a, b, c);
}

static Word ordered_add(uint a, uint b, thread uint& counts) {
    const uint am = a & positive_mask, bm = b & positive_mask;
    if (am == 0 || bm == 0) {
        counts += 1u << 12;
        return {am != 0 ? a : bm != 0 ? b : a & b, 0};
    }
    const uint ea = am >> 23, eb = bm >> 23;
    if (ea >= 24 && ea <= 252 && eb >= 24 && eb <= 252 && ((a ^ b) & sign_bit) == 0) {
        return {as_type<uint>(as_type<float>(a) + as_type<float>(b)), 0};
    }
    counts += 1;
    return exact_add(a, b);
}

static uint take(Word word, thread uint& flags) {
    flags |= word.status;
    return word.status == 0 ? word.bits : 0;
}

kernel void lfm3_ordered(
    const device ushort* input [[buffer(0)]],
    const device ushort* weight [[buffer(1)]],
    device atomic_uint* status [[buffer(2)]],
    constant uint4& dims [[buffer(3)]],
    device ushort* output [[buffer(4)]],
    device uint* routes [[buffer(5)]],
    uint group [[threadgroup_position_in_grid]],
    ushort lane [[thread_index_in_simdgroup]]
) {
    // Admission is complete and immutable for the nonfinite bit. Overflow is
    // sticky, but never changes the control flow of another SIMD group.
    if ((atomic_load_explicit(status, memory_order_relaxed) & 2) != 0) return;
    const uint token = group / dims.z;
    const uint row = group % dims.z;
    uint partial = 0, flags = 0, counts = 0;
    for (uint k = lane; k < dims.y; k += 32) {
        partial = take(ordered_fma(input[token * dims.y + k], weight[row * dims.y + k],
                                   partial, counts), flags);
    }
    for (ushort offset = 16; offset >= 4; offset >>= 1) {
        const uint other = simd_shuffle_down(partial, offset);
        if (lane < offset) partial = take(ordered_add(partial, other, counts), flags);
    }
    const uint p0 = simd_shuffle(partial, ushort(0));
    const uint p1 = simd_shuffle(partial, ushort(1));
    const uint p2 = simd_shuffle(partial, ushort(2));
    const uint p3 = simd_shuffle(partial, ushort(3));
    uint bits = 0;
    if (lane == 0) {
        const uint left = take(ordered_add(p0, p1, counts), flags);
        const uint right = take(ordered_add(p2, p3, counts), flags);
        const uint sum = take(ordered_add(left, right, counts), flags);
        bits = take(ordered_add(sum, 0, counts), flags);
    }

    // Routing census is observational and this correctness witness is untimed
    // for promotion. 2080 total operations per output cannot carry past 12 bits.
    const uint receipt = simd_sum(counts);
    const bool failed = simd_any(flags != 0);
    if (lane == 0) {
        const ushort rounded = ushort((bits + 0x7fff + ((bits >> 16) & 1)) >> 16);
        if (failed || (rounded & 0x7fff) >= 0x7f80) {
            atomic_fetch_or_explicit(status, 4u, memory_order_relaxed);
        } else {
            output[group] = rounded;
            routes[group] = receipt;
        }
    }
}

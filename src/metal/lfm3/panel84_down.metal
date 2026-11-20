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
    const uint weight_count = dims.w;
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
        counts += 1u << 16;
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
        counts += 1u << 16;
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

static bool normal_pair(uint a, uint b) {
    if ((a & 0x30000u) != 0x10000u || (b & 0x30000u) != 0x10000u) return false;
    const int minimum = int(a & 255u) + int(b & 255u) - 254;
    const int maximum = int((a >> 8) & 255u) + int((b >> 8) & 255u) - 254;
    return minimum >= -112 && maximum <= 109;
}

static float plain_dot(const device ushort* a, const device ushort* w, uint k, ushort lane) {
    float partial = 0.0f;
    for (uint i = lane; i < k; i += 32) {
        partial = fma(widen(a[i]), widen(w[i]), partial);
    }
    for (ushort offset = 16; offset >= 4; offset >>= 1) {
        const float other = simd_shuffle_down(partial, offset);
        if (lane < offset) partial = partial + other;
    }
    const float p0 = simd_shuffle(partial, ushort(0));
    const float p1 = simd_shuffle(partial, ushort(1));
    const float p2 = simd_shuffle(partial, ushort(2));
    const float p3 = simd_shuffle(partial, ushort(3));
    return ((p0+p1)+(p2+p3)) + 0.0f;
}

// Per-output body retained from the qualified scalar SIMD kernel.
static void original_one(
    const device ushort* input, const device ushort* weight, device atomic_uint* status,
    constant uint4& dims, device ushort* output, device uint* routes,
    const device uint* input_bounds, const device uint* weight_bounds,
    uint group, uint token, uint row, ushort lane
) {
    if (normal_pair(input_bounds[token],weight_bounds[row])) {
        const uint bits=as_type<uint>(plain_dot(input+token*dims.y,weight+row*dims.y,dims.y,lane));
        if (lane==0) {
            const ushort rounded=ushort((bits+0x7fff+((bits>>16)&1))>>16);
            if ((rounded&0x7fff)>=0x7f80) atomic_fetch_or_explicit(status,4u,memory_order_relaxed);
            else {
                output[group]=rounded;
                routes[group]=0x80000000u;
            }
        }
        return;
    }
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
    // for promotion. 7712 total operations per output cannot carry past 16 bits.
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


static float4 four_dots(const device ushort* a, const device ushort* w, uint k, ushort lane) {
    float4 partial(0.0f);
    for (uint i = lane; i < k; i += 32) {
        const float weight = widen(w[i]);
        partial.x = fma(widen(a[i]), weight, partial.x);
        partial.y = fma(widen(a[k+i]), weight, partial.y);
        partial.z = fma(widen(a[2u*k+i]), weight, partial.z);
        partial.w = fma(widen(a[3u*k+i]), weight, partial.w);
    }
    for (ushort offset = 16; offset >= 4; offset >>= 1) {
        const float4 other(
            simd_shuffle_down(partial.x, offset), simd_shuffle_down(partial.y, offset),
            simd_shuffle_down(partial.z, offset), simd_shuffle_down(partial.w, offset));
        if (lane < offset) partial = partial + other;
    }
    float4 sums;
    for (uint t = 0; t < 4; ++t) {
        const float p0 = simd_shuffle(partial[t], ushort(0));
        const float p1 = simd_shuffle(partial[t], ushort(1));
        const float p2 = simd_shuffle(partial[t], ushort(2));
        const float p3 = simd_shuffle(partial[t], ushort(3));
        sums[t] = ((p0+p1)+(p2+p3)) + 0.0f;
    }
    return sums;
}

static void panel(const device ushort* input, const device ushort* weight,
                  device atomic_uint* status, uint4 dims, device ushort* output,
                  device uint* routes, uint base_token, uint base_row, ushort lane) {
    // Each component owns one original lane's accumulator for one token/output.
    float4 partial[8];
    #pragma unroll
    for (uint t = 0; t < 8; ++t) partial[t] = float4(0.0f);
    const uint k = dims.y;
    for (uint i = lane; i < k; i += 32) {
        const float4 w(widen(weight[(base_row+0u)*k+i]), widen(weight[(base_row+1u)*k+i]),
                       widen(weight[(base_row+2u)*k+i]), widen(weight[(base_row+3u)*k+i]));
        #pragma unroll
        for (uint t = 0; t < 8; ++t) {
            const float a = widen(input[(base_token+t)*k+i]);
            partial[t] = fma(float4(a), w, partial[t]);
        }
    }
    #pragma unroll
    for (uint t = 0; t < 8; ++t) {
        float4 p = partial[t];
        for (ushort offset = 16; offset >= 4; offset >>= 1) {
            const float4 other(simd_shuffle_down(p.x,offset), simd_shuffle_down(p.y,offset),
                               simd_shuffle_down(p.z,offset), simd_shuffle_down(p.w,offset));
            if (lane < offset) p = p + other;
        }
        #pragma unroll
        for (uint r = 0; r < 4; ++r) {
            const float p0 = simd_shuffle(p[r], ushort(0));
            const float p1 = simd_shuffle(p[r], ushort(1));
            const float p2 = simd_shuffle(p[r], ushort(2));
            const float p3 = simd_shuffle(p[r], ushort(3));
            const float sum = ((p0+p1)+(p2+p3))+0.0f;
            if (lane == 0) {
                const uint bits = as_type<uint>(sum);
                const ushort rounded = ushort((bits+0x7fffu+((bits>>16)&1u))>>16);
                const uint index = (base_token+t)*dims.z+base_row+r;
                if ((rounded & 0x7fffu) >= 0x7f80u) {
                    atomic_fetch_or_explicit(status,4u,memory_order_relaxed);
                } else {
                    output[index] = rounded;
                    routes[index] = 0xc0000000u;
                }
            }
        }
    }
}

kernel void lfm3_ordered(
    const device ushort* input [[buffer(0)]],
    const device ushort* weight [[buffer(1)]],
    device atomic_uint* status [[buffer(2)]],
    constant uint4& dims [[buffer(3)]],
    device ushort* output [[buffer(4)]],
    device uint* routes [[buffer(5)]],
    const device uint* input_bounds [[buffer(6)]],
    const device uint* weight_bounds [[buffer(7)]],
    uint group [[threadgroup_position_in_grid]],
    ushort lane [[thread_index_in_simdgroup]]
) {
    if ((atomic_load_explicit(status,memory_order_relaxed)&2u)!=0) return;
    if (dims.x!=128 || dims.y!=7680 || dims.z!=2048) {
        if (lane==0) atomic_fetch_or_explicit(status,4u,memory_order_relaxed);
        return;
    }
    const uint rows = dims.z/4u;
    const uint t0 = 8u*(group/rows);
    const uint r0 = 4u*(group%rows);
    bool native = true;
    for (uint t = 0; t < 8; ++t) {
        for (uint r = 0; r < 4; ++r) {
            native = normal_pair(input_bounds[t0+t],weight_bounds[r0+r]) && native;
        }
    }
    if (!native) {
        for (uint t = 0; t < 8; ++t) {
            for (uint r = 0; r < 4; ++r) {
                const uint token=t0+t, row=r0+r;
                original_one(input,weight,status,dims,output,routes,input_bounds,weight_bounds,
                             token*dims.z+row,token,row,lane);
            }
        }
    } else {
        panel(input,weight,status,dims,output,routes,t0,r0,lane);
    }
}

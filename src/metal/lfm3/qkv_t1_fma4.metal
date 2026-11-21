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

static void projection(
    const device ushort* input,
    const device ushort* weight,
    device atomic_uint* status,
    uint4 dims,
    device ushort* output,
    device uint* routes,
    const device uint* input_bounds,
    const device uint* weight_bounds,
    uint group,
    ushort lane
) {
    // Admission is complete and immutable for the nonfinite bit. Overflow is
    // sticky, but never changes the control flow of another SIMD group.
    if ((atomic_load_explicit(status, memory_order_relaxed) & 2) != 0) return;
    const uint token = group / dims.z;
    const uint row = group % dims.z;
    if (dims.y==2048 && normal_pair(input_bounds[token],weight_bounds[row])) {
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

static void fma4(
    const device ushort* input, const device ushort* weights, device atomic_uint* status,
    device ushort* output, device uint* routes, ushort lane
) {
    float4 partial(0.0f);
    for (uint k=lane; k<2048u; k+=32u) {
        const float a=widen(input[k]);
        const float4 w(widen(weights[k]),widen(weights[2048u+k]),
                       widen(weights[4096u+k]),widen(weights[6144u+k]));
        partial=fma(float4(a),w,partial);
    }
    for (ushort off=16; off>=4; off>>=1) {
        const float4 other(simd_shuffle_down(partial.x,off),simd_shuffle_down(partial.y,off),
                           simd_shuffle_down(partial.z,off),simd_shuffle_down(partial.w,off));
        if (lane<off) partial=partial+other;
    }
    for (uint r=0; r<4u; ++r) {
        const float p0=simd_shuffle(partial[r],ushort(0));
        const float p1=simd_shuffle(partial[r],ushort(1));
        const float p2=simd_shuffle(partial[r],ushort(2));
        const float p3=simd_shuffle(partial[r],ushort(3));
        if (lane==0) {
            const uint bits=as_type<uint>(((p0+p1)+(p2+p3))+0.0f);
            const ushort rounded=ushort((bits+0x7fff+((bits>>16)&1))>>16);
            if ((rounded&0x7fff)>=0x7f80) atomic_fetch_or_explicit(status,4u,memory_order_relaxed);
            else {
                output[r]=rounded;
                routes[r]=0x80000000u;
            }
        }
    }
}

kernel void lfm3_ordered(
    const device ushort* input [[buffer(0)]],
    const device ushort* q [[buffer(1)]],
    const device ushort* k [[buffer(2)]],
    const device ushort* v [[buffer(3)]],
    device atomic_uint* status [[buffer(4)]],
    device ushort* output_q [[buffer(5)]],
    device ushort* output_k [[buffer(6)]],
    device ushort* output_v [[buffer(7)]],
    device uint* routes_q [[buffer(8)]],
    device uint* routes_k [[buffer(9)]],
    device uint* routes_v [[buffer(10)]],
    const device uint* input_bounds [[buffer(11)]],
    const device uint* tags_q [[buffer(12)]],
    const device uint* tags_k [[buffer(13)]],
    const device uint* tags_v [[buffer(14)]],
    uint group [[threadgroup_position_in_grid]],
    ushort lane [[thread_index_in_simdgroup]]
) {

    const uint role = group < 512 ? 0 : (group < 640 ? 1 : 2);
    const uint first = role == 0 ? 0 : (role == 1 ? 512 : 640);
    const uint row = 4u*(group-first);
    const uint rows = role == 0 ? 2048 : 512;
    const device ushort* bank = role == 0 ? q : (role == 1 ? k : v);
    device ushort* output = role == 0 ? output_q : (role == 1 ? output_k : output_v);
    device uint* routes = role == 0 ? routes_q : (role == 1 ? routes_k : routes_v);
    const device uint* weight_bounds = role == 0 ? tags_q : (role == 1 ? tags_k : tags_v);
    if ((atomic_load_explicit(status,memory_order_relaxed)&2u)!=0) return;
    bool admitted = true;
    for (uint r=0; r<4u; ++r)
        admitted = normal_pair(input_bounds[0],weight_bounds[row+r]) && admitted;
    if (admitted) {
        fma4(input,bank+row*2048u,status,output+row,routes+row,lane);
    } else {
        for (uint r=0; r<4u; ++r)
            projection(input,bank,status,uint4(1,2048,rows,0),output,routes,
                       input_bounds,weight_bounds,row+r,lane);
    }
}

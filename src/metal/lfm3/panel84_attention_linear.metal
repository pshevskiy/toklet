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
                    routes[index] = 0x80000000u;
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
    uint group [[threadgroup_position_in_grid]],
    ushort lane [[thread_index_in_simdgroup]]
) {
    if ((atomic_load_explicit(status,memory_order_relaxed)&2u)!=0u) return;
    if (dims.x!=128u || dims.y!=2048u || (dims.z!=2048u && dims.z!=512u)) {
        if (lane==0) atomic_fetch_or_explicit(status,4u,memory_order_relaxed);
        return;
    }
    // The host admitted every input/weight pair with the sealed row tags.
    const uint rows = dims.z/4u;
    const uint t0 = 8u*(group/rows);
    const uint r0 = 4u*(group%rows);
    panel(input,weight,status,dims,output,routes,t0,r0,lane);
}

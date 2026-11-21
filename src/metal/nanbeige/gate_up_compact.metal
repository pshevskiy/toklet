// Fixed 128×3072 by 10752×3072 affine G64 Q4. One block computes M32×N32.
// Four subgroups each own a 2×2 set of 8×8 products. Every output sees the
// released FP32 simdgroup MMA sequence (K32 outer slabs, four K8 steps).
kernel void nanbeige_ffn_compact(
    device const uint* words [[buffer(0)]],
    device const ushort* scales [[buffer(1)]],
    device const ushort* biases [[buffer(2)]],
    device const ushort* input [[buffer(3)]],
    device ushort* output [[buffer(4)]],
    device atomic_uint* status [[buffer(5)]],
    constant MatrixShape& s [[buffer(6)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]],
    uint subgroup [[simdgroup_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]]) {
    // Store BF16 bits, like the resident operands, and widen in registers.
    // The stride pads each row by eight BF16 elements.
    constexpr uint stride = 40u;
    threadgroup ushort x_tile[32u * stride];
    threadgroup ushort w_tile[32u * stride];
    // The host checks this lane layout on the compiled pipeline before use.
    uint quad = simd_lane / 4u;
    uint frag_m = (quad & 4u) + (simd_lane / 2u) % 4u;
    uint frag_n = (quad & 2u) * 2u + (simd_lane % 2u) * 2u;
    simdgroup_float8x8 c00 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_float8x8 c01 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_float8x8 c10 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_float8x8 c11 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    uint first_token = group.y * 32u;
    uint first_row = group.x * 32u;
    uint tile_m = (subgroup / 2u) * 16u;
    uint tile_n = (subgroup % 2u) * 16u;
    uint row = first_row + lane / 4u;
    uint column_in_tile = (lane % 4u) * 8u;
    for (uint k = 0; k < s.inputs; k += 32u) {
        for (uint i = lane; i < 32u * 32u; i += 128u) {
            uint token = first_token + i / 32u;
            ushort x = input[(s.input_row + token) * s.inputs + k + i % 32u];
            x_tile[(i / 32u) * stride + i % 32u] = x;
        }
        uint packed = words[row * (s.inputs / 8u) + (k + column_in_tile) / 8u];
        uint meta = row * (s.inputs / 64u) + k / 64u;
        float scale = widen(scales[meta]);
        float bias = widen(biases[meta]);
        for (uint j = 0; j < 8u; ++j) {
            float v = scale * float((packed >> (4u * j)) & 15u) + bias;
            w_tile[(lane / 4u) * stride + column_in_tile + j] = narrow(v);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = 0; i < 32u; i += 8u) {
            simdgroup_float8x8 a0, a1, b0, b1;
            uint a = (tile_m + frag_m) * stride + i + frag_n;
            uint b = (tile_n + frag_n) * stride + i + frag_m;
            a0.thread_elements()[0] = widen(x_tile[a]);
            a0.thread_elements()[1] = widen(x_tile[a + 1u]);
            a1.thread_elements()[0] = widen(x_tile[a + 8u * stride]);
            a1.thread_elements()[1] = widen(x_tile[a + 8u * stride + 1u]);
            // B is physically [output, input]; its MMA fragment is [K, N].
            b0.thread_elements()[0] = widen(w_tile[b]);
            b0.thread_elements()[1] = widen(w_tile[b + stride]);
            b1.thread_elements()[0] = widen(w_tile[b + 8u * stride]);
            b1.thread_elements()[1] = widen(w_tile[b + 9u * stride]);
            simdgroup_multiply_accumulate(c00, a0, b0, c00);
            simdgroup_multiply_accumulate(c01, a0, b1, c01);
            simdgroup_multiply_accumulate(c10, a1, b0, c10);
            simdgroup_multiply_accumulate(c11, a1, b1, c11);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    uint low = (first_token + tile_m + frag_m) * s.outputs + first_row + tile_n + frag_n;
    uint high = low + 8u * s.outputs;
    store_word(output, low, c00.thread_elements()[0], status);
    store_word(output, low + 1u, c00.thread_elements()[1], status);
    store_word(output, low + 8u, c01.thread_elements()[0], status);
    store_word(output, low + 9u, c01.thread_elements()[1], status);
    store_word(output, high, c10.thread_elements()[0], status);
    store_word(output, high + 1u, c10.thread_elements()[1], status);
    store_word(output, high + 8u, c11.thread_elements()[0], status);
    store_word(output, high + 9u, c11.thread_elements()[1], status);
}

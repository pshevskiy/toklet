// Preserve the released BF16 boundaries at both projection stores and at
// sigmoid, SiLU and the final product. One lane owns the same output pair
// throughout both matrices and the epilogue.
inline void fused_swiglu_store(
    device ushort* output, uint index, float gate, float up, device atomic_uint* status) {
    ushort g_word = narrow(gate), u_word = narrow(up);
    if ((g_word & 0x7f80u) == 0x7f80u || (u_word & 0x7f80u) == 0x7f80u)
        atomic_fetch_or_explicit(status, 1u, memory_order_relaxed);
    float g = widen(g_word), e = precise::exp(-abs(g));
    float sigmoid = rounded((g < 0.0f ? e : 1.0f) / (1.0f + e));
    store_word(output, index, rounded(g * sigmoid) * widen(u_word), status);
}

kernel void nanbeige_ffn_fused_gate_up_swiglu_dense_k64(
    device const uint4* weights [[buffer(0)]],
    device const ushort* input [[buffer(1)]],
    device ushort* output [[buffer(2)]],
    device atomic_uint* status [[buffer(3)]],
    constant MatrixShape& s [[buffer(4)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]],
    uint subgroup [[simdgroup_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]]) {
    constexpr uint stride = 72u;
    // Four-word aligned storage matches the shipped single-bank K64 route.
    threadgroup ushort4 x_vectors[32u * stride / 4u];
    threadgroup ushort* x_tile = reinterpret_cast<threadgroup ushort*>(x_vectors);
    uint quad = simd_lane / 4u;
    uint frag_m = (quad & 4u) + (simd_lane / 2u) % 4u;
    uint frag_n = (quad & 2u) * 2u + (simd_lane % 2u) * 2u;
    simdgroup_float8x8 g00 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_float8x8 g01 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_float8x8 g10 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_float8x8 g11 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_float8x8 u00 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_float8x8 u01 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_float8x8 u10 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_float8x8 u11 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    uint first_token = group.y * 32u;
    uint first_row = group.x * 32u;
    uint tile_m = (subgroup / 2u) * 16u;
    uint tile_n = (subgroup % 2u) * 16u;
    // Each vector contains Gate then Up in exact b0/b1 lane order.
    uint weight_base = group.x * (s.inputs / 8u);
    for (uint k = 0; k < s.inputs; k += 64u) {
        for (uint i = lane; i < 32u * 16u; i += 128u) {
            uint token = first_token + i / 16u;
            uint source = (s.input_row + token) * s.inputs + k + (i % 16u) * 4u;
            uint target = (i / 16u) * stride + (i % 16u) * 4u;
            x_vectors[target / 4u] = *reinterpret_cast<device const ushort4*>(input + source);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        #pragma unroll
        for (uint i = 0; i < 64u; i += 8u) {
            simdgroup_float8x8 a0, a1, b0, b1;
            uint a = (tile_m + frag_m) * stride + i + frag_n;
            uint weight = ((weight_base + (k + i) / 8u) * 2u + subgroup % 2u) * 32u + simd_lane;
            uint4 pair = weights[weight];
            a0.thread_elements()[0] = widen(x_tile[a]);
            a0.thread_elements()[1] = widen(x_tile[a + 1u]);
            a1.thread_elements()[0] = widen(x_tile[a + 8u * stride]);
            a1.thread_elements()[1] = widen(x_tile[a + 8u * stride + 1u]);
            // Keep identical K8 and fragment operand order for each bank.
            ushort4 gate_words = as_type<ushort4>(pair.xy);
            b0.thread_elements()[0] = widen(gate_words.x);
            b0.thread_elements()[1] = widen(gate_words.y);
            b1.thread_elements()[0] = widen(gate_words.z);
            b1.thread_elements()[1] = widen(gate_words.w);
            simdgroup_multiply_accumulate(g00, a0, b0, g00);
            simdgroup_multiply_accumulate(g01, a0, b1, g01);
            simdgroup_multiply_accumulate(g10, a1, b0, g10);
            simdgroup_multiply_accumulate(g11, a1, b1, g11);

            ushort4 up_words = as_type<ushort4>(pair.zw);
            b0.thread_elements()[0] = widen(up_words.x);
            b0.thread_elements()[1] = widen(up_words.y);
            b1.thread_elements()[0] = widen(up_words.z);
            b1.thread_elements()[1] = widen(up_words.w);
            simdgroup_multiply_accumulate(u00, a0, b0, u00);
            simdgroup_multiply_accumulate(u01, a0, b1, u01);
            simdgroup_multiply_accumulate(u10, a1, b0, u10);
            simdgroup_multiply_accumulate(u11, a1, b1, u11);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    uint low = (first_token + tile_m + frag_m) * s.outputs + first_row + tile_n + frag_n;
    uint high = low + 8u * s.outputs;
    fused_swiglu_store(output, low, g00.thread_elements()[0], u00.thread_elements()[0], status);
    fused_swiglu_store(output, low + 1u, g00.thread_elements()[1], u00.thread_elements()[1], status);
    fused_swiglu_store(output, low + 8u, g01.thread_elements()[0], u01.thread_elements()[0], status);
    fused_swiglu_store(output, low + 9u, g01.thread_elements()[1], u01.thread_elements()[1], status);
    fused_swiglu_store(output, high, g10.thread_elements()[0], u10.thread_elements()[0], status);
    fused_swiglu_store(output, high + 1u, g10.thread_elements()[1], u10.thread_elements()[1], status);
    fused_swiglu_store(output, high + 8u, g11.thread_elements()[0], u11.thread_elements()[0], status);
    fused_swiglu_store(output, high + 9u, g11.thread_elements()[1], u11.thread_elements()[1], status);
}

// Preserve the released BF16 boundaries at both projection stores and at
// sigmoid, SiLU and the final product. One lane owns the same output pair
// throughout both matrices and the epilogue.
inline void prepacked_swiglu_store(
    device ushort* output, uint index, float gate, float up,
    device atomic_uint* status) {
    ushort g_word = narrow(gate), u_word = narrow(up);
    if ((g_word & 0x7f80u) == 0x7f80u || (u_word & 0x7f80u) == 0x7f80u)
        atomic_fetch_or_explicit(status, 1u, memory_order_relaxed);
    float g = widen(g_word);
    float sigmoid = steel_sigmoid(g);
    store_word(output, index, rounded(g * sigmoid) * widen(u_word), status);
}

// Route contract: exact T128, K3072, offset zero. Each record contains
// the two A0 and two A1 BF16 words for one lane in [T16 block, K8, lane] order.
kernel void nanbeige_pack_dense_gate_up_input(
    device const ushort2* input [[buffer(0)]],
    device ushort4* packed [[buffer(1)]],
    constant MatrixShape& s [[buffer(2)]],
    uint2 index [[thread_position_in_grid]]) {
    uint lane = index.x % 32u;
    uint k8 = index.x / 32u;
    uint block = index.y;
    uint record = block * (s.inputs / 8u) * 32u + index.x;
    uint quad = lane / 4u;
    uint frag_m = (quad & 4u) + (lane / 2u) % 4u;
    uint frag_n = (quad & 2u) * 2u + (lane % 2u) * 2u;
    uint a = ((s.input_row + block * 16u + frag_m) * s.inputs + k8 * 8u + frag_n) / 2u;
    packed[record] = ushort4(input[a], input[a + 4u * s.inputs]);
}

kernel void nanbeige_ffn_fused_gate_up_swiglu_prepacked_k64(
    device const uint4* weights [[buffer(0)]],
    device const ushort4* packed [[buffer(1)]],
    device ushort* output [[buffer(2)]],
    device atomic_uint* status [[buffer(3)]],
    constant MatrixShape& s [[buffer(4)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint subgroup [[simdgroup_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]]) {
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
    uint block = (first_token + tile_m) / 16u;
    uint tile_n = (subgroup % 2u) * 16u;
    // Each vector contains Gate then Up in exact b0/b1 lane order.
    uint weight_base = group.x * (s.inputs / 8u);
    for (uint k = 0; k < s.inputs; k += 64u) {
        #pragma unroll
        for (uint i = 0; i < 64u; i += 8u) {
            simdgroup_float8x8 a0, a1, b0, b1;
            uint record = (block * (s.inputs / 8u) + (k + i) / 8u) * 32u + simd_lane;
            ushort4 a_words = packed[record];
            uint weight = ((weight_base + (k + i) / 8u) * 2u + subgroup % 2u) * 32u + simd_lane;
            uint4 pair = weights[weight];
            a0.thread_elements()[0] = widen(a_words.x);
            a0.thread_elements()[1] = widen(a_words.y);
            a1.thread_elements()[0] = widen(a_words.z);
            a1.thread_elements()[1] = widen(a_words.w);
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
    }
    uint low = (first_token + tile_m + frag_m) * s.outputs + first_row + tile_n + frag_n;
    uint high = low + 8u * s.outputs;
    prepacked_swiglu_store(output, low, g00.thread_elements()[0], u00.thread_elements()[0], status);
    prepacked_swiglu_store(output, low + 1u, g00.thread_elements()[1], u00.thread_elements()[1], status);
    prepacked_swiglu_store(output, low + 8u, g01.thread_elements()[0], u01.thread_elements()[0], status);
    prepacked_swiglu_store(output, low + 9u, g01.thread_elements()[1], u01.thread_elements()[1], status);
    prepacked_swiglu_store(output, high, g10.thread_elements()[0], u10.thread_elements()[0], status);
    prepacked_swiglu_store(output, high + 1u, g10.thread_elements()[1], u10.thread_elements()[1], status);
    prepacked_swiglu_store(output, high + 8u, g11.thread_elements()[0], u11.thread_elements()[0], status);
    prepacked_swiglu_store(output, high + 9u, g11.thread_elements()[1], u11.thread_elements()[1], status);
}

// Two released 8×8 M products per SIMD group, sharing the packed K32×N32
// reconstruction. Each accumulator sees the same ascending K sequence.
kernel void nanbeige_ffn_m32(
    device const uint* words [[buffer(0)]],
    device const ushort* scales [[buffer(1)]],
    device const ushort* biases [[buffer(2)]],
    device const ushort* input [[buffer(3)]],
    device ushort* output [[buffer(4)]],
    device atomic_uint* status [[buffer(5)]],
    constant MatrixShape& s [[buffer(6)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]],
    uint subgroup [[simdgroup_index_in_threadgroup]]) {
    threadgroup float x_tile[32 * 32];
    threadgroup float w_tile[32 * 32];
    threadgroup float results[32 * 32];
    simdgroup_float8x8 lower = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_float8x8 upper = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    uint first_token = group.y * 32u, first_row = group.x * 32u;
    uint tile_m = (subgroup / 4u) * 8u, tile_n = (subgroup % 4u) * 8u;
    for (uint k = 0; k < s.inputs; k += 32u) {
        for (uint i = lane; i < 32u * 32u; i += 256u) {
            uint token = first_token + i / 32u;
            x_tile[i] = token < s.tokens
                ? widen(input[(s.input_row + token) * s.inputs + k + i % 32u])
                : 0.0f;
        }
        if (lane < 128u) {
            uint row = first_row + lane / 4u;
            uint column = k + (lane % 4u) * 8u;
            uint w = words[row * (s.inputs / 8u) + column / 8u];
            uint meta = row * (s.inputs / 64u) + k / 64u;
            float scale = widen(scales[meta]), bias = widen(biases[meta]);
            for (uint j = 0; j < 8u; ++j)
                w_tile[lane * 8u + j] = rounded(scale * float((w >> (4u * j)) & 15u) + bias);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = 0; i < 32u; i += 8u) {
            simdgroup_float8x8 a0, a1, b;
            simdgroup_load(a0, x_tile + tile_m * 32u + i, 32u);
            simdgroup_load(a1, x_tile + (tile_m + 16u) * 32u + i, 32u);
            simdgroup_load(b, w_tile + tile_n * 32u + i, 32u, ulong2(0), true);
            simdgroup_multiply_accumulate(lower, a0, b, lower);
            simdgroup_multiply_accumulate(upper, a1, b, upper);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    simdgroup_store(lower, results + tile_m * 32u + tile_n, 32u);
    simdgroup_store(upper, results + (tile_m + 16u) * 32u + tile_n, 32u);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = lane; i < 32u * 32u; i += 256u) {
        uint token = first_token + i / 32u;
        if (token < s.tokens)
            store_word(output, token * s.outputs + first_row + i % 32u, results[i], status);
    }
}

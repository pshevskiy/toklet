// Append to unchanged original source; arithmetic body/store helper unchanged.
// Shape [tokens,columns,chunk_rows,total_rows,first_row].
template <uint Bits>
void gemma_product_strided(device const uint* codes, device const uint* scales,
    device const uint* biases, device const uint* input, device uint* output,
    constant uint* shape, device atomic_uint* status, uint group, uint lane, uint lanes) {
    uint token = group / shape[2], row = group % shape[2];
    if (token >= shape[0]) return;
    constexpr uint codes_per_word = 32 / Bits;
    constexpr uint code_mask = (1u << Bits) - 1;
    uint columns = shape[1], groups = columns / 64;
    float2 sum = float2(0.0f);
    for (uint g = 0; g < groups; ++g) {
        uint index = row * groups + g;
        float scale = gemma_coefficient(scales, index);
        float bias = gemma_coefficient(biases, index);
        for (uint j = lane; j < 64; j += lanes) {
            uint column = g * 64 + j;
            uint word = codes[row * (columns / codes_per_word) + column / codes_per_word];
            uint code = (word >> (Bits * (column % codes_per_word))) & code_mask;
            float weight = fma(scale, float(code), bias);
            float x = as_type<float>(input[token * columns + column]);
            float product = weight * x;
            gemma_add(sum, product);
            gemma_add(sum, fma(weight, x, -product));
        }
    }
    gemma_store_dot(sum, output, status, token * shape[3] + shape[4] + row, lane, lanes);
}

kernel void gemma_affine4_product_strided(
    device const uint* codes [[buffer(0)]],
    device const uint* scales [[buffer(1)]],
    device const uint* biases [[buffer(2)]],
    device const uint* input [[buffer(3)]],
    device uint* output [[buffer(4)]],
    constant uint* shape [[buffer(5)]],
    device atomic_uint* status [[buffer(6)]],
    uint group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint lanes [[threads_per_simdgroup]]) {
    gemma_product_strided<4>(codes, scales, biases, input, output, shape, status, group, lane, lanes);
}

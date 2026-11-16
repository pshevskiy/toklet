// Appended to the unchanged original gemma_affine.metal in a cold library.
// Only entry name and output store indexing differ from its BF16 entry.
kernel void gemma_bf16_product_strided(
    device const uint* coefficients [[buffer(0)]],
    device const uint* input [[buffer(3)]],
    device uint* output [[buffer(4)]],
    constant uint* shape [[buffer(5)]],
    device atomic_uint* status [[buffer(6)]],
    uint group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint lanes [[threads_per_simdgroup]]) {
    uint token = group / shape[2], row = group % shape[2];
    if (token >= shape[0]) return;
    uint columns = shape[1];
    float3 sum = float3(0.0f);
    for (uint column = lane; column < columns; column += lanes) {
        float weight = gemma_coefficient(coefficients, row * columns + column);
        float x = as_type<float>(input[token * columns + column]);
        float product = weight * x;
        gemma_bf16_add(sum, product);
        gemma_bf16_add(sum, fma(weight, x, -product));
    }
    for (uint hop = 1; hop < lanes; hop *= 2) {
        float high = simd_shuffle_down(sum.x, hop);
        float middle = simd_shuffle_down(sum.y, hop);
        float low = simd_shuffle_down(sum.z, hop);
        if (lane % (2 * hop) == 0 && lane + hop < lanes) {
            gemma_bf16_add(sum, high);
            gemma_bf16_add(sum, middle);
            gemma_bf16_add(sum, low);
        }
    }
    if (lane == 0) {
        float value = gemma_bf16_round(sum);
        output[token * shape[3] + shape[4] + row] = as_type<uint>(value);
        if (!isfinite(value)) atomic_fetch_or_explicit(status, 1u, memory_order_relaxed);
    }
}

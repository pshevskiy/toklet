// Compiled only by the explicit affine8 prefill experiment.
// One SIMD group owns one output channel for up to four activation rows.
kernel void mlx_product_tile4_8_0(
    device const uchar* weights [[buffer(0)]],
    device const uint* scales [[buffer(1)]],
    device const uint* biases [[buffer(2)]],
    device const half* linear_bias [[buffer(3)]],
    device const half* input [[buffer(4)]],
    device half* output [[buffer(5)]],
    constant uint* s [[buffer(6)]],
    device atomic_uint* status [[buffer(7)]],
    uint group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint width [[threads_per_simdgroup]]) {
    uint first = 4 * (group / s[2]), row = group % s[2];
    uint active = min(4u, s[0] - first), groups = s[1] / s[3];
    float2 sums[4] = {float2(0.f), float2(0.f), float2(0.f), float2(0.f)};
    for (uint g = 0; g < groups; ++g) {
        uint metadata_index = row * groups + g;
        float scale = mlx_metadata<0>(scales, metadata_index);
        float bias = mlx_metadata<0>(biases, metadata_index);
        for (uint j = lane; j < s[3]; j += width) {
            uint column = g * s[3] + j;
            float weight = fma(scale, float(mlx_code<8>(weights, s[1], row, column)), bias);
            #pragma unroll
            for (uint t = 0; t < 4; ++t) {
                if (t < active) {
                    float x = float(input[s[7] + (first + t) * s[5] + column]);
                    float product = weight * x;
                    mlx_add(sums[t], product);
                    mlx_add(sums[t], fma(weight, x, -product));
                }
            }
        }
    }
    #pragma unroll
    for (uint t = 0; t < 4; ++t) {
        if (t < active) {
            for (uint hop = width / 2; hop > 0; hop /= 2) {
                float hi = simd_shuffle_down(sums[t].x, hop);
                float lo = simd_shuffle_down(sums[t].y, hop);
                if (lane < hop) {
                    mlx_add(sums[t], hi);
                    mlx_add(sums[t], lo);
                }
            }
            if (lane == 0) {
                mlx_add(sums[t], float(linear_bias[row]));
                mlx_store(output, (first + t) * s[6] + row,
                    mlx_half_store_value(sums[t]), true, status);
            }
        }
    }
}

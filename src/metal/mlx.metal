#include <metal_stdlib>
using namespace metal;
#pragma clang fp contract(off)

// D: 0=F16, 1=BF16, 2=F32. Original metadata stays resident; no expansion.
template<uint D> inline float mlx_metadata(device const uint* data, uint index) {
    if (D == 2) return as_type<float>(data[index]);
    ushort bits = reinterpret_cast<device const ushort*>(data)[index];
    if (D == 1) return as_type<float>(uint(bits) << 16);
    return float(as_type<half>(bits));
}
template<uint B> inline uint mlx_code(device const uchar* data, uint row_bytes, uint row, uint column) {
    uint bit = column * B, index = row * row_bytes + bit / 8, shift = bit % 8;
    uint word = uint(data[index]);
    if (shift + B > 8) word |= uint(data[index + 1]) << 8;
    return (word >> shift) & ((1u << B) - 1);
}
inline void mlx_add(thread float2& sum, float x) {
    float total = sum.x + x;
    sum.y += abs(sum.x) >= abs(x) ? (sum.x - total) + x : (x - total) + sum.x;
    sum.x = total;
}
inline float mlx_half_store_value(float2 sum) {
    // Preserve the low component's side of an FP16 midpoint before Half RNE.
    // FP32 output retains ordinary nearest rounding instead.
    float value = sum.x + sum.y;
    if (!isfinite(value)) return value;
    float tail = abs(sum.x) >= abs(sum.y) ? (sum.x - value) + sum.y : (sum.y - value) + sum.x;
    uint bits = as_type<uint>(value);
    if (tail != 0.0f && (bits & 1u) == 0) {
        if (value == 0.0f) return copysign(as_type<float>(1u), tail);
        bits = ((tail > 0.0f) == (value > 0.0f)) ? bits + 1u : bits - 1u;
        value = as_type<float>(bits);
    }
    return value;
}
inline void mlx_store(device void* output, uint index, float value, bool half_output, device atomic_uint* status) {
    if (!isfinite(value)) atomic_fetch_or_explicit(status, 1u, memory_order_relaxed);
    if (half_output) {
        half h = half(value);
        if (!isfinite(h)) atomic_fetch_or_explicit(status, 1u, memory_order_relaxed);
        reinterpret_cast<device half*>(output)[index] = h;
    } else {
        reinterpret_cast<device float*>(output)[index] = value;
    }
}
// Shape: [batch, columns, output rows, group size, half output,
//         input stride, output stride, input offset], strides/offset in scalars.
// One queried SIMD group per (token, output row); no fixed execution width.
template<uint B, uint D> inline void mlx_product(
    device const uchar* weights, device const uint* scales, device const uint* biases,
    device const half* linear_bias, device const half* input, device void* output,
    constant uint* s, device atomic_uint* status, uint group, uint lane, uint width) {
    uint token = group / s[2], row = group % s[2], columns = s[1], groups = columns / s[3];
    if (token >= s[0]) return;
    float2 sum = float2(0.0f);
    for (uint g = 0; g < groups; ++g) {
        uint metadata_index = row * groups + g;
        float scale = mlx_metadata<D>(scales, metadata_index);
        float bias = mlx_metadata<D>(biases, metadata_index);
        for (uint j = lane; j < s[3]; j += width) {
            uint column = g * s[3] + j;
            // One FP32 fused affine reconstruction; separate product/reduction.
            float weight = fma(scale, float(mlx_code<B>(weights, columns * B / 8, row, column)), bias);
            float x = float(input[s[7] + token * s[5] + column]), product = weight * x;
            mlx_add(sum, product);
            mlx_add(sum, fma(weight, x, -product));
        }
    }
    for (uint hop = width / 2; hop > 0; hop /= 2) {
        float hi = simd_shuffle_down(sum.x, hop), lo = simd_shuffle_down(sum.y, hop);
        if (lane < hop) { mlx_add(sum, hi); mlx_add(sum, lo); }
    }
    if (lane == 0) {
        mlx_add(sum, float(linear_bias[row]));
        mlx_store(output, token * s[6] + row, s[4] != 0 ? mlx_half_store_value(sum) : sum.x + sum.y, s[4] != 0, status);
    }
}
// Gather shape has the same fields, with s[2] the embedding vocabulary size.
template<uint B, uint D> inline void mlx_gather(
    device const uchar* weights, device const uint* scales, device const uint* biases,
    device const uint* tokens, device void* output, constant uint* s,
    device atomic_uint* status, uint index) {
    if (index >= s[0] * s[1]) return;
    uint token = index / s[1], column = index % s[1], row = tokens[token];
    if (row >= s[2]) { atomic_fetch_or_explicit(status, 1u, memory_order_relaxed); return; }
    uint metadata_index = row * (s[1] / s[3]) + column / s[3];
    float value = fma(mlx_metadata<D>(scales, metadata_index),
        float(mlx_code<B>(weights, s[1] * B / 8, row, column)), mlx_metadata<D>(biases, metadata_index));
    mlx_store(output, token * s[6] + column, value, s[4] != 0, status);
}

#define MLX_PAIR(B,D) \
kernel void mlx_product_##B##_##D(device const uchar* w[[buffer(0)]],device const uint* sc[[buffer(1)]],device const uint* bi[[buffer(2)]],device const half* lb[[buffer(3)]],device const half* x[[buffer(4)]],device void* out[[buffer(5)]],constant uint* s[[buffer(6)]],device atomic_uint* status[[buffer(7)]],uint group[[threadgroup_position_in_grid]],uint lane[[thread_index_in_simdgroup]],uint width[[threads_per_simdgroup]]) { mlx_product<B,D>(w,sc,bi,lb,x,out,s,status,group,lane,width); } \
kernel void mlx_gather_##B##_##D(device const uchar* w[[buffer(0)]],device const uint* sc[[buffer(1)]],device const uint* bi[[buffer(2)]],device const uint* tokens[[buffer(3)]],device void* out[[buffer(4)]],constant uint* s[[buffer(5)]],device atomic_uint* status[[buffer(6)]],uint index[[thread_position_in_grid]]) { mlx_gather<B,D>(w,sc,bi,tokens,out,s,status,index); }
#define MLX_DTYPES(B) MLX_PAIR(B,0) MLX_PAIR(B,1) MLX_PAIR(B,2)
MLX_DTYPES(2)
MLX_DTYPES(3)
MLX_DTYPES(4)
MLX_DTYPES(5)
MLX_DTYPES(6)
MLX_DTYPES(8)

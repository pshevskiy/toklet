#include <metal_stdlib>
using namespace metal;
#pragma clang fp contract(off)

// Direct FP32 affine reconstruction, original packed group64/BF16 storage.
// Cold admission excludes flushed metadata/weights. FP32 input subnormal loss
// is bounded below the absolute numerical gate by the admitted weight domain.
float gemma_coefficient(device const uint* words, uint index) {
    ushort bits = reinterpret_cast<device const ushort*>(words)[index];
    return as_type<float>(uint(bits) << 16);
}
void gemma_add(thread float2& sum, float value) {
    float next = sum.x + value;
    sum.y += abs(sum.x) >= abs(value) ? (sum.x - next) + value : (value - next) + sum.x;
    sum.x = next;
}
void gemma_store_dot(float2 sum, device uint* output, device atomic_uint* status,
    uint index, uint lane, uint lanes) {
    // Increasing pair offsets support any admitted queried SIMD width.
    for (uint hop = 1; hop < lanes; hop *= 2) {
        float hi = simd_shuffle_down(sum.x, hop);
        float lo = simd_shuffle_down(sum.y, hop);
        if (lane % (2 * hop) == 0 && lane + hop < lanes) {
            gemma_add(sum, hi);
            gemma_add(sum, lo);
        }
    }
    if (lane == 0) {
        float value = sum.x + sum.y;
        output[index] = as_type<uint>(value);
        if (!isfinite(value)) atomic_fetch_or_explicit(status, 1u, memory_order_relaxed);
    }
}

// Shape [active tokens, input width, output rows]. One queried SIMD group per
// token/output row. No dense expansion, padded tensor rows or linear bias.
template <uint Bits>
void gemma_product(device const uint* codes, device const uint* scales,
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
    gemma_store_dot(sum, output, status, token * shape[2] + row, lane, lanes);
}

kernel void gemma_affine_product(
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
    gemma_product<4>(codes, scales, biases, input, output, shape, status, group, lane, lanes);
}

kernel void gemma_affine_product_8(
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
    gemma_product<8>(codes, scales, biases, input, output, shape, status, group, lane, lanes);
}

float gemma_bf16_add_error(float a, float b, float rounded) {
    return abs(a) >= abs(b) ? (a - rounded) + b : (b - rounded) + a;
}

void gemma_bf16_add(thread float3& sum, float value) {
    float high = sum.x + value;
    float error = gemma_bf16_add_error(sum.x, value, high);
    float middle = sum.y + error;
    sum.z += gemma_bf16_add_error(sum.y, error, middle);
    sum.x = high;
    sum.y = middle;
}

// Round the retained three terms together. Collect nonoverlapping partials
// in increasing magnitude, then round once at the first inexact addition.
// A smaller partial on the same side distinguishes a rounded midpoint.
float gemma_bf16_round(float3 sum) {
    if (sum.z == 0.0f) return sum.x + sum.y;
    float parts[3];
    uint count = 0;
    for (uint term = 0; term < 3; ++term) {
        float value = sum[term];
        uint kept = 0;
        for (uint i = 0; i < count; ++i) {
            float next = value + parts[i];
            float error = gemma_bf16_add_error(value, parts[i], next);
            if (error != 0.0f) parts[kept++] = error;
            value = next;
        }
        parts[kept++] = value;
        count = kept;
    }
    float value = parts[--count];
    while (count != 0) {
        float part = parts[--count];
        float next = value + part;
        float error = gemma_bf16_add_error(value, part, next);
        value = next;
        if (error == 0.0f) continue;
        if (count != 0 && ((error > 0.0f && parts[count - 1] > 0.0f)
            || (error < 0.0f && parts[count - 1] < 0.0f))) {
            float twice = error + error;
            float adjusted = value + twice;
            if (adjusted - value == twice) value = adjusted;
        }
        break;
    }
    return value;
}

// Row-major BF16 coefficients widen exactly for multiplication. The device
// needs FP32 arithmetic and aligned ushort loads, not a native BF16 ALU.
kernel void gemma_bf16_product(
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
        output[token * shape[2] + row] = as_type<uint>(value);
        if (!isfinite(value)) atomic_fetch_or_explicit(status, 1u, memory_order_relaxed);
    }
}

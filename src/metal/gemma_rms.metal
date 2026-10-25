#include <metal_stdlib>
using namespace metal;
#pragma clang fp contract(off)

// Kahan convention: sum - correction approximates the accumulated exact sum.
void rms_add(thread float& sum, thread float& correction, float value) {
    float adjusted = value - correction;
    float next = sum + adjusted;
    correction = (next - sum) - adjusted;
    sum = next;
}
// Decode original stored bits, including subnormals, into a NORMAL positive
// mantissa [1,2) and integer exponent. Call only for nonzero finite magnitudes.
float rms_mantissa(uint bits, thread int& exponent) {
    uint fraction = bits & 0x7fffffu;
    uint field = (bits >> 23) & 255u;
    if (field != 0) {
        exponent = int(field) - 127;
        return as_type<float>(0x3f800000u | fraction);
    }
    uint highest = 31u - clz(fraction);
    exponent = int(highest) - 149;
    return as_type<float>(0x3f800000u | ((fraction << (23u - highest)) & 0x7fffffu));
}
// Avoid flushing a tiny intermediate that a large learned scale would amplify
// to a normal output. Integer exponent bookkeeping postpones scaling to ldexp.
float rms_weighted(uint xb, uint wb, uint scale_bits, float denominator) {
    uint sign = (xb ^ wb) & 0x80000000u;
    if ((xb & 0x7fffffffu) == 0 || (wb & 0x7fffffffu) == 0)
        return as_type<float>(sign);
    int xe, we, se;
    float xm = rms_mantissa(xb, xe), wm = rms_mantissa(wb, we);
    float sm = rms_mantissa(scale_bits, se);
    float result = ldexp(((xm / sm) * wm) / denominator, xe + we - se);
    return as_type<float>(as_type<uint>(result) ^ sign);
}
// Row-major [rows,width]; direct learned [width], never implicit +1. One
// queried SIMD group per row, coalesced columns, no materialized q/variance.
template<bool Learned>
void rms_row(device const uint* input, device const uint* weights,
             device uint* output, constant uint* shape,
             device atomic_uint* status, uint row, uint lane, uint lanes) {
    if (row >= shape[0]) return;
    uint width = shape[1], start = row * width;
    uint maximum = shape[2]; // once-rounded host sqrt(exact FP32 epsilon)
    for (uint col = lane; col < width; col += lanes)
        maximum = max(maximum, input[start + col] & 0x7fffffffu);
    maximum = simd_max(maximum);
    float scale = as_type<float>(maximum);
    float sum = 0.0f, correction = 0.0f;
    for (uint col = lane; col < width; col += lanes) {
        float q = as_type<float>(input[start + col]) / scale;
        rms_add(sum, correction, q * q);
    }
    // Increasing pair offsets also handle a non-power-of-two queried width.
    for (uint hop = 1; hop < lanes; hop *= 2) {
        float other_sum = simd_shuffle_down(sum, hop);
        float other_correction = simd_shuffle_down(correction, hop);
        if (lane % (2 * hop) == 0 && lane + hop < lanes) {
            rms_add(sum, correction, other_sum);
            rms_add(sum, correction, -other_correction);
        }
    }
    float total = simd_broadcast(sum - correction, 0);
    float epsilon_ratio = as_type<float>(shape[2]) / scale;
    float denominator = sqrt(total / float(width) + epsilon_ratio * epsilon_ratio);
    for (uint col = lane; col < width; col += lanes) {
        uint xb = input[start + col];
        float value;
        if (Learned) value = rms_weighted(xb, weights[col], maximum, denominator);
        else value = (as_type<float>(xb) / scale) / denominator;
        output[start + col] = as_type<uint>(value);
        if (!isfinite(value)) atomic_fetch_or_explicit(status, 1u, memory_order_relaxed);
    }
}
#define RMS_KERNEL(NAME, LEARNED) \
kernel void NAME(device const uint* input [[buffer(0)]], \
                 device const uint* weights [[buffer(1)]], \
                 device uint* output [[buffer(2)]], \
                 constant uint* shape [[buffer(3)]], \
                 device atomic_uint* status [[buffer(4)]], \
                 uint row [[threadgroup_position_in_grid]], \
                 uint lane [[thread_index_in_simdgroup]], \
                 uint lanes [[threads_per_simdgroup]]) { \
    rms_row<LEARNED>(input, weights, output, shape, status, row, lane, lanes); \
}
RMS_KERNEL(gemma_rms_unlearned, false)
RMS_KERNEL(gemma_rms_learned, true)

#include <metal_stdlib>
using namespace metal;
#pragma clang fp contract(off)

struct RmsParams { uint rows; uint hidden; uint epsilon_bits; };

// One thread owns one complete contiguous row. FP32 serial reduction is a
// correctness reference; no SIMD reduction or performance tuning is claimed.
kernel void toklet_rms_norm(device const half* input [[buffer(0)]],
                            device const half* weight [[buffer(1)]],
                            device half* output [[buffer(2)]],
                            constant RmsParams& p [[buffer(3)]],
                            uint row [[thread_position_in_grid]]) {
    if (row >= p.rows) return;
    uint base = row * p.hidden;
    float total = 0.0f;
    for (uint feature = 0; feature < p.hidden; ++feature) {
        float value = float(input[base + feature]);
        total = total + value * value;
    }
    float inverse = 1.0f / sqrt(total / float(p.hidden) + as_type<float>(p.epsilon_bits));
    for (uint feature = 0; feature < p.hidden; ++feature) {
        float value = (float(input[base + feature]) * inverse) * float(weight[feature]);
        output[base + feature] = half(value);
    }
}

#include <metal_stdlib>
using namespace metal;
#pragma clang fp contract(off)

// Shape: rows, experts, selected, hidden, intermediate, then strides for
// input, routing, intermediate slots, expert output slots, reduced output.
inline half store_half(float value, device atomic_uint *status) {
    half result = half(value);
    if (!isfinite(value) || !isfinite(float(result))) {
        atomic_fetch_or_explicit(status, 2u, memory_order_relaxed);
        return half(0);
    }
    return result;
}

inline float ordered_dot(device const half *a, device const half *b, uint count) {
    float sum = 0;
    for (uint j = 0; j < count; ++j) sum += float(a[j]) * float(b[j]);
    return sum;
}

kernel void expert_gate_up(device const half *input [[buffer(0)]],
                           device const half *weights [[buffer(1)]],
                           device half *gate [[buffer(3)]], device half *up [[buffer(4)]],
                           device const uint *ids [[buffer(8)]],
                           device atomic_uint *status [[buffer(10)]],
                           constant uint *s [[buffer(11)]], uint index [[thread_position_in_grid]]) {
    if (index >= s[0] * s[2] * s[4]) return;
    uint slot = index / s[4], column = index % s[4], row = slot / s[2];
    uint expert = ids[row * s[6] + slot % s[2]];
    uint out = slot * s[7] + column;
    if (expert >= s[1]) {
        atomic_fetch_or_explicit(status, 1u, memory_order_relaxed);
        gate[out] = up[out] = half(0);
        return;
    }
    uint weight = (expert * 2 * s[4] + column) * s[3];
    gate[out] = store_half(ordered_dot(input + row * s[5], weights + weight, s[3]), status);
    up[out] = store_half(ordered_dot(input + row * s[5], weights + weight + s[4] * s[3], s[3]), status);
}

kernel void expert_glu(device const half *gate [[buffer(3)]], device const half *up [[buffer(4)]],
                       device half *activated [[buffer(5)]], device atomic_uint *status [[buffer(10)]],
                       constant uint *s [[buffer(11)]], uint index [[thread_position_in_grid]]) {
    if (index >= s[0] * s[2] * s[4]) return;
    uint offset = (index / s[4]) * s[7] + index % s[4];
    float value = float(gate[offset]);
    float exponential = exp(-abs(value));
    float sigmoid = value >= 0 ? 1.0f / (1.0f + exponential) : exponential / (1.0f + exponential);
    half silu = store_half(value * sigmoid, status);
    activated[offset] = store_half(float(silu) * float(up[offset]), status);
}

kernel void expert_down(device const half *weights [[buffer(2)]],
                        device const half *activated [[buffer(5)]], device half *values [[buffer(6)]],
                        device const uint *ids [[buffer(8)]], device atomic_uint *status [[buffer(10)]],
                        constant uint *s [[buffer(11)]], uint index [[thread_position_in_grid]]) {
    if (index >= s[0] * s[2] * s[3]) return;
    uint slot = index / s[3], column = index % s[3], row = slot / s[2];
    uint expert = ids[row * s[6] + slot % s[2]];
    uint out = slot * s[8] + column;
    if (expert >= s[1]) {
        atomic_fetch_or_explicit(status, 1u, memory_order_relaxed);
        values[out] = half(0);
        return;
    }
    uint weight = (expert * s[3] + column) * s[4];
    values[out] = store_half(ordered_dot(activated + slot * s[7], weights + weight, s[4]), status);
}

kernel void expert_reduce(device const half *values [[buffer(6)]], device half *output [[buffer(7)]],
                          device const uint *ids [[buffer(8)]], device const float *gates [[buffer(9)]],
                          device atomic_uint *status [[buffer(10)]], constant uint *s [[buffer(11)]],
                          uint index [[thread_position_in_grid]]) {
    if (index >= s[0] * s[3]) return;
    uint row = index / s[3], column = index % s[3];
    uint seen = 0;
    float sum = 0;
    for (uint slot = 0; slot < s[2]; ++slot) {
        uint expert = ids[row * s[6] + slot];
        float gate = gates[row * s[6] + slot];
        if (expert >= s[1] || (seen & (1u << expert)) || !isfinite(gate) || gate < 0) {
            atomic_fetch_or_explicit(status, 1u, memory_order_relaxed);
            return;
        }
        seen |= 1u << expert;
        float product = float(values[(row * s[2] + slot) * s[8] + column]) * gate;
        sum += product;
    }
    output[row * s[9] + column] = store_half(sum, status);
}

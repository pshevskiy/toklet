#include <metal_stdlib>
using namespace metal;

#pragma clang fp contract(off)
kernel void toklet_silu(device const half* input [[buffer(0)]],
                       device half* output [[buffer(1)]],
                       constant uint& count [[buffer(2)]],
                       uint index [[thread_position_in_grid]]) {
    if (index >= count) return;
    const float x = float(input[index]);
    const float e = exp(-abs(x));
    const float sigmoid = x >= 0.0f ? 1.0f / (1.0f + e) : e / (1.0f + e);
    output[index] = half(x * sigmoid);
}

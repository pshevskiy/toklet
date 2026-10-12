#include <metal_stdlib>
using namespace metal;
#pragma clang fp contract(off)
// shape: logical rows, outputs, FP32 row stride, Half row stride.
kernel void toklet_projection_bias(device const half *bias [[buffer(0)]],
                                   device float *result [[buffer(1)]],
                                   constant uint4 &shape [[buffer(2)]],
                                   uint index [[thread_position_in_grid]]) {
    if (index >= shape.x * shape.y) return;
    result[(index / shape.y) * shape.z + index % shape.y] = float(bias[index % shape.y]);
}
kernel void toklet_projection_store(device const float *result [[buffer(0)]],
                                    device half *output [[buffer(1)]],
                                    constant uint4 &shape [[buffer(2)]],
                                    uint index [[thread_position_in_grid]]) {
    if (index >= shape.x * shape.y) return;
    output[(index / shape.y) * shape.w + index % shape.y] = half(result[(index / shape.y) * shape.z + index % shape.y]);
}

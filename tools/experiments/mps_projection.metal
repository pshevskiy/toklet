#include <metal_stdlib>
using namespace metal;

// Parameters are four UInt32 values: tokens, outputs, float row stride,
// Half row stride. Both strides count elements, including physical padding.
// Bounds are checked on the host, and rounded dispatch tails return early.
kernel void projection_bias(device const half *bias [[buffer(0)]],
                            device float *result [[buffer(1)]],
                            constant uint4 &shape [[buffer(2)]],
                            uint index [[thread_position_in_grid]]) {
    if (index >= shape.x * shape.y) return;
    uint row = index / shape.y;
    uint column = index % shape.y;
    result[row * shape.z + column] = float(bias[column]);
}

kernel void projection_store(device const float *result [[buffer(0)]],
                             device half *stored [[buffer(1)]],
                             constant uint4 &shape [[buffer(2)]],
                             uint index [[thread_position_in_grid]]) {
    if (index >= shape.x * shape.y) return;
    uint row = index / shape.y;
    uint column = index % shape.y;
    stored[row * shape.w + column] = half(result[row * shape.z + column]);
}

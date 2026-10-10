#include <metal_stdlib>
using namespace metal;

kernel void toklet_add_one(device const float* input [[buffer(0)]],
                           device float* output [[buffer(1)]],
                           constant uint& count [[buffer(2)]],
                           uint index [[thread_position_in_grid]]) {
    if (index < count) output[index] = input[index] + 1.0f;
}

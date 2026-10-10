#include <metal_stdlib>
using namespace metal;
#pragma clang fp contract(off)

// Each lane reads/writes only its own state element; delta is a distinct
// immutable allocation. FP32 addition, then one Half round-to-nearest-even.
kernel void toklet_residual(device half* state [[buffer(0)]],
                            device const half* delta [[buffer(1)]],
                            constant uint& count [[buffer(2)]],
                            uint index [[thread_position_in_grid]]) {
    if (index >= count) return;
    float value = float(state[index]) + float(delta[index]);
    state[index] = half(value);
}

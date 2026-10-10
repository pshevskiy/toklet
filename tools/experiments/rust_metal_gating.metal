#include <metal_stdlib>
using namespace metal;
#pragma clang fp contract(off)

// Both operands are already stored Half, including the SiLU stage boundary.
// A finite Half product is exact in FP32; store one Half RNE, preserving zero
// signs. Inputs and output are three distinct owned contiguous allocations.
kernel void toklet_gating(device const half* gate [[buffer(0)]],
                          device const half* up [[buffer(1)]],
                          device half* output [[buffer(2)]],
                          constant uint& count [[buffer(3)]],
                          uint index [[thread_position_in_grid]]) {
    if (index >= count) return;
    float value = float(gate[index]) * float(up[index]);
    output[index] = half(value);
}

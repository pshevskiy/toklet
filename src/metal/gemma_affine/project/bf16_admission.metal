#include <metal_stdlib>
using namespace metal;
// One queried SIMD group per row. Read every actual FP32 input bit, including
// subnormals and signed zero, without floating-point arithmetic or FTZ. Shape
// is [tokens,columns,largest_admitted_positive_finite_fp32_bits]. Distinct status
// bits: 1=input nonfinite, 2=max-proof miss requiring unchanged host replay.
kernel void gemma_bf16_admit_max(
    device const uint* input [[buffer(0)]],
    constant uint* shape [[buffer(1)]],
    device atomic_uint* status [[buffer(2)]],
    uint row [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint lanes [[threads_per_simdgroup]]) {
    if (row >= shape[0]) return;
    uint maximum = 0;
    for (uint column = lane; column < shape[1]; column += lanes)
        maximum = max(maximum, input[row * shape[1] + column] & 0x7fffffffu);
    maximum = simd_max(maximum);
    if (lane == 0) {
        uint flags = maximum >= 0x7f800000u ? 1u : (maximum > shape[2] ? 2u : 0u);
        if (flags != 0) atomic_fetch_or_explicit(status, flags, memory_order_relaxed);
    }
}

// Conservative supplement; the max-only entrypoint above stays byte-exact.
// The same scalar helper is used by production admission and bounded diagnostics.
uint bf16_l1_upper(device const uint* input, uint columns) {
    uint upper = 0u;
    for (uint column = 0; column < columns; ++column) {
        uint magnitude = input[column] & 0x7fffffffu;
        if (magnitude == 0u) continue;
        magnitude = max(magnitude, 0x00800000u);
        if (upper == 0u) { upper = magnitude; continue; }
        float rounded = as_type<float>(upper) + as_type<float>(magnitude);
        uint bits = as_type<uint>(rounded);
        if (bits < 0x00800000u || bits > 0x7f7ffffdu) return 0x7f800000u;
        upper = bits + 2u;
    }
    return upper;
}
struct Bf16L1Decision { uint upper; uint flags; uint path; };
Bf16L1Decision bf16_l1_decision(device const uint* input,
    constant uint* shape, uint maximum) {
    if (maximum >= 0x7f800000u) return {0u,1u,2u};
    if (maximum <= shape[2]) return {0u,0u,0u};
    uint upper = bf16_l1_upper(input,shape[1]);
    return {upper,upper > shape[3] ? 2u : 0u,1u};
}
// Shape [tokens,columns,max_input_bits,l1_input_bits]; status1=fatal,2=replay.
// Only max-miss rows reread inputs in source order on lane0. No later barrier
// or SIMD operation follows divergence. Neither production path clears status.
kernel void gemma_bf16_admit_max_l1(
    device const uint* input [[buffer(0)]],
    constant uint* shape [[buffer(1)]],
    device atomic_uint* status [[buffer(2)]],
    uint row [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint lanes [[threads_per_simdgroup]]) {
    if (row >= shape[0]) return;
    uint maximum = 0u;
    for (uint column = lane; column < shape[1]; column += lanes)
        maximum = max(maximum, input[row * shape[1] + column] & 0x7fffffffu);
    maximum = simd_max(maximum);
    if (lane == 0) {
        Bf16L1Decision result = bf16_l1_decision(input+row*shape[1],shape,maximum);
        if (result.flags != 0u) atomic_fetch_or_explicit(status,result.flags,memory_order_relaxed);
    }
}
// Explicit diagnostic entry only: [maximum,upper,flags,path] per active row.
// Production admission allocates/writes no trace. Both call the same helper.
kernel void gemma_bf16_admit_l1_diagnostic(
    device const uint* input [[buffer(0)]],
    constant uint* shape [[buffer(1)]],
    device uint* trace [[buffer(2)]],
    uint row [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint lanes [[threads_per_simdgroup]]) {
    if (row >= shape[0]) return;
    uint maximum = 0u;
    for (uint column = lane; column < shape[1]; column += lanes)
        maximum = max(maximum,input[row*shape[1]+column]&0x7fffffffu);
    maximum = simd_max(maximum);
    if (lane == 0) {
        Bf16L1Decision result = bf16_l1_decision(input+row*shape[1],shape,maximum);
        trace[4*row]=maximum;trace[4*row+1]=result.upper;
        trace[4*row+2]=result.flags;trace[4*row+3]=result.path;
    }
}

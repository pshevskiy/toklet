#include <metal_stdlib>
using namespace metal;
#pragma clang fp contract(off)
// Private output is meaningful only when its dedicated status remains zero.
// Status1 is fatal nonfinite input/output; status2 requests reference replay.
void residual_flag(device atomic_uint* status, uint flag) {
    atomic_fetch_or_explicit(status, flag, memory_order_relaxed);
}
kernel void gemma_fp32_residual_add(
    device const uint* left [[buffer(0)]], device const uint* right [[buffer(1)]],
    device uint* output [[buffer(2)]], constant uint* shape [[buffer(3)]],
    device atomic_uint* status [[buffer(4)]], uint i [[thread_position_in_grid]]) {
    if (i >= shape[0]) return;
    uint a = left[i], b = right[i], am = a & 0x7fffffffu, bm = b & 0x7fffffffu;
    // Finiteness precedes every identity, including zero operands.
    if (am >= 0x7f800000u || bm >= 0x7f800000u) {
        output[i] = 0; residual_flag(status, 1u); return;
    }
    // All four signed-zero pairs: only -0 + -0 yields -0 under nearest-even.
    if (am == 0 && bm == 0) { output[i] = (a & b) & 0x80000000u; return; }
    if (am == 0) { output[i] = b; return; }
    if (bm == 0) { output[i] = a; return; }
    if (am < 0x00800000u || bm < 0x00800000u) {
        output[i] = 0; residual_flag(status, 2u); return;
    }
    float value = as_type<float>(a) + as_type<float>(b);
    uint word = as_type<uint>(value), magnitude = word & 0x7fffffffu;
    output[i] = word;
    if (magnitude >= 0x7f800000u) { residual_flag(status, 1u); return; }
    if (magnitude == 0 && am == bm) { output[i] = 0; return; }
    // Opposite unequal normal inputs may cancel to a flushed subnormal. Route
    // every non-identity subnormal/zero result to the unchanged CPU reference.
    if (magnitude < 0x00800000u) { output[i] = 0; residual_flag(status, 2u); }
}
kernel void gemma_fp32_scalar_multiply(
    device const uint* input [[buffer(0)]], device uint* output [[buffer(1)]],
    constant uint* shape [[buffer(2)]], device atomic_uint* status [[buffer(3)]],
    uint i [[thread_position_in_grid]]) {
    if (i >= shape[0]) return;
    uint x = input[i], scalar = shape[1], xm = x & 0x7fffffffu, sm = scalar & 0x7fffffffu;
    // Reject nonfinite operands before 0/1 shortcuts (Inf*0 remains fatal).
    if (xm >= 0x7f800000u || sm >= 0x7f800000u) {
        output[i] = 0; residual_flag(status, 1u); return;
    }
    uint sign = (x ^ scalar) & 0x80000000u;
    if (xm == 0 || sm == 0) { output[i] = sign; return; }
    if (sm == 0x3f800000u) { output[i] = x ^ (scalar & 0x80000000u); return; }
    if (xm < 0x00800000u || sm < 0x00800000u) {
        output[i] = 0; residual_flag(status, 2u); return;
    }
    float value = as_type<float>(x) * as_type<float>(scalar);
    uint word = as_type<uint>(value), magnitude = word & 0x7fffffffu;
    output[i] = word;
    if (magnitude >= 0x7f800000u) { residual_flag(status, 1u); return; }
    if (magnitude < 0x00800000u) { output[i] = 0; residual_flag(status, 2u); }
}

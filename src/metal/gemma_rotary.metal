#include <metal_stdlib>
using namespace metal;
#pragma clang fp contract(off)
// UInt32 storage preserves inactive FP32 bits, including signed zero. Shape is
// [tokens, heads, head_dim, active_pairs]. Coefficients are [tokens,pairs,2]
// with [cosine,sine] prepared in FP64 on the host then rounded once to FP32.
kernel void gemma_proportional_rotary(
    device const uint* input [[buffer(0)]],
    device const uint* coefficients [[buffer(1)]],
    device uint* output [[buffer(2)]],
    constant uint* shape [[buffer(3)]],
    device atomic_uint* status [[buffer(4)]],
    uint pair [[thread_position_in_grid]]) {
    uint half_dim = shape[2] / 2;
    if (pair >= shape[0] * shape[1] * half_dim) return;
    uint coordinate = pair % half_dim;
    uint token = pair / (shape[1] * half_dim);
    uint head = (pair / half_dim) % shape[1];
    uint first = (token * shape[1] + head) * shape[2] + coordinate;
    uint second = first + half_dim;
    if (coordinate >= shape[3]) {
        output[first] = input[first];
        output[second] = input[second];
        return;
    }
    uint index = 2 * (token * shape[3] + coordinate);
    float c = as_type<float>(coefficients[index]);
    float s = as_type<float>(coefficients[index + 1]);
    float a = as_type<float>(input[first]);
    float b = as_type<float>(input[second]);
    float ac = a * c, bs = b * s, bc = b * c, as = a * s;
    float x = ac - bs, y = bc + as;
    output[first] = as_type<uint>(x);
    output[second] = as_type<uint>(y);
    if (!isfinite(x) || !isfinite(y))
        atomic_fetch_or_explicit(status, 1u, memory_order_relaxed);
}

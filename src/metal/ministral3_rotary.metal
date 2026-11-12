#include <metal_stdlib>
using namespace metal;
#pragma clang fp contract(off)

// Tight Half [tokens,heads,128], split-half pairs [i,i+64]. Immutable cached
// coefficients are FP32 [capacity,64,{cos,sin}], gains are Half [capacity].
// Shape is [tokens,heads,relative_table_row], never an absolute model admission.
// Separate entrypoints specialize the query gain outside per-pair arithmetic.
template<bool Query>
void ministral3_rotate_pair(
    device const half* input,
    device const float* coefficients,
    device const half* gains,
    device half* output,
    constant uint* shape,
    device atomic_uint* status,
    uint pair) {
    constexpr uint half_dim = 64;
    if (pair >= shape[0] * shape[1] * half_dim) return;
    uint feature = pair % half_dim;
    uint token = pair / (shape[1] * half_dim);
    uint head = (pair / half_dim) % shape[1];
    uint first = (token * shape[1] + head) * 128 + feature;
    uint second = first + half_dim;
    uint row = shape[2] + token;
    uint coefficient = row * 128 + 2 * feature;
    float cosine = coefficients[coefficient];
    float sine = coefficients[coefficient + 1];
    float a = float(input[first]), b = float(input[second]);
    float ac = a * cosine, bs = b * sine;
    float bc = b * cosine, asine = a * sine;
    float x = ac - bs, y = bc + asine;
    half hx = half(x), hy = half(y);
    bool finite = isfinite(a) && isfinite(b) && isfinite(cosine) && isfinite(sine)
        && isfinite(x) && isfinite(y) && isfinite(float(hx)) && isfinite(float(hy));
    if (Query) {
        // Explicit Half rotation store precedes widening and multiplication by
        // the already-Half gain. Neither gain nor store boundaries are fused.
        float gain = float(gains[row]);
        float scaled_x = float(hx) * gain;
        float scaled_y = float(hy) * gain;
        hx = half(scaled_x);
        hy = half(scaled_y);
        finite = finite && isfinite(gain) && isfinite(scaled_x) && isfinite(scaled_y)
            && isfinite(float(hx)) && isfinite(float(hy));
    }
    output[first] = hx;
    output[second] = hy;
    if (!finite) atomic_fetch_or_explicit(status, 1u, memory_order_relaxed);
}

kernel void ministral3_rotary_keys(
    device const half* input [[buffer(0)]],
    device const float* coefficients [[buffer(1)]],
    device const half* gains [[buffer(2)]],
    device half* output [[buffer(3)]],
    constant uint* shape [[buffer(4)]],
    device atomic_uint* status [[buffer(5)]],
    uint pair [[thread_position_in_grid]]) {
    ministral3_rotate_pair<false>(input, coefficients, gains, output, shape, status, pair);
}

kernel void ministral3_rotary_queries(
    device const half* input [[buffer(0)]],
    device const float* coefficients [[buffer(1)]],
    device const half* gains [[buffer(2)]],
    device half* output [[buffer(3)]],
    constant uint* shape [[buffer(4)]],
    device atomic_uint* status [[buffer(5)]],
    uint pair [[thread_position_in_grid]]) {
    ministral3_rotate_pair<true>(input, coefficients, gains, output, shape, status, pair);
}

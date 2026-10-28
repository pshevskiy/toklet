#include <metal_stdlib>
using namespace metal;
#pragma clang fp contract(off)

// Isolated profile: canonical host FP64 inverse frequencies rounded to FP32,
// FP32 absolute angle/trig and separate pair products/sums, then Half storage.
// Shape [rows, heads, dimension, absolute_offset]. No per-layer host waits.
kernel void llama3_rotary(
    device const half* input [[buffer(0)]],
    device const float* inverse [[buffer(1)]],
    device half* output [[buffer(2)]],
    constant uint* shape [[buffer(3)]],
    device atomic_uint* status [[buffer(4)]],
    uint pair [[thread_position_in_grid]]) {
    uint half_dim = shape[2] / 2;
    if (pair >= shape[0] * shape[1] * half_dim) return;
    uint feature = pair % half_dim;
    uint token = pair / (shape[1] * half_dim);
    uint head = (pair / half_dim) % shape[1];
    uint first = (token * shape[1] + head) * shape[2] + feature;
    uint second = first + half_dim;
    float angle = float(shape[3] + token) * inverse[feature];
    float sine = sin(angle), cosine = cos(angle);
    float a = float(input[first]), b = float(input[second]);
    float ac = a * cosine, bs = b * sine;
    float bc = b * cosine, as = a * sine;
    float x = ac - bs, y = bc + as;
    half hx = half(x), hy = half(y);
    output[first] = hx;
    output[second] = hy;
    if (!isfinite(x) || !isfinite(y) || !isfinite(float(hx)) || !isfinite(float(hy)))
        atomic_fetch_or_explicit(status, 1u, memory_order_relaxed);
}

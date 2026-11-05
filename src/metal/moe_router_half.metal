#include <metal_stdlib>
using namespace metal;
#pragma clang fp contract(off)

kernel void moe_router_half(device const half *input [[buffer(0)]],
                            device uint *ids [[buffer(1)]],
                            device float *gates [[buffer(2)]],
                            device atomic_uint *status [[buffer(3)]],
                            constant uint *shape [[buffer(4)]],
                            uint row [[thread_position_in_grid]]) {
    if (row >= shape[0]) return;
    const uint experts = shape[1], selected = shape[2];
    const uint start = row * shape[3], out = row * shape[4];
    float scores[32];
    for (uint expert = 0; expert < experts; ++expert) {
        scores[expert] = float(input[start + expert]);
        if (!isfinite(scores[expert])) {
            atomic_fetch_or_explicit(status, 1u, memory_order_relaxed);
            return;
        }
    }
    uint used = 0;
    float maximum = 0.0f, mass = 0.0f;
    for (uint slot = 0; slot < selected; ++slot) {
        uint best = 0;
        float score = -INFINITY;
        // Ascending traversal and strict comparison break ties by lowest ID.
        for (uint expert = 0; expert < experts; ++expert) {
            if (!(used & (1u << expert)) && scores[expert] > score) {
                score = scores[expert];
                best = expert;
            }
        }
        used |= 1u << best;
        if (slot == 0) maximum = score;
        ids[out + slot] = best;
        const float gate = exp(score - maximum);
        gates[out + slot] = gate;
        mass += gate;
    }
    if (!isfinite(mass) || mass <= 0.0f) {
        atomic_fetch_or_explicit(status, 2u, memory_order_relaxed);
        return;
    }
    for (uint slot = 0; slot < selected; ++slot) gates[out + slot] /= mass;
}

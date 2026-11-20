#include <metal_stdlib>
using namespace metal;

// Inject the authenticated existing integer arithmetic source here.
__LFM3_DYADIC__

static bool finite_word(ushort x) { return (x & 0x7fffu) < 0x7f80u; }
static float widen(ushort x) { return as_type<float>(uint(x) << 16); }

static Word product_f32(ushort a, ushort b, thread uint& route) {
    const uint sign = uint((a ^ b) & 0x8000u) << 16;
    if ((a & 0x7fffu) == 0 || (b & 0x7fffu) == 0) {
        route = 0u;
        return {sign, 0u};
    }
    const int ea = int((a >> 7) & 255u);
    const int eb = int((b >> 7) & 255u);
    const int exponent = ea + eb - 254;
    if (ea > 0 && eb > 0 && exponent >= -126 && exponent <= 125) {
        route = 1u;
        return {as_type<uint>(widen(a) * widen(b)), 0u};
    }
    route = 2u;
    return exact_fma(a, b, sign);
}

kernel void lfm3_admit_elementwise(
    const device ushort* gate [[buffer(0)]],
    const device ushort* up [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    device atomic_uint* status [[buffer(3)]],
    uint index [[thread_position_in_grid]]
) {
    if (index < count && (!finite_word(gate[index]) || !finite_word(up[index]))) {
        atomic_fetch_or_explicit(status, 2u, memory_order_relaxed);
    }
}

kernel void lfm3_silu_product_witness(
    const device ushort* gate [[buffer(0)]],
    const device ushort* up [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    device atomic_uint* status [[buffer(3)]],
    const device ushort* table [[buffer(4)]],
    device ushort* activation [[buffer(5)]],
    device ushort* product [[buffer(6)]],
    device uint* routes [[buffer(7)]],
    uint index [[thread_position_in_grid]]
) {
    if (index >= count || (atomic_load_explicit(status, memory_order_relaxed) & 2u) != 0) return;
    const ushort a = table[gate[index]];
    if (!finite_word(a)) {
        atomic_fetch_or_explicit(status, 8u, memory_order_relaxed);
        return;
    }
    uint route;
    const Word word = product_f32(a, up[index], route);
    const ushort rounded = ushort((word.bits + 0x7fffu + ((word.bits >> 16) & 1u)) >> 16);
    if (word.status != 0 || !finite_word(rounded)) {
        atomic_fetch_or_explicit(status, 4u, memory_order_relaxed);
        return;
    }
    activation[index] = a;
    product[index] = rounded;
    routes[index] = route;
}

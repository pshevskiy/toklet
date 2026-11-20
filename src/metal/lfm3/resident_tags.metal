// The production host domain tag is a BF16 exponent extent and two class flags.
kernel void lfm3_resident_tags(
    const device ushort* product [[buffer(0)]],
    const device uint* gate_routes [[buffer(1)]],
    const device uint* up_routes [[buffer(2)]],
    const device uint* bridge_routes [[buffer(3)]],
    device uint* tags [[buffer(4)]],
    device uint* census [[buffer(5)]],
    device atomic_uint* down_status [[buffer(6)]],
    uint token [[threadgroup_position_in_grid]],
    ushort lane [[thread_index_in_simdgroup]]
) {
    uint min_e = 255, max_e = 0;
    bool normal = false, exceptional = false, bad = false;
    uint gate_native = 0, gate_panel = 0, up_native = 0, up_panel = 0;
    uint zero = 0, hardware = 0, integer = 0;
    const uint first = token * 7680u;
    for (uint i = first + lane; i < first + 7680u; i += 32u) {
        const uint raw = uint(product[i]) & 0x7fffu;
        const uint exponent = raw >> 7;
        if (raw && exponent > 0 && exponent < 255) {
            min_e = min(min_e, exponent);
            max_e = max(max_e, exponent);
            normal = true;
        } else if (raw) {
            exceptional = true;
            bad |= exponent == 255;
        }
        const uint gate = gate_routes[i], up = up_routes[i], bridge = bridge_routes[i];
        gate_panel += uint(gate == 0xc0000000u);
        gate_native += uint(gate == 0x80000000u || gate == 0xc0000000u);
        up_panel += uint(up == 0xc0000000u);
        up_native += uint(up == 0x80000000u || up == 0xc0000000u);
        bad |= (gate != 0x80000000u && gate != 0xc0000000u && (gate & 4095u) + (gate >> 12) > 2080u);
        bad |= (up != 0x80000000u && up != 0xc0000000u && (up & 4095u) + (up >> 12) > 2080u);
        zero += uint(bridge == 0u);
        hardware += uint(bridge == 1u);
        integer += uint(bridge == 2u);
        bad |= bridge > 2u;
    }
    const uint tag = (simd_any(normal) ? 0x10000u : 0u)
                   | (simd_any(exceptional) ? 0x20000u : 0u)
                   | (simd_max(max_e) << 8) | simd_min(min_e);
    const uint counts[7] = {simd_sum(gate_native), simd_sum(gate_panel), simd_sum(up_native),
                            simd_sum(up_panel), simd_sum(zero), simd_sum(hardware), simd_sum(integer)};
    if (lane == 0) {
        tags[token] = tag;
        for (uint i = 0; i < 7; ++i) census[token * 7 + i] = counts[i];
    }
    if (simd_any(bad) && lane == 0) {
        atomic_fetch_or_explicit(down_status, 2u, memory_order_relaxed);
    }
}

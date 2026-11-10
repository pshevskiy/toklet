// Same serial arithmetic as kernels.metal::rms. Only output placement changes.
// Exactly one queried SIMD group owns a contiguous [width] row. No scratch,
// cross-row sharing, reduction reassociation or implicit execution-width constant.
kernel void rms_parallel_store(device const half* x [[buffer(0)]],
                               device const half* w [[buffer(1)]],
                               device half* out [[buffer(2)]],
                               constant uint* s [[buffer(3)]],
                               device atomic_uint* status [[buffer(4)]],
                               uint row [[threadgroup_position_in_grid]],
                               uint lane [[thread_index_in_simdgroup]],
                               uint lanes [[threads_per_simdgroup]]) {
    if (row >= s[0]) return;
    uint base = row * s[1];
    float inv = 0.0f;
    if (lane == 0) {
        float total = 0.0f;
        if (s[3] != 0) {
            float2 sum = float2(0.0f);
            for (uint i = 0; i < s[1]; ++i) {
                float v = float(x[base + i]);
                sum = compensated_add(sum, v * v);
            }
            total = sum.x + sum.y;
        } else {
            for (uint i = 0; i < s[1]; ++i) {
                float v = float(x[base + i]);
                total = total + v * v;
            }
        }
        inv = 1.0f / sqrt(total / float(s[1]) + as_type<float>(s[2]));
    }
    inv = simd_broadcast(inv, 0);
    for (uint i = lane; i < s[1]; i += lanes)
        checked_half(out, base + i, (float(x[base + i]) * inv) * float(w[i]), status);
}

kernel void toklet_qmm_bf16_scan(
    device const ushort* output [[buffer(0)]],
    device atomic_uint* status [[buffer(1)]],
    constant uint& words [[buffer(2)]],
    uint i [[thread_position_in_grid]]) {
    if (i < words && (output[i] & 0x7f80u) == 0x7f80u) {
        atomic_fetch_or_explicit(status, 1u, memory_order_relaxed);
    }
}

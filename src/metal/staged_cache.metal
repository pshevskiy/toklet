#include <metal_stdlib>
using namespace metal;
// Tight active Half storage is inspected as IEEE words. Preserve subnormals and
// signed zero; only either all-ones exponent (NaN or infinity) is invalid.
kernel void validate_half_kv(device const ushort* k [[buffer(0)]],
                             device const ushort* v [[buffer(1)]],
                             constant uint& count [[buffer(2)]],
                             device atomic_uint* status [[buffer(3)]],
                             uint item [[thread_position_in_grid]]) {
    if (item >= count) return;
    if ((k[item] & 0x7c00u) == 0x7c00u || (v[item] & 0x7c00u) == 0x7c00u)
        atomic_fetch_or_explicit(status, 1u, memory_order_relaxed);
}

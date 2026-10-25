#include <metal_stdlib>
using namespace metal;
#pragma STDC FP_CONTRACT OFF

// Original low-nibble packed4 rows; BF16 group64 metadata. Adjacent lanes
// reconstruct adjacent columns. Inputs and immutable weights never alias output.
kernel void gemma_table_gather(
    device const uint *codes [[buffer(0)]],
    device const ushort *scales [[buffer(1)]],
    device const ushort *biases [[buffer(2)]],
    device const uint *ids [[buffer(3)]],
    device uint *output [[buffer(4)]],
    constant uint *shape [[buffer(5)]],
    device atomic_uint *status [[buffer(6)]],
    uint index [[thread_position_in_grid]]) {
    uint tokens = shape[0], columns = shape[1], rows = shape[2];
    if (index >= tokens * columns) return;
    uint selected = index / columns, column = index % columns;
    uint row = ids[selected];
    if (row >= rows) {
        atomic_fetch_or_explicit(status, 1u, memory_order_relaxed);
        return;
    }
    uint group = row * (columns / 64u) + column / 64u;
    float scale = as_type<float>(uint(scales[group]) << 16u);
    float bias = as_type<float>(uint(biases[group]) << 16u);
    uint packed = codes[row * (columns / 8u) + column / 8u];
    float code = float((packed >> (4u * (column % 8u))) & 15u);
    float value = fma(scale, code, bias);
    if (!isfinite(value)) atomic_fetch_or_explicit(status, 2u, memory_order_relaxed);
    output[index] = as_type<uint>(value);
}

// Append to the same precise released source as dense.metal.
inline ushort dense_affine_word(
    device const uint* packed, device const ushort* scales, device const ushort* biases,
    uint row, uint col, uint inputs) {
    uint offset = row * (inputs / 8u) + col / 8u;
    uint meta = row * (inputs / 64u) + col / 64u;
    uint q = (packed[offset] >> (4u * (col % 8u))) & 15u;
    float product = widen(scales[meta]) * float(q);
    return narrow(product + widen(biases[meta]));
}

// One dispatch writes directly to the final immutable pair. No temporary dense bank.
kernel void nanbeige_prepare_dense_gate_up(
    device const uint* gate [[buffer(0)]],
    device const ushort* gate_scales [[buffer(1)]],
    device const ushort* gate_biases [[buffer(2)]],
    device const uint* up [[buffer(3)]],
    device const ushort* up_scales [[buffer(4)]],
    device const ushort* up_biases [[buffer(5)]],
    device uint4* destination [[buffer(6)]],
    constant MatrixShape& s [[buffer(7)]],
    uint index [[thread_position_in_grid]]) {
    if (index >= (s.outputs / 32u) * (s.inputs / 8u) * 64u) return;
    uint lane = index % 32u;
    uint n_half = (index / 32u) % 2u;
    uint k8 = (index / 64u) % (s.inputs / 8u);
    uint n32 = (index / 64u) / (s.inputs / 8u);
    uint quad = lane / 4u;
    uint m = (quad & 4u) + (lane / 2u) % 4u;
    uint n = (quad & 2u) * 2u + (lane % 2u) * 2u;
    uint col = k8 * 8u + m;
    uint row = n32 * 32u + n_half * 16u + n;
    ushort4 g(
        dense_affine_word(gate, gate_scales, gate_biases, row, col, s.inputs),
        dense_affine_word(gate, gate_scales, gate_biases, row + 1u, col, s.inputs),
        dense_affine_word(gate, gate_scales, gate_biases, row + 8u, col, s.inputs),
        dense_affine_word(gate, gate_scales, gate_biases, row + 9u, col, s.inputs));
    ushort4 u(
        dense_affine_word(up, up_scales, up_biases, row, col, s.inputs),
        dense_affine_word(up, up_scales, up_biases, row + 1u, col, s.inputs),
        dense_affine_word(up, up_scales, up_biases, row + 8u, col, s.inputs),
        dense_affine_word(up, up_scales, up_biases, row + 9u, col, s.inputs));
    destination[index] = uint4(as_type<uint2>(g), as_type<uint2>(u));
}

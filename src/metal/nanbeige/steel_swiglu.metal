// The pinned MLX BF16 sigmoid rounds each stable-math primitive.
inline float steel_sigmoid(float gate) {
    float exponential = rounded(precise::exp(abs(gate)));
    float denominator = rounded(1.0f + exponential);
    float reciprocal = rounded(1.0f / denominator);
    return gate < 0.0f ? reciprocal : rounded(1.0f - reciprocal);
}

kernel void nanbeige_swiglu_steel(
    device ushort* gate [[buffer(0)]],
    device const ushort* up [[buffer(1)]],
    device atomic_uint* status [[buffer(2)]],
    constant UnaryShape& s [[buffer(3)]],
    uint i [[thread_position_in_grid]]) {
    if (i >= s.tokens * s.columns) return;
    float g = widen(gate[i]);
    float sigmoid = steel_sigmoid(g);
    store_word(gate, i, rounded(g * sigmoid) * widen(up[i]), status);
}

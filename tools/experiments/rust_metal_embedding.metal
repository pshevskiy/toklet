#include <metal_stdlib>
using namespace metal;

struct EmbeddingParams { uint count; uint hidden; uint vocab; };

kernel void toklet_embedding(device const ushort *table [[buffer(0)]],
                             device const uint *tokens [[buffer(1)]],
                             device ushort *output [[buffer(2)]],
                             constant EmbeddingParams &params [[buffer(3)]],
                             uint index [[thread_position_in_grid]]) {
    if (index >= params.count) return;
    uint token = tokens[index / params.hidden];
    if (token >= params.vocab) return;
    // Raw bit copy preserves signed zero and Half subnormals exactly.
    output[index] = table[token * params.hidden + index % params.hidden];
}

// MLX rotation arithmetic with direct append to the head-major cache.
kernel void nanbeige_rotate_cache_steel(
    device ushort* query [[buffer(0)]],
    device const ushort* key [[buffer(1)]],
    device const ushort* value [[buffer(2)]],
    device ushort* cache_k [[buffer(3)]],
    device ushort* cache_v [[buffer(4)]],
    device atomic_uint* status [[buffer(5)]],
    constant CacheShape& s [[buffer(6)]],
    uint i [[thread_position_in_grid]]) {
    const uint pairs = NB_HEAD_DIM / 2u;
    uint row = i / ((NB_Q_HEADS + NB_KV_HEADS) * pairs);
    uint head = (i / pairs) % (NB_Q_HEADS + NB_KV_HEADS), j = i % pairs;
    uint p = s.past + row;
    float theta = float(s.position_offset + p) *
        precise::exp2(-float(j) / float(pairs) * as_type<float>(0x41d07ca0u));
    float c = fast::cos(theta), sn = fast::sin(theta);
    if (head < NB_Q_HEADS) {
        uint from = (row * NB_Q_HEADS + head) * NB_HEAD_DIM + j;
        float a = widen(query[from]), b = widen(query[from + pairs]);
        store_word(query, from, fma(a, c, (-sn) * b), status);
        store_word(query, from + pairs, fma(a, sn, b * c), status);
    } else {
        head -= NB_Q_HEADS;
        uint from = (row * NB_KV_HEADS + head) * NB_HEAD_DIM + j;
        uint to = ((s.slot * NB_KV_HEADS + head) * s.capacity + p) * NB_HEAD_DIM + j;
        float a = widen(key[from]), b = widen(key[from + pairs]);
        store_word(cache_k, to, fma(a, c, (-sn) * b), status);
        store_word(cache_k, to + pairs, fma(a, sn, b * c), status);
        cache_v[to] = value[from];
        cache_v[to + pairs] = value[from + pairs];
    }
}

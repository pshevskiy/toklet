#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;
#pragma clang fp contract(off)

// Session storage is BF16 bits throughout. Float is used for the specified
// reductions, even on Apple GPU families without native BF16 matrix hardware.
inline float widen(ushort x) {
    return as_type<float>(uint(x) << 16);
}

inline ushort narrow(float x) {
    uint bits = as_type<uint>(x);
    if ((bits & 0x7fffffffu) > 0x7f800000u)
        return ushort((bits >> 16) | 0x0040u);
    return ushort((bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16);
}

inline float rounded(float x) {
    return widen(narrow(x));
}

inline void store_word(device ushort* output, uint index, float x, device atomic_uint* status) {
    ushort word = narrow(x);
    if ((word & 0x7f80u) == 0x7f80u)
        atomic_fetch_or_explicit(status, 1u, memory_order_relaxed);
    output[index] = word;
}

inline uint code(device const uint* words, uint column) {
    return (words[column / 8u] >> (4u * (column % 8u))) & 15u;
}

// Common packed affine arguments are input rows, input width, output width and
// input row offset. The separate untied head can retain just the last row.
struct MatrixShape { uint tokens, inputs, outputs, input_row; };

kernel void nanbeige_embedding(
    device const uint* words [[buffer(0)]],
    device const ushort* scales [[buffer(1)]],
    device const ushort* biases [[buffer(2)]],
    device const uint* tokens [[buffer(3)]],
    device ushort* output [[buffer(4)]],
    device atomic_uint* status [[buffer(5)]],
    constant MatrixShape& s [[buffer(6)]],
    uint i [[thread_position_in_grid]]) {
    if (i >= s.tokens * s.inputs) return;
    uint token = tokens[i / s.inputs], column = i % s.inputs;
    uint g = token * (s.inputs / 64u) + column / 64u;
    // The pinned MLX embedding control observes a single final BF16 store,
    // including affine cancellation across the two terms.
    float value = fma(widen(scales[g]),
        float(code(words + token * (s.inputs / 8u), column)), widen(biases[g]));
    store_word(output, i, value, status);
}

// One SIMD group shares a vector slice across four output rows. Each lane
// consumes one 16-activation slice per input block and adds its affine
// partials in input order before the SIMD reduction.
kernel void nanbeige_vector(
    device const uint* words [[buffer(0)]],
    device const ushort* scales [[buffer(1)]],
    device const ushort* biases [[buffer(2)]],
    device const ushort* input [[buffer(3)]],
    device ushort* output [[buffer(4)]],
    device atomic_uint* status [[buffer(5)]],
    constant MatrixShape& s [[buffer(6)]],
    uint group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint width [[threads_per_simdgroup]]) {
    uint first = group * 4u;
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (uint column = lane * 16u; column < s.inputs; column += width * 16u) {
        float x[16];
        float sum = 0.0f;
        for (uint inner = 0; inner < 16u; inner += 4u) {
            uint offset = s.input_row * s.inputs + column + inner;
            float a = widen(input[offset]);
            float b = widen(input[offset + 1u]);
            float c = widen(input[offset + 2u]);
            float d = widen(input[offset + 3u]);
            x[inner] = a;
            x[inner + 1u] = b;
            x[inner + 2u] = c;
            x[inner + 3u] = d;
            sum += rounded(rounded(rounded(a + b) + c) + d);
        }
        for (uint r = 0; r < 4u; ++r) {
            uint row = first + r;
            const device uint* w = words + row * (s.inputs / 8u) + column / 8u;
            uint metadata = row * (s.inputs / 64u) + column / 64u;
            float dot = 0.0f;
            for (uint pair = 0; pair < 2u; ++pair) {
                uint packed = w[pair];
                for (uint i = 0; i < 8u; i += 4u) {
                    uint j = pair * 8u + i;
                    float four = x[j] * float((packed >> (4u * i)) & 15u);
                    four += x[j + 1u] * float((packed >> (4u * (i + 1u))) & 15u);
                    four += x[j + 2u] * float((packed >> (4u * (i + 2u))) & 15u);
                    four += x[j + 3u] * float((packed >> (4u * (i + 3u))) & 15u);
                    dot += four;
                }
            }
            acc[r] += widen(scales[metadata]) * dot + widen(biases[metadata]) * sum;
        }
    }
    for (uint r = 0; r < 4u; ++r) {
        float total = simd_sum(acc[r]);
        if (lane == 0u) store_word(output, first + r, total, status);
    }
}

// Tile is 16 activation rows ×32 output rows ×32 input columns. Eight SIMD
// groups each own an 8×8 accumulator. The weights are reconstructed once per
// tile, rounded to BF16, and widened exactly for the FP32 matrix primitive.
// Each 32-column slab stays in the same affine group.
kernel void nanbeige_tiled(
    device const uint* words [[buffer(0)]],
    device const ushort* scales [[buffer(1)]],
    device const ushort* biases [[buffer(2)]],
    device const ushort* input [[buffer(3)]],
    device ushort* output [[buffer(4)]],
    device atomic_uint* status [[buffer(5)]],
    constant MatrixShape& s [[buffer(6)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]],
    uint subgroup [[simdgroup_index_in_threadgroup]]) {
    threadgroup float x_tile[16 * 32];
    threadgroup float w_tile[32 * 32];
    threadgroup float results[16 * 32];
    simdgroup_float8x8 acc = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    uint first_token = group.y * 16u, first_row = group.x * 32u;
    uint tile_m = (subgroup / 4u) * 8u, tile_n = (subgroup % 4u) * 8u;
    for (uint k = 0; k < s.inputs; k += 32u) {
        for (uint i = lane; i < 16u * 32u; i += 256u) {
            uint token = first_token + i / 32u;
            x_tile[i] = token < s.tokens
                ? widen(input[(s.input_row + token) * s.inputs + k + i % 32u])
                : 0.0f;
        }
        // One thread per packed word, eight codes per thread. The first
        // half of threads stages exactly 32×32 weights.
        if (lane < 128u) {
            uint row = first_row + lane / 4u;
            uint column = k + (lane % 4u) * 8u;
            uint w = words[row * (s.inputs / 8u) + column / 8u];
            uint meta = row * (s.inputs / 64u) + k / 64u;
            float scale = widen(scales[meta]), bias = widen(biases[meta]);
            for (uint j = 0; j < 8u; ++j)
                w_tile[lane * 8u + j] = rounded(scale * float((w >> (4u * j)) & 15u) + bias);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = 0; i < 32u; i += 8u) {
            simdgroup_float8x8 a, b;
            simdgroup_load(a, x_tile + tile_m * 32u + i, 32u);
            simdgroup_load(b, w_tile + tile_n * 32u + i, 32u, ulong2(0), true);
            simdgroup_multiply_accumulate(acc, a, b, acc);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    simdgroup_store(acc, results + tile_m * 32u + tile_n, 32u);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = lane; i < 16u * 32u; i += 256u) {
        uint token = first_token + i / 32u;
        if (token < s.tokens)
            store_word(output, token * s.outputs + first_row + i % 32u, results[i], status);
    }
}

struct UnaryShape { uint tokens, columns; };

kernel void nanbeige_rms(
    device const ushort* input [[buffer(0)]],
    device const ushort* gain [[buffer(1)]],
    device ushort* output [[buffer(2)]],
    device atomic_uint* status [[buffer(3)]],
    constant UnaryShape& s [[buffer(4)]],
    uint row [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]],
    uint subgroup [[simdgroup_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint width [[threads_per_threadgroup]]) {
    threadgroup float partials[8];
    threadgroup float factor;
    float sum = 0.0f;
    for (uint i = lane; i < s.columns; i += width) {
        float x = widen(input[row * s.columns + i]);
        sum = fma(x, x, sum);
    }
    sum = simd_sum(sum);
    if (simd_lane == 0u) partials[subgroup] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0u) {
        sum = 0.0f;
        for (uint j = 0; j < 8u; ++j) sum += partials[j];
        factor = precise::rsqrt(sum / float(s.columns) + 1e-5f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = lane; i < s.columns; i += width) {
        uint index = row * s.columns + i;
        float normalized = rounded(widen(input[index]) * factor);
        store_word(output, index, normalized * widen(gain[i]), status);
    }
}

constant uint NB_Q_HEADS = 48u;
constant uint NB_KV_HEADS = 8u;
constant uint NB_HEAD_DIM = 128u;
constant float NB_ATTN_SCALE = 0.08838834764831845f;
struct CacheShape { uint tokens, past, capacity, slot, position_offset; };

kernel void nanbeige_rotate_cache(
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
    if (row >= s.tokens) return;
    uint head = (i / pairs) % (NB_Q_HEADS + NB_KV_HEADS), j = i % pairs;
    uint p = s.past + row;
    float theta = float(s.position_offset + p) *
        fast::exp2(-float(j) / float(pairs) * fast::log2(70000000.0f));
    float c = fast::cos(theta), sn = fast::sin(theta);
    if (head < NB_Q_HEADS) {
        uint from = (row * NB_Q_HEADS + head) * NB_HEAD_DIM + j;
        float a = widen(query[from]), b = widen(query[from + pairs]);
        store_word(query, from, fma(-b, sn, a * c), status);
        store_word(query, from + pairs, fma(a, sn, b * c), status);
    } else {
        head -= NB_Q_HEADS;
        uint from = (row * NB_KV_HEADS + head) * NB_HEAD_DIM + j;
        uint to = ((s.slot * NB_KV_HEADS + head) * s.capacity + p) * NB_HEAD_DIM + j;
        float a = widen(key[from]), b = widen(key[from + pairs]);
        store_word(cache_k, to, fma(-b, sn, a * c), status);
        store_word(cache_k, to + pairs, fma(a, sn, b * c), status);
        cache_v[to] = value[from];
        cache_v[to + pairs] = value[from + pairs];
    }
}

// Four SIMD groups per query head split the visible positions. Each computes
// a streaming softmax partial in FP32. Cross-SIMD merge rescales by each
// maximum. Causality includes only up through the query's absolute KV index.
kernel void nanbeige_attention(
    device const ushort* query [[buffer(0)]],
    device const ushort* cache_k [[buffer(1)]],
    device const ushort* cache_v [[buffer(2)]],
    device ushort* output [[buffer(3)]],
    device atomic_uint* status [[buffer(4)]],
    constant CacheShape& s [[buffer(5)]],
    uint group [[threadgroup_position_in_grid]],
    uint split [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float partials[4][128];
    threadgroup float maxima[4], totals[4];
    uint row = group / NB_Q_HEADS, h = group % NB_Q_HEADS;
    uint front = group * NB_HEAD_DIM + lane * 4u;
    uint head = h / (NB_Q_HEADS / NB_KV_HEADS);
    uint base = (s.slot * NB_KV_HEADS + head) * s.capacity * NB_HEAD_DIM + lane * 4u;
    uint visible = s.past + row + 1u;
    float best = -INFINITY, sum = 0.0f;
    float q[4], acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    bool vector = s.tokens == 1u;
    for (uint i = 0; i < 4u; ++i)
        q[i] = widen(query[front + i]) * (vector ? NB_ATTN_SCALE : 1.0f);

    for (uint p = split; p < visible; p += 4u) {
        uint offset = base + p * NB_HEAD_DIM;
        float score = 0.0f;
        for (uint i = 0; i < 4u; ++i) score = fma(q[i], widen(cache_k[offset + i]), score);
        score = simd_sum(score);
        score *= vector ? M_LOG2E_F : NB_ATTN_SCALE * M_LOG2E_F;
        float peak = max(best, score);
        float keep = fast::exp2(best - peak), next = fast::exp2(score - peak);
        sum = fma(sum, keep, next);
        for (uint i = 0; i < 4u; ++i)
            acc[i] = fma(next, widen(cache_v[offset + i]), acc[i] * keep);
        best = peak;
    }

    if (lane == 0u) {
        maxima[split] = best;
        totals[split] = sum;
    }
    for (uint i = 0; i < 4u; ++i) partials[split][lane * 4u + i] = acc[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (split != 0u) return;

    best = max(max(maxima[0], maxima[1]), max(maxima[2], maxima[3]));
    sum = 0.0f;
    for (uint i = 0; i < 4u; ++i) acc[i] = 0.0f;
    for (uint j = 0; j < 4u; ++j) {
        float scale = fast::exp2(maxima[j] - best);
        sum = fma(totals[j], scale, sum);
        for (uint i = 0; i < 4u; ++i)
            acc[i] = fma(partials[j][lane * 4u + i], scale, acc[i]);
    }
    for (uint i = 0; i < 4u; ++i) store_word(output, front + i, acc[i] / sum, status);
}

kernel void nanbeige_add(
    device ushort* residual [[buffer(0)]],
    device const ushort* delta [[buffer(1)]],
    device atomic_uint* status [[buffer(2)]],
    constant UnaryShape& s [[buffer(3)]],
    uint i [[thread_position_in_grid]]) {
    if (i < s.tokens * s.columns)
        store_word(residual, i, widen(residual[i]) + widen(delta[i]), status);
}

kernel void nanbeige_swiglu(
    device ushort* gate [[buffer(0)]],
    device const ushort* up [[buffer(1)]],
    device atomic_uint* status [[buffer(2)]],
    constant UnaryShape& s [[buffer(3)]],
    uint i [[thread_position_in_grid]]) {
    if (i >= s.tokens * s.columns) return;
    float g = widen(gate[i]), e = precise::exp(-abs(g));
    float sigmoid = rounded((g < 0.0f ? e : 1.0f) / (1.0f + e));
    store_word(gate, i, rounded(g * sigmoid) * widen(up[i]), status);
}

kernel void nanbeige_copy(
    device const ushort* input [[buffer(0)]],
    device ushort* output [[buffer(1)]],
    constant uint2& extent [[buffer(2)]],
    uint i [[thread_position_in_grid]]) {
    if (i < extent.y) output[extent.x + i] = input[i];
}

inline bool lower_id_better(float x, uint id, float best, uint best_id) {
    return x > best || (x == best && id < best_id);
}

kernel void nanbeige_greedy(
    device const ushort* input [[buffer(0)]],
    device uint* output [[buffer(1)]],
    constant uint& rows [[buffer(2)]],
    uint lane [[thread_index_in_threadgroup]]) {
    threadgroup float scores[256];
    threadgroup uint ids[256];
    float best = -INFINITY;
    uint best_id = UINT_MAX;
    for (uint row = lane; row < rows; row += 256u) {
        float x = widen(input[row]);
        if (lower_id_better(x, row, best, best_id)) {
            best = x;
            best_id = row;
        }
    }
    scores[lane] = best;
    ids[lane] = best_id;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint hop = 128u; hop > 0u; hop /= 2u) {
        if (lane < hop &&
            lower_id_better(scores[lane + hop], ids[lane + hop], scores[lane], ids[lane])) {
            scores[lane] = scores[lane + hop];
            ids[lane] = ids[lane + hop];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0u) output[0] = ids[0];
}

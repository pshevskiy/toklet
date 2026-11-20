#include <metal_stdlib>
using namespace metal;
#pragma clang fp contract(off)

// One admitted layout. Q/context: [query,head,feature]. K/V: [kv_head,key,feature].
// The shared plane is [head,query,key], padded by the host outside these extents.
constant uint kQueries = 128;

constant uint kWidth = 128;
constant uint kHeads = 16;
constant uint kRows = kQueries * kHeads;

constant uint kSpectrum = 0x3db504f3u;
constant uint kQkDomain = 1u;
constant uint kNormDomain = 2u;
constant uint kValueDomain = 4u;
constant uint kGeometry = 8u;
struct Geometry {
    uint keys;
    uint query_start;
    uint key_start;
    uint window;
};

static bool supported(constant Geometry &g) {
    if (g.query_start != 128 && g.query_start != 256 && g.query_start != 384) return false;
    if (g.window != 0 && g.window != 256) return false;
    uint first = g.window == 0 || g.query_start < g.window
        ? 0 : g.query_start - (g.window - 1);
    return g.key_start == first && g.keys == g.query_start + kQueries - first;
}


static float widen(ushort word) {
    return as_type<float>(uint(word) << 16);
}

static ushort narrow(float value) {
    uint bits = as_type<uint>(value);
    return ushort((bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16);
}

static bool bf_normal_or_zero(ushort value) {
    uint raw = uint(value) & 0x7fffu;
    uint e = raw >> 7;
    return raw == 0 || (e != 0 && e != 255);
}

static bool product_domain(ushort a, ushort b) {
    if (!bf_normal_or_zero(a) || !bf_normal_or_zero(b)) {
        return false;
    }
    if (((uint(a) & 0x7fffu) == 0) || ((uint(b) & 0x7fffu) == 0)) {
        return true;
    }
    int sum = int((a >> 7) & 255u) + int((b >> 7) & 255u) - 254;
    return sum >= -112 && sum <= 109;
}

// Exact fused add when finite BF16 operands produce a nonzero magnitude
// strictly below half the minimum binary32 subnormal, or an exact zero.
// Classification uses raw bits so an implementation's FTZ cannot affect it.
static bool qk_accumulate(ushort a, ushort b, thread float &sum) {
    if (product_domain(a, b)) {
        sum = fma(widen(a), widen(b), sum);
        return true;
    }
    uint left = uint(a) & 0x7fffu;
    uint right = uint(b) & 0x7fffu;
    uint ea = left >> 7;
    uint eb = right >> 7;
    if (ea == 255 || eb == 255) return false;
    bool zero_product = left == 0 || right == 0;
    // Field zero conservatively bounds a nonzero BF16 subnormal by 2^-126.
    if (!zero_product && int(ea) + int(eb) - 254 > -152) return false;
    uint bits = as_type<uint>(sum);
    if ((bits & 0x7fffffffu) == 0) {
        uint product_sign = (uint(a ^ b) & 0x8000u) << 16;
        // Exact zero addition keeps a negative zero only for two negative
        // zero summands. A tiny nonzero exact sum rounds to its own sign.
        sum = as_type<float>(zero_product ? product_sign & bits : product_sign);
    }
    return true;
}

// Reciprocals of positive normal FP32 denominators in [1,512] are normal.
// For m>2^23 the quotient is strictly between 2^23 and 2^24. Divide the
// exact integers and apply RN-even before packing. MSL reciprocal is unused.
static float reciprocal_rne(float denominator) {
    uint bits = as_type<uint>(denominator);
    uint exponent = (bits >> 23) & 255u;
    uint mantissa = (bits & 0x7fffffu) | 0x800000u;
    if (mantissa == 0x800000u) {
        return as_type<float>((254u - exponent) << 23);
    }
    ulong numerator = 1ul << 47;
    uint quotient = uint(numerator / mantissa);
    uint twice_remainder = uint((numerator % mantissa) << 1);
    if (twice_remainder > mantissa || (twice_remainder == mantissa && (quotient & 1u))) {
        ++quotient;
    }
    return as_type<float>(((253u - exponent) << 23) | (quotient - 0x800000u));
}

static float binary_power(int exponent) {
    return as_type<float>(uint(exponent + 127) << 23);
}

// Scalar adaptation of the nonpositive range of SLEEF xexpf (AdvSIMD/FMA).
// Copyright Naoki Shibata and contributors 2010–2024; ARM Ltd. 2010–2024.
// Boost Software License 1.0, source from the pinned Toklet CPU adaptation.
static bool source_exp(float delta, thread float &result) {
    if (delta < -104.0f) {
        result = 0.0f;
        return true;
    }
    if (delta >= -0x1p-26f) {
        // n=0 and the final RN FP32 sum returns 1 even after the quadratic term.
        result = 1.0f;
        return true;
    }
    float exponent_value = rint(delta * 0x1.715476p+0f); // nearest-even LOG2_E
    int exponent = int(exponent_value);
    int lower = exponent >= 0 ? exponent / 2 : (exponent - 1) / 2;
    float remainder = fma(exponent_value, -0.69314575f, delta);
    remainder = fma(exponent_value, -1.4286068e-6f, remainder);
    float poly = fma(1.9852762e-4f, remainder, 0.0013930436f);
    poly = fma(poly, remainder, 0.008333361f);
    poly = fma(poly, remainder, 0.041666485f);
    poly = fma(poly, remainder, 0.16666667f);
    poly = fma(poly, remainder, 0.5f);
    float value = 1.0f + fma(remainder * remainder, poly, remainder);
    float first = value * binary_power(lower);
    int upper = exponent - lower;
    // Before executing the final power-of-two multiplication, prove both
    // its output and the later multiplication by 1/denominator stay normal.
    int biased = int((as_type<uint>(first) >> 23) & 255u);
    if (biased + upper < 10) {
        return false; // lower than 2^-117, including possible FP32/BF16 subnormals
    }
    result = first * binary_power(upper);
    return true;
}

kernel void lfm3_late_qk(device const ushort *q [[buffer(0)]],
                         device const ushort *k [[buffer(1)]],
                         device ushort *plane [[buffer(2)]],
                         device atomic_uint *status [[buffer(3)]],
                         constant Geometry &g [[buffer(4)]],
                         uint group [[threadgroup_position_in_grid]],
                         uint sid [[simdgroup_index_in_threadgroup]],
                         uint lane [[thread_index_in_simdgroup]],
                         uint simd_width [[threads_per_simdgroup]]) {
    if (simd_width != 32 || !supported(g)) {
        if (lane == 0) atomic_fetch_or_explicit(status, kGeometry, memory_order_relaxed);
        return;
    }
    // Four independent output pairs per 128-thread group.
    uint pair = group * 4 + sid;
    if (pair >= kRows * g.keys) return;
    uint row = pair / g.keys;
    uint key = pair % g.keys;
    uint h = row / kQueries;
    uint token = row % kQueries;
    uint q0 = (token * kHeads + h) * kWidth;
    uint k0 = (h / 4 * g.keys + key) * kWidth;
    float sum = 0.0f;
    bool safe = true;
    for (uint i = lane; i < kWidth; i += 32) {
        if (!qk_accumulate(q[q0+i], k[k0+i], sum)) safe = false;
    }
    if (simd_any(!safe)) {
        if (lane == 0) atomic_fetch_or_explicit(status, kQkDomain, memory_order_relaxed);
        return;
    }
    sum = sum + simd_shuffle_xor(sum, 16);
    sum = sum + simd_shuffle_xor(sum, 8);
    sum = sum + simd_shuffle_xor(sum, 4);
    float lane1 = simd_shuffle(sum, 1);
    float lane2 = simd_shuffle(sum, 2);
    float lane3 = simd_shuffle(sum, 3);
    if (lane != 0) return;
    float dot = (sum + lane1) + (lane2 + lane3);
    // The CPU explicitly adds its empty tail's positive zero.
    if ((as_type<uint>(dot) & 0x7fffffffu) == 0) dot = 0.0f;
    ushort score = narrow(dot);
    // Check pre-mask even when the key is in the future. Restrict scaled
    // scores to a BF16 lattice of at least 2^-126 for exact normal deltas.
    uint magnitude = uint(score) & 0x7fffu;
    uint exponent = magnitude >> 7;
    if (!bf_normal_or_zero(score) || (magnitude != 0 && exponent < 12)) {
        atomic_fetch_or_explicit(status, kQkDomain, memory_order_relaxed);
        return;
    }
    float scaled = widen(narrow(widen(score) * as_type<float>(kSpectrum)));
    uint pos = g.query_start + token;
    uint kpos = g.key_start + key;
    uint first = g.window == 0 || pos < g.window ? 0 : pos - (g.window - 1);
    float mask = first <= kpos && kpos <= pos ? 0.0f : widen(0xff7fu);
    float masked = widen(narrow(scaled + mask));
    if (!isfinite(masked)) {
        atomic_fetch_or_explicit(status, kQkDomain, memory_order_relaxed);
        return;
    }
    plane[pair] = narrow(masked);
}

kernel void lfm3_late_probability(device ushort *plane [[buffer(0)]],
                                  device atomic_uint *status [[buffer(1)]],
                                  constant Geometry &g [[buffer(2)]],
                                  uint row [[threadgroup_position_in_grid]],
                                  uint tid [[thread_index_in_threadgroup]]) {
    if (row >= kRows) return;
    threadgroup float exps[512];
    threadgroup float lanes[4];
    threadgroup bool good[4];
    threadgroup bool ready;
    threadgroup float maximum;
    threadgroup float inverse;
    uint start = row * g.keys;
    if (tid == 0) {
        ready = atomic_load_explicit(status, memory_order_relaxed) == 0;
        float local_max = -INFINITY;
        if (ready) {
            for (uint key = 0; key < g.keys; ++key) {
                local_max = max(local_max, widen(plane[start + key]));
            }
        }
        maximum = local_max;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (!ready) return;
    if (tid < 4) {
        float sum = 0.0f;
        bool valid = true;
        for (uint key = tid; key < g.keys; key += 4) {
            float e = 0.0f;
            valid = source_exp(widen(plane[start + key]) - maximum, e) && valid;
            exps[key] = e;
            sum += e;
        }
        lanes[tid] = sum;
        good[tid] = valid;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float total = (lanes[0] + lanes[2]) + (lanes[1] + lanes[3]);
        ready = good[0] && good[1] && good[2] && good[3] &&
                total >= 1.0f && total <= float(g.keys);
        if (ready) {
            inverse = reciprocal_rne(total);
        } else {
            atomic_fetch_or_explicit(status, kNormDomain, memory_order_relaxed);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (ready) {
        for (uint key = tid; key < g.keys; key += 32) {
            plane[start + key] = narrow(exps[key] * inverse);
        }
    }
}

kernel void lfm3_late_context(device const ushort *plane [[buffer(0)]],
                              device const ushort *v [[buffer(1)]],
                              device ushort *context [[buffer(2)]],
                              device atomic_uint *status [[buffer(3)]],
                              constant Geometry &g [[buffer(4)]],
                              uint item [[thread_position_in_grid]]) {
    if (item >= kRows * kWidth || atomic_load_explicit(status, memory_order_relaxed) != 0) return;
    uint row = item / kWidth;
    uint feature = item % kWidth;
    uint h = row / kQueries;
    uint token = row % kQueries;
    uint v0 = (h / 4) * g.keys * kWidth + feature;
    float partial[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    uint aligned = (g.keys / 4) * 4;
    for (uint index = 0; index < g.keys; ++index) {
        // T128 CPU V assigns only the final incomplete group to lane 0.
        uint lane = index < aligned ? index & 3 : 0;
        ushort p_word = plane[row * g.keys + index];
        float p = widen(p_word);
        ushort v_word = v[v0 + index * kWidth];
        if (!product_domain(p_word, v_word)) {
            atomic_fetch_or_explicit(status, kValueDomain, memory_order_relaxed);
            return;
        }
        float term = p * widen(v_word);
        partial[lane] = partial[lane] + term;
    }
    float sum = ((partial[0] + partial[1]) + partial[2]) + partial[3];
    ushort word = narrow(sum);
    if (!bf_normal_or_zero(word)) {
        atomic_fetch_or_explicit(status, kValueDomain, memory_order_relaxed);
        return;
    }
    context[(token * kHeads + h) * kWidth + feature] = word;
}

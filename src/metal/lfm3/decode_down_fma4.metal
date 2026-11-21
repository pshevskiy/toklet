#include <metal_stdlib>
using namespace metal;

// Host inserts the shared integer header here.
// Shared Metal / portable C++ integer implementation.
#ifdef __METAL_VERSION__
#define LIMB ulong
#define CLZ(x) (uint((x) >> 32) != 0 ? clz(uint((x) >> 32)) : 32 + clz(uint(x)))
#else
#include <cstdint>
using uint = std::uint32_t;
using ushort = std::uint16_t;
#define LIMB std::uint64_t
#define CLZ(x) __builtin_clzll(x)
#endif

struct Term {
    uint magnitude;
    int exponent;
    bool negative;
};

struct Word {
    uint bits;
    uint status; // 0 success, 2 nonfinite input, 4 overflow.
};

#ifdef __METAL_VERSION__
constant uint positive_mask = 0x7fffffff;
constant uint sign_bit = 0x80000000;
#else
constexpr uint positive_mask = 0x7fffffff;
constexpr uint sign_bit = 0x80000000;
#endif

static Term unpack(uint bits, bool bf16) {
    const uint field = (bits >> 23) & 255;
    const uint sig = (bits & 0x7fffff) | (field == 0 ? 0 : 0x800000);
    const int exponent = field == 0 ? -149 : int(field) - 150;
    return {bf16 ? sig >> 16 : sig, exponent + (bf16 ? 16 : 0), (bits & sign_bit) != 0};
}

struct Wide {
    LIMB words[9];
};

static Wide place(Term value) {
    Wide out = {};
    const uint offset = uint(value.exponent + 266);
    const uint word = offset / 64;
    const uint shift = offset % 64;
    out.words[word] = LIMB(value.magnitude) << shift;
    if (shift != 0) out.words[word + 1] = LIMB(value.magnitude) >> (64 - shift);
    return out;
}

static Word round_wide(Wide value, bool negative) {
    const uint sign = negative ? sign_bit : 0;
    int top = 8;
    while (top >= 0 && value.words[top] == 0) --top;
    if (top < 0) return {sign, 0};
    const uint highest = uint(64 * top + 63 - int(CLZ(value.words[top])));
    const uint discard = highest > 140 ? highest - 23 : 117;
    const uint word = discard / 64;
    const uint shift = discard % 64;
    LIMB reduced = value.words[word] >> shift;
    if (shift != 0 && word < 8) reduced |= value.words[word + 1] << (64 - shift);
    const uint middle = discard - 1;
    const uint mg = middle / 64;
    const uint mb = middle % 64;
    bool sticky = (value.words[mg] & ((LIMB(1) << mb) - 1)) != 0;
    for (uint i = 0; i < mg; ++i) sticky = sticky || value.words[i] != 0;
    if (((value.words[mg] >> mb) & 1) && (sticky || (reduced & 1))) ++reduced;
    int exponent = int(discard) - 243;
    if (reduced == (LIMB(1) << 24)) {
        reduced >>= 1;
        ++exponent;
    }
    if (exponent > 127) return {sign | 0x7f800000, 4};
    if (reduced < 0x800000) return {sign | uint(reduced), 0};
    return {sign | (uint(exponent + 127) << 23) | (uint(reduced) & 0x7fffff), 0};
}

static Word combine(Term a, Term b) {
    Wide x = place(a), y = place(b);
    const bool same = a.negative == b.negative;
    bool swap = false;
    if (!same) {
        int i = 8;
        while (i >= 0 && x.words[i] == y.words[i]) --i;
        if (i < 0) return {0, 0};
        swap = x.words[i] < y.words[i];
    }
    Wide result;
    bool carry = false;
    for (int i = 0; i < 9; ++i) {
        const LIMB left = swap ? y.words[i] : x.words[i];
        const LIMB right = swap ? x.words[i] : y.words[i];
        const LIMB first = same ? left + right : left - right;
        const LIMB final_word = same ? first + LIMB(carry) : first - LIMB(carry);
        carry = same ? first < left || final_word < first : left < right || first < final_word;
        result.words[i] = final_word;
    }
    return round_wide(result, swap ? b.negative : a.negative);
}

static Word exact_fma(ushort a, ushort b, uint c) {
    const uint ax = uint(a) << 16;
    const uint bx = uint(b) << 16;
    if ((ax & positive_mask) >= 0x7f800000 || (bx & positive_mask) >= 0x7f800000
        || (c & positive_mask) >= 0x7f800000) return {0, 2};
    const Term x = unpack(ax, true), y = unpack(bx, true);
    return combine({x.magnitude * y.magnitude, x.exponent + y.exponent, x.negative != y.negative},
                   unpack(c, false));
}

static Word exact_add(uint a, uint b) {
    if ((a & positive_mask) >= 0x7f800000 || (b & positive_mask) >= 0x7f800000) return {0, 2};
    return combine(unpack(a, false), unpack(b, false));
}


kernel void lfm3_admit(
    const device ushort* input [[buffer(0)]],
    const device ushort* weight [[buffer(1)]],
    device atomic_uint* status [[buffer(2)]],
    constant uint4& dims [[buffer(3)]],
    uint index [[thread_position_in_grid]]
) {
    const uint input_count = dims.x * dims.y;
    const uint weight_count = dims.w;
    if ((index < input_count && (input[index] & 0x7fff) >= 0x7f80) ||
        (index < weight_count && (weight[index] & 0x7fff) >= 0x7f80)) {
        atomic_fetch_or_explicit(status, 2u, memory_order_relaxed);
    }
}

static float widen(ushort bits) { return as_type<float>(uint(bits) << 16); }

static Word ordered_fma(ushort a, ushort b, uint c, thread uint& counts) {
    const uint abs_c = c & positive_mask;
    const int ca = int((a >> 7) & 255);
    const int cb = int((b >> 7) & 255);
    const int cc = int(abs_c >> 23);
    const int product_exponent = ca + cb - 254;
    // Even at the first normal word of a binade, the smaller halfway gap
    // around c is at least 2^(cc-127-25). Each BF16 magnitude, including
    // subnormals with field zero, is strictly below 2^(field-126).
    if (cc != 0 && product_exponent + 2 <= cc - 152) {
        counts += 1u << 16;
        return {c, 0};
    }
    // Both exact terms are on the 2^-126 grid and cannot overflow.
    // Opposite signs can cancel exactly to +0 under nearest-even, or
    // leave at least a minimum normal. The device witness tests zero.
    if (ca > 0 && cb > 0 && product_exponent >= -112 && product_exponent <= 124 &&
        (abs_c == 0 || (cc >= 24 && cc <= 252))) {
        return {as_type<uint>(fma(widen(a), widen(b), as_type<float>(c))), 0};
    }
    counts += 1;
    return exact_fma(a, b, c);
}

static Word ordered_add(uint a, uint b, thread uint& counts) {
    const uint am = a & positive_mask, bm = b & positive_mask;
    if (am == 0 || bm == 0) {
        counts += 1u << 16;
        return {am != 0 ? a : bm != 0 ? b : a & b, 0};
    }
    const uint ea = am >> 23, eb = bm >> 23;
    if (ea >= 24 && ea <= 252 && eb >= 24 && eb <= 252 && ((a ^ b) & sign_bit) == 0) {
        return {as_type<uint>(as_type<float>(a) + as_type<float>(b)), 0};
    }
    counts += 1;
    return exact_add(a, b);
}

static uint take(Word word, thread uint& flags) {
    flags |= word.status;
    return word.status == 0 ? word.bits : 0;
}

static bool normal_pair(uint a, uint b) {
    if ((a & 0x30000u) != 0x10000u || (b & 0x30000u) != 0x10000u) return false;
    const int minimum = int(a & 255u) + int(b & 255u) - 254;
    const int maximum = int((a >> 8) & 255u) + int((b >> 8) & 255u) - 254;
    return minimum >= -112 && maximum <= 109;
}

static float plain_dot(const device ushort* a, const device ushort* w, uint k, ushort lane) {
    float partial = 0.0f;
    for (uint i = lane; i < k; i += 32) {
        partial = fma(widen(a[i]), widen(w[i]), partial);
    }
    for (ushort offset = 16; offset >= 4; offset >>= 1) {
        const float other = simd_shuffle_down(partial, offset);
        if (lane < offset) partial = partial + other;
    }
    const float p0 = simd_shuffle(partial, ushort(0));
    const float p1 = simd_shuffle(partial, ushort(1));
    const float p2 = simd_shuffle(partial, ushort(2));
    const float p3 = simd_shuffle(partial, ushort(3));
    return ((p0+p1)+(p2+p3)) + 0.0f;
}

// Per-output body retained from the qualified scalar SIMD kernel.
static void original_one(
    const device ushort* input, const device ushort* weight, device atomic_uint* status,
    constant uint4& dims, device ushort* output, device uint* routes,
    const device uint* input_bounds, const device uint* weight_bounds,
    uint group, uint token, uint row, ushort lane
) {
    if (normal_pair(input_bounds[token],weight_bounds[row])) {
        const uint bits=as_type<uint>(plain_dot(input+token*dims.y,weight+row*dims.y,dims.y,lane));
        if (lane==0) {
            const ushort rounded=ushort((bits+0x7fff+((bits>>16)&1))>>16);
            if ((rounded&0x7fff)>=0x7f80) atomic_fetch_or_explicit(status,4u,memory_order_relaxed);
            else {
                output[group]=rounded;
                routes[group]=0x80000000u;
            }
        }
        return;
    }
    uint partial = 0, flags = 0, counts = 0;
    for (uint k = lane; k < dims.y; k += 32) {
        partial = take(ordered_fma(input[token * dims.y + k], weight[row * dims.y + k],
                                   partial, counts), flags);
    }
    for (ushort offset = 16; offset >= 4; offset >>= 1) {
        const uint other = simd_shuffle_down(partial, offset);
        if (lane < offset) partial = take(ordered_add(partial, other, counts), flags);
    }
    const uint p0 = simd_shuffle(partial, ushort(0));
    const uint p1 = simd_shuffle(partial, ushort(1));
    const uint p2 = simd_shuffle(partial, ushort(2));
    const uint p3 = simd_shuffle(partial, ushort(3));
    uint bits = 0;
    if (lane == 0) {
        const uint left = take(ordered_add(p0, p1, counts), flags);
        const uint right = take(ordered_add(p2, p3, counts), flags);
        const uint sum = take(ordered_add(left, right, counts), flags);
        bits = take(ordered_add(sum, 0, counts), flags);
    }

    // Routing census is observational and this correctness witness is untimed
    // for promotion. 7712 total operations per output cannot carry past 16 bits.
    const uint receipt = simd_sum(counts);
    const bool failed = simd_any(flags != 0);
    if (lane == 0) {
        const ushort rounded = ushort((bits + 0x7fff + ((bits >> 16) & 1)) >> 16);
        if (failed || (rounded & 0x7fff) >= 0x7f80) {
            atomic_fetch_or_explicit(status, 4u, memory_order_relaxed);
        } else {
            output[group] = rounded;
            routes[group] = receipt;
        }
    }
}



static void fma4(
    const device ushort* input, const device ushort* weights, device atomic_uint* status,
    device ushort* output, device uint* routes, ushort lane
) {
    float4 partial(0.0f);
    for (uint k=lane; k<7680u; k+=32u) {
        const float a=widen(input[k]);
        const float4 w(widen(weights[k]),widen(weights[7680u+k]),
                       widen(weights[15360u+k]),widen(weights[23040u+k]));
        partial=fma(float4(a),w,partial);
    }
    for (ushort off=16; off>=4; off>>=1) {
        const float4 other(simd_shuffle_down(partial.x,off),simd_shuffle_down(partial.y,off),
                           simd_shuffle_down(partial.z,off),simd_shuffle_down(partial.w,off));
        if (lane<off) partial=partial+other;
    }
    for (uint r=0; r<4u; ++r) {
        const float p0=simd_shuffle(partial[r],ushort(0));
        const float p1=simd_shuffle(partial[r],ushort(1));
        const float p2=simd_shuffle(partial[r],ushort(2));
        const float p3=simd_shuffle(partial[r],ushort(3));
        if (lane==0) {
            const uint bits=as_type<uint>(((p0+p1)+(p2+p3))+0.0f);
            const ushort rounded=ushort((bits+0x7fff+((bits>>16)&1))>>16);
            if ((rounded&0x7fff)>=0x7f80) atomic_fetch_or_explicit(status,4u,memory_order_relaxed);
            else {
                output[r]=rounded;
                routes[r]=0xc0000000u;
            }
        }
    }
}

kernel void lfm3_ordered(
    const device ushort* input [[buffer(0)]],
    const device ushort* weight [[buffer(1)]],
    device atomic_uint* status [[buffer(2)]],
    constant uint4& dims [[buffer(3)]],
    device ushort* output [[buffer(4)]],
    device uint* routes [[buffer(5)]],
    const device uint* input_bounds [[buffer(6)]],
    const device uint* weight_bounds [[buffer(7)]],
    uint group [[threadgroup_position_in_grid]],
    ushort lane [[thread_index_in_simdgroup]]
) {
    if ((atomic_load_explicit(status, memory_order_relaxed)&2u)!=0) return;
    if (dims.x!=1u || dims.y!=7680u || dims.z!=2048u || group>=512u) {
        if (lane==0) atomic_fetch_or_explicit(status,4u,memory_order_relaxed);
        return;
    }
    const uint row=4u*group;
    bool native=true;
    for (uint r=0; r<4u; ++r)
        native=normal_pair(input_bounds[0],weight_bounds[row+r]) && native;
    if (native) {
        fma4(input,weight+row*dims.y,status,output+row,routes+row,lane);
    } else {
        for (uint r=0; r<4u; ++r)
            original_one(input,weight,status,dims,output,routes,input_bounds,weight_bounds,row+r,0,row+r,lane);
    }
}

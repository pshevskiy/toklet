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

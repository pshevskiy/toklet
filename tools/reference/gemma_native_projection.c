/* Independent, bounded CPU replay of the declared packed affine SIMD order.
 * Not linked into Toklet. Compile without fast math or implicit contraction.
 * Storage: little-endian codes [rows, columns/8], BF16 scale/bias
 * [rows, columns/64], FP32 input [tokens, columns], output [tokens, rows].
 * The caller owns disjoint, sufficiently sized buffers for the entire call.
 */
#include <math.h>
#include <fenv.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>
#pragma STDC FP_CONTRACT OFF

static float bf16(const uint16_t *words, size_t index) {
    uint32_t bits = (uint32_t)words[index] << 16;
    float value;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

static void add(float *high, float *low, float value) {
    float next = *high + value;
    float correction = fabsf(*high) >= fabsf(value)
        ? (*high - next) + value : (value - next) + *high;
    *low = *low + correction;
    *high = next;
}

int gemma_native_projection(const uint32_t *codes, const uint16_t *scales,
                            const uint16_t *biases, const float *inputs,
                            float *outputs, size_t tokens, size_t rows,
                            size_t columns, size_t lanes) {
    if (!codes || !scales || !biases || !inputs || !outputs ||
        tokens == 0 || tokens > 32 || rows == 0 || rows > 128 ||
        columns < 64 || columns > 12288 || columns % 64 ||
        lanes == 0 || lanes > 64 || (lanes & (lanes - 1))) return 1;
    if (fegetround() != FE_TONEAREST) return 3;
    size_t groups = columns / 64;
    for (size_t token = 0; token < tokens; ++token) {
        for (size_t row = 0; row < rows; ++row) {
            float high[64] = {0}, low[64] = {0};
            for (size_t lane = 0; lane < lanes; ++lane) {
                for (size_t group = 0; group < groups; ++group) {
                    float scale = bf16(scales, row * groups + group);
                    float bias = bf16(biases, row * groups + group);
                    if (!isfinite(scale) || !isfinite(bias)) return 2;
                    for (size_t local = lane; local < 64; local += lanes) {
                        size_t column = group * 64 + local;
                        uint32_t packed = codes[row * (columns / 8) + column / 8];
                        uint32_t code = (packed >> (4 * (column % 8))) & 15;
                        float weight = fmaf(scale, (float)code, bias);
                        float input = inputs[token * columns + column];
                        float product = weight * input;
                        if (!isfinite(input) || !isfinite(weight) || !isfinite(product)) return 2;
                        add(&high[lane], &low[lane], product);
                        add(&high[lane], &low[lane], fmaf(weight, input, -product));
                    }
                }
            }
            for (size_t hop = 1; hop < lanes; hop *= 2) {
                for (size_t lane = 0; lane + hop < lanes; lane += 2 * hop) {
                    add(&high[lane], &low[lane], high[lane + hop]);
                    add(&high[lane], &low[lane], low[lane + hop]);
                }
            }
            float value = high[0] + low[0];
            if (!isfinite(value)) return 2;
            outputs[token * rows + row] = value;
        }
    }
    return 0;
}

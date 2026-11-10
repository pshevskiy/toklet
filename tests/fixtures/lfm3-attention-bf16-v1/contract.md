# LFM3 CPU eager attention qualification

Use the modified parent's pinned eager functions, with training and dropout
disabled, Torch 2.9.1 (5811a8d7da873dd699ff6687092c225caffcf1bb), on the
M1 Pro no-BFDOT CPU. This explicitly selects eager, batch one, 16 Q heads,
4 KV heads, width 128, full causal or window 256, no padding. Q is token-major
before transposition; K/V use per-head contiguous source rows. Source arithmetic
stores QK, scale, additive mask, probabilities and context in BF16. The mask
uses BF16 finite minimum, the scale is the FP32 Python scalar, and softmax
operates in FP32. Validate all raw Q/K/V values including masked keys, exclude
head-stride padding, and reject nonfinite QK, scale or context. Mask addition may
legitimately produce negative infinity. No cache publication or KV expansion.

Freeze bitwise BF16 correspondence for every stored stage, including masked
scores and final output. Check the existing 1e-5 + 1e-4*abs(reference) pre-store
FP32 diagnostic against independent FP64 QK, softmax on exact masked source
scores, and P@V on exact source probabilities. BF16 rounding is a distinct
boundary. Shape, span and finite-input failures must precede writes, with
arithmetic failure leaving discardable output. Zero heap allocations once
caller scratch and output exist. Compare exact segmentation variants on the
same logical operands, and exercise a wrapped real BF16 cache without publishing.
Use component release timings on the actual 16/4/128 shape for 1/32/128
query chunks and short/deep full versus window cases. Keep runtime admission
and all whole-model acceptance criteria unchanged.

Frozen deterministic input cases are in the independently authored Torch
producer. They include small BMM threshold at Q*K=3/4, vector reduction tails,
signed zero and subnormal terms, cancellation, mask boundaries and deep prefix
windows. Synthetic component inputs isolate arithmetic, not model quality.

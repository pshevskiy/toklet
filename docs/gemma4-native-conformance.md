# Proposed Gemma4 E2B Mac conformance at 512 positions

**Status: design awaiting specification and review.** This policy would test the
existing hybrid arithmetic on one named Apple Silicon Mac with capacity 512 and
chunks of 1–32. It covers the original Q4/group64/BF16 affine payload for
`mlx-community/gemma-4-e2b-it-4bit` at
`238767527555cb75a05732a84dff5d6ba0dd6809`. Gemma3 qualification does not
transfer. The 11 original-BF16 cross-precision failures, 27 short mathematical
logit failures, four long mathematical logit failures and eight long staged-V
observations remain rejected under their original contracts. Held-out quality,
product admission and the declared 131,072-position window have separate gates.

## Arithmetic to specify in the independent reference

- Reconstruct a weight by widening the group's BF16 scale and bias, unpacking
  low nibble first, and evaluating FP32 fused multiply-add with the unsigned
  code. For projection, each lane visits groups in order and columns at the
  queried lane stride. Accumulate the rounded product and its FMA residual
  separately using the shader's two-component compensated add. Merge both
  components at increasing power-of-two offsets and round their final sum.
  The token and PLE gathers use the same affine reconstruction. [Projection
  shader](../src/metal/gemma_affine.metal) · [Gather shader](../src/metal/gemma_affine/partitioned/gather.metal)
- Preserve every stored intermediate. The main embedding is multiplied by
  FP32-rounded sqrt(hidden) before projection. Keep the subsequent projection
  scale, per-layer RMS, lookup scale, addition and merge scale separate. RMS
  sums squares in FP64, uses the widened **FP32** epsilon `1e-6`, then divides
  and multiplies by the direct scale before an FP32 store. V uses unit scale.
  Residual additions and the layer scalar operate in FP32. Gated GELU computes
  its tanh approximation in FP64, stores the activation in FP32, then multiplies
  in FP32. The final softcap instead computes division by 30, tanh and
  multiplication by 30 in FP64 with **one** final FP32 store. [Host graph](../src/model/gemma_packed/forward.rs) · [PLE](../src/cpu/per_layer.rs) · [Softcap](../src/cpu/softcap.rs)
- RoPE uses absolute positions and half-head pairing. The frequency exponent
  uses the full head dimension. Local heads rotate 256 of 256 dimensions at
  theta 10,000. Global heads rotate 128 of 512 at theta 1,000,000. Angles and
  sin/cos are FP64, trigonometric values are cast before FP32 products and
  sums, and untouched dimensions are copied. Preserve all 15 KV producers
  and consumer assignments. Attention uses scale one, causal/global or
  causal/window-512 inclusion, ordered FP64 QK sums stored in FP32, FP64
  max-shifted exponential normalization stored in FP32, and ordered FP64
  weighted-V sums stored in FP32. [RoPE](../src/cpu/proportional_rotary.rs) · [Span attention](../src/cpu/span_attention.rs)

**Blocking prerequisites.** Before observing fresh target arrays, record the
actual pipeline SIMD width, compiler settings (including Metal 2.3 safe/precise
projection), gather settings, device, binary and runtime. Specify rounding,
contraction, subnormal handling and a justified host math-provider contract
for each affected operation. Implement an independent original-byte reference
with its own unpacking, reductions, provider checks and causal propagation.
Validate on hand-computable and tiny whole graphs. An unsupported provider
bound, unverified reduction tree or ambiguous expected row blocks collection.
Do not fit coefficients to earlier failures or select a reference center using
target intermediates.

## Frozen comparisons for a fresh run

1. Use `[2,100,257,100]` and the authenticated first 512 frozen corpus IDs.
   Compare all 262,144 logits at **every** short position and at absolute
   positions **0, 31, 255, 433 and 511** on the long input. Compare the full
   normalized owner-13 V row at position 433 against the independently
   propagated reference. Require every finite coupled value to satisfy
   `abs(actual-reference) <= 1e-5 + 1e-4*abs(reference)`. No excluded IDs,
   coordinates or margin exceptions. [Short discrepancy](gemma-precision-localization.md) · [512 fixture](gemma-hybrid512.md)
2. Separately replay the short tied head from the actual final-normalized
   inputs and the softcap from actual uncapped logits. At owner 13/433 replay
   the complete V projection from the actual normalized input, RMS from actual
   raw V, and compare retained V bytes exactly. Require exact deterministic
   stores where specified and pre-reviewed provider semantics elsewhere.
   These witnesses localize defects and cannot replace the coupled gates.
   [Existing V witness](gemma-v-boundary.md)
3. Reuse the two 512-token partitions, `16×32` and `1+16×31+15`. Require
   bit-identical sampled rows and retained KV across schedules, exact cache
   owners, positions and slots, correct commit epochs, unchanged state after
   rejecting token 513, and reset/fresh-prefix agreement. This domain never
   evicts a local position. Keep the separate wrap rejection on record.
4. Require cold controls that detect a swapped nibble, lost product residual,
   wrong RMS epsilon, shifted/shared V slot and misplaced softcap rounding.
   Require the checker to reject missing rows, nonfinite values and coherent
   corruption of a coupled row even if state and local checks pass.

Seal inputs, source ranges, operator specification, checker, selected rows,
model/config/index and runtime hashes before the fresh run. Bound captures,
require clean worker exits and matching endpoint identities, and measure RSS
and physical peak below 6 GiB per worker. Review raw evidence and all failed
coordinates before a scoped conformance decision. A passing result could
support a later audited Mac owner and app journey. Unsupported arithmetic or
a failed coupled value leaves that decision open.

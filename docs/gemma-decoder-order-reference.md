# Gemma dense decoder order witness

This independently freezes the dense `DecoderLayer.__call__`,
`MLP.__call__` and compiled `geglu` order before native layer composition.
The pinned MLX-LM source is revision
`0d4d3d04f8ab7ce4a51e7209e8615755a48ba94d`; `gemma4_text.py` hashes to
`73ffa75e9069ce7b2f864ccd8f0267bb80c5b5d4e2b73102b8a2a93fe03f936d`.
The [acceptance receipt](../benchmarks/results/2026-10-24-gemma-decoder-order-reference-v1.json)
binds the fixture, raw observations, commands, runtime, replay and review inputs.

## Numerical contract

Inputs and weights are FP32. Hidden states and supplied attention outputs use
`[batch,tokens,H]`, FFN gate/up/down weights use `[I,H]`, `[I,H]`, `[H,I]`,
and PLE gate/projection weights use `[P,H]`, `[H,P]`. The already composed
per-layer input is `[batch,tokens,P]`. Norm weights are `[H]`; a layer scalar
has shape `[1]`. All matrix storage is row major.

The exact order is:

1. Normalize hidden state with direct learned RMS weights and pass it to attention.
2. Normalize the supplied attention result, then add the original hidden residual.
3. Pre-normalize that sum; project gate and up; apply tanh GELU to gate, multiply
   by up and down-project. Post-normalize, then add the attention residual.
4. If all three PLE modules and the per-layer input exist, gate the updated
   hidden state, apply tanh GELU, multiply by PLE, project to H, post-normalize,
   then add the saved FFN residual.
5. Multiply the entire result by the learned layer scalar when present.

The FP64 ideal and staged FP32 contracts are separate. The staged contract uses
FP64 linear products, RMS reductions and tanh arithmetic, with FP32 storage after
each operator, residual addition and product. It is a readable numerical contract,
not a statement about a backend's internal reduction order or fusion. RMS uses
`mean(x*x)+f32(1e-6)` and direct learned scale, without adding one to weights.

The fixture has12 bounded cases: empty no-call, hand, batched, distinct small
FFN widths6/12, each missing PLE guard operand, all PLE modules disabled,
synthetic absent scalar, H1536 with small I/P, and P256 with small H/I. Negative,
unequal norm weights, distinct nonzero residuals and scalar−1.25 expose misplaced
normalization, gates and scalar multiplication. These split shapes do not model
one complete E2B layer. Scalar absence and partially absent PLE modules exercise
call guards; they do not claim valid checkpoint configurations. Constructor/config
selection of double-wide FFNs remains unqualified.

## Source and evidence boundary

The unchanged selected source bodies execute with real FP32 MLX Linear, RMSNorm
and GELU modules on explicit CPU and GPU devices. The compiled `geglu` decorator
is retained; only annotations are deferred. An attention stand-in supplies a
fixed result, records the actual normalized input, checks all incoming argument
identities and returns distinct shared-KV/offset sentinels. Source output must
forward that returned tuple unchanged. This proves decoder call order and
forwarding, not attention mathematics, cache operations or rotary precision.

Separate stage probes are evaluated and retained before later fallible operations.
They are not internal captures of the unchanged helper's lazy/fused execution.
The unchanged MLP output and complete decoder result are also captured. Empty
sequences use the mathematical no-call policy. Raw arrays, partial failure logs,
owned process cleanup and source/protected-binary guards remain in the capture.

An independent stdlib `fsum`/`sqrt`/`tanh` replay verifies all540 arrays and75,720
values. Staged FP32 replay is exact; the largest ideal replay difference is
1.3642420526593924e−12, within1e−13+1e−13×abs(expected).
Each CPU/GPU capture retains201 raw arrays and passes402 comparisons against
both contracts. Maximum error/policy ratios are0.0486313/0.0588058 CPU and
0.0174533/0.0275974 GPU for ideal/staged contracts. Eight portable, three owner
and three runtime-admission tests and all six owned commands pass.
The unchanged upstream comparison policy is1e−5+1e−4×abs(expected).

The [producer](../tools/reference/gemma_decoder_order.py),
[portable regressions](../tools/reference/test_gemma_decoder_order.py),
[owned controller](../tools/experiments/gemma_decoder_order_reference.py) and
[owner regressions](../tools/experiments/test_gemma_decoder_order_reference.py)
are versioned. This witness does not qualify MoE, BF16/quantized weights, a
native layer/session, full model, device fit, quality, performance or long replies.
The eight [long rotary discrepancies](gemma-rotary-precision.md) remain unwaived.

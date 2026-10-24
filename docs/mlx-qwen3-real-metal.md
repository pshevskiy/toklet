# Real Qwen3 native Mac execution

24 October 2026. Toklet's resident Rust/Metal graph now runs the inspected
[Qwen3-0.6B eight-bit checkpoint](mlx-qwen3-checkpoint.md) on Apple M1 Pro.
The [independent real reference](mlx-qwen3-real-oracle.md) was frozen before
target edits; its manifest and numerical gates remain unchanged.

## Admission, storage and execution

The existing [Qwen3 config module](../src/mlx/qwen3.rs) owns graph and artifact
validation. [Metal admission](../src/metal/mod.rs) adds one exact geometry:
hidden 1024, FFN 3072, 28 layers, 16 query/eight KV heads of width 128, vocabulary
151936, theta 1e6 and epsilon 1e-6. Context is capped at 512; the source's 40960
positions are not admitted generally. Tiny Qwen3 admission is preserved, and
unqualified geometry rejects before GPU creation. Sliding windows, alternate
rotary schemes, biases and Qwen3 MoE do not gain support.

The shared graph and existing kernels need no changes. Packed matrices and BF16
affine metadata remain resident; floating BF16 norm vectors convert once to
Half. Immutable Q/K norm vectors apply per head before RoPE. Query/context width
2048 is independent of hidden width 1024, including the output projection back
to hidden space. Tied embedding/output storage occurs once. Resident weights
total 633,366,062 bytes: exact payload plus seven Half guards on each of 113 norms.
Session KV and scratch remain separately owned and bounded. Reference attention
is required; Online and implicit default-session requests reject explicitly.

This is architecture admission and functional qualification, with no new shader,
fusion, generic framework, per-element dispatch or numerical-policy change.
The installed apps and their three download profiles are unchanged.

## Completed functional checks

The [real-model suite](../tests/mlx_qwen3_real_model_metal.rs) passes 912 exposed
stage snapshots/14,276,352 scalar comparisons across six authored cases, chunk
sizes 1/3/full, rotary offsets 0/5 and selected layers 0/1/27. Raw Q/K projections,
per-head normalization and rotary outputs are included. Attention scores and
probabilities are not exposed by the target diagnostic interface; context and
subsequent graph boundaries are compared. Maximum absolute stage error 1.0
passes the unchanged per-stage limits. Logit comparisons never inherit that
larger stage allowance.

Both full-logit and token-only readbacks select `[1096,374,264,3491]` exactly on
the predeclared independent FP64 teacher history. Four greedy full distributions
and three scoring-prefix distributions pass their frozen gates; maximum observed
decode/scoring logit error is 0.03388547897338867. Token-only publishes no full
logits. Pending-token rejection, fully cached prompts, edited-prefix replay,
reset, partial prefill and capacity overflow preserve their expected contracts.

An actual 512-capacity/chunk128 session commits 128 input positions, discards the
384-token pending tail, restores it and completes 512 positions. Append and
replacement overflow leave the exact committed history and output unchanged.
A fresh session with the same recipe reproduces the complete output distribution
exactly. This is partial-prefill discard/resume, not generation-callback
cancellation, physical lifecycle testing or an independent 512-position numerical
oracle. Independent numerical coverage remains the short cases through 17 tokens
and offset 5.

Exact session admission and limit-minus-one rejection pass for both readbacks.
Diagnostic admission likewise accepts its exact limit and rejects one byte below
on an empty fresh session without changing admitted state. Full-logit session
and diagnostic admissions are 707,649,798/712,997,036 bytes; token-only values are
706,434,350/711,781,588. Maximum short diagnostic admission is 747,182,964 bytes;
the 512 state witness admits 853,205,778 bytes. These are logical numerical
budgets, not measured process peaks, whole-app memory or phone-fit results.

The [receipt](../benchmarks/results/2026-10-24-qwen3-real-metal-v1.json) records
Rust 1.90 formatting, strict all-target default/Metal Clippy, 162 CPU passes
(13 ignored) and 181 Metal-feature passes (99 ignored). All 18 explicit optimized
GPU tests pass: four real Qwen3, three tiny Qwen3, three Qwen MLX, two tiny Llama,
legacy Half/Q4 release checks, the shared native bridge and three CLI cases.
Protected release binaries, shaders and Swift remain unchanged. The native/CLI
regressions here use existing tiny fixtures; real app/text integration remains
separate. Astra reviewed source and the bounded trial owner before execution.

Retain selected real Qwen3 Reference execution on Mac. Held-out quality,
representative/sustained performance, actual generation cancellation, broader
context numerical qualification, tokenizer/thinking/chat semantics, app catalog
integration and iPhone trials remain open in the
[active plan](engine-expansion-plan.md).

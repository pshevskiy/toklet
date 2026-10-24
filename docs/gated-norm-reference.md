# Independent gated output normalization reference

This witness fixes mean RMS normalization, direct learned scale and subsequent
SiLU gate multiplication for the inspected Qwen3.5 linear-attention output.
It is a small operator reference; a native composition, hybrid graph or new
model has not been qualified by this task.

Hidden states and gates are FP32 `[tokens,heads,width]`; learned weight is FP32
`[width]`, shared across rows/heads. The independent NumPy producer widens to
FP64, divides each row by `sqrt(mean(hidden²)+epsilon)`, multiplies the learned
weight directly, then multiplies stable SiLU of the gate. This norm has no
`1+weight` conversion and does not normalize after gating. The five tiny cases
cover empty input, scalar, asymmetric heads/weights, epsilon-dominated input and
128-wide rows. Saturated and tiny negative gates are included.

The inline JSON format is bounded to 256 KiB and requires an external file hash.
It validates exact cases/fields/shapes, finite numbers, non-Boolean dimensions
and exact FP32 representability of input values. Producer/runtime identity and
scope flags are explicit. The frozen fixture is
[math.json](../tests/fixtures/gated-norm-v1/math.json).

Separate completed CPU/GPU processes authenticate the three pinned MLX-LM
sources and actual NumPy/MLX/native/Metal runtime before numerical imports.
Only `precise_swiglu` and `Qwen3NextRMSNormGated` source definitions execute.
The original Qwen3.5 file is an authenticated alias/layout witness, not a model
execution. Each comparison retains normalized output, SiLU gate and gated output
immediately before constructing later calls. Every numeric array is saved before
dtype/shape/finiteness checking; failure preserves preceding observations.
Empty input makes no upstream call. The ungated class branch supplies an
additional normalization witness; FP32 gated execution is the target convention.

Three ownership, eight portable mathematical/format/failure and three runtime
admission tests pass. CPU and GPU each pass all 12 comparisons at the unchanged
`1e-5 + 1e-4*abs(expected)` policy. Maximum error/policy ratios are approximately
0.001797 CPU and 0.001493 GPU. These are tiny numerical observations on the Mac,
not timing, memory, quality or device qualification. Exact commands, raw arrays,
logs, captured source, runtimes and endpoint guards are bound by
[the receipt](../benchmarks/results/2026-10-24-gated-norm-reference-v1.json).

The existing native RMSNorm and gated-SiLU functions may be reusable, but their
composition still requires its own frozen-fixture checks. New graph/session,
Metal scheduling, model/chat admission and long-context/device work remain open.
The [current priorities](gemma4-implementation-plan.md) are Gemma4 support and
131072-token replies, with their separate context and memory gates.

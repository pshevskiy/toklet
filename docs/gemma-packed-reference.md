# Gemma packed table/projection reference

The [receipt](../benchmarks/results/2026-10-25-gemma-packed-reference-v1.json)
freezes a small independent four-bit/group64/BF16-coefficient witness before
native table/projection composition. No checkpoint payload is read. The existing
scalar affine reference accepts FP32 inputs and reconstructs FP32 weights; the
existing Half-activation Metal path cannot establish Gemma's BF16 behavior.

## Shapes and contracts

Three-row tables have widths64,128,1536 and8960. The latter two match E2B's
main embedding and35×256 per-layer embedding strides, with a tiny synthetic
vocabulary. Literal low-bit-first packed patterns cover every unsigned code0..15
and byte/word/group/row boundaries. Per-row/per-group BF16 scales and biases
include unequal, negative and zero scales. Lookup IDs2,0,2,1 check repeated and
reordered rows. Two projection rows contain FP32 values that also exercise BF16
input rounding. Arrays are contiguous row-major; packed U32 storage is
`[3,width/8]`, metadata `[3,width/64]`, inputs `[2,width]`, gathers `[4,width]`
and projection outputs `[2,3]`.

The standard-library producer freezes distinct contracts:

- Exact BF16 coefficients promoted to FP32, affine multiply/add in FP64, then
  one FP32 reconstruction; ordered FP64 dot products with FP32 output.
- Ideal affine/projection math with stable FP64 summation.
- BF16 gather alternatives: one final BF16 rounding, or BF16 product rounding
  followed by BF16 addition rounding.
- A diagnostic projection over once-BF16-rounded weights and BF16-rounded
  inputs, accumulated in ordered FP64 and rounded to BF16.

Those alternatives describe different arithmetic. Agreement with one does not
imply agreement with the others or qualify a complete checkpoint's precision.
The frozen [ledger](../tests/fixtures/gemma-packed-v1/math.json) binds the
[raw arrays](../tests/fixtures/gemma-packed-v1/math.bin); its binary is616,964
bytes. The source producer and independent replay are bound in the receipt.

## Actual MLX observations

The owned runner admits the installed MLX0.32.3/NumPy2.3.2 source/native/Metal
runtime before importing it, then checks actual module import origins. Actual
`QuantizedEmbedding` and `QuantizedLinear` constructors are used with exactly
the frozen packed parameters. Promoted-FP32 and original-BF16 metadata/input
paths are separate. Embedding calls, linear calls and the embedding's tied
`as_linear` calls run sequentially; completed raw bytes and actual dtype/shape
are retained before validation. Parameter identities remain unchanged.

Each backend captures24 arrays/85,600 values in256,800 bytes, with60 comparison
rows. All28 required FP32/tied comparisons per backend pass: FP32 gathers are
exact; projections use unchanged `1e-5 + 1e-4×abs(reference)` gates; tied and
linear outputs are identical within each precision/backend. Maximum required
gate ratios are0.0465303 on CPU and0.00501317 on GPU. These are numerical
comparison ratios, not speed measurements.

BF16 results are retained as diagnostics. CPU gathers exactly match separate
product/add rounding at all four widths; GPU gathers exactly match once-rounded
affine reconstruction. Their differences affect4,24,752 and4336 gathered values
at the respective widths. Each backend has28 rejected diagnostic comparisons,
including BF16 projections against the FP32/rounded-weight alternatives. The
worst diagnostic gate ratios are4781.76 on CPU and280.402 on GPU. No tolerance
is relaxed and neither BF16 checkpoint execution nor CPU/GPU equivalence is
qualified. Capture completion and numerical agreement are separate fields.

## Checks and remaining work

The final six owned commands pass: seven portable tests, three owner tests,
three runtime-admission tests, export, CPU capture and GPU capture. Mocked
construction/evaluation failure, wrong shape/dtype, nonfinite output and
interruption preserve earlier and rejected current raw observations. Source-copy
substitution and failed/raised child cases retain logs and failed qualification.
All captured inputs/outputs have execution endpoint guards and clean teardown.

The first preliminary portable run rejected an insufficient witness: its
width64 dyadic scales happened to reconstruct exactly in BF16. That failure and
the matching source snapshots remain unaccepted. Before freezing any fixture,
non-dyadic scale seeds were chosen to exercise the required rounding distinction;
the final gate is unchanged. The subsequent preliminary tests and final owned
checks pass. Rust/shader/header inputs remain unchanged from the checked adapter
commit; this task changes reference tooling/fixtures/docs only.

Next, compose packed table reads and projections into a bounded native scalar
path without full-model expansion. Actual payload integrity/value admission,
general graph/session/tokenizer/Chat semantics, resident Metal execution,
real-model quality, performance and independent Mac/phone fit remain open.
The earlier15 short-graph GPU and8 long-rotary discrepancies remain unwaived.

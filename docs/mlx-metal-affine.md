# Resident Metal MLX affine operators

The opt-in [operator surface](../src/metal/mlx.rs) and
[separate shader library](../src/metal/mlx.metal) implement packed matrix products
and embedding gathers on Apple Silicon. These are numerical building blocks for
M8.2. They do not yet execute a complete Qwen graph or qualify model quality,
end-to-end latency, mobile memory or any additional architecture.

## Layout and execution

The [MLX storage layout](mlx-affine.md) remains unchanged. Original U32 weights,
scales and biases reside in three separate shared Metal buffers; metadata buffers
round up to four bytes, with initialized zero padding. F16/BF16 metadata uses
16-bit indexing and FP32 metadata uses 32-bit indexing. Neither embeddings nor
projection weights expand into a retained floating table. A context caches 18
bit-width/metadata-dtype pairs, each with a matrix-product and gather pipeline.
Existing engine defaults keep their original shader library and pipelines.

A product consumes contiguous Half `[batch, columns]` activations and a separate
Half linear-bias vector, producing contiguous Half or FP32 `[batch, rows]` output.
It assigns one queried SIMD group to each output row/token; supported reduction
widths are positive powers of two. Width is queried rather than hardcoded.
Gathers map token IDs to packed embedding rows and validate IDs before submission
and again in the shader before weight access. Packing and metadata specializations
are selected outside numerical loops.

Each weight reconstruction uses explicit FP32 fused multiply-add. Products retain
their FMA residual, with two-component compensated per-lane accumulation and SIMD
reduction. Before Half conversion, rounding to an odd FP32 significand preserves
the side of a Half midpoint; FP32 outputs keep ordinary nearest rounding. This is
a separate precision contract from both native-dtype MLX kernels and the scalar
reference's FP64 affine multiply/add. Exact dyadic fixture agreement is not a
universal bit-equivalence claim for arbitrary metadata or complete models.

## Restricted GPU value domain

The first actual M1 Pro trial exposed FP32 subnormal flushing despite Safe/Precise
compilation: a `0 * code + minimum FP32 subnormal` gather returned zero. The
original failing source and raw output are retained. Its scalar expectation and
the broader CPU storage contract remain unchanged.

GPU upload now rejects FP32-subnormal scale/bias values, including BF16 values
that become FP32 subnormals. For **every possible code** in each group, fused
reconstruction must be finite and either zero or have magnitude at least
`2^-79`. This conservative rule includes unused codes. It can reject artifacts
whose actual code stream would be safe; a more permissive rule needs separate
implementation and qualification.

Every admitted nonzero binary32 weight is a multiple of `2^-102`. Half inputs are
multiples of `2^-24`; exact products, residuals and reduction differences therefore
lie on the `2^-126` lattice, which is FP32 normal or zero. Half bias values also
lie on that lattice. Overflow is still rejected at runtime. Metadata/domain
validation occurs once before upload, scans groups/codes without retaining
expanded weights and does not change the scalar parser's accepted value domain.

## Ownership and failure boundaries

The context owns device/queue/pipelines; an affine operator borrows that context,
owns immutable packed buffers and reuses admitted input, bias, output, token and
status buffers. Original CPU payload slices need not remain live after upload.
The public surface is a bounded operator probe, not a full-model/session wrapper.
Each call creates command resources, uploads its small changing operands and uses
one checked completion boundary. No allocation-free steady-state claim is made.

All buffers remain live through command completion; CPU payload access occurs
before submission or after the checked wait. Output status and guard regions are
validated before writing any caller output. Shape/input/domain errors, nonfinite
results and command failures preserve caller output. Operators have no KV prefix
to commit and can reuse scratch after a completed nonfinite result.

Admission counts actual rounded packed-buffer sizes, both output precisions,
input/bias/tokens/status and guard allocations, plus a 64 MiB library/metadata
estimate within the caller's initial Mac budget. This is not measured process
peak, total driver memory or an iPhone budget. The later model/session integration
must reuse shared scratch and admit its own KV, tokenizer and application state.

## Validation scope

The [explicit GPU tests](../tests/mlx_metal.rs) use immutable hashed MLX storage
fixtures and independent ordered FP64 products over reconstructed FP32 weights.
The 54-case Cartesian set covers every supported width/group/metadata dtype,
batches 1/3/9/1, both output types, resident reuse and output guards. Additional
cases cover three-row F16/BF16 metadata padding, shrinking batches, both signs and
both sides/exact midpoint for Half rounding, failure/reuse, Half subnormal
conversion, restricted-domain rejection and the exact FP32-normal residual floor.
Ordinary CPU/Metal-feature tests remain independent of GPU availability.

The final M1 Pro release run passes all four explicit tests, including 864
completed fixture calls and 390,096 exact value comparisons. Two focused admission
tests, formatting, strict CPU/Metal-feature Clippy, 126 CPU tests (8 ignored),
140 Metal-feature tests (63 ignored) and offline fixture hashes pass. Protected
historical CPU binaries retain their hashes. The
[task receipt](../benchmarks/results/2026-10-22-native-mlx-metal-affine-v1.json)
binds sources, checks, original failure and raw results. No timing benchmark or
model/phone qualification is inferred from these functional checks.

Complete model assembly, strided execution integration, state/quality/memory
qualification, CLI/native-app selection and physical iPhone checks remain open
in the [M8 plan](mlx-implementation-plan.md).

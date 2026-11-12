# Bounded Granite mixed affine conversion

Status: conversion, saved-byte checks and Astra review pass; no actionable findings. This
produces a candidate for the selected Heretic Granite 4.2 3B track. Full-model
execution, codec, numerical/quality, physical-fit, request/overhead and managed
adoption gates remain open.

The input is `saidutta69/granite-4.2-3b-heretic` at
`db83ced6c830ad52b32880d026a4f8e3f888c316`. Both complete BF16 shards authenticate
against pinned source hashes while their actual tensor rows are consumed. The
7,319,475,200-byte source body exceeds the engine ceiling; the producer streams
it without constructing a model. The original IBM index still rejects because
its declared byte total is inconsistent. No source index is repaired.

The explicit group-64 recipe has 201 affine4 matrices, 81 affine8 matrices and
81 unchanged BF16 norm vectors. Output/down projections and the untied head
use eight bits. Packed codes are little-endian U32, low bits first; scale and
bias arrays are separate BF16 tensors. Each row is contiguous, groups run along
the input dimension, and the safetensors body starts at an eight-byte boundary.
The child body is 2,737,853,440 bytes; the complete file is 2,737,958,760 bytes.
Its SHA256 is
`02f938b9f2a82d61c8609e19d2d65fcd7b32e52cc5b1cc7834b9843b9c4eec1e`.

The cold Python producer uses at most 256 complete rows and 8,192 columns per
tile, with at most 4 MiB of BF16 input. It admits the complete source/runtime
inventory before importing NumPy 2.3.2 and MLX 0.32.3. Runtime hashing reads at
most 1 MiB at a time, checks regular-file descriptor/path identity, and repeats
inventory, hashes and actual module/native-image origins at the endpoint.
Existing MiniCPM5 helpers and their format/shape contracts stay unchanged.

Every affine tile completes GPU quantization and readback before an independent
NumPy FP32 decoder extracts codes, expands BF16 metadata, multiplies and adds.
Its bytes must exactly equal upstream MLX dequantization with FP32 metadata.
Every output component is written and read back, then hashed as a whole tensor.
Norms are copied exactly, finite values are required, and complete parent and
child hashes close before the data file is promoted from its partial name.
Ancillary metadata and config are written afterward; a failed run remains an
unqualified partial candidate and must use fresh output/evidence paths on retry.
The config adds only the explicit storage recipe and an exact legacy default
RoPE theta alias for the reference adapter. No tokenizer payload is qualified.

The full run covers 4,704 tiles, 282 matrices, 81 norms and 3,659,530,240 matrix
values with exact independent/upstream FP32 reconstruction. A separate saved-data
audit rereads actual source/child tiles, all norms, whole objects and journal
coverage. That offline audit reconciles the captured floating-point comparisons;
it does not recompute them. The authenticated conversion manifest binds the
runtime, source, child and individual tensor/component hashes.

All 48 CPU/GPU literal and maximum-width controls pass, including zeros,
constants, low-bit packing, rounding ties, negative-only groups and BF16 values
above FP16 range. Device-specific expected bytes were frozen before execution.
Thirty-three portable tests pass in normal and optimized Python. The full
conversion's largest observed charge is 455,131,136 bytes, including the 32 MiB
helper reserve, below the separate 512 MiB setup/audit cap. The monitor is sampled
and retains conservative lifetime/cohort charges; it is not a continuous allocator
cap. No full model or compiler ran alongside this conversion.

Six checks use the previously checked Metal-enabled CLI against the actual child.
Metal metadata admits contexts 128 and 512; 1,024 correctly rejects at the
existing 512-position implementation limit. Scalar FP32 expansion exceeds 6 GiB
at all three contexts. Metadata acceptance does not prove GPU loading, memory
fit or successful inference. Unchanged Rust/dependency/fixture inputs allow reuse
of the prior fmt, strict CPU/Metal Clippy, 737 CPU/883 Metal tests and explicit
Granite GPU fixture; the reuse receipt records exact checked input identities.

The earlier whole-file runtime hashing attempt exceeded the audit cap despite
passing numerical controls. Streaming hashing preserves all non-timing case
fields. The initial index preflight incorrectly excluded the publisher's optional
parameter count; it now validates that strict integer against the BF16 byte
total. An initial CLI expectation incorrectly assumed 1,024-position Metal
support; correcting the expectation preserves the engine's original rejection.
All failed receipts remain retained.

Conversion clocks are diagnostic phases, not engine throughput or speedups.
No quantization-quality, full-model fit, summary-overhead or adoption claim
follows. The next gate is real-model source/child graph validation within 6 GiB,
followed by the original quality, session, request and overhead campaigns.

[Portable report](../benchmarks/results/2026-11-12-granite-mixed-affine-v1.json),
[frozen contract](../artifacts/granite-conversion-2026-11-12/prospective-conversion-contract-1.json),
[full conversion](../artifacts/granite-conversion-2026-11-12/full-conversion-1/manifest.json),
[saved-data audit](../artifacts/granite-conversion-2026-11-12/full-audit-1.json),
[actual CLI checks](../artifacts/granite-conversion-2026-11-12/child-inspection-2.json),
[Rust check reuse](../artifacts/granite-conversion-2026-11-12/rust-check-reuse-2.json).

Quantization source semantics are inspected from the pinned
[MLX CPU implementation](https://github.com/ml-explore/mlx/blob/v0.32.3/mlx/backend/cpu/quantized.cpp)
and [Metal kernels](https://github.com/ml-explore/mlx/blob/v0.32.3/mlx/backend/metal/kernels/quantized.h).
The [configuration milestone](granite-config.md) separately records strict source
admission and tiny-graph qualification. Full logs and retained failures stay in
the linked artifact evidence. [Review/commit](../artifacts/granite-conversion-2026-11-12/publication-checks.json).

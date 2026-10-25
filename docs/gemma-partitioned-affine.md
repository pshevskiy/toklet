# Resident packed projection chunks with transactional publication

`Context::partitioned_affine` builds a concrete `PartitionedAffine` from borrowed
original packed matrix rows. The format remains packed4/group64/BF16, with FP32
activations and outputs. Columns remain bounded by the existing affine operator;
the new parent allows at most 262144 output rows, 64 chunks, 16384 rows per chunk
and 32 token slots. It does not raise the existing single-owner geometry limit.
The [independent tiny table witness](gemma-table-reference.md) was frozen before
the native implementation, and [immutable row views](mlx-row-views.md) preserve
the original source bytes without reconstructing a dense table.

Each child owns its resident packed codes, scales, biases, input, output and
status buffers. All children borrow one Context with cached device, queue and
pipelines. Original source slices are needed only during construction; the parent
retains neither an extra CPU model nor expanded dense weights. There is no
architecture or dtype selection within a numerical inner loop.

The cold plan checks every child layout and size and the aggregate allocation
before numerical allocation. It includes each child's six buffers, padding and
guards, the live child structs exactly once, unused owner-vector slots, the
parent struct and both CPU staging arrays at their actual allocated capacities.
Full staging is `[capacity, total_rows]`; reusable chunk staging is
`[capacity, largest_chunk_rows]`. Actual reservations are admitted again before
upload. Every device buffer length and the complete coefficient codebook are
checked before GPU upload. The chunk count is bounded before materializing owners.

This is owner-only accounting. The caller must separately admit the Context
allowance, caller output and source/loading overlap, and every other live model,
session and application allocation. Individual chunk fit is insufficient to
admit a complete model. This task does not qualify actual full-head residency or
a mobile memory budget.

`apply` validates shape and finite input and performs the existing numerical
admission for every child before staging begins. A later child's rejection thus
preserves all warmed device buffers, CPU staging and caller output, and the owner
remains reusable. Once execution begins the parent is marked poisoned. Each
completed child writes reusable chunk scratch; token-strided copies place its
rows in full staging. Caller output is copied only after all children succeed.
An error after execution begins leaves the caller untouched and the complete
owner poisoned, retaining its child resources until teardown. The existing
child command-completion/lifetime rules are unchanged.

This composition deliberately waits and reads back per chunk. It establishes
ownership and publication correctness; it is not an optimized GEMM, fused head,
resident full graph, or latency improvement claim.

The [qualification receipt](../benchmarks/results/2026-10-25-gemma-partitioned-affine-v1.json)
records these checks on Apple M1 Pro, MacBookPro18,3 with 16 GiB RAM and
macOS 26.6.2 (25G83), SIMD width 32:

- Eight tiny source/partition configurations, ten owner lifetimes and sixty
  execution/rejection steps, with 1501 successful output values checked against
  independent exact-rational and ordered FP64 computations rounded to FP32.
- Complete captures of all six buffers for every live child, original byte
  spans, odd BF16 metadata padding, guards, inactive capacity, token strides,
  full CPU staging and reusable chunk scratch. Allocation identities remain
  distinct within each live owner and stable across calls.
- Repeated full batches, three single-token partitions and full-batch resume;
  adjacent aggregate-budget rejection; domain and invalid-input rejection;
  later-child numerical rejection before either child stages; and recovery.
- A test-only composition failure after two children complete and change
  staging. The caller stays untouched, retry is blocked with identical retained
  state, and a fresh owner with the same geometry reproduces the baseline.
  This does not inject a driver or in-flight GPU failure.
- Twenty-two reader and actual-capture mutation tests reject corrupt source
  bytes/padding, guards/status, allocation changes, wrong results and strides,
  missing outcomes, incorrect partial staging, caller publication and lost poison.

Formatting, strict lint with and without Metal, 351 CPU tests, 381 Metal-feature
tests and both iOS compile targets pass. An optimized aggregate-plan unit test
and all three optimized row-view fixture tests pass. Ordinary CPU tests require
neither a model download nor an Apple GPU. The exact release diagnostic is
snapshotted and all owned commands/process groups are closed. Native diagnostic
maximum RSS was 25772032 bytes and peak physical footprint 15188568 bytes;
replay and regression stayed below their separate 512 MiB allowances. These
whole-process resource observations include diagnostic work and do not measure
inference latency or full-model residency.

The original single-owner full-head geometry rejection remains intact. Actual
262144-row tied-head execution, large per-layer table gather, source/loading
overlap and model-wide admission, resident graph/KV/scheduling, numerical/source
discrepancies, representative generation performance, app and physical iPhone
qualification, and 131072 output tokens plus a nonempty prompt remain open.
No historical source or precision rejection is waived and no app capability or
context limit changes.

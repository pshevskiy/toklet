# Transactional scalar gathers from original packed tables

`Matrix::gather_rows(ids, output)` reconstructs 1–32 selected rows into contiguous
FP32 caller storage `[ids.len(), columns]`. It borrows the same immutable packed
codes, scales and biases as `copy_row`, preserving repeated IDs and request order.
It allocates no scratch, expands no whole table and adds no owner or unsafe code.

Validate the nonempty bounded request, checked output-size multiplication, exact
destination shape and every row index before numerical work. Select the metadata
dtype and packed width once, outside the row/scalar loops. Preflight every
selected row for finite reconstruction, then publish all rows in a separate
infallible pass over the immutable source. A bad later index or overflowing later
reconstruction leaves the entire output unchanged. A later valid request can use
the same Matrix immediately; there is no mutable model/session state to poison.

The existing single-row check and publication loops are now two small concrete
private helpers used by both operations. Their numerical code, feature order,
coefficient reads, packed decoding and existing `copy_row` errors stay unchanged.
The second concrete use justifies this factoring. Matvec and GPU/shader code are
unchanged. No dtype or bit-width selection occurs per scalar.

This is the existing scalar contract: read metadata into FP32, compute affine
multiply/add in FP64, then round to FP32. It preserves negative zero according to
that contract and does not emulate upstream native BF16 gather arithmetic.

The [pre-target independent witness](gemma-table-reference.md) supplies complete
original tables, selected rows and failure cases. Native checks reproduce all
960 canonical gathered values including 192 negative zeros, duplicate IDs and
their order. Successful one-ID and 32-ID requests compare complete repeated and
reordered rows against those same frozen values. Empty/excessive requests, wrong
destination sizes and invalid later IDs preserve every destination bit. A finite
BF16 table whose later selected row reconstructs beyond finite FP32 leaves all
1024 destination bytes unchanged; selecting its valid rows afterwards succeeds.

A private control test covers all 54 already admitted bit-width/metadata/group
combinations against the independently qualified existing single-row operation.
It checks the new batch control/publication behavior and does not newly qualify
every numerical format or GPU execution.

Formatting, strict default/Metal checks, 356 CPU tests, 387 Metal-feature tests
and both iOS compile targets pass. The exact optimized 54-format control test and
all four optimized fixture methods pass. Selected current test/CLI binaries are
snapshotted, compile-time inputs and original release executables stay unchanged
during validation, and all twelve owned commands/process groups close. The
[receipt](../benchmarks/results/2026-10-26-mlx-table-gather-v1.json) binds exact
source, fixtures, executed selected binaries and raw results. No download, Apple
GPU, real-model suite or new latency/resource benchmark is required for this
scalar storage change.

This qualifies the bounded scalar multirow mechanism. It does not yet qualify
actual large PLE table selections, resident GPU gather/table storage, whole
graph/KV/scheduling, original BF16 source parity, quality, representative inference
performance, app/iPhone admission or 131072-output-plus-prompt behavior. Existing
precision/source failures, artifact eligibility and context limits remain intact.

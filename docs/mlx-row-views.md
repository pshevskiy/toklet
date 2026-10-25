# Immutable row views over original packed matrices

`Matrix::row_range(start..end)` selects contiguous rows of an already validated
MLX affine matrix. It returns the same concrete Matrix type over borrowed
original code, scale and bias slices. It allocates no tensor, reconstructs no
weights, changes no numerical loop and introduces no dynamic dispatch or unsafe
code. This is a cold storage operation, not a new model architecture or GPU head.

Validate nonempty increasing row bounds before subtraction. Build the selected
layout with the same column count, bit width, group size and metadata dtype;
check source byte-offset multiplication and use bounded slice access. Code
stride is the original row byte count. Metadata stride is the validated complete
metadata byte count divided by original rows. Nonzero rows and group divisibility
make that division exact. Finite metadata validation is inherited from the
immutable parent, without rescanning every coefficient.

The view borrows the original source lifetime, so it can outlive its temporary
parent Matrix value while the original bytes remain alive. Rust prevents those
bytes being freed or mutably borrowed during the view's use. Nested views resolve
to the same original source. Parent and child share allocations; a selected
layout's logical byte count is not an additional owned model allocation.
BF16/F16 views may begin at two-byte offsets. No additional native-pointer
alignment guarantee is assumed.

The [independent pre-target fixture](gemma-table-reference.md) was committed
before this native change. Tests authenticate all three frozen files. Native
single-row views reconstruct 960 selected values including 192 negative zeros;
partitioned scalar matvecs reproduce all 360 output values across the four
partition sizes. Invalid empty, reversed, outside and extreme ranges return the
exact row-range error. Nested views preserve complete expected rows.

A private structural test covers all 54 combinations of six admitted bit widths,
three metadata dtypes and three group sizes. It checks original-byte pointer
sharing, strides, lengths, nested offsets and lifetime beyond a temporary parent,
using deliberately offset byte-backed sources. These are storage-view checks;
they do not newly qualify every numeric format on the GPU.

Formatting, both strict lint configurations, 351 CPU / 379 Metal-feature tests
and both iOS compile targets pass. Optimized checks pass the exact structural
test and all three fixture integration methods. Actual executed test binaries
are snapshotted; source closure and original release binaries remain unchanged
through the owned checks, and all twelve owned processes/groups are closed.
No explicit GPU/model suite or latency/resource benchmark is added: existing
numerical loops, reconstruction and GPU constructors are unchanged. The
[receipt](../benchmarks/results/2026-10-25-mlx-row-views-v1.json) binds exact
sources, frozen inputs, executed binaries and results.

The existing full-head single-owner geometry rejection remains unchanged.
There is no native transactional multi-row gather in this task: individually
successful `copy_row` calls do not make an entire request atomic. Chunked GPU
head/table construction still needs admitted simultaneous owners, source/loading
overlap, scratch/caller output, Context and device buffer limits, original-source
identity and independent complete numerical results. Native activation parity,
resident graph/KV/scheduling, source/cross-precision discrepancies, model quality,
measured inference performance, app/physical iPhone admission and long replies
remain open.

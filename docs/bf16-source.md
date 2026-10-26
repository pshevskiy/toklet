# Bounded BF16 reference-file ownership

The [reference owner](../tools/reference/bf16_source.py) now authenticates one
regular file and supplies bounded BF16 row views without mapping or expanding a
whole tensor. This supports the next [unquantized Gemma reference](gemma-unquantized-layout.md)
task. It is a Python reference mechanism, with no Rust engine, app, official
checkpoint, model arithmetic, quality or full-window qualification.

## Ownership, layout and limits

An owner receives an externally fixed size and full SHA-256. It opens one
`O_NOFOLLOW | O_NONBLOCK` descriptor, rejects nonregular/wrong-size inputs, and
authenticates through bounded1MiB `pread` calls. It retains that descriptor and
the original FD/path identity for its entire lifetime. Every row read checks
that same identity before and after access; it never adopts a newer baseline.
Short reads, IO failures, path replacement and changed/restored contents poison
the owner permanently. Invalid caller ranges and cancellation do not themselves
poison immutable weights. Deadlines cover authentication, reads and conversion.
Close is idempotent, and cleanup failure preserves an existing exception.

Concrete views describe row-major BF16 data with absolute file offsets. Scalar
and narrow-vector widths are accepted; there is no inherited group64 constraint.
Views permit at most262144 rows and12288 columns. A single contiguous read or
discontiguous gather admits at most128 aggregate rows, validating IDs and checked
extents before allocation. The largest raw tile is3145728 bytes. Gather can
retain a bytearray plus its immutable result: up to6291456 raw bytes, separately
from NumPy conversion buffers, runtime residency and caller-retained tiles.
These bounds are not a complete model-memory estimate.
They are IO tile bounds, not input/output context limits; callers must chunk
complete model workloads instead of clipping them to128 rows.

The official adapter checks the exact source size/hash and frozen layout hash,
then resolves a selected tensor's name, shape and offset once. It excludes the
60 unused shared-consumer tensors recorded by the header task. Generic row
extents alone do not qualify an official model. The positive official-file
adapter exercise remains a separate acceptance task.

`read_rows` and `gather` reuse the existing `affine4_tiles.bf16` conversion,
preserving exact FP32 bits and nonfinite rejection. No new rounding recipe or
numerical kernel is introduced. The returned arrays are independent of the file.
The fixed-size, domain-separated read chain encodes sequence/offset/length and
the hash of the exact returned bytes. It commits only after endpoint and deadline
checks. The chain records consumption; it is not an independent per-row proof
against the original whole-file hash.

## Actual checks, 26 October 2026

Thirteen [offline ownership tests](../tools/reference/test_bf16_source.py) cover
raw rows/scalars/gathers, bounds, symlinks/FIFOs, authentication mutation, source
replacement/restoration/truncation, short reads/IO failures, cancellation before
and after `pread`, audit publication, idempotent close and primary-error retention.
They need no numerical runtime, model download or GPU.

The [cold numerical witness](../tools/reference/bf16_source_reference.py) checks
37 exact FP32 values over full/split/gather/scalar schedules against independent
stdlib-generated bit fixtures. Signed zero and BF16 subnormals retain exact
bits; infinity/NaN inputs reject and poison the source. Conversion-end
cancellation rejects without poisoning immutable weights. Actual `<f4` dtypes
and two-dimensional shapes are captured before their gates.

Runtime admission uses the existing NumPy2.3.2/MLX0.32.3 contract plus the full
Python source/native closure. It checks source bytes before NumPy import and
actual module/native-image origins afterward. MLX/text package files are in
the shared admitted inventory; those packages do not execute in this witness.
The cold process includes admission and endpoint instrumentation: physical peak
289637192 bytes, RSS325877760, below its512MiB diagnostic policy. This is not
model fit or a latency benchmark.

An independent sparse recipe constructs a4294967360-byte logical file with
20480 allocated bytes and authenticates its complete SHA. Raw/gather reads at
offset4294967304 match the explicit source bytes, with four audited reads/30
returned bytes. Physical peak29639376/RSS34144256 fits128MiB. This qualifies
offset handling beyond4GiB, not real checkpoint execution or tensor values.

All eight v1/v2/v3 collection groups close. V2 adds explicit dtype/shape checks,
and V3 separately checks each nonfinite word. Both reuse
v1's ownership and large-offset evidence only after verifying unchanged
owner/test sources. The earlier witnesses are verified against their archived copies;
they are not silently rebound to the updated source. Original arrays/receipts and
all three capture versions remain retained. Core Rust/shaders/existing tests/examples
and protected release binaries are unchanged.

See the [small fixture](../tests/fixtures/bf16-source-v1/README.md) and
[public evidence](../benchmarks/results/2026-10-26-bf16-source-v1.json).
The separately completed official-payload acquisition authenticates the baseline
file. Next exercise the positive layout-bound row path against independently
frozen real rows, then compose and qualify reference
arithmetic and whole-process memory. The full1025-position quality suite,
historical precision gates and model-driven app/context delivery remain open.

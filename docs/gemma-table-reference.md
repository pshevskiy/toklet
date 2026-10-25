# Independent packed-table row and partition reference

A small pre-target reference now freezes the storage and numerical boundaries
needed before Gemma table gathers and a partitioned output head. It covers two
complete four-bit/group-64/BF16 tables, repeated row gathers and contiguous head
partitions. It changes no numerical Rust and executes no GPU. The old full-head
single-owner geometry rejection remains unchanged.

## Representation and numerical contracts

The version-one fixture uses row-major codes `[rows, columns/2]` U8, with the
first column in the low nibble and the next in the high nibble. Scales and affine
biases are little-endian BF16 `[rows, columns/64]`. Reconstruction promotes the
coefficients, evaluates the affine value in FP64 and rounds once to FP32. This
is the existing scalar reconstruction contract, not upstream BF16 gather math.

Complete tables have shapes 19×128 and 11×64. Every row/group has a documented
deterministic code/metadata recipe, including negative, positive and zero
coefficients. Store complete reconstructed FP32 values alongside original
packed bytes. Preserve 192 negative-zero weights as raw bits; rational zero
alone cannot encode their signs.

Three complete input rows cover zeros, cyclic small signed values and
sparse column/group boundaries. Preserve exact rational dot results, ordered
FP64 sums and final FP32 outputs separately. The producer uses Fraction dots;
the independent checker enumerates source values separately, accumulates
mathematical dots in Decimal400 and performs exact rational IEEE rounding at
each ordered-FP64 addition. It imports no producer or engine code.

Both consumers refer to one immutable source table. Gather IDs are
`[0,6,7,last,7]`; repeated and out-of-order rows must retain their exact values
and signs. Head partition maxima 1, 7, 8 and 19 cover every source row exactly
once with explicit output placement for all three tokens. Each code/metadata
view records its source-relative byte offset, length, source identity, exact
padding count and padded digest. Twelve metadata views begin at two-byte rather
than four-byte offsets. The K=64 case exposes BF16 alignment and trailing padding
that K=128 alone would miss. This is a source-storage witness, not a qualified
native row-view API or a device allocation.

## Admission and failure boundaries

The materializing reference deliberately limits tables to 32 rows, columns to
64 or 128, and gather requests to 32 rows. Partition maxima are positive and at
most 16384. These are bounded witness policies, not production model limits.
Larger overflow arithmetic is checked without creating a large partition list.
Exact positive geometry and complete reconstructed source length are validated
before any gather mutation.

A separate finite BF16 table reconstructs an overflowing later selected row.
The request `[1,0,2,1]` must leave the entire destination unchanged, including
earlier and repeated rows. The independent checker proves the failing value
exceeds finite FP32. Ten explicit rejections cover invalid partition bounds,
empty/reversed/out-of-range/overflowing intervals, allocation overflow, invalid
row IDs, excessive materialization and truncated source rows.

Capture failure retains the completed first case, its eight arrays and the
second case's failed attempt. Report-write failure after complete capture retains
the full binary, partial report, failed manifest and original error identity.
Manifest file capture is independent so a missing report does not discard the
binary identity. No failed or partial capture receives successful-fixture credit.

## Verification and scope

Sixteen portable and two retention methods pass, including coherent packed/gather
payload mutations, wrong padding, overlapping chunks, misplaced output indices,
duplicate JSON, extra overflow-array descriptors, doubled source accounting,
late invalid IDs, truncated rows and atomic overflow. Two fresh captures are
byte identical and both independent complete replays pass. All six owned child
processes and their groups are closed. Maximum successful-process RSS is
35,143,680 bytes and physical footprint is 23,888,616 bytes, each below separate
512 MiB allowances. These are reference-process observations, not model fit or
inference latency.

The [frozen fixture](../tests/fixtures/gemma-table-math-v1/manifest.json) has
20 arrays / 22,664 bytes under a 64 KiB binary cap. Its three files are frozen
before native changes; the [receipt](../benchmarks/results/2026-10-25-gemma-table-reference-v1.json)
binds their bytes, sources, captures, tests, resources and unchanged numerical
code. Exact scripts/interpreter are bound; no full standard-library/native
runtime closure is claimed. No engine build or unrelated benchmark is needed
for this pure reference task.

Next implement only the concrete immutable borrowed row access needed by
consumers, then separately admit and qualify chunked resident head storage and
table gathers. All source and temporary overlap, caller output, scratch, Context
allowance and device buffer limits need admission. Sharing source bytes in a
fixture does not establish native tied ownership or simultaneous GPU residency.
The actual PLE table's 1,321,205,760 bytes still need bounded device storage.

The [real projection result](gemma-real-projections.md) remains limited to its
276 fixed-input calls. Native table/head execution, activation parity, full
resident graph/KV, original source/cross-precision discrepancies, broader quality,
inference performance, app/iPhone fit and 131072 output tokens plus nonempty
prompt remain open. G2, G3 and G4 are not complete.

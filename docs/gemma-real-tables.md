# Complete real Gemma table upload and GPU lookup

The [public evidence record](../benchmarks/results/2026-10-26-gemma-real-tables-v1.json)
qualifies bounded source upload and resident lookup of both complete original
Gemma4 E2B embedding tables on the Mac. It combines the existing
[bounded uploader](gemma-stream-upload.md) and [lookup workspace](gemma-table-gather-gpu.md);
no production numerical function or shader changes. Complete Gemma generation,
whole-model admission and app/phone support remain open in the
[active plan](gemma4-implementation-plan.md).

The pinned checkpoint is `mlx-community/gemma-4-e2b-it-4bit` revision
`238767527555cb75a05732a84dff5d6ba0dd6809`. Its complete 3550670554-byte file
has SHA256 `038e39a37a7667373d2c3991375446b10c96ae1d717a68674870343db376b76e`.
Header, tensor geometry, original ranges and exact consumed chunks are verified.
The original packed4/group64/BF16 tables have 262144 rows: embedding width 1536
and per-layer embedding width 8960. Both tables stay resident simultaneously,
with sixteen 16384-row chunks each. Affine capacity 1 is separate from the two
lookup workspaces' capacity 32. There is no retained full CPU source table or
expanded model copy.

Before GPU execution, the independently owned preparation phase streams every
original source chunk through a 64 KiB window, records original and guarded GPU
buffer digests, and reconstructs 35 selected complete rows per table with the
previous exact rational/IEEE rounding oracle. All 41984 original four-token values
match the [frozen complete independent lookup reference](gemma-lookup-reference.md)
bitwise. The resulting 367360 expected values and source manifests are frozen and
bound to their first retained hashes before native execution. Replay reconstructs
them again from the original checkpoint. Target results never define expectations.

Requests alternate between the two tables: original IDs `[2,100,257,100]`, the
first and last row of every resident chunk (32 IDs), their reverse, the final
single row, 32 duplicates of row 16384, then the original four IDs again. All 12
requests pass, preserving order, repeated IDs and global row offsets. Every one
of the 1102080 returned values matches exactly. Both original weight owners remain
unchanged through the sequence; 198 simultaneous device allocation identities are
checked for distinctness and reuse.

The 32 source callbacks retain exact consumed offsets/lengths/digests and three
stable, disjoint source allocations per construction. Complete cold/final matrix
buffer digests and guards are retained. Cold snapshots and requests retain the caller, full CPU staged/chunk storage
and all ID/output/status workspace words in 84 binary arrays (12 cold and
72 request arrays), 32624592 bytes total. A workspace output describes its last executed child;
inactive output is sentinel-filled and the CPU chunk tail can retain earlier
values. Replay simulates those details independently. This does not capture every
intermediate child output separately. Completion-safe buffer identities/hashes
and original outcomes are flushed before raw capture, and partial preparation,
raw files and first errors are retained on failure.

Memory is admitted before either table allocation. The conservative combined
policy is 1767039660 bytes: 1552708704 bytes of requested resident owners,
37585648 bytes of lookup workspaces including two 16 MiB pipeline allowances,
82575432 bytes of maximum source staging/descriptors, Context policy and bounded
header/reference/caller/hash/serialization metadata. The staging allowance remains
counted even though it is released before lookup workspace creation. Constructors
receive exact sub-budgets and recheck actual capacities before callback/upload.
These are policy/requested bytes, not measured physical residency or free memory.

On Apple M1 Pro, MacBookPro18,3 with 16 GiB RAM, macOS 26.6.2 (25G83), the controlled
native process peaks at RSS 1578057728 bytes and physical footprint 1661945536 bytes,
below its 3 GiB process ceiling. Replay peaks at 82624512/61801120 bytes and the
regression process at 169082880/93815528 bytes (RSS/physical), below 512 MiB each.
The 18.47-second native run includes loading, validation, hashing and diagnostic
capture; it is not an inference latency or throughput benchmark.

Astra reviewed source, independent oracle/replay, controlled execution and final
evidence. Formatting, strict default/Metal checks, CPU 356/16 ignored, Metal 392/121
ignored, iOS device/simulator compilation, optimized ordinary ledger/writer-failure
2 and scalar gather fixture 4 pass. Eight portable checks run before preparation;
the independent actual replay passes, and 8 portable plus 29 actual-capture/reader
methods (one baseline and 28 mutations) give 37 passing tests.
Controls cover original/secondary preparation failures, publication failures,
source offsets/bytes/reuse/alias/wrap, resident identity/guards, complete workspace
and caller values, request order, missing records and forbidden capability grants.
All 19 owned build/run process groups are absent after completion. The receipt
binds exact scripts/interpreter, sources, binaries and outputs; it does not claim
a complete standard-library or native-library runtime closure.

Retain this actual-table boundary. The original 5 arrays/57 cross-precision values,
27 ideal-FP64 native logit discrepancies per schedule, 15 dense and 15 packed source
GPU intermediates and 8 long-rotary comparisons remain unwaived. Resident full text
graph, shared/rotating KV and attention scheduling, source BF16 parity, quality,
completed generation benchmarks, app integration, iPhone fit and 131072-output
requirements remain open. No download eligibility or context/output limit changes.

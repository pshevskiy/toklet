# Pinned Qwen3.5-9B CPU storage

Q2.B adds a separate immutable CPU payload store for the [pinned9B source](qwen35-9b-cold.md).
Required checks, actual-source verification and loading cohorts pass; Astra
approves scoped retention after the reader timing fix, with no remaining findings. No numerical model, GPU upload, codec, session, request
collector, context expansion or managed route is added.

The9B public loader retains its concrete `nine_b::Plan`, separate from the4B
store/plan type. It requires explicit text-only policy and exact config/header/
container identities. It reads a single stream, hashes all5,038,160,953bytes,
checks EOF and validates every packed component and promoted small tensor before
publishing a store, historical content identity or report. Failure drops all
unpublished owners. The original source's unaligned120,889payload origin grants
no typed-alignment assumption, and sequential byte counters preserve offsets
above4GiB.

An internal borrowed `SourceRecipe` supplies logical descriptors, physical spans
and payload bytes to the common cold streaming mechanism. Returned components
move into source-specific stores; there is no matrix clone or full FP32 expansion.
The4B public API, source pins, tied-head plan, numerical binding and consuming Metal
uploader keep their types and behavior. The9B store exposes a separate original
head; it cannot be passed as a4B store or uploaded through its4B-only method.
No numerical hot loop or backend selection changes.

| Cold ownership | Bytes |
| --- | ---: |
| Original packed codes | 4,476,370,944 |
| Original BF16 scales and biases | 559,546,368 |
| Exact FP32 small vectors and convolutions | 4,245,504 |
| Retained tensor capacity | 5,040,162,816 |
| Largest temporary BF16 conversion source | 65,536 |
| Streaming buffer | 65,536 |
| Cold metadata allowance | 67,108,864 |
| Requested load policy | 5,107,402,752 |
| Mac working-set ceiling | 6,442,450,944 |

These policy reservations are distinct from observed process memory. Capacity
arithmetic is checked, actual vector capacities enter admission, and a one-byte
undersized budget rejects before reading tensor payload. Header/config work is
covered by the metadata allowance. Retained small tensors are exact BF16-to-FP32
promotions, including signed zero; their original temporary buffers are released.

The independent211-byte fixture has different2x64embedding/3x64head tables, a
small vector and a four-tap convolution. Physical order differs from logical
order. Its Python generator packs original integer cells and records independent
expected rows and exact small-value bits. Private test constructors supply its
artificial recipe; every pinned public loader rejects it. Nine new portable
controls cover all original bytes/rows, owner separation, promotion bits/capacity,
minus-one/overflow admission, truncation/tail/hash/IO errors, nonfinite/overflowing
coefficients, short/interrupted reads, malformed spans and cross-source rejection.

Required formatting and strict CPU/Metal all-target Clippy pass, with661CPU/
795Metal tests (44/322ignored). The targeted run passes31tests/6ignored.
Seventeen Python controls cover arithmetic, generator reproduction, strict receipt
and number types, process/source/binary/input/log binding and scope rejection.
Fresh native60 passes three arm64 compile/export/link/package slices,46exports
per slice, and Mac CPU/Metal C execution smokes. No iPhone runtime follows.

Both actual9B storage captures pass with927components,250matrices,153small
vectors and24convolutions. Independent validation hashes the entire source and
both descriptor endpoints, compares every retained component, and reconstructs
750rows/3,858,432FP32 values bitwise per capture. Row bodies match between the
two processes. Child lifetime physical peaks are5,052,813,248 and5,052,862,424bytes,
within6GiB. A nonmutating last-byte corruption reader rejects before publication.
These are CPU storage results under explicit child-only `MallocLargeCache=0`;
default application allocator behavior and complete model/application fit remain
unqualified. No process-global allocator setting changes.

The same new release binary passes two actual4B load/capture checks and corruption
rejection. All924components and747rows/2,698,752values match the existing oracle;
row-body bits also match prior4B evidence. This does not rehabilitate the
[128/256-input numerical rejection](qwen35-context256.md).

The new independent9B reader passes a copied valid actual capture and rejects
seven damaged/resealed variants: changed row bytes, rewritten row digest with
changed bytes, scope escalation, boolean work count, a head-name substitution,
reversed load/capture durations and capture time exceeding the owned run.
The original capture remains unchanged. Read-only receipts bind the exact test,
binary, source inputs, allocator, completed process and measured physical peak.
The reader requires `0 < load_ns <= load_and_capture_ns <= owned_elapsed_ns`.
Astra's first review found the missing timing enclosure; the targeted reader fix
and all actual controls now pass, with no production finding or recollection.
Original consumer versions are preserved. Reusing the original collection requires
an explicit SHA256-bound update naming only the two reader/test consumer paths,
their exact old/new hashes and reviewed snapshots. Every producer input remains
unchanged; an update cannot authorize any Rust/fixture/probe change.

Loading measurements use one standalone probe source linked separately against
preserved native59 and new native60 Rust libraries, with matched opt3/unwind
profiles. Compile-time selection chooses4B or9B outside the loop. Each process
constructs/drops13new stores (3warmups+10samples); config/file-open time is
excluded from `load_ns`, and logging follows drop. Two reversed old/new4B pairs
check the shared mechanism; two independent9B cohorts report new-source loading.
Warmups exercise the file cache; full OS residency and absence of paging are not
qualified. Supplementary native I/O/page-in and VM observations remain descriptive.
Raw samples, IQR, standard deviation, native process peak, live charge guard,
hardware/power/thermal snapshots and source endpoints are retained. No4B/9B
request-speed ratio,1% bound or tail/sustained qualification follows.

All six cohorts close successfully:18warmups and60recorded samples, with exact
source/binary endpoints. Matched4B load medians are7.590→7.513s and7.554→7.540s
(−1.02%/−0.18%); no repeated5% regression is observed. New9B medians are22.731/
22.675s, with IQRs0.077/0.141s. Helper-process lifetime physical peaks are5.051GB;
maximum observed conservative charge sums including the collector are5.070GB.
These new-store loading baselines retain the cache-state limits above and do not
establish an engine or request speedup. Source verification, tensor conversion
and allocation all remain inside the timed load boundary.

Initial executable/incremental cache cleanups missed their requested headroom
targets while preserving all71protected files; they reclaimed about2.1GB and the
2GiB guard holds. The initial fixture's zero-sign expectation was corrected before
tests without changing its source bytes. The first standalone probe build lacked
the provider's host dependency search path; the corrected build pins both target
and host dependencies. Failed attempts and raw logs remain recorded.

[Contract](../artifacts/qwen35-9b-storage-2026-11-09/acceptance-contract.json),
[checks/review/commit](../artifacts/qwen35-9b-storage-2026-11-09/publication-checks.json),
[Astra](../artifacts/qwen35-9b-storage-2026-11-09/astra-review.json),
[required checks](../artifacts/qwen35-9b-storage-2026-11-09/required-checks-1/checks.json),
[actual source controls](../artifacts/qwen35-9b-storage-2026-11-09/native-checks-1.json),
[independent checks after reader fix](../artifacts/qwen35-9b-storage-2026-11-09/independent-checks-2.json),
[damaged/resealed controls](../artifacts/qwen35-9b-storage-2026-11-09/resealed-checks-2.json),
[consumer-only reuse audit](../artifacts/qwen35-9b-storage-2026-11-09/reader-fix-audit.json),
[explicit consumer update](../artifacts/qwen35-9b-storage-2026-11-09/reader-consumer-update.json),
[measurement contract](../artifacts/qwen35-9b-storage-2026-11-09/measurement-contract.json),
[loading results](../artifacts/qwen35-9b-storage-2026-11-09/loading-summary.json),
[cache-state scope](../artifacts/qwen35-9b-storage-2026-11-09/cache-state-clarification.json),
[native60](../artifacts/qwen35-9b-storage-2026-11-09/native60-publication.json),
[Mac C smokes](../artifacts/qwen35-9b-storage-2026-11-09/native-smoke-60/checks.json).

Next gates remain explicit untied numerical head binding with independent tiny/
real-source oracles, bounded reference layout support above4GiB, measured GPU/
session/codec/application fit with the unchanged512MiB codec policy, request
metrics/overhead, quality and route adoption or rejection. Metadata or cold CPU
storage success cannot establish any of those gates.

The next code milestone is explicit untied numerical head binding with independent
tiny fixtures, while preserving tied4B constructors.
[Next-scope analysis](../artifacts/qwen35-9b-storage-2026-11-09/next-scope-analysis.json).

# Immutable packed GPU weights and separate lookup sessions

`gemma_affine::Weights` owns original packed buffers independently of
construction Context and mutable session scratch. This is a concrete prerequisite
for a resident Gemma model: model weights can live beside a Context without a
self-referential Rust owner, while each lookup workspace owns its own state.
The [evidence record](../benchmarks/results/2026-10-26-gemma-immutable-weights-v1.json)
qualifies this ownership boundary; the
[active Gemma plan](gemma4-implementation-plan.md) remains open.

Each immutable chunk contains a validated layout and three original packed4,
group64, BF16 buffers. The owner retains the underlying Metal device and cold
metadata, including a conservative bound over all sixteen reconstruction codes
(even unused codes). It owns no activation/status buffers, pipeline, borrowed
Context, complete CPU source table or expanded model. Total rows are bounded at
262144, chunks at 64, and each chunk at 16384 rows. Existing fixed-format column
and codebook validation remains unchanged.

`Context::weights_from_fill` uses three reusable initialized byte arrays. The
callback must overwrite every active byte on success. Admission checks all
resident original buffers, padding/guards, actual chunk Vec slots, owner/device
handles, actual maximum source capacities and their descriptors. Device buffer
limits are checked before allocation and callbacks; actual source capacities are
readmitted before upload effects. Callback-captured storage and Context/session
policies are separate. Errors and unwind drop unpublished owners without rolling
back external callback effects. Construction submits no GPU commands.

`Gather::apply_weights` borrows weights for a request and validates the underlying
retained device identity, width, selected rows and complete caller destination
before changing session state. Its construction Context may already have dropped.
The legacy `Gather::apply` retains its stricter exact-Context and matrix-state
checks. Both select concrete borrowed views at the request boundary and use the
same existing shader, command retention, completion/guard checks and transactional
publication. No view collection or dynamic numerical dispatch is introduced.
A failure after staging begins leaves the caller untouched and the workspace
sticky poisoned; a fresh workspace can reuse the unchanged weights. Preflight
rejection preserves a reusable workspace. This does not add
unchecked Send/Sync or qualify concurrent execution.

Required formatting, strict default/Metal Clippy, CPU 356/16 ignored, Metal
395/123 ignored, iOS device/simulator checks and optimized 2 ownership plans,
1 real-table plan and 4 gather fixtures pass. Exact frozen sources, includes,
fixtures, isolated executable snapshots and both protected release binaries are
bound. Thirteen source-check commands and thirteen actual commands are closed;
all owned processes and groups are absent.

Before any new GPU execution, four tiny and four complete transformed-prior
replay controls pass. They are reader tests only. The unchanged legacy gather
passes its full 52-step witness and 29 reader/capture tests. The new tiny witness
passes four construction-Context teardowns, 40 request steps, 22272 exact returned
values, eight published immutable owners/20 chunks and 27 callbacks. Five
construction failures preserve original errors/panic payload and the warmed
owner; budget-minus-one invokes zero callbacks. Width mismatch preserves both
owners, warmed session and caller, followed by valid reuse. Four fresh-workspace
recoveries and four legacy controls pass, with 34 new capture/mutation tests.
The unchanged independent tiny source fixture defines all numerical expectations.

Both complete original Gemma4 tables are then loaded together, 262144 rows each,
widths 1536/8960, sixteen 16384-row chunks per table. The construction Context and
its autorelease pool drop before a fresh same-device lookup Context is created.
All 32 source callbacks and original chunk digests match the independently frozen
[previous real-table oracle](gemma-real-tables.md). That accepted capture and its
three original oracle files retain their literal first-capture size/SHA pins;
replay reconstructs expectations from the original checkpoint again. Historical
Affine geometry/policy remains historical identity and does not define the new
owner's admission. Target outputs never define expected values.

All 12 alternating original/boundary/reverse/single/duplicate/repeated schedules
match 1102080 returned values bitwise. All 84 cold/request arrays, 32624592 bytes,
full source/guarded-weight hashes and 102 distinct live device buffer identities
are checked. Complete immutable buffers remain unchanged. Thirty-four additional
actual-capture/mutation tests pass, including coordinated numerical-view changes.
This does not capture every internal child output separately.

Requested resident immutable owners total 1547704672 bytes, 5004032 fewer than
prior coupled owners because per-matrix activation/status scratch is excluded.
This is an accounting difference, not a measured physical-memory or speed gain.
The separate gather policies total 37585648 bytes; maximum source staging and
descriptors are 82575432 bytes. Conservative aggregate policy is 1829144492 bytes,
including two 64 MiB Context allowances even though the construction and lookup
Contexts do not overlap, plus source staging and bounded parser/caller/capture
state. It is not exact physical residency or a whole-model/device budget.

On MacBookPro18,3, Apple M1 Pro, 16 GiB, macOS 26.6.2 (25G83), the complete real
native process peaks at RSS 1655652352 / physical 1663059696 bytes, below 3 GiB.
Its independent replay peaks at 86261760 / 65340088 bytes and regressions at
149504000 / 91554584, below 512 MiB each. Tiny native RSS/physical are
26329088 / 15024704. These are separate process peaks; the 17.86-second native
diagnostic includes load/hash/capture and is not a generation benchmark.

Retain this immutable weight/session ownership boundary. Driver/in-flight errors
and a physical foreign-device configuration are not injected on this single-device
Mac. Whole graph/KV/scheduling, original source/precision/long-rotary discrepancies,
quality, representative performance, app/iPhone qualification and
131072-output-plus-nonempty-prompt remain open. No model capability, search/download
eligibility, numerical tolerance or context/output limit changes.

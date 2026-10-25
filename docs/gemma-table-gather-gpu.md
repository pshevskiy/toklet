# Small resident packed GPU table gathers

`metal::gemma_affine::Gather` supplies a reusable lookup workspace over an
existing immutable `PartitionedAffine`. It borrows the matrix per request and
reads the original resident packed buffers. It does not upload a second table.
The earlier [scalar gather](mlx-table-gather.md),
[partitioned affine](gemma-partitioned-affine.md) and original affine shader
retain their numerical contracts. This is a small standalone mechanism for later
Gemma composition, not full model or app qualification.

## Shapes, storage and ownership

Construction supports columns 64–12288 divisible by 64 and capacity 1–32. The matrix
already admits packed 4-bit, low-nibble-first codes, group 64 and BF16 scale/bias
metadata. Gather capacity is independent of affine token capacity. Existing
resident matrix capacity 1 can therefore serve 32 selected rows.

ID, output and status buffers contain U32 words with active capacities of
capacity, capacity*columns and one, respectively, plus seven guard words each. Two fixed CPU F32 staging arrays
contain capacity*columns values each. Admission checks arithmetic, every device
buffer limit, actual CPU vector capacities and the concrete owner. A separate
16 MiB policy covers the opaque compiled pipeline; it is not measured pipeline
residency. Existing matrix, Context, caller and loading overlap are separate.
A matrix must be admitted as a whole; fitting one chunk does not admit a model.

The workspace caches its own Safe/Precise Metal 2.3 pipeline at construction.
Existing affine Context initialization acquires no extra eager compilation.
The numerical kernel selects no architecture or dtype per scalar. Adjacent GPU
lanes traverse adjacent columns and reconstruct each original weight with an
explicit FMA. Launches use the actual pipeline execution width.

## Transactions and command lifetimes

Before changing warmed state, apply checks the complete ID/destination shape,
every row index, exact Context identity and columns, and readiness of the parent
and every child. Invalid requests leave both workspace and caller unchanged.

Two fixed 32-element routing arrays preserve duplicate IDs, arbitrary order and
original caller slots. Each nonempty chunk receives local IDs and writes the
same reused output buffer. Completed output is checked for finite values, zero
status and untouched guards/inactive capacity before entering fixed CPU scratch.
Caller output is published only after all selected chunks succeed. Failure after
staging begins poisons the workspace; recreation leaves original weights intact.
There is no per-request Rust routing/staging heap allocation. Metal command and
encoder creation may allocate.

The shared command helper requires retained resource references. Gather stores
the command before committing it; encoded original weight and scratch buffers
remain retained after the request borrow ends, including an unfinished error
return. Shared CPU views require checked completion. The diagnostic injects a
composition failure after completed chunks, not a driver or in-flight failure.

## Verification surface

The frozen [independent table witness](gemma-table-reference.md) provides two
original tables 19×128 and 11×64. The GPU diagnostic uses maximum chunk rows 7/8,
repeated canonical selections, reordered/duplicate selections at 1/32-ID bounds,
six malformed requests, foreign Context/column mismatch and poisoned/pending
matrix checks. It alternates unchanged affine projection and lookup on the same
weights. A failure after two nonempty chunks must preserve changed internal
staging, untouched caller output and sticky poison; a new workspace then recovers.

The independent checker reconstructs every complete source/ID/output/status
buffer, padding and guard, both workspace CPU arrays and all original affine
scratch. It verifies live allocation identities through reuse and replacement.
A portable regression checks first-affine replay before any GPU invocation;
mutation tests run against the actual full capture afterward. Build, diagnostic,
reader and publication owners preserve first outputs, original causes, copied
executed scripts/binaries and process-group cleanup.

The native workload policy includes two Context allowances, three simultaneous
matrices, two overlapping Gather owners/pipelines during recreation, the largest
caller, 384 bytes of routing arrays and separate host/capture policies. This
conservative ledger and measured process peaks answer different questions.

## Remaining qualification

Actual real-model head/PLE selections and bounded large-table upload remain next.
The original PLE representation alone is 1321205760 bytes; this tiny diagnostic
cannot establish its loading overlap, model-wide admission or phone fit.
Resident whole graph, activation/KV/scheduling, historical source/precision and
rotary discrepancies, representative latency/quality, app/phone and long replies
remain open. Existing capabilities and context limits are unchanged. The pinned
E2B configuration has 131072 total positions and cannot grant 131072 output tokens
plus a nonempty prompt.

## Accepted diagnostic results — 26 October 2026

The [public evidence record](../benchmarks/results/2026-10-26-gemma-table-gather-gpu-v1.json)
binds the exact source/build and actual capture packets. Apple M1 Pro,
MacBookPro18,3, 16 GiB, macOS 26.6.2 (25G83), Rust 1.90 release build; the recorded
pipeline execution width is 32. There are 52 lookup steps across four cases,
19072 successfully published lookup values and eight alternating affine calls.
All complete buffer/staging comparisons are bit exact, including negative zeros.
Five portable checks precede hardware and 29 actual-capture/reader regressions
pass afterward. No driver or unfinished-command failure was injected.

Formatting, strict default/Metal lint, 356 CPU tests, 388 Metal-feature tests,
iOS device/simulator compile checks and optimized plan 1/fixture 4 pass. Explicit
hardware/real-model tests remain ignored in ordinary suites (CPU 16, Metal 119).
The first source-check attempt failed strict lint because the existing fixture
was loaded twice; its original snapshots/logs remain retained without success
credit. The correction shares the test-only module and its parent visibility;
no fixture body, numerical code or tolerance changed.

The conservative aggregate native policy is 335671488 bytes. Its components are
134217728 for two Contexts, 4436 for the largest original matrix, 6396 for the two
extra rejection matrices, 33653488 for overlapping Gather owners/pipelines,
16896 caller bytes, 384 routing bytes, 33554432 fixture/host policy and 134217728
capture/serialization policy. The policy is not an exact physical measurement.

| Process | Maximum resident set, bytes | Peak physical footprint, bytes | Allowance |
| --- | ---: | ---: | ---: |
| Native GPU diagnostic | 36274176 | 25510536 | 1 GiB |
| Independent replay | 71761920 | 54346424 | 512 MiB |
| Capture regressions | 107626496 | 61801240 | 512 MiB |

Native execution took 0.55 s including diagnostic work. This is not a latency or
throughput benchmark. There are no new performance, device-budget or app claims.
Retain the standalone mechanism and explicit failure semantics for later model
composition; do not treat these results as large-table or whole-graph admission.

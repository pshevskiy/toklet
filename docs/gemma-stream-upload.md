# Bounded source upload for resident packed chunks

`Context::partitioned_affine_from_fill` loads a complete `PartitionedAffine`
through one reusable codes/scales/biases source workspace. It prepares large
table loading without retaining a second full CPU table. The numerical
[affine](gemma-metal-affine.md), [partitioned projection](gemma-partitioned-affine.md)
and [gather](gemma-table-gather-gpu.md) implementations and shaders are unchanged.
This document describes a standalone loading mechanism, not a real-model or
whole-graph admission result.

The constructor accepts the existing validated packed 4-bit/group 64/BF16 layout,
maximum chunk rows, token capacity, working-byte budget and a generic cold
callback. At each chunk it passes the first row, child layout and exact mutable
byte slices for the three tensors. Architecture names, files, tokenization and
config interpretation stay outside numerical kernels.

## Ownership and peak admission

All child geometry and device buffer limits are checked before source reservation
or callback invocation. Working admission includes the entire resident GPU/CPU
owner plus all three maximum source Vec capacities and their descriptors at
once. Actual chunk slots, resident scratch and source capacities are re-admitted
before the first callback or GPU upload. Context, existing model allocations,
callback-captured state, file/header buffers and caller output are separate.

Source storage is reserved and initialized once, then reused across every child.
Only active prefixes reach the callback or upload, including short final chunks
and odd BF16 metadata tails. Existing affine validation and synchronous
shared-buffer copies consume each completed prefix. No GPU command is submitted
by this loading mechanism. The returned owner retains all GPU chunks and its
usual resident scratch; source staging is released. Its `owned_bytes()` keeps
its previous resident-only meaning.

The PLE shape has 1321205760 original source bytes; a 16384-row source chunk uses
82575360 bytes rather than a full duplicate. This arithmetic plan is not an actual
allocation, device-fit or model-memory qualification.

## Callback and failure contract

Every successful callback must completely overwrite all three active prefixes.
Initialized reuse provides memory safety but cannot authenticate file contents
or prove a callback filled every byte. The artifact adapter is responsible for
source identity, tensor spans and complete reads, including stable source
endpoints and any IO/conversion workspace admission.

Only a complete owner is returned. A first/late source error, invalid completed
chunk or unwinding callback panic drops the unpublished partial construction.
The original error propagates without replacement. External callback effects,
such as file position and counters, are not rolled back. The old borrowed-Matrix
constructor's whole-table codebook preflight stays unchanged; streamed loading
can reject a later chunk after earlier synchronous uploads.

## Verification surface

Two pure plan tests independently account for GPU/resident/source bytes, all
owner descriptors and extra capacities, adjacent budgets, overflow/format
rejection and the large PLE shape without GPU or large allocation.

The explicit tiny GPU diagnostic uses both frozen original tables with maximum
chunk rows 7/8. It records every callback's active bytes before/after fill,
exact source slices and stable allocation addresses; independent replay checks
full admitted intervals for overlap and address wrap. It observes all six
resident buffers per child, guards and padding, CPU scratch and unchanged
projection outputs. Source storage remains independent of the returned owner.

Tests exercise exact-budget construction, budget-minus-one with zero callbacks,
first/late source errors, late finite metadata outside the codebook domain and
an intentional caught callback panic. Original warmed resident weights and
scratch are recorded before assertions after each error and after fresh recovery.
Construction atomicity is verified; rollback of external callback effects and
immediate release of driver-internal allocation pools are not claimed.

The independent projection oracle and existing diagnostic observation helpers
are reused without changing their bodies. The new source adapter has no numerical
reconstruction or projection implementation to compare against itself. No actual
model is opened by ordinary CPU or Metal-feature tests.

## Remaining qualification

Original real head/PLE upload and gather remain a separate task using the
[frozen complete lookup rows](gemma-lookup-reference.md). It must authenticate
the original checkpoint and count all simultaneous resident weights, loading
scratch, Context, gathers and callers before allocating. A single chunk fitting
is insufficient. Full graph/KV scheduling, existing precision/source/rotary
discrepancies, representative quality/performance, app/phone and long-output
qualification remain open. No model eligibility or capability limit changes.

## Results — 26 October 2026

The [public evidence record](../benchmarks/results/2026-10-26-gemma-stream-upload-v1.json)
binds the exact source, binary, complete captures and independent readers. On
Apple M1 Pro/MacBookPro18,3 with 16 GiB and macOS 26.6.2 (25G83), five complete
owners publish 13 chunks across four geometry cases and recovery. Twenty source
callbacks and 25 completed projection requests produce 777 bit-exact values.
Every source prefix, resident device buffer, guard, padding word and CPU scratch
transition matches the frozen independent data. Source intervals stay disjoint
and stable within each construction. Five portable checks and 31 actual-capture
mutation/reader regressions pass. The one intentional callback panic is caught;
its original payload and the unchanged existing owner are retained.

Formatting, strict default/Metal lint, 356 CPU tests, 390 Metal-feature tests,
iOS device/simulator compilation and optimized upload-plan 2/fixture 4 checks
pass. Ordinary suites retain 16 CPU and 120 Metal ignored hardware/model checks.

The first explicit GPU attempt correctly rejected invalid metadata, but the test
expected `Error::Options` rather than the existing `Error::Unsupported` category.
It stopped at that assertion after recording the original cause and unchanged
owner. The failed attempt, original source snapshots, binary and outputs remain
retained without success credit. The sole correction changes the test category;
production code, rejection thresholds, numerical fixtures and tolerances stay
unchanged. Fresh source checks and the corrected GPU run pass.

The conservative tiny diagnostic policy totals 167820292 bytes: 67108864 Context,
47256 resident-owner policy, 648 source staging/descriptors, 228 caller,
33554432 fixture/host policy and 67108864 observation/serialization policy.
These policy allowances are not exact physical residency.

| Process | Maximum resident set, bytes | Peak physical footprint, bytes | Allowance |
| --- | ---: | ---: | ---: |
| Native GPU diagnostic | 20234240 | 9552448 | 1 GiB |
| Independent replay | 29884416 | 19415664 | 512 MiB |
| Capture regressions | 39043072 | 27411128 | 512 MiB |

Retain the bounded uploader and explicit construction failure contract for later
real-table qualification. This is a correctness/memory-ownership result; it is
not an end-to-end timing or model-fit benchmark.

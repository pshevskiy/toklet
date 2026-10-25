# Complete real tied head on Metal

The original Gemma4 E2B tied head now passes a fixed four-token standalone Metal
qualification on Apple M1 Pro. All sixteen packed chunks remain resident
together; the existing parent executes them sequentially and publishes caller
output after all chunks complete. This does not qualify the whole decoder,
original BF16 execution, inference performance or the iPhone application.

The pinned checkpoint is `mlx-community/gemma-4-e2b-it-4bit` at revision
`238767527555cb75a05732a84dff5d6ba0dd6809`. Original codes are U32
`[262144,192]`, 201326592 bytes; scales and biases are each BF16
`[262144,24]`, 12582912 bytes. No dense head is expanded. Each of sixteen
16384-row children uses the unchanged packed4/group64/BF16 affine operator with
FP32 activations and outputs. One Context supplies cached device/queue/pipelines.

The [complete independent reference](gemma-head-reference.md) supplies all 6144
FP32 inputs and 1048576 pre-softcap output values for tokens `[2,100,257,100]`.
Four native scalar ordered-FP64 dot rows rounded to FP32 are also retained as a
separate reference. GPU results pass the existing `1e-5 + 1e-4*abs(reference)`
gate against both contracts. Maximum observed difference is zero against these
rounded FP32 arrays. This is not an equality claim for unrounded ideal FP64,
native graph activations or original BF16 source execution.

Two full four-token calls, four single-token calls and full-batch resume pass.
Repeats, parts and resume are bitwise identical to the first native batch. The
independent replay checks 1048576 scalar/mathematical comparisons and 8388608
GPU/reference comparisons. It authenticates the original checkpoint, streams its
complete packed spans into expected weight-buffer digests, checks all 96 live
allocation identities and every whole-buffer digest, guard and inactive input or
output slot, and reconstructs complete token-strided caller publication and both
CPU staging arrays. The actual GPU weights are checked by complete-buffer
cryptographic observations, not retained as another plaintext model copy.

All active outputs of every safely readable child are saved separately from the
transactional caller. Original completion/error and safe buffer observations are
published before fallible raw writes. An unfinished or non-completed child is
never read. On an unaccepted apply failure, a safe child observation is not
automatically a result of that failed request: an unexecuted child may retain old
data. This run has no apply or driver failure. The earlier
[tiny composition failure](gemma-partitioned-affine.md) remains the qualification
for poisoning after two completed children; a separate ordinary writer test
checks that a failed raw writer cannot erase the previously published cause.

Before large allocation, the workload plan counts all simultaneous resident
children and parent CPU staging, Context allowance, original CPU loading overlap,
reference and decoded arrays, scalar/caller/baseline storage, bounded conversion
and hash workspace, parsed metadata and record/serialization policy allowances.
Actual vector capacities are admitted again before upload. Owner-only bytes are
235543088 on the exact 64-bit build; the actual complete workload policy totals
572937540 bytes. The difference from the planned 568767812 bytes is the source
reader's 8 MiB reference-vector capacity for a 4218880-byte binary. Policy
allowances and unused process allowance do not promise free device memory.

CPU code/scale/bias vectors are explicitly dropped after the scalar reference and
single resident upload, before GPU reuse. A one-byte-short aggregate owner budget
is rejected before upload. The original single-owner full-head geometry rejection
also remains unchanged and occurs before large source spans are read.

The [receipt](../benchmarks/results/2026-10-26-gemma-real-head-v1.json) records
these completed results on MacBookPro18,3, Apple M1 Pro, 16 GiB RAM, macOS 26.6.2
(25G83), SIMD width 32:

| Observation | Result |
| --- | --- |
| Complete recorded arrays / raw bytes | 123 / 37748736 |
| Reuse calls / simultaneously resident chunks | 7 / 16 |
| Native process maximum RSS / peak physical footprint | 498171904 / 497746832 bytes |
| Replay maximum RSS / peak physical footprint | 74170368 / 45089512 bytes |
| Regression maximum RSS / peak physical footprint | 107495424 / 50528952 bytes |
| Separate native / checker process limits | 1 GiB / 512 MiB |
| Actual-capture mutation and reader tests | 24 passed |

The 24 checks include coherent wrong results beyond numerical gates, adjacent
repeat/partition drift within the tolerance gate, altered original spans or
weight digests, changed/aliased allocations, guards/status/inactive capacity,
wrong parent/chunk staging, missing outcomes and raw-reader bounds. Required
source checks pass: formatting, strict lint with and without Metal, 351 CPU tests,
382 Metal-feature tests, both iOS compile targets, the optimized writer test and
all three optimized row-view tests. The selected unit/CLI/affine/row-view binaries and executed release driver are
snapshotted. All twelve source-check and five diagnostic processes/groups close
successfully.
Production numerical/shader/admission code and prior tiny test bodies are
unchanged. Process observations include scalar reference work, hashing and raw
capture and are not inference latency or full-model residency measurements.

Large per-layer table gather, resident full graph/KV/scheduling, full native
activation and continuation parity, historical cross-precision/source and long
rotary discrepancies, quality, representative performance, app/physical iPhone
admission and 131072 output tokens plus a nonempty prompt remain open. No model
capability, app eligibility or context limit is raised.

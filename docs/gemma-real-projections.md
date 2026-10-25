# Real Gemma packed projections on Metal

All 276 geometry-fitting projections in the frozen four-token Gemma E2B trace
now pass explicit Apple M1 Pro execution. Complete device outputs match both
rounded staged-FP32 mathematical outputs and separately computed native ordered
FP64 dots rounded to FP32. This qualifies those standalone projections; it does
not execute the complete decoder or grant Gemma support to Chat or the iPhone.

## Inputs, layouts and ownership

Use the authenticated original `mlx-community/gemma-4-e2b-it-4bit` checkpoint at
revision `238767527555cb75a05732a84dff5d6ba0dd6809`, with the
[frozen complete projection references](gemma-projection-trace.md). The model
container is 3,550,670,554 bytes at SHA256
`038e39a37a7667373d2c3991375446b10c96ae1d717a68674870343db376b76e`.
The owned controller authenticates the complete file before/after its owned
commands; independent replay authenticates it before streaming verification; the native diagnostic authenticates its exact header
and reports every original tensor span's offset, shape, dtype, byte count and
hash. No converted weights substitute for original packed U32 codes and BF16
scales/biases.

The trace uses token IDs `[2,100,257,100]` at positions 0..3. Preserve the
206 captured-state input reconstructions and 70 explicitly derived mathematical
GELU products. These are frozen reference inputs, not evidence that a native
activation path produces the same inputs. In particular, the acknowledged
Python `x**3` activation recipe remains distinct from native activation parity.
Expected outputs are the complete pre-norm/pre-scale/pre-residual linear states.
The mathematical reference uses FP64 accumulation then FP32 output rounding;
the new native ordered reference also rounds its final dot to FP32. Neither is
an unrounded exact-dot or ideal-history graph qualification.

Use one cached Context and one matrix owner at a time. Four-token FP32 input
and output rows are contiguous. Upload each matrix's original row-major
four-bit/group-64/BF16 tensors once, reuse capacity four for two full calls,
four single-token partitions and a full resumed call, then drop that owner.
The underlying shader, admission thresholds and production projection API are
unchanged. This task adds an ignored diagnostic and independent capture checks.

The tied head has 262144 rows, exceeding the unchanged 16384-row limit. Its
constructor must return exactly `Gemma affine fixed geometry/format outside bounds`
before reading any large head spans. Retain the rejection; no empty-input or
other constructor failure may stand in for it. Head/table chunking remains next.

## Complete observations and failure retention

The native test retains 3036 complete arrays / 76,021,760 bytes: all native
ordered rows and every actual device output, under a 128 MiB binary cap.
Each array is at most 196,608 bytes. All six complete device allocations are
observed as word counts, byte counts, SHA256 digests and seven trailing guard
words. Independent reconstruction includes original tensor padding, inactive
input/output capacity and status. GPU weights are retained as cryptographic
observations, not additional plaintext copies of the complete GPU allocations.

A completed device output is captured separately from the transactional caller
output, then their bytes are compared. On an apply failure, only a retained
command with checked Completed status permits shared-buffer reads. The original
apply outcome and completed device metadata are flushed before a fallible raw
append. A portable native regression rejects writes through a read-only file
and verifies that the original outcome survives. The controller independently
retains raw/partial binary and text on failure, preserves first identities and
verifies every owned child/process group is absent.

## Checks and measurements

Formatting, both strict lint configurations, 347 CPU / 375 Metal-feature tests,
iOS device/simulator compilation and the exact optimized failure-writer test
pass. Ordinary tests do not open the model or require a GPU. The first source
check attempt failed only the new test's equivalent boolean expression under
strict Clippy; it is retained separately and receives no successful-run credit.
The corrected fresh check target passes without relaxing any check.

Actual GPU execution completes 276 cases and 1932 steps: 552 full repeats,
1104 partitions and 276 resumes. Independent replay checks 7,602,176 repeated
GPU output values under 15,204,352 dual-reference comparisons, plus 3,801,088
native-ordered versus rounded-mathematical comparisons. Maximum absolute error
and gate fraction are zero for these rounded FP32 outputs. Repeats, partitions
and resumes agree byte for byte. The unchanged numerical gate remains
`1e-5 + 1e-4 * abs(reference)`; no tolerance or earlier failed result changes.

Eighteen replay regressions pass. Five cover malformed JSON, bounded growth/
short reads and cleanup after the second file-open fails. Thirteen exercise
the actual capture, including typed counters, original span identities, weight/
guard/status digests, wrong partition input, missing records and extra metadata.
Coherently changed payloads/digests specifically fail the repeat, partition and
numerical gates rather than merely failing their hashes.

Hardware is MacBookPro18,3, Apple M1 Pro, 16 GiB, macOS 26.6.2 (25G83), with
queried SIMD width 32. The largest exercised owner totals 10,838,428 bytes,
including its 240-byte Rust structure and six guarded numerical allocations.
This excludes the separate Context policy allowance, caller/reference storage
and host tensor buffers. One-owner reuse does not qualify simultaneous owners.

| Owned process | Maximum RSS bytes | Peak physical footprint bytes | Separate allowance |
| --- | ---: | ---: | ---: |
| Native diagnostic | 65,585,152 | 57,918,064 | 1 GiB each |
| Independent replay | 95,862,784 | 92,095,280 | 512 MiB each |
| Replay regressions | 136,380,416 | 107,184,968 | 512 MiB each |

The resource policy concerns these diagnostic processes, not whole-model
residency or device admission. Scalar oracle work, hashing, upload and exhaustive
readback make this unsuitable as an inference-latency benchmark. No performance
claim follows. Exact scripts/interpreter and checked source inputs are bound;
a full standard-library/native-runtime closure is not claimed. The
[receipt](../benchmarks/results/2026-10-25-gemma-real-projections-v1.json) binds
build checks, actual outputs, process closure and retained failed evidence.

## Remaining work

Head/table chunking and gathers, resident whole-model weights/KV/attention,
native activation parity, pointwise composition and scheduling remain open.
Preserve all existing source/cross-precision failures: five mathematical arrays /
57 values, 27 native FP64-ideal logit failures per schedule, 15 dense and 15 packed
source-GPU intermediate failures, and eight upstream long-position rotary
failures. This standalone trace does not waive them. Broader quality, measured
inference performance, app/ABI integration, physical iPhone qualification and
131072 output tokens plus a nonempty prompt are still required. G2, G3 and G4
are not complete.

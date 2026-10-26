# Gemma chunk scratch independent of total context

This is the first capacity-separation mechanism in the
[active context work](long-reply-engine-plan.md). It changes host allocation
without increasing Gemma's current 32-position executable ceiling. Model-owned
cache geometry, history capacity, rotary positions, scalar arithmetic and
sampled/committed/pending-token semantics remain unchanged. Larger-context and
real E2B boundary qualification remain separate work.

## Allocation and API

The [raw model](../src/model/gemma_packed.rs) now exposes
`planned_session_bytes_with_chunk` and `new_session_with_chunk`. They validate
`1 <= chunk_rows <= layout.positions` before allocation. Existing raw methods
delegate with the full current position bound, preserving their accepted chunk
sizes. A session reports its per-call limit through `chunk_capacity()`.

Activations, producer K/V proposals and both private/published logits use chunk
rows. Attention's two diagnostic buffers use `heads * chunk_rows * positions`,
preserving the existing reduction and masking order. Thus logits require
`2 * chunk_rows * vocab * sizeof(f32)`, and attention scratch requires
`2 * heads * chunk_rows * positions * sizeof(f32)`. These are independent of
the model's immutable weight allocation. Source/history/cache ownership is not
silently resized by this task.

Generation's existing `prefill_chunk_size` now selects physical host scratch.
Scalar construction, generation admission and hybrid preadmission use the same
checked calculation. The extra session field is included in metadata size.
Actual GPU projection/gather pools still admit 32 rows and remain fully charged;
shrinking those pools is separate work. Historical admission receipts retain
their original source and physical-allocation scope.

The [forward boundary](../src/model/gemma_packed/forward.rs) rejects a request
larger than its chunk scratch before backend readiness or numerical writes,
using a distinct chunk-limit error. Total context overflow retains the existing
capacity error. All producer frames remain unpublished until every layer and
the final head/softcap succeed. Failed work leaves prior KV/history/logits
intact, while private scratch can be discarded and reused for a retry.

## Verification scope

The new tiny [Rust checks](../src/model/gemma_packed/tests/chunk.rs) use chunk
bounds 1, 3, 8 and 32 through the complete existing 32-token fixture. Every
successful chunk is compared against both independent arithmetic contracts.
They include the final two-row chunk for bound three and local-window crossings,
exact/minus-one budgets, invalid bounds 0/33, oversized calls after a prefix,
total-context overflow, the exact injected final failure, retry and reset.
Steady forward, rejection, late failure and reset allocation counts are checked.

The [stdlib replay](../tools/reference/check_gemma_chunk_scratch.py) independently
reconstructs all 23 scratch dimensions, cache/staging payloads and session byte
accounting. It checks every emitted row against the original frozen fixture;
target rows do not become expected values. Six semantic
[corruption controls](../tools/reference/test_check_gemma_chunk_scratch.py)
exercise numerical values, coherent memory changes, missing/rejected state
records and wrapper provenance.

Generation tests retain exact-prefix reuse, pending-token hiding/discard,
shortened/changed-prompt replay, cancellation, stop and output-failure behavior.
The synthetic GPU control checks actual admission and source ownership with
three-row host scratch. Its reader requires explicit chunk-aware admission;
legacy records keep the old default interpretation and existing controls.

## Actual checks, 26 October 2026

Formatting and strict all-target Clippy pass for default and Metal builds. The
full default CPU suite passes 357 tests, with 16 explicitly ignored. The scoped
Metal build passes 16 Gemma tests with three ignored; its synthetic GPU control
is then run explicitly and passes. Nine portable hybrid-reader methods and six
semantic CPU-transcript controls pass. Larger real-model suites and the completed
original BF16 captures are not rerun by this task.

Independent replay checks 48 emitted row records and 4352 scalar comparisons
against both frozen contracts; the largest error-to-bound ratio is
`0.017896043200363743`. Numerical gates and arithmetic are unchanged.

| Chunk rows | Tiny test session bytes | Private + published logits bytes | Attention scratch bytes |
| ---: | ---: | ---: | ---: |
| 1 | 49632 | 136 | 512 |
| 3 | 77040 | 408 | 1536 |
| 8 | 145560 | 1088 | 4096 |
| 32 | 474456 | 4352 | 16384 |

These are the instrumented tiny test-build session's actual ownership, excluding
the immutable model and other caller/process ownership. They are not E2B or
release-process footprint estimates. Exact/minus-one admission and no-allocation
steady forwards, invalid requests, late failure and reset pass; allocation
addresses remain stable. The default 32-row numerical buffers are preserved;
the new bound field adds one `usize` to session metadata, which is accounted for.

The GPU control independently checks 35 matrices/105 unchanged buffer owners,
source mismatch, exact/minus-one budgets and two injected pre-command failures.
Three-row hybrid generation admits 101347012 bytes including model, backend
pools, metadata and opaque policies. GPU pools still use their existing 32-row
bounds. In-flight GPU failure and complete real E2B qualification remain false.

On Apple M1 Pro / MacBookPro18,3 / 16 GiB / macOS 26.6.2 (25G83), direct CPU and
synthetic GPU test processes peak at 4276608 and 13795928 physical bytes; RSS
peaks are 14434304 and 35012608 respectively. Both fit the 512 MiB diagnostic
policy. Compilation is separate from these runtime measurements. No latency,
speed or phone-fit claim follows. All 14 collection process groups close;
protected release binaries are unchanged. See the
[public evidence](../benchmarks/results/2026-10-26-gemma-chunk-scratch-v1.json).

## Remaining delivery work

This mechanism does not qualify the whole real-E2B 32-position boundary, a
511/512/513 or 1024/1025 boundary, the complete frozen quality workload or the
selected model's full shared input/output window. Next separate actual total
context/cache/history capacity from the remaining model-wide bound, then
qualify larger state/replay/failure and process-memory boundaries. Keep original
arithmetic contracts, the eleven new cross-precision failures and prior
source/BF16/GPU/rotary gaps. No app/profile/download eligibility changes here.

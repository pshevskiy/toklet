# Rotary/cache-write fusion experiment

The hypothesis is that combining K rotary storage with K/V cache writes can
reduce dispatch and intermediate traffic without changing stored numerical
values. This is an individual candidate against Split execution with matched
projection, normalization and full-logit readback. Both frozen Q4 quality
rejections remain unchanged. The candidate is numerically qualified but rejected for negligible request gain.

## Mechanism and ownership

K/V input is Half `[rows, kv_heads, head_dim]`; separate session-owned caches
are `[kv_heads, capacity, head_dim]`. A thread owns one split-half rotary pair
and copies the matching raw V words. Absolute rotary position is independent
of cache placement. Each pair uses the existing FP32 trigonometry and operation
sequence, explicitly rounds both K outputs to Half and preserves finite checks
before writing cache. V copying retains the split kernel's raw-word semantics;
upstream projection owns its finite checks. No head expansion or cache-prefix
copy is introduced. The active chunk may be written before validation, but
committed positions advance only after the complete command succeeds.

Ordinary fusion reads K/V and writes K/V cache: eight bytes per K element,
versus twelve across split rotary and cache kernels. It removes four bytes of
rotated-K write/read traffic and one dispatch per layer. Diagnostics optionally
materialize actual fused rotated K, preserving all stage shapes and order.
Existing rotated-K scratch remains admitted for late diagnostics, so no
resident-memory saving is claimed. Both modes retain one completed command per
chunk. Profile admission removes one event per layer: larger Q4 uses454 instead
of 482. Larger Half profiling still exceeds the existing 1,024-event ceiling.

## Numerical and state qualification

The primitive has 1,280 exact split/fused cases spanning rows 1/3/8/9/17,
KV heads 1/2, dimensions 4/8/64/128, placement 0/5, absolute positions 0/17/513/2031,
zero/four untouched suffix slots and both diagnostic flags. Whole K/V caches,
readonly inputs, prefix/stale slots, the last capacity slot and guards are
checked. Raw V includes nonfinite words without adding new rejection semantics.
Four isolated K MAX/+Inf/-Inf/NaN cases require nonfinite status while preserving
all positions outside the active chunk and optional diagnostic omission.

Tiny Half and compensated Q4 each preserve 4,550 complete stage arrays against
Split. The inventory independently checks all 35 identities, rows, columns and
complete lengths. Ordinary Half/Q4 at offsets 0/5 compare 12 generated outputs per
case, full distributions/history, eight allocation-free warmed steps, exact
budget admission and one-byte-short rejection, prefix edits/shortening/full
reuse, cancellation, hidden stops and output failure. The failure/profile test
preserves the last committed selection, requires replay after completion failure
and enables diagnostics after profile admission.

Larger Half/Q4 each pass 969 unchanged independent selected-stage comparisons,
covering layers 0/1/27 and complete selected arrays. Maximum gate fractions remain
0.5792791429143427 and 0.7321240893134191. Scores/probabilities are outside that
inventory. Saved reports explicitly record Fused rotary/cache and FullLogits.
Native lifetime peaks are 3,150,187,256 Half and 1,341,539,840 Q4 bytes, below 6 GiB.
These are short numerical checks, not lossy quality acceptance.

Formatting, strict all-target Clippy in both configurations, 87 CPU/94 Metal
ordinary tests and179 Python experiment tests pass. All six native checks
execute one passing test each. A stale alias in copied Python collector tests
was fixed after the first run reported five errors; the complete rerun passes.
Astra's source review is clear. Exact prototype sources and immutable
native/worker binaries are retained before measurement.

## Controlled measurement

The fixed preselected larger Q4 model uses compensated eight-row projections
and compensated RMS in both sessions. The frozen 512-token prompt produces 128
outputs. One immutable model is shared with separate KV/scratch. Both variants
use FullLogits and CPU greedy. Three alternating warmup pairs precede ten
balanced measured pairs. Timing includes completed GPU work, readback, selection
and a preallocated token callback. All 128 IDs, committed history and every final
logit bit are compared after timers. The pending final sample is then discarded.
The bounded owned collector verifies mode, geometry, state, timings, memory,
frozen pins and cleanup; exclusive files preserve raw events and failures.
Builds, tests and large hashes do not overlap measurement.

## Result and discard decision

Apple M1 Pro, 16 GiB, macOS 26.6.2, Rust 1.90.0 and Cargo release defaults.
AC power, low-power mode zero and unchanged power/thermal endpoints are recorded;
no thermal/performance warning is recorded. Continuous sensors and energy are
unmeasured. The source parent is `b2ba643`, with exact prototype
sources retained rather than attributing their changes to that committed parent.

| Variant | Median complete request | MAD | Measured range |
| --- | ---: | ---: | ---: |
| Split | 78.9783902085 s | 16.083 ms | 78.853700500–78.995453750 s |
| Fused | 78.9668645830 s | 8.659 ms | 78.690431792–78.986358084 s |

Median improvement is **0.014593%**, only 11.526 ms. It is below Split dispersion
and far below the frozen 5% threshold. Reject the candidate; no promising result
warrants a confirmation run. Fewer dispatches and intermediate bytes do not
produce a useful complete-request improvement on this workload. This result
does not identify the cause or establish reliable tail percentiles.

All 26 requests retain all 128 generated IDs, exact committed histories and
every final-logit bit. Persistent inventories remain equal and fixed. Combined
admission is 1,691,328,140 bytes; joint native lifetime peak is 1,534,068,848 bytes.
Ten measured pairs total 1578.940851544 seconds of completed repeated requests.
The [saved paired run](../../benchmarks/results/2026-10-14-metal-rotary-paired-v1.json)
has 15 exact events, clean completion, unchanged identities and empty cleanup
errors. Larger [Half](../../benchmarks/results/2026-10-14-rotary-half-parity-v1.json)
and [Q4](../../benchmarks/results/2026-10-14-rotary-q4-parity-v1.json) numerical reports
remain separate from this performance result.

The experimental kernel, API, tests and harness are removed from the live
package. Only task-owned files whose bytes still matched the measured source
snapshot were restored to `b2ba643`; no unrelated edits were overwritten.
The bound local prototype snapshot and immutable binaries preserve reproduction.
All ordinary formatting/Clippy/tests are checked again after discard. The
[qualification receipt](../../benchmarks/results/2026-10-14-rotary-cache-qualification-v1.json)
distinguishes measured prototype paths from restored sources and binds both.
Protected CPU binaries are unchanged. Astra reviewed source, native checks,
saved-data replay, provenance and the discard decision.

## Local reproduction

The ignored source snapshot is `artifacts/m5-rotary-prototype-source-v1/`;
retained binaries are `artifacts/m5-rotary-native-binaries-v1/`. Symlinks share
existing models/artifacts/fixtures without duplicating model payloads. Rebuild
only into an absolute isolated `CARGO_TARGET_DIR`, keeping protected CPU binaries
unchanged. To replay from the repository root with a fresh exclusive output:

```sh
.venv/bin/python artifacts/m5-rotary-prototype-source-v1/tools/experiments/metal_rotary_cache.py --binary artifacts/m5-rotary-native-binaries-v1/metal_rotary_cache --output benchmarks/results/2026-10-14-metal-rotary-rerun.json
```

No kernel is adopted, no quality gate or default changes, and overall M5 release
acceptance remains open under both Q4 failures. Residual/post-RMS fusion and the
remaining full release evidence are still pending.

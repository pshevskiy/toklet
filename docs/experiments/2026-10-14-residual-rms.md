# Residual/post-attention RMS fusion experiment

The hypothesis is that fusing the attention residual addition with its following
RMSNorm can reduce dispatch and input traffic without changing either Half
storage boundary. This individual experiment compares Split and Fused with
matched projection, normalization and full-logit readback. Both frozen Q4 quality
rejections remain unchanged. The candidate is numerically qualified but rejected for slower complete requests.

## Mechanism and ownership

Residual, delta and normalized activation use contiguous Half `[rows, hidden]`;
post-attention weights use `[hidden]`. Each fused row has one owner. It adds x and
delta in FP32, explicitly rounds/stores/checks the residual Half, then squares
that rounded value in the same feature order. Original retains ordered FP32
summation; Compensated retains the existing two-component sum. Final division,
epsilon, square root, inverse, weight multiplication and checked Half output are
unchanged. The rounded residual remains resident for the later MLP addition.
Only this within-layer residual/post-RMS pair is fused. No cross-layer or final
normalization fusion is introduced.

Split residual reads x/delta and writes x, then RMS reads x twice and weights
once and writes norm: fourteen bytes per activation including normalization weights, excluding
uniform/status traffic. Fused reuses the rounded value for square accumulation, removing one
two-byte x read and one dispatch per layer. Existing scratch remains admitted;
no resident-memory saving is claimed. Diagnostics snapshot both actual fused
outputs in the existing stage order. One completed command per chunk and failure
publication rules are preserved. Profile admission removes one event per layer:
larger Q4 uses 454 instead of 482; larger Half still exceeds the existing 1,024
counter-event limit and remains unsupported for profiling.

## Numerical and state qualification

The guarded primitive compares 4,379,072 elements across 88 cases and both
Original/Compensated modes at epsilon 1e-6/1e-5. All 63,488 finite Half x words
cross fifteen signed/zero/subnormal/extreme delta values at hidden 1; separate
safe all-finite zero-delta sweeps require status exactly zero. Other cases span
rows 1/3/9/17 and hidden 8/64/128/896/1536. Complete residual bits and finite
normalized bits must match Split. Unpublished nonfinite outputs require matching
classification/status; all output/input guards and readonly delta/weight buffers
are checked. No valid-output tolerance is relaxed.

A separately SHA-pinned four-case larger dynamic-range capture at hidden 1536
compares both modes exactly to Split and preserves all 66 storage differences
between Original and Compensated. Two isolated finite MAX+MAX residual cases
require nonfinite status with intact guards. Sticky failure elsewhere cannot
hide a false finite rejection or a missed overflow.

Tiny Half and compensated Q4 each preserve 4,550 complete stage arrays, with
independent checks for all 35 identities, shapes and full lengths. Ordinary
Half/Q4 at offsets 0/5 compare 12 generated outputs per case, full logits/history,
eight warmed steps with zero Rust allocations, exact budget admission and
one-byte-short rejection, prefix edits/shortening/full reuse, cancellation,
hidden stops and output failure. The failure/profile test preserves the last
committed selection, requires replay after completion failure and admits late
diagnostics after profiling.

Larger Half/Q4 each pass 969 unchanged selected-stage gate comparisons across
layers 0/1/27. Scores/probabilities remain outside this selected inventory.
Maximum gate fractions remain 0.5792791429143427 and 0.7321240893134191. Reports
explicitly record Fused residual/RMS and FullLogits. Native lifetime peaks are
3,147,549,456 Half and 1,340,851,712 Q4 bytes, below 6 GiB. These short numerical
checks do not establish lossy model quality or release performance.

Formatting, strict all-target Clippy in both configurations, 87 CPU/94 Metal
ordinary tests and 179 Python experiment tests pass. All seven native checks
execute one passing test each. Astra's source review is clear. Exact prototype
sources and immutable native/worker binaries are retained before measurement.

## Controlled measurement

The fixed preselected larger Q4 model uses compensated eight-row projections
and compensated RMS in both sessions, with FullLogits and CPU greedy. One
immutable model is shared with separate session-owned KV/scratch. The frozen
512-token prompt produces 128 outputs. Three alternating warmup pairs precede
ten balanced measured pairs. Complete warm generation includes GPU completion,
readback, CPU selection and the preallocated token callback. All 128 IDs,
committed histories and every final-logit bit are checked after timers. The
pending final sample is then discarded. The bounded owned collector verifies
modes, geometry, state, completed timings, fixed memory, frozen pins and cleanup,
retaining exclusive raw files and failure causes. Builds, tests and large hashes
do not overlap measurement.

## Result and discard decision

Apple M1 Pro, 16 GiB, macOS 26.6.2, Rust 1.90.0 and Cargo release defaults.
AC power, low-power mode zero and unchanged power/thermal endpoints are recorded;
no thermal/performance warning is recorded. Continuous sensors and energy are
unmeasured. The source parent is `5de31e1`; exact prototype sources
are retained separately from that committed parent.

| Variant | Median complete request | MAD | Measured range |
| --- | ---: | ---: | ---: |
| Split | 78.920911979 s | 46.957 ms | 78.773302209–79.006145541 s |
| Fused | 79.357371687 s | 44.824 ms | 79.071335875–79.464723625 s |

Fused is **0.553034% slower**, adding 436.460 ms to the median. Every measured
pair is slower. Reject the candidate; no benefit meets the frozen 5% threshold
or warrants a confirmation run. Fewer dispatches and input bytes do not produce
useful complete-request latency on this workload. This result does not identify
the cause or establish reliable tail percentiles.

All 26 requests retain all 128 generated IDs, exact committed histories and
every final-logit bit. Persistent inventories remain equal and fixed. Combined
admission is 1,691,328,140 bytes; joint native lifetime peak is 1,535,330,440 bytes.
Ten measured pairs total 1,582.416389125 seconds of completed repeated requests.
The [saved paired run](../../benchmarks/results/2026-10-14-metal-residual-paired-v1.json)
has 15 exact events, clean completion, unchanged identities and empty cleanup
errors. Larger [Half](../../benchmarks/results/2026-10-14-residual-half-parity-v1.json)
and [Q4](../../benchmarks/results/2026-10-14-residual-q4-parity-v1.json) numerical
reports remain separate from this performance result.

The experimental kernel, API, tests and harness are removed from the live
package. All ten owned paths were verified against the measured snapshot before
restoring `5de31e1`; no unrelated edits were overwritten. Bound local
prototype sources and immutable binaries preserve reproduction. Ordinary
formatting/Clippy/tests pass again after discard. The
[qualification receipt](../../benchmarks/results/2026-10-14-residual-rms-qualification-v1.json)
distinguishes measured prototype paths from restored sources and binds both,
including the frozen reduction witness, actual FP32 reference arrays and oracle
helpers. Protected CPU binaries are unchanged. Astra reviewed source, native
checks, saved-data replay, provenance and the discard decision.

## Local reproduction

The ignored source snapshot is `artifacts/m5-residual-prototype-source-v1/`;
retained binaries are `artifacts/m5-residual-native-binaries-v1/`. Symlinks share
models/artifacts/fixtures and pinned reference helpers without duplicating model
payloads. Rebuild only into an absolute isolated `CARGO_TARGET_DIR`. To replay
from the repository root with a fresh exclusive output:

```sh
.venv/bin/python artifacts/m5-residual-prototype-source-v1/tools/experiments/metal_residual_rms.py --binary artifacts/m5-residual-native-binaries-v1/metal_residual_rms --output benchmarks/results/2026-10-14-metal-residual-rerun.json
```

No fusion is adopted, no quality gate or default changes, and overall M5 release
acceptance remains open under both frozen Q4 failures. This completes the last
named fusion experiment; M5.4 synthesis/closure follows final task acceptance.

# Q4 compensated projection and larger GPU geometry

This task adds an explicit experimental `ProjectionMode::Q4Compensated` and
admits the previously selected Qwen2.5-1.5B geometry. CPU and the Original Metal
projection mode remain defaults. This is a numerical component qualification,
not M5 completion, a speed claim, or larger Q4 quality acceptance.

The packed Q4 v1 representation, quantizer, model selection, corpus and frozen
gates are unchanged. Each projection still reads row-local groups directly,
reconstructs the same rounded FP32 weight, adds bias after the complete dot,
and stores one Half activation. One queried SIMD group owns each output row.
There are no expanded model weights or additional persistent activation buffers.

## Numerical mechanism

The candidate uses two-component FP32 Neumaier sums, an explicit FMA product
residual, and a shuffle tree merging disjoint lane subtrees. Width is queried
from its compiled pipeline; launch bounds and shader tails remain checked.
Ordinary non-contracted arithmetic and the existing precise compiler policy
remain in place. Original and Q4Tiled retain their arithmetic.

The first candidate passed projection bounds and cancellation but still failed
the expanded tiny decoder. A bounded capture proved that layer 1's final offset
row had an exactly matching Half attention-context input. O projection feature 19
was `0.16259765625` instead of `0.1627197265625`. Independent replay gives:

| Calculation | Value before Half |
| --- | ---: |
| Ordered FP32 dot | 0.1626587063074112 |
| FP64 dot of decoded FP32 weights and Half inputs | 0.16265869280323386 |
| Exact Half midpoint | 0.16265869140625 |

Both independent dots round directly to the upper Half. FP64-to-FP32-to-Half
loses the small positive residual, lands on the midpoint, and selects the lower
Half. The initial compensated candidate had that same double-rounding error.

The corrected storage boundary first rounds the represented two-component sum
to an odd FP32 significand when inexact, moving one ULP toward its recovered
residual if necessary. Half nearest-even conversion then preserves which side
of the midpoint the sum occupies. Nonfinite handling stays with `checked_half`.
This preserves information in the represented sum; it does not claim exact
arithmetic throughout the decoder.

The native rounding test covers 72 signed cases at even/odd midpoint, exponent,
subnormal, zero and finite-overflow boundaries. All exact FP64-to-Half bits,
status and inactive guards pass. Exactly 22 cases distinguish direct FP64-to-Half
from FP32-then-Half. The initial assertion incorrectly predicted 24 witnesses;
NumPy independently confirmed 22 because the zero boundary's smallest product
is already a full FP32 ULP. Initial count-only failures are retained; the 72
numerical comparisons and all decoder gates were unchanged.

The corrected candidate passes all 4550 complete stage comparisons from the
existing six-case expanded tiny fixture with full/8/3/1 chunks, including
offset 5 and irregular final chunks. The Original/Tiled expanded failure remains
separately runnable and unresolved on those modes.

## Larger geometry and remaining failure

The added geometry is exactly hidden 1536, MLP 8960, query 12/KV 2, head 128 and
28 layers. Existing vocabulary, rotary, epsilon, context and chunk<=128 bounds
remain. Q4 profiling needs 482 events within the existing 1024-event limit.
Half needs 1070 and is rejected before diagnostic allocation or admission
mutation; ordinary execution remains usable after rejection.

The native suite compares 969 complete selected-stage arrays over the four
frozen short cases and chunk1/3/full, including offset 5. This is 204 distinct
selected arrays repeatedly compared; attention scores/probabilities are outside
this diagnostic inventory. Payload hashes, probe bindings, shapes, per-row
greedy IDs, model/config identity and final source endpoints are checked.

The first larger Half run passes with maximum gate fraction 0.579279 and native
lifetime peak 3,148,073,720 bytes. Q4 actual resident weights are 1,223,992,516
bytes and bounded 482-event profiling works. Original Q4 fails 33 comparisons,
maximum fraction 3.345601. Compensation without the storage correction fails 33;
the corrected candidate fails 34, maximum fraction 3.236828. All failures are
retained as failed reports. The tiny correction does not fix the larger decoder.
Further identical-input normalization and rotary localization is required before
larger quality/performance work can be accepted.

## Reproduction and decision

Use an isolated `CARGO_TARGET_DIR=artifacts/m5-checks` for builds. Explicit native
tests run sequentially on the M1 Pro, without concurrent builds or benchmark
work. The qualification receipt binds the final sources, binaries, fixture
identities, raw logs and reports. The ordinary format/lint/default/Metal suites
and targeted native regressions are recorded there.

Retain this bounded opt-in candidate for numerical investigation and its passing
expanded tiny component; no performance adoption or default change is made.
Larger Half geometry is qualified by its frozen storage envelope. Larger Q4 is
experimental and fails its unchanged numerical gate. The 0.5B Q4 held-out
rejection is unchanged. M5.2, M5.3, M5.4, M5.6 and the M5 exit remain open.

Astra accepted the code and evidence for this component scope after verifying all
1,587 bindings and check totals. See the [qualification receipt](../../benchmarks/results/2026-10-13-q4-compensated-storage-qualification-v1.json),
[midpoint replay](../../benchmarks/results/2026-10-13-q4-midpoint-localization-v1.json),
[passing larger Half run](../../benchmarks/results/2026-10-13-larger-half-gpu-parity-v3.json),
and [failed larger Q4 run](../../benchmarks/results/2026-10-13-larger-q4-gpu-parity-v3.json).
The larger v3 runs precede a documentation-comment-only source change; they are
not claimed to use the final rebuilt executable. All 15 final native regressions
use the rebuilt checked binaries.

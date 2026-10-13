# Compensated RMS reduction for the larger Q4 decoder

The separate opt-in `NormalizationMode::Compensated` clears the larger Q4
numerical gate when combined with `ProjectionMode::Q4Compensated`. Original
normalization and projection remain defaults. This qualifies short numerical
execution, not held-out quality, performance or the M5 exit.

## Identical-input localization

The previous larger candidate failed 34 comparisons under frozen gates. A bounded
capture of the four existing first-layer cases retains exact FP32 bits for
embedding, input normalization, Q/K projections and rotary outputs, plus the
source-bound 1536-element Half norm vector. Source/config/packed identities are
pinned before loading and checked at the endpoint. The source vector read is
bounded by the fixed safetensors header, BF16 dtype, shape and offsets; no second
model is retained. The capture is diagnostic and supplies no acceptance gate.

Independent CPU replay of all 12 input rows finds:

| Arithmetic | Result |
| --- | --- |
| Ordered FP32 square summation and remaining RMS arithmetic | Exact GPU Half outputs |
| Frozen NumPy FP32 reference | 66 Half values differ from GPU; maximum 0.00048828125 |
| Compensated square sum, original FP32 remaining arithmetic | Exact frozen NumPy and full FP64 Half outputs |

For identical captured K projections, CPU FP32 rotary matches GPU throughout.
Only three Q rotary values differ across these cases, by at most 0.00048828125.
This supports changing the square-sum reduction first. Rotary and SiLU are
unchanged; no cause is inferred from a downstream error alone.

## Implementation

Each Half square has an exact FP32 product. The candidate uses the existing
two-component Neumaier addition for their positive sum, then rounds the sum to
FP32 once. Division by hidden width, epsilon addition, square root, reciprocal,
multiply order and final Half conversion remain unchanged. The uniform mode
branch occurs once before the feature loop. All three decoder RMS sites pass
the mode through the single checked encoding helper.

`Session::with_execution_modes` admits explicit projection and normalization
choices. Existing constructors delegate to Original normalization. Model weights,
KV layout, persistent buffers, command scheduling and profile-event count are
unchanged. The session field is included in the existing metadata admission.
The Q4 representation, quantizer, checkpoint and every frozen numerical/quality
gate are unchanged.

## Native evidence and decision

The isolated RMS test checks 43,008 exact Half values against an independent CPU
calculation: ordered FP32 for Original, or FP64 square sum rounded to FP32 and
the original FP32 remaining arithmetic for Compensated. It includes all captured
rows, zero and high dynamic-range inputs, completed status and inactive guards.
The combined compensated modes pass all 4,550 complete expanded tiny stage
comparisons under the unchanged fixture gates.

The final-source larger v5 run passes 969 complete selected-stage comparisons, all required
row greedy IDs and offset/chunk checks. Maximum gate fraction is
0.7321240893134191; native lifetime peak is 1,344,226,768 bytes. Q4 weights occupy
1,223,992,516 bytes and the existing 482-event profile remains bounded. The prior
failed v1/v2/v3 reports are retained. Attention scores/probabilities are outside
this selected-stage inventory; no full held-out or task quality is inferred.

Retain this explicit numerical candidate for matched quality evaluation. There
is no speed claim or default change. The 0.5B Q4 held-out rejection is unchanged;
M5.2 and the M5 exit remain pending their quality and performance gates.

See the [identical-input replay](../../benchmarks/results/2026-10-13-larger-rms-rotary-localization-v1.json),
[passing larger Q4 v5 run](../../benchmarks/results/2026-10-13-larger-q4-gpu-parity-v5.json),
and [qualification receipt](../../benchmarks/results/2026-10-13-compensated-rms-qualification-v1.json).

The final-source larger Half Original-normalization run also passes all 969
comparisons (maximum fraction 0.5792791429143427; native peak 3,147,942,648
bytes). The final 17 native regressions cover original Half/Q4 state, failure,
capacity and allocation contracts as well as both compensated mechanisms.
They run sequentially without concurrent builds or benchmark work.

Astra accepted this code task and short numerical evidence after independently
verifying all 1,412 bindings, 87 CPU tests, 94 Metal-feature tests and 17 final
native regressions. Formatting and strict Clippy pass in both configurations.

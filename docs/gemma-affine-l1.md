# Independent L1 projection admission reference

The tighter per-token L1 proof covers all seven previously inconclusive
FF-down envelopes in the frozen four-token Gemma trace, under the existing
error allowance. The proposed combined classification is 269 legacy passes,
seven additional L1 passes and one full-head geometry rejection. This task
freezes independent evidence before any engine admission change. It executes
neither GELU nor projections and does not qualify a resident Gemma graph.

## Proof and ownership

For each row of K finite FP32 inputs, accumulate an upward FP64 bound
`A >= sum(abs(x_i))`. Skip exact zero terms; the first positive term is exact.
For every subsequent positive addition, compute the nearest FP64 sum and then
advance one representable value toward positive infinity, including exact sums.
This conservative outward step preserves small positive terms even when
nearest FP64 summation loses them.
Keep independent token rows separate: a batch takes the maximum row A, never
the sum across tokens.

Using the authenticated all-codebook maximum weight, compute upward
`S >= maxW*A`, then upward `U >= 2*K*2^-53*S`. Require the unchanged
`S <= F32_MAX/4` and `U <= 1e-5/16`, with the existing K/weight/codebook domain.
Finite FP32 products are exact in FP64, and `gamma_K <= 2*K*2^-53` bounds the
ordered-FP64 summation discrepancy against an exact dot. The L1 proof bounds
absolute products without multiplying the maximum input by all K coordinates.
It does not prove every GPU rounding error or alter the dual numerical gate
`1e-5 + 1e-4*abs(reference)`.

The future engine change must keep the current max proof first and try L1 only
after its rejection. Both failing must preserve the old error and pre-staging
transaction behavior. No engine code changes in this reference task.

## Frozen synthetic witnesses

Nineteen cases contain 625,152 bytes of FP32 inputs and 47 output references.
Each conceptual matrix uses original four-bit/group64/BF16 representation:
all packed codes and scales are zero; every output row's BF16 bias equals the
listed weight. The fixture stores the inputs and this explicit reconstruction
recipe without redundant zero-filled matrices.

The producer computes separate ordered FP64 dots and exact Fraction dots.
An independent checker reconstructs inputs, uses Decimal precision 400 for
mathematical dots, and verifies upward bounds with exact fractions. It does
not import the producer. Fifteen cases satisfy L1; four reject. Twelve cases
are newly admitted compared with the old max proof.

Cases cover sparse and dense cancellation, actual projection widths,
within/across-lane and group positions, independent two-token rows, adjacent
FP32 inputs on each side of the admission cutoff, zero weights with maximum
inputs, tiny inputs, and widely separated magnitudes with signed zeros.
The historical ordered-zero/exact-one conflict and finite-product overflow
remain rejected. A naive FP32 sum loses the deliberately retained small
cancellation residual; it is not used as either oracle.

## Complete captured-input replay

Authenticate the [coordinate audit](gemma-affine-admission.md#coordinate-correlated-product-refinement),
its parent codebook receipt, full raw trace and compact array ledger. Reuse
the accepted all-codebook maxima and metadata identity; do not repeat the
unchanged 72-million-pair reconstruction audit. Read complete selected states,
preserve the attention context token axis and two rounded residual additions.

For the 70 uncaptured GELU products, sum per-coordinate outward FP32 product
envelopes. These remain sufficient bounds, not observed product activations.
The proposed proof admits FF-down layers 0..6 in addition to all 269 existing
passes. All 277 invocation records replay independently. The one full tied
head and both whole tables remain above the existing owner row limit.
The earlier coordinate audit's seven inconclusive cases remain immutable
historical results; only the new proposed L1 evidence resolves their bounds.

## Verification and retained failure

Thirteen portable methods and two retention methods pass. A late raw-read
failure regression executes the real publication/trace path, retaining the
completed first record, incomplete current attempt and original error.
Two fresh captures are byte identical and both independent replays pass.
Six owned commands complete with clean source/output guards and process-group
closure. The fixture is frozen before target edits.

The first attempt retains one failed portable assertion: Python 3.12's built-in
sum compensates, whereas the diagnostic requires ordinary ordered additions.
The test now explicitly adds in order, matching the unchanged producer and
checker. That failed run is not accepted or overwritten.

Maximum successful-process RSS is 58,949,632 bytes; maximum physical footprint
is 38,159,008 bytes. Both are below separate 512 MiB reference-process policies.
These are not whole-model residency measurements. Exact scripts/interpreter
are bound; a full standard-library/native-runtime closure is not claimed.
This reference-only task does not rerun unrelated Rust or GPU suites.
The [receipt](../benchmarks/results/2026-10-25-gemma-affine-l1-v1.json) binds
sources, captures, resource/process checks and retained failure.

## Remaining work

Implement the separately reviewed allocation-free L1 fallback and qualify
newly admitted inputs on the actual GPU before using it in graph composition.
Table/head chunking, native activation parity, resident weights/KV/scratch,
shared scheduling, quality/performance, app/iPhone admission and 131072 output
tokens plus a nonempty prompt remain open. All earlier cross-precision,
source-GPU and long-rotary failures remain unwaived. No G2/G3/G4 completion or
capability change follows from this reference evidence.

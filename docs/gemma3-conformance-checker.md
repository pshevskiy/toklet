# Independent Gemma3 actual-input checker

29 October2026. The checker implementation and historical diagnostic pass.
**Fresh native qualification remains pending.** The
[committed policy](gemma3-native-conformance.md) was reviewed and committed as
`ae7a457` before this code. The engine, source3 reference, teacher histories and
literal numeric coefficients are unchanged.

## What changed

The saved-byte reader now keeps the complete coupled-trajectory statistics while
checking a separate universal fatal route. All selected score/context/head and
teacher groups retain their existing coupled source3 MAIN gate. Other selected
stages use independent actual-input operations over original weights. Every
value retains a fatal check; no failing-coordinate whitelist exists.

Two concrete helpers keep this readable: `gemma3_conformance_math.py` implements
the arithmetic contracts; `gemma3_conformance.py` resolves original weight rows
and the fixed stage dependency table. The reader retains raw descriptors,
schedule/state/codec/resource checks and endpoint authentication. No model graph
or source producer is imported as an oracle. Source3 and WIDE identities remain
authenticated, including all3,232 legacy coefficient dictionaries.

Basic operations and RMS require exact declared nonzero store results, with the
existing numeric signed-zero convention. Projection requires true faithful Half
bracketing plus the unchanged local numeric gate. Intrinsics use independently
certified true-result bounds and published ULP contracts propagated through the
actual FP32 operations. GELU checks discrete outputs; rotary requires the same
sin/cos candidate pair to explain both coordinates. Softmax uses a conservative
ordered interval enclosure and the unchanged local gate, without claiming to
reconstruct one hidden exp vector. Unsupported proof/domain cases fail closed.

Original Q4/group64/BF16 bytes are decoded one small matrix at a time. Finite
Q4×BF16 products are exact FP32, making the separate add equivalent to original
FMA reconstruction; nonfinite product cases are explicitly unsupported. Exact
dot ambiguity uses rational fallback. Numeric writable owners have a conservative
63MiB planning envelope below64MiB, separate from mapped files and runtime costs.

## Checks and measured diagnostic

All five private development attempts pass and remain retained. The final code
passes16 operator controls and12 reader controls again from the working tree.
Controls include independent Neumaier/gamma boundaries, exact cancellation,
faithful third-neighbor rejection, pi/range-reduction proof, intrinsic boundary
ambiguity, discrete GELU holes, a jointly invalid rotary pair, masks, all400 stage
identities, original-byte corruption and final eligibility labels. Exact code and
23,191 runtime files were authenticated before/after both owned jobs.

| Historical diagnostic route | Groups | Values | Fatal failures |
| --- | ---: | ---: | ---: |
| Coupled attention/output | 624 | 13,185,444 | 0 |
| Actual-input operations | 5,808 | 5,351,844 | 0 semantic /0 numeric |
| Total | 6,432 | 18,537,288 | 0 |

The original coupled aggregate still rejects84 MAIN values and has98 WIDE
diagnostic exceedances. `native_store_comparison_passed` stays false; the distinct
`native_conformance_passed` is true. Both old92- and84-value rejection receipts
are hash-bound and unchanged. Historical eligibility and numerical qualification
stay false even when the new method passes.

The finite reader finishes in252.355647s;2,378 direct-process observations report
maximum RSS743,981,056 bytes and lifetime physical counter368,150,712 bytes,
below6GiB. These are reader observations, not model inference speed or a
continuous/process-tree peak guarantee. Its owned PID/group is absent afterward.
All6,496 raw record hashes/extents,508 original weight-range hashes and9,132
masked +0 words pass independent Astra audit. Eight projection fallback branches
are reported in aggregate and supported by reviewed code/controls; individual
fallback certificates were not retained/replayed independently.

## Evidence and scope

The [machine-readable result](../benchmarks/results/2026-10-29-gemma3-conformance-checker-v1.json)
pins the proposal, code review, portable controls, complete diagnostic and
independent saved-result review. These local artifacts retain full raw data and
closed-owner evidence; the tracked result is an index, not their replacement.

The checker code task is complete. The bounded real Mac native task still needs
a separately frozen fresh capture, independent checker and final Astra review.
This diagnostic grants no original-unquantized10k/100 adoption quality,
full-window support, production Chat, whole-app fit, iPhone or performance claim.
Those requirements remain explicit in B3/B5/M8 and M9 as applicable.

# Localizing the Qwen3 C512 precision gap

This reader examines the complete frozen [C512 CPU captures](qwen3-context512-reference.md).
It changes no producer math, numerical gate or native implementation.

The [existing native qualifier](../tests/mlx_qwen3_real_model_metal.rs) compares
the saved canonical FP32 capture from the [exporter](../tools/reference/mlx_qwen3_real.py).
The two-interval analysis is an additional precision diagnostic; it introduces
no dual-oracle target acceptance rule.

The original asymmetric failure classification compares FP32 values against the
FP64 reference using its unchanged absolute and relative tolerance. Separately,
each saved reference value defines its own native target-acceptance interval.
The diagnostic radius conservatively includes both exact algebraic evaluation
and the evaluated binary64 bound expanded outward by one binary64 ULP. Exact
rational endpoints and gaps avoid a false disjoint result from rounded endpoints.
The guard belongs only to this diagnostic; target acceptance tolerances stay
unchanged. Inclusive overlap is necessary only: it does not prove a representable
Half/FP32 value exists or that an implementation computes the correct result.

The reader streams at most65536 elements per block. Whole-file descriptor/path
endpoints and hashes, plus hashes of the actual consumed array bytes, must match
the frozen identities before the report is complete. Failure retains partial
findings with their authentication status. Every rejected value includes raw
coordinates, semantic token/query/key/head/vocabulary fields, both values,
original bound, exact guarded gap and each contract's raw greedy ID/margin at
that position. Causally masked attention coordinates are marked explicitly.
Case-specific counts are separate from unique stage coordinates and unique
coordinate/value pairs across the original and future-tail cases.

## Actual result, 27 October 2026

All22 cold tests pass:10 new exact-interval/stream controls and12 existing
reference controls. The full reader authenticates and localizes every7282
case-specific failure, with complete file/path endpoints for all four captures.
There are3641 unique stage coordinates and3641 unique coordinate/value pairs;
each case has3641 failures over40 affected positions from25 through504.
The complete [raw report](../artifacts/m8-qwen3-context-precision-checks-v1/analyze.json)
retains every coordinate, value, interval and token choice/margin.

Ten case-specific entries have disjoint guarded intervals, corresponding to five
unique pairs, all at position65 and repeated unchanged in the future-tail case:

| State and feature/vocabulary index | FP32 contract | FP64 contract |
| --- | ---: | ---: |
| Layer27 V projection,25 | 0.4248046875 | 0.76513671875 |
| Layer27 V projection,98 | -0.38525390625 | -0.0714111328125 |
| Layer27 attention output projection,625 | 3.724609375 | 3.83984375 |
| Layer27 attention output projection,664 | 3.1640625 | 3.2734375 |
| Logits,83172 | -3.765871286392212 | -3.9448165893554688 |

Both contracts select token1879 at that position; margins are6.518730163574219
and6.510900497436523. Disjoint intervals rule out simultaneous agreement with
both guarded bounds. They do not establish failure against the existing
canonical FP32 target contract. Layer27 remains the first failing captured layer;
layers2–26 were not observed.

External process peaks are295,912,312 physical/349,224,960 RSS bytes, below512MiB.
Both owned groups close. A separate standard-library publisher independently
reads all7282 saved value pairs from the frozen raw files, reconstructs exact
rational interval decisions, semantic coordinates and raw greedy IDs/margins,
and verifies all duplicate/count summaries. Producer math, gates, Rust sources
and protected release binaries remain unchanged. See the
[public receipt](../benchmarks/results/2026-10-27-qwen3-context512-precision-v1.json).

The [native C512 diagnosis](qwen3-context512-native.md) now completes all
chunk/raw/state replay but retains18,400 canonical failures. Captured arrays are
byte-identical across schedules; source quality and numerical qualification
remain open. Next isolate selected layer27 Q/K/V dots with the same saved native
and reference input rows, authenticated packed weights and passing controls.
An all-layer CPU trace or source-framework run is not a prerequisite for that
bounded diagnosis. A later source-parity run cannot substitute the same quantized
checkpoint for an unquantized quality baseline, and the full quality workload
must remain intact. No production precision change or gate widening follows.

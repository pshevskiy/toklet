# LFM3 source-profile BF16 rotary

Status: Astra approves scoped CPU component retention, with no findings. Full LFM3 execution and
adoption remain open. The whole model roadmap and original gates remain active.

## Source and numerical behavior

The authenticated [child/parent source](lfm3-source-qualification.md) fixes
head dimension 128. Full attention uses theta 1,000,000 with 32 active pairs;
sliding attention uses theta 10,000 with 64. Pair i joins i+64. Every exponent
uses denominator 128, including partial rotation; inactive frequencies are +0.
All coordinates execute rotation, so signed zeros need not copy exactly.

The child computes an FP32 exponent, FP64 power rounded to FP32 and FP32
reciprocal. The pinned parent Torch FP32 power recipe agrees bitwise for both
profiles on the named Mac/runtime. Absolute position and frequency multiply in
FP32. Source FP32 sine/cosine round to BF16; inputs and coefficients then widen
exactly to FP32. Separate products and additions precede one nearest-even BF16
store. Rust retains this ordering without fast-math or fused multiply-add.

The existing FP64 mathematical proportional rotary function remains the
independent mathematical path. Existing model positional defaults are unchanged.
The new component lives in src/model/lfm3_rotary.rs; it is not a model factory.

## Ownership and admission

Two immutable profiles can be shared by all six full/five sliding attention
layers and all heads. Coefficients are contiguous [capacity,64,cosine/sine] BF16;
frequencies are an inline 64-element FP32 array. At capacity 2,048, each table has
524,288 B of coefficient storage plus its owner object. No per-head copies or
session KV state are retained by these owners.

Admission checks nonzero capacity, the declared 262,144 source position limit,
checked byte arithmetic and the caller's remaining model budget before reserving
storage. Actual Vec capacity is counted before publication. Callers separately
account weights, KV, scratch, codec and application owners under the unchanged
6 GiB engine/512 MiB codec/32 MiB application policy. Component admission does not
prove whole-model memory fit.

Apply uses caller BF16 input/output slices shaped [tokens,heads,128], absolute
offsets and the same immutable table for every chunk. Geometry, extent and all
finite input checks precede writes. Zero-token calls reject. Arithmetic or BF16
store overflow may leave partial output; discard it and retry with valid input.
No cache/session state is published by this operation.

## Independent evidence

The [prospective contract](../artifacts/lfm3-rotary-2026-11-10/prospective-contract-1.json)
predates source/candidate outcomes. Its exact frequency, BF16 trig and final
BF16 gates remain unchanged. The parent and child recipes agree on 128 frequency
words; scalar libm and source agree on 525,312 BF16 trig values across positions
0..2,047 and 4 sparse positions per profile.

The versioned fixture contains 40 cases at offsets 0/255/511/1,023/2,031, lengths
1/17 and 4/16 heads. Every one of 460,800 output words agrees with the separate
scalar FP32/BF16 arithmetic. The unrounded FP32 diagnostic against independent
FP64 trig/rotation at the contract FP32 angles passes 1e-5 + 1e-4 * abs(reference),
with maximum absolute error 8.242e-7. This diagnostic precedes BF16 coefficient
and output rounding and does not demand that BF16 equal unrounded FP32.

The source producer verifies the pinned Torch 2.9.1 / NumPy 2.3.2 runtime before and
after execution; remote model code is never imported. The strengthened producer
adds an explicit runtime-manifest pin and explicit FP32 zeros. Its replay has
identical fixture outcomes except the producer hash. Both closed CPU campaigns
stay within the separate 512 MiB aggregate audit guard; maximum conservative final
RSS sum is 261,816,320 B. Final per-PID peaks cover that PID's sampled RSS; final
physical peak is unavailable. The 2 GiB disk floor remains unchanged.

[Source probe](../artifacts/lfm3-rotary-2026-11-10/source-probe-report-1.json),
[fixture campaign](../artifacts/lfm3-rotary-2026-11-10/fixture-checks-2.json),
[fixture correspondence](../artifacts/lfm3-rotary-2026-11-10/fixture-correspondence-1.json).

## Checks, measurement and remaining work

Fresh fmt, both strict all-target Clippy configurations, 693 CPU tests (50
ignored), 839 Metal-enabled tests (338 ignored) and nine depth-example tests
pass. Five optimized source tests explicitly include sparse positions 4,095,
65,535, 131,071 and 262,143. These compare those rows, not every source-limit row.
Forty cases additionally match whole/chunked execution schedules. Both profiles
perform 48 repeated apply calls and 16 rejections with zero Rust allocations;
owner bytes and pointers stay fixed. Four portable tests pass ordinarily and
with Python optimization; six actual source/runtime admission controls and ten
actual console argument controls reject.

Native72 compiles three fresh arm64 slices with 46 exports each and strict C
links; fresh CPU and Metal C smokes pass on the existing tiny model. These do
not qualify LFM3 GPU execution. The app still links native46.

On Apple M1 Pro / MacBookPro18,3 with 16 GiB, macOS 26.6.2 (25G83), Rust 1.90.0
Homebrew / LLVM 21.1.1, all 18 release cohorts pass the frozen resource/condition
checks. Both runs use three warmups and ten samples at chunks 1/32/128 and
absolute offsets 0/511/1,023, with reversed second-run order. Endpoints report
AC power, Low Power Mode off, nominal thermal state, and no active compilers.
Raw samples, medians, ranges and median absolute deviations are retained.

| Metric | Observed range across the two runs and three offsets |
| --- | --- |
| First two-table construction in a fresh process | 1.664–1.800 ms |
| Repeated two-table construction medians | 1.640–1.793 ms |
| Warm eleven-layer Q/K pass, one token | 0.0665–0.0703 ms |
| Same operator, 32 tokens | 2.112–2.226 ms |
| Same operator, 128 tokens | 8.437–8.695 ms |

Two admitted owners occupy 1,049,152 B including inline frequencies and objects;
caller input/output storage is another 10,240 / 327,680 / 1,310,720 B. The largest
conservative final controller+child RSS-peak sum is 34,619,392 B, below the
separate 512 MiB component audit allowance; final physical peak is unavailable.
These are operator baselines. Construction excludes process startup/deallocation;
warm time includes finite validation and BF16 stores, across six full/five
sliding layers with 16 Q and 4 KV heads. Weights, KV, codec, sampling, TTFT,
whole-model throughput, summary overhead and model memory fit are unmeasured.

[Required checks](../artifacts/lfm3-rotary-2026-11-10/required-checks-1/checks.json),
[optimized source tests](../artifacts/lfm3-rotary-2026-11-10/release-source-checks-1.json),
[zero allocations](../artifacts/lfm3-rotary-2026-11-10/allocation-targeted-1.log),
[native build](../artifacts/lfm3-rotary-2026-11-10/native72-publication.json),
[native smokes](../artifacts/lfm3-rotary-2026-11-10/native-smoke-72/checks.json),
[timing contract](../artifacts/lfm3-rotary-2026-11-10/performance-contract-1.json),
[raw timing/check index](../artifacts/lfm3-rotary-2026-11-10/performance-1/checks.json),
[review/commit](../artifacts/lfm3-rotary-2026-11-10/publication-checks.json).

Next qualify the remaining LFM3 source norms/gates/window semantics and compose
the graph. Complete stock-to-modified and modified-to-native 10,000-target/
100-task quality comparisons, original output caps, real-model memory and
performance/adoption gates remain mandatory. Screen capture and iPhone runtime
remain deferred; the linked app package is unchanged.

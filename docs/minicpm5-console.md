# MiniCPM5: bounded console evaluation

S1.B checks and matched request measurements pass in the bounded scope below.
Astra approves scoped retention; [review/commit](../artifacts/minicpm5-heretic8-payload-2026-11-10/publication-checks.json).
This enables an explicit diagnostic example, with no managed model route, codec,
full quality or adoption qualification.

## Source and numerical scope

The publisher-labelled uncensored [Heretic MLX8 conversion](https://huggingface.co/bumblebuttpow/MiniCPM5-2B-heretic-abliterated-MLX-8bit/tree/50ab1b226834c290cdf68e460eb899e5cff7c61d)
is pinned at `50ab1b22`. Its complete 2,674,327,346-byte object matches SHA-256
`6fadca0fcdefe8b77fc144851ee2c2fd3c8949ccc0ec756b8ada27e3156c8910`.
The exact config/index/header have 973 physical entries, representing 381 logical
tensors and 2,516,756,480 parameters. Group64 affine8 matrices retain BF16 scale
and bias metadata; embeddings and output head are distinct. Source files are
unchanged. The default-RoPE declaration uses the [existing source admission](minicpm5-default-rope.md).

Pinned headers from the declared insraq parent report 381 BF16 tensors with the
same names and logical shapes. This resolves the header declaration, while the
conversion card's FP16 wording, whole-parent hashes and exact conversion lineage
remain unverified. Original mondk4 access failures remain recorded; this is a
separate candidate.

Independent NumPy/BLAS FP32 and FP64 profiles reconstruct packed rows through
bounded `pread` tiles. They retain Half activation/norm boundaries, FP32 output
weights/logits and ordinary split-half RoPE, with an explicit output head. No
expanded full model or whole-file mapping is retained. The unchanged tiny
independent oracle contributes 780 stage-array checks across tied/untied forms.

The [prospective contract](minicpm5-console-contract.json) predates native results.
Six cases, layers 0/1/41, offsets 0/5/509, four teacher-forced prefixes and three
scoring prefixes produce 349 arrays per profile. Gates remain
`atol=max(1e-4,2*cases-only pooled FP32/FP64 max error), rtol=1e-3`; teacher/scoring
logits use that same cases-only gate. Neither equivalent-prefix variation nor
native outcomes enlarge it.

First GPU capture compares **52,896,768 values** across both profiles, with zero
failures and maximum gate fraction 0.97220634. These are comparisons against two
profiles, rather than 52.9M unique native values. The 2,614 native slices include
functional-only native-generation and 512-boundary observations. Exact numeric
logits agree across chunks 1/3/32 and unchanged causal prefixes. Full/token
readbacks select the same native histories. Reset, overflow preservation and
partial discard/resume at 512 pass. Offset 509 starts with three local keys;
neither it nor a functional 512-token check qualifies a 512-key numerical trajectory.

## Ownership and observed memory

The 512-capacity/32-chunk pilot, 6 GiB Mac ceiling and 512 MiB codec allowance remain
fixed. Codec work is not part of this raw-ID experiment. The independent source
ledger is 2,566,914,048 B; actual source peakRSS is 614,760,448 B. Native learned
buffers own 2,674,218,150 B, including layout padding, counted once.

The instrumented native capture admits at most 2,878,862,568 B including 32 MiB
application allowance. Its exact final `wait4` peakRSS is 2,507,980,800 B. Maximum
live-PID lifetime physical counter is 2,791,508,440 B; final-after-exit physical
peak is unavailable. Planning/admission, ownership and observed counters remain
separate. These observations do not qualify a different device or app journey.

## Console commands and timing boundaries

Use Rustup 1.90, NumPy 2.3.2 and the recorded controlled build environment. The
system/Homebrew Clippy was incompatible with this Rust cache; its failed attempt
and corrected checks are retained. Ordinary CPU tests require no model download
or GPU. The registered example requires the explicit `metal` feature.

```sh
cargo build --locked --offline --release --features metal --example minicpm5_probe
```

Generate independent profiles before native outcomes, with BLAS thread limits
set to 1 (`OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `VECLIB_MAXIMUM_THREADS` and
`MKL_NUM_THREADS`). Supply the authenticated local model directory and a fresh
output directory. Calculate the external manifest SHA after generation.

```sh
.venv/bin/python tools/reference/minicpm5_reference.py MODEL FRESH_REFERENCE docs/minicpm5-console-contract.json
shasum -a 256 FRESH_REFERENCE/manifest.json
target/release/examples/minicpm5_probe MODEL FRESH_REFERENCE/manifest.json MANIFEST_SHA FRESH_CAPTURE
target/release/examples/minicpm5_probe bench MODEL FRESH_REFERENCE/manifest.json MANIFEST_SHA FRESH_FULL full-logits
target/release/examples/minicpm5_probe bench MODEL FRESH_REFERENCE/manifest.json MANIFEST_SHA FRESH_TOKEN greedy-token
```

These commands assume the default Cargo target directory. Qualified local binary
paths, hashes and exact arguments are recorded in the evidence. Outputs are
exclusive. The capture reader binds metadata to the completed owned receipt,
checks the flushed journal/raw hashes, compares both original source profiles
and checks exact native state/schedules. A finite payload with refreshed hashes
and journal still fails the original gates. Portable clock/state checks and
resealed actual-output controls remain separate from model execution.

The [timing contract](minicpm5-console-benchmark.json) freezes a raw 4-token prompt
and 4 teacher tokens, including the final EOS. Decisions use contexts 4/5/6/7;
7 tokens finish in KV and the eighth remains pending. One persistent 512/32 session
resets history before each request. Both readback paths use the same weights,
precision, token IDs and cache semantics. Three warmups and ten measured samples
are retained per cohort, with full/token/token/full ordering across two runs.

Request time starts before `set_tokens` and ends after the final pending append.
Reset and JSON publication are outside it. Token-decisionTTFT is an internal
raw-ID event; streamed or visible-text latency is unavailable. Decode rate is
`3/(last decision-first decision)`; prefill rate uses 4 tokens and completed
prefill wall time. Phase clocks enclose completed GPU work and host boundaries.
The sampling clock measures the CPU `greedy_token` accessor; token-only GPU
selection is already inside its completed GPU phase. It is not double-counted.
Operator/allocation/energy and visible-frame observations remain unavailable.

Power/thermal eligibility, raw samples, median/IQR/stdev, exact final peakRSS and
conditional live-PID physical counters accompany the matched cohorts. No tail
percentiles or continuous thermal stability follow from ten requests. A request
effect must clear 5% and observed variation/dispersion in both runs before a
scoped performance claim; a benchmark never grants quality or model adoption.
Ordinary telemetry's 1% overhead bound remains unqualified and opt-in.

## Matched request results

All four cohorts pass eligibility and exact work/selection checks: 40 measured
requests and 12 warmups, with AC power, nominal thermal state and Low Power Mode
off at all eight boundaries. All raw samples are retained.

| Readback | Run | Request median | Internal decisionTTFT median | Decode median |
| --- | ---: | ---: | ---: | ---: |
| Full logits | 1 | 500.319 ms | 192.485 ms | 9.746 tokens/s |
| Token only | 1 | 503.813 ms | 193.384 ms | 9.665 tokens/s |
| Token only | 2 | 504.340 ms | 193.511 ms | 9.653 tokens/s |
| Full logits | 2 | 501.122 ms | 192.556 ms | 9.733 tokens/s |

Token-only request time regresses 0.698%/0.642% in the two pairs. Between-run
request variation is 0.160%/0.105%; largest within-cohort request IQR is 0.261%.
The 5% request-improvement screen is not cleared. **Keep existing defaults and
both explicit options; no engine optimization is adopted.** This conclusion
applies only to the fixed raw 4..7-context workload.

Completed GPU intervals dominate these requests (about 495 ms full versus 500 ms
token-only median). The CPU selection-accessor medians are about 2.1 ms full versus
0.0006 ms token-only. These observations point toward GPU work as the next
profiling target; medians of separate phases are not additive identities or
operator-level attribution. No kernel/dispatch counts are fabricated.

Full/token planned session admission is 2,785,878,866/2,784,834,426 B, including the
same driver allowance, before 32 MiB application reserve. The 1,044,440 B planned
reduction does not establish an observed process-memory saving. Across cohorts,
exact final peakRSS reaches 2,743,009,280 B and sampled lifetime physical peak
2,733,050,280 B; final physical counters remain unavailable.

The offline reader reproduces every cohort's statistics/environment exactly.
All 36 raw evidence files, numerical operations, build and prospective
contracts remain unchanged after the review fixes. One positive
and 39 resealed actual-benchmark damage controls pass. Its CLI validates completed
owned reports independently of power eligibility:

```sh
.venv/bin/python -S -O tools/bench/minicpm5_performance.py BENCH_JSON OWNED_JSON docs/minicpm5-console-benchmark.json full-logits FRESH_DERIVED_JSON
```

[Raw cohorts and statistics](../artifacts/minicpm5-heretic8-payload-2026-11-10/performance-summary-1.json),
[review-fix revalidation](../artifacts/minicpm5-heretic8-payload-2026-11-10/astra-fix-revalidation-6.json),
[actual benchmark controls](../artifacts/minicpm5-heretic8-payload-2026-11-10/performance-controls-2/checks.json).

## Checks, failures and remaining gates

Fresh root formatting, strict CPU/Metal Clippy, 670 CPU/810 Metal tests pass
(46/331 ignored). Four example tests exercise cold identity/token/history and
partial-journal behavior without loading a model. The earlier unchanged-input
native63 reuse audit predates registration. Fresh native64 checks all three
arm64 slices, 46 exports each, strict links/package hashes and CPU/Metal C smokes.
The original capture/common frontend paths remain byte-identical after adding
the benchmark entry point; runtime Rust/shaders/ABI do not change. No app
framework link changes.

Failed source access, an initial hardlink cache preflight, a wrong-directory
Python invocation, the Clippy mismatch and a release disk-admission rejection
are preserved. Only inactive generated nonexecutable compiler objects were
reclaimed; hardlinks, binaries, static archives, model data, source and raw
results stay intact. A post-check snapshot observed free disk below 2 GiB before
recovery; its cause is unverified. Do not claim uninterrupted disk-floor
compliance from guarded jobs or infer an engine memory effect from free disk.

Astra found two offline-consumer gaps: disjoint durations could each fit an
owned clock while their sum exceeded it, and balanced capture ledgers could
underreport known owners. Readers now bound load plus all completed work, and
sum the six sequential host phases within each wall clock. GPU intervals remain
overlapping. Seconds convert with exact integer arithmetic to nearest ns,
with at most 0.5 ns quantization per field and a `2^20`-second duration bound.
The capture ledger independently derives every owner from fixed graph shapes,
guards, diagnostics and pinned 64-bit Apple metadata sizes. Ten capture-reader
and eleven request-reader portable tests pass; actual resealed controls reject
41 capture and 39 request damages, including the two findings. Saved comparisons
and statistics reproduce exactly; no new model run is needed. An initial huge
integer conversion test and an overly broad frozen-input audit failed before
correction; their logs remain retained. The final staged whitespace check also
found an extra EOF blank line in the source helper. Removing its single LF
preserves the exact AST/tokens; the old manifest remains immutable. The reader
accepts only that reviewed pair of exact SHA hashes, preserving all other
producer/package pins. [Reuse proof](../artifacts/minicpm5-heretic8-payload-2026-11-10/producer-whitespace-reuse-1.json).
Astra approves both fixes and this scoped retention with no unresolved findings;
[review result](../artifacts/minicpm5-heretic8-payload-2026-11-10/astra-review-result.json).

Whole-parent identity/conversion lineage, codec/template/stop behavior, original
10k-target/100-task quality, deeper numerical trajectories, managed adoption and
physical iPhone work remain open or deferred. Screen-capture qualification is
deferred by the user's engine/CLI priority; this suite needs no sharing picker.

[Source identity](../artifacts/minicpm5-heretic8-payload-2026-11-10/payload-verified.json),
[independent source](../artifacts/minicpm5-heretic8-payload-2026-11-10/reference-checks-1.json),
[first capture](../artifacts/minicpm5-heretic8-payload-2026-11-10/native-short-summary-1.json),
[fresh required checks](../artifacts/minicpm5-heretic8-payload-2026-11-10/required-checks-4/checks.json),
[actual capture controls](../artifacts/minicpm5-heretic8-payload-2026-11-10/capture-controls-3/checks.json),
[disk observation](../artifacts/minicpm5-heretic8-payload-2026-11-10/post-check-disk-observation.json).

[Native64 package](../artifacts/minicpm5-heretic8-payload-2026-11-10/native64-publication.json),
[CPU/Metal C smokes](../artifacts/minicpm5-heretic8-payload-2026-11-10/native-smoke-64/checks.json),
[capture-path reuse](../artifacts/minicpm5-heretic8-payload-2026-11-10/capture-path-reuse-1.json).

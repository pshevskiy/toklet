# Gemma current-path cost observations

15 November 2026. G1 diagnostic work for the active
[competitive GPU goal](mlx-gpu-performance-plan.md). These observations identify
the next optimization target; they do not qualify a speedup, summary overhead
or a win over an external engine.

## Current binary and workload

The named host is Apple M1 Pro/MacBookPro18,3, 16 GiB, macOS 26.6.2/25G83.
The reused release binary has SHA-256
`da693435917e603244428dd746e8a4788032d345013595b0066615c71308e175`.
All 4,106 relevant compiled source/test/example inputs match its frozen source.
It uses the authenticated official Gemma 4 E2B BF16 checkpoint, FP32 graph/KV,
capacity 512, chunk 32 and explicit batched-single-row submission. The workload
is the unchanged 384-input/128-output golden, with reset, no EOS stop and state
512 history/511 committed/one pending token. Decode evaluations total 127.

All actual runs use the existing owned-process guard with the 6 GiB ceiling,
32 MiB helper reserve and 2 GiB disk floor. Preparation requires another 3 GiB
disk reservation. AC power, Low Power Mode off and thermal state zero are
checked repeatedly. Existing Ollama app/server identities are allowed only
while its local residency API reports no loaded models; foreign named engine
or build work rejects admission. These observations do not prove continuous
global exclusivity or a continuous allocator cap.

## CPU sampling

Three complete warmups precede one sampled request. Eight-second prefill and
ten-second decode sampling commands use `/usr/bin/sample` at one millisecond
intervals. Their enclosing parent-clock brackets fall wholly before the first
token and between first/last tokens respectively. The prefill command's bracket
is longer than eight seconds because sampling/symbol processing is included.
Exact output IDs, work and state pass for all four requests; all six model files
authenticate before and after. No engine source is changed by this collection.

The denominator is the main thread only: 5,776 pre-first-token samples and 7,366
decode samples. A tree parser partitions exclusive sample weights once, so
recursive/inlined appearances are not summed as separate work. Other threads
are excluded. Ten parser controls pass, including recursive frames, other-thread
exclusion, exclusive parent remainders and malformed/truncated trees.

| Main-thread sample bucket | Pre-first-token share | Decode share |
| --- | ---: | ---: |
| CPU span attention | 46.68% | 28.77% |
| Metal completion wait, including GPU execution | 46.57% | 59.76% |
| Other projection host work | 1.54% | 5.08% |
| Identified projection fill/copy | 0.40% | 3.39% |
| CPU GELU | 2.61% | 0.86% |
| CPU RMSNorm | 1.11% | 0.07% |
| Other identified/unknown main-thread work | 1.09% | 2.08% |

These are statistical sample shares in two windows, not exact CPU durations,
GPU occupancy, whole-request shares or an additive speed forecast. Inlined
validation and copies can remain in other projection work. An absent stack
sample does not prove zero cost. The sampled request clocks include profiler
effects and are excluded from acceptance timings.

Peak guard charge is 5,075,048,616 bytes including the helper reserve. The
collection closes with no live owned process or cleanup error.
[Fixed protocol](../artifacts/mlx-gpu-gap-2026-11-15/g1-cpu-protocol-1.json),
[raw/checked capture](../artifacts/mlx-gpu-gap-2026-11-15/g1-cpu-sample-1/result.json),
[sample analysis](../artifacts/mlx-gpu-gap-2026-11-15/g1-cpu-analysis-1.json),
[guard](../artifacts/mlx-gpu-gap-2026-11-15/g1-cpu-guard-1/result.json).

## Completed GPU frames

The existing CLI runs plain and profiled generation sequentially on one owner,
checks complete output/state plus every selected logit bit, and exports a
bounded selected frame. Each case below passes the strict offline report.

| Evaluation | Position/input rows | Frame wall ms | GPU encoder union ms | Host wait ms | Unattributed host ms | Commands/waits |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| First prefill, 1, diagnostic fix | 0/32 | 597.288 | 363.901 | 441.517 | 121.721 | 278 |
| Final prefill, 12 | 352/32 | 1453.541 | 427.811 | 604.701 | 818.651 | 278 |
| First decode, 13 | 384/1 | 203.984 | 52.281 | 118.489 | 64.289 | 277 |
| Last decode, 139 | 510/1 | 224.476 | 55.338 | 119.907 | 83.724 | 277 |

Each frame has 349 projection dispatches and two CPU gathers. GPU intervals
overlap host waits; they are not added or subtracted to invent queue time.
Unattributed host time is not an exact attention timer. These selected frames
are not whole-request operator totals or independent performance cohorts.
Peak guard charge is 5,001,975,256 bytes including the helper reserve; completion
and cleanup pass.
[Capture/replay](../artifacts/mlx-gpu-gap-2026-11-15/g1-gpu-captures-2/result.json),
[guard](../artifacts/mlx-gpu-gap-2026-11-15/g1-gpu-guard-2/result.json).

First prefill uses the new diagnostic binary
`1955f486db7a54568bd5d8fdd8535975d2e96b8ee5414b87519dda8af8f50943`;
the other three frames use the earlier verified binary above. The diagnostic
change leaves numerical functions and kernels unchanged, with an
[exact inverse/source proof](../artifacts/mlx-gpu-gap-2026-11-15/g1-diagnostic-numerical-invariance-1.json).
Its complete 384/128 plain/profile outputs and all 262,144 selected prefix-logit
values agree bit for bit; the frame has 32 committed tokens. Peak guard charge is
4,999,828,856 bytes including helper, with clean completion. The different
frames/binaries are not pooled or compared as an overhead experiment.
[Early capture](../artifacts/mlx-gpu-gap-2026-11-15/g1-gpu-captures-3/result.json),
[early guard](../artifacts/mlx-gpu-gap-2026-11-15/g1-gpu-guard-3/result.json).

## Early-prefill diagnostic gap and fix

The first attempt at ordinal one fails with `selected logits unavailable` and
cleans up. Generation correctly hides ordinary sampling logits until all queued
prompt rows are evaluated. The old generic profile witness incorrectly needs
those ordinary logits at every selected ordinal. The failed attempt remains
[separate evidence](../artifacts/mlx-gpu-gap-2026-11-15/g1-early-prefill-coverage-gap-1.json).

The fix adds a borrowed diagnostic view carrying its committed prefix
length and updates the profile witness to require that exact prefix. It does
not expose queued rows through ordinary `logits()`, advance state, add numerical
work or change kernels/scheduling. Frozen scalar-prefix, no-allocation, replay
invalidation and private-witness controls cover the new behavior. Fmt, strict
CPU/Metal all-target Clippy, 847 CPU tests and 1,019 Metal tests pass (53/365
ignored respectively), as does the release build. The actual early capture
above passes. Astra approves the scoped diagnostic fix and observations with no findings.
Publication is tracked in the [receipt](../artifacts/mlx-gpu-gap-2026-11-15/g1-publication-1.json).
[Full checks](../artifacts/mlx-gpu-gap-2026-11-15/g1-required-checks-progress-1.json).

## Summary component measurements

The unchanged production collector was compiled into a standalone release probe
with the existing release dependencies. Its production source prefix is byte
identical. A genuine successful 384/128 record and its manifest reproduce the
saved JSON values exactly before timing. Each process runs three warmup and ten
measured repetitions per component in forward and reverse component order.
Two processes retain all 280 measured and 84 warmup samples; the second adds
explicit empty-Ollama residency/birth checks to the power/thermal and foreign
engine/build admission used by the first. No model runs inside this probe.

The second process observes the following medians. Values are per operation;
callback accounting is per token, with 128-token batches. These component
orders are not independent complete-request confirmation cohorts.

| Component | Forward order | Reverse order |
| --- | ---: | ---: |
| Disabled `Collector::record` | 1.24 ns | 1.24 ns |
| Complete enabled `Collector::record` | 3.569 us | 3.555 us |
| Record validation plus session/memory checks | 51.83 ns | 52.18 ns |
| Summary JSON serialization | 3.461 us | 3.469 us |
| Bounded ring insertion/copy | 46.34 ns | 46.31 ns |
| Token clock/interval accounting | 133.34 ns | 133.51 ns |
| Empty black-box callback baseline | 0.31 ns | 0.31 ns |

Enabled insertion includes the actual validation, serialization, distinct
monotone request IDs and bounded overwrites. Each process completes 425,984
insertions, retains 252 rows and records exactly 425,732 expected overwrites,
zero invalid records and zero export errors. A row is 2,263 bytes. The collector
retains its existing 9,000-byte value and 1,032,192-byte ring; these are logical
bytes, not a measured allocator peak. The confirming guard's maximum sampled
charge is 91,996,160 bytes including the 32 MiB helper reserve, with clean exit.

The callback component copies the real token event accounting block, surrounds
empty decoder/writer bodies with the same clocks, and checks 128 callbacks/127
intervals. It excludes actual decoding, output I/O/flush and callback delivery.
Its roughly 17.1 us per 128-token batch is a component observation, not measured
request overhead. Separate component medians must not be added or differenced
as a causal decomposition. Standalone loops also omit inference cache/scheduler
effects. This evidence does not establish that the old 3/8 overhead failures
were noise, or qualify any new 1% result.

The first preparation is stopped by sandbox process-observation denial; a
subsequent build finds an ambiguous integer in the artifact-only repetition
index. Both failures are preserved. The corrected artifact leaves production
code untouched, and both complete successful runs remain in the analysis.
[Fixed component protocol](../artifacts/mlx-gpu-gap-2026-11-15/g1-summary-probe-2/protocol.json),
[raw confirming output](../artifacts/mlx-gpu-gap-2026-11-15/g1-summary-probe-2/run-2/stdout.log),
[guard](../artifacts/mlx-gpu-gap-2026-11-15/g1-summary-probe-2/run-2/result.json),
[complete replay/dispersion](../artifacts/mlx-gpu-gap-2026-11-15/g1-summary-component-analysis-1.json).

## Decision and remaining work

Prioritize GPU-resident attention and dependent projection scheduling. CPU
attention is a directly observed large stack bucket; matrix GPU work and
completion boundaries also remain substantial. Do not spend the next kernel
iteration on small normalization/rotary stack buckets solely because their
standalone implementations exist. Qualify one accurate attention/block
mechanism before composing a full resident graph.

Defer the proposed typed-ring rewrite: measured insertion costs a few
microseconds, and no request-scale benefit has been demonstrated. Preserve the
bounded collector and original schema. Revisit callback specialization or
export-only serialization if a retained candidate demonstrates a useful effect;
file export is already outside inference. Run matched A/A diagnostics to assess
variation before the unchanged fixed-N overhead campaign on that candidate.
Neither new diagnostics nor A/A can replace its original eight comparisons.

G1's CPU/GPU and summary component measurements are complete. All 422 relative
links and whitespace pass; Astra approves the evidence and decision with no
findings. No production source changes in this measurement task. The prefix-witness code is already pushed as `42efe9c8`, with all
seven exact-head workflows/15 jobs passing. G0 common-artifact/cache/output
comparability and G2's accurate resident attention are next. The active goal
still requires actual matched wins against both external engines and all
original 1% overhead gates; the three old failures remain binding.

## G2 attention feasibility follow-up

An artifact-only GPU primitive keeps FP32 Q and three immutable K/V segments on
the device through dependent validation, QK, softmax and probability–V encoders.
The four dispatches share one command and one completion wait. There are no
intermediate CPU readbacks. Compensated three-term dot/context accumulation and
product-error recovery preserve the FP32 score/probability/context boundaries;
softmax uses Safe/Precise FP32 exponential and a compensated denominator. This
is a feasibility experiment, not a public owner, integrated block or replacement
for the existing CPU reference.

On M1 Pro, eight frozen D3–D512/full/local/translated/cancellation cases pass in
two segment layouts: all 932,968 values meet the original
`1e-5 + 1e-4 * abs(expected)` bound, with 111,608 exact masked zeros. Maximum
error uses 0.011921 of that bound. Contiguous and three-segment outputs are
bitwise equal. The expectations precede this GPU implementation.

A second input family extracts authenticated saved staged-FP32 graph states
computed from official original BF16 rows. Layers 0/4/13/14/15/34 cover both
head dimensions, early/late producers and shared KV. Original/future sequences
supply four-row prefill and final-row decode over four keys: 24 cases, each in
two layouts. All 376,320 values pass, with 1,152 exact masked zeros and maximum
bound fraction 0.009036. Context expectations are unchanged historical arrays;
independent scalar-fsum score/probability replay reproduces those contexts
exactly before GPU execution. There are 1,170 probabilities between 0.01 and
0.99 in the input family. This strengthens short own-input coverage; it is not
a fresh BF16 runtime, long-context graph or quality result.

Four actual GPU fault injections reject a NaN query, masked NaN value, masked
infinite key and masked score overflow from finite inputs. All input words and
buffer/status guards remain unchanged. They check primitive status handling;
they do not qualify session-prefix recovery or transactional KV publication.
Astra found the first overflow injection also overflowed visible scores. That
non-isolating attempt is retained. The corrected injection zeroes feature zero
in all Q/K rows, then sets only query/head zero and future key six to MAX. An
independent certificate checks all 42 pairs: exactly that masked pair overflows,
all 24 visible pairs and every other pair remain finite. The corrected actual
GPU run returns rejection status one with unchanged inputs/guards.
The independent raw reader also rejects eight corruptions, including a masked
1e-6 probability that passes the pointwise tolerance but violates exact zero.
Normal and optimized-Python replay agree for both input families and faults.

All seven execution guards close without survivors. Maximum sampled charge is
117,030,912 bytes including helper; the separate terminal direct-pair maximum
is 152,109,056 bytes, under the 512 MiB probe cap. Production sources, defaults,
model gates and numerical reference remain unchanged. Artifact-only release
builds pass; ordinary source checks are reused only for unchanged inputs.
Astra approves scoped primitive feasibility after the isolated-overflow correction,
with no remaining findings. Production ownership/integration remains unqualified.

[Fixture protocol](../artifacts/mlx-gpu-gap-2026-11-15/g2-attention-feasibility-1/protocol.json),
[fixture replay and eight controls](../artifacts/mlx-gpu-gap-2026-11-15/g2-attention-replay-1.json),
[own-input protocol](../artifacts/mlx-gpu-gap-2026-11-15/g2-attention-own-input-1/protocol.json),
[own-input and initial GPU fault replay](../artifacts/mlx-gpu-gap-2026-11-15/g2-attention-own-replay-1.json),
[isolated masked-overflow correction](../artifacts/mlx-gpu-gap-2026-11-15/g2-masked-overflow-replay-2.json).

Next implement a bounded reusable workspace and checked completion/poisoning
boundary for this one mechanism, then test genuine deeper own inputs and a
complete block. Keep whole-graph/full quality/state and request-performance
qualification open. These untimed probes establish no speedup or external win.

## Reusable workspace implementation

The [engine attention workspace](gemma-gpu-attention.md) now implements bounded
reuse and a checked borrowed-buffer encode seam on the existing GPU context.
Required checks pass; independent replay covers 2,844,296 values including actual
32-row capacity controls, and 32 warmed operations allocate no Rust memory.
Astra approves the scoped owner/encoder with no findings. Ordinary graph execution and request-performance gates
remain unchanged. [Hardware priorities](apple-silicon-gpu-goal.md).

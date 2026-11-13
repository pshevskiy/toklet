# Toklet improvement index

Updated **13 November 2026**. This is the compact index of retained capabilities,
measured gains and material restrictions. The [full dated history](archive/2026-11-09/improvements.md)
preserves every previous entry, source revision, failed check and evidence link.
Read it only when the current task needs that detail.

Measurements apply to the named hardware, route, workload and timing boundary.
Gains from different experiments cannot be added together. Current task and
qualification scope live in [TODO](../TODO.md); acceptance policy is in
[PLAN](../PLAN.md#validation-and-performance-policy).

The [LFM3 external request observations](lfm3-engine-comparison.md) record
78 requests across two 3+10 cohorts per engine, with preserved raw observations,
model/executable identities and clock/memory scope. Ollama lacks authoritative
IDs; vLLM matches exact prompt/cache input but only 3/13 output-ID/text rows
match CPU per cohort. No speedup ratio, quality or adoption is qualified.
All 26 genuine vLLM timing captures replay consistently. Source/guard/import
controls and scoped content checks pass; Astra approves scoped publication with
no findings. [Checks/review/commit](../artifacts/lfm3-engine-comparison-2026-11-11/comparison-publication-checks.json).

The [optional vLLM request timeline](lfm3-vllm-timeline.md) replays sealed captures
and separates server queue, scheduled-to-first, first-to-last and overall-rate
boundaries. Missing fields remain unavailable; inconsistent arithmetic withholds
derived decode rate. Twenty-four portable controls in both modes, five pinned-source
cases and nine guarded offline CLI cases pass under 512 MiB after correcting
audit/helper errors. Required checks bind 5,177 unchanged inputs and both 52-test
CI commands pass. Astra's empty-entry and stop-metadata findings are fixed;
follow-up approves scoped retention with no remaining findings; actual engine, quality
and performance qualification remain separate.

The [external HTTP benchmark observer](lfm3-http-metrics.md) validates bounded
Ollama/vLLM streams and separates client clocks, API counters and unavailable
state. Twenty-eight portable controls in both modes and twelve synthetic HTTP/CLI
cases pass within 512 MiB. All 26 genuine saved Ollama replies preserve counters,
text and original event clocks. Input mismatch and truncated streams cannot
qualify a comparison. Required checks bind to 5,177 unchanged inputs. Astra's
rejected-byte, counter-overflow and stop-ID findings are fixed, and scoped
retention is approved. [Review/commit](../artifacts/lfm3-engine-comparison-2026-11-11/http-observer-publication-checks.json).
No model startup,
engine overhead, quality, display or performance qualification follows.

The real LFM3 BF16 comparison reaches 1,025 positions with 68,054,016
exact source words and two late-abort retries. The source and native runs
stay within their respective memory caps. Ordinary one-token continuation
at the verified depth has a 1.285-second median on M1 Pro.
Astra approves scoped numerical retention. The complete original quality
results below pass; result review and managed adoption remain open.
[Scope and measurements](lfm3-context1025.md).

The original LFM3 quality campaign completes both 10,000-target loss pairs and
both 100-task comparisons under unchanged gates. Stock-to-modified NLL increases
0.00781 nats per token; task passes rise from 35 to 37. Native retains all
37 passes, all 10,000 greedy loss IDs and all 5,094 sampled task decisions, with
3.66e-15 maximum absolute NLL delta. Both isolated grading windows confirm the
guest stopped. The guarded loss retry stays within 6 GiB; interrupted output
remains unqualified. Astra approves scoped CPU qualification and retention;
managed adoption remains open.
[Complete decision and evidence](lfm3-quality.md).

The isolated LFM3 finite-widening experiment removes redundant BF16 conversion
work after complete public validation, preserving FMA/reduction/store order.
Fmt, strict CPU/Metal Clippy, 734 CPU/880 Metal tests, targeted release/allocation
checks, three native package slices and both C smokes pass. Full parity matches
854 observations/68,054,016 words, thirty wrapped views and two late retries.
Earlier failed checks, interrupted output and runner corrections remain in the
linked evidence. Full primary ABBA matches all 52 requests, including 40 measured;
medians 349.54→248.15 s and 349.58→246.60 s improve 29.0%/29.5%, above 0.63%
maximum median variation. The primary gate passes; final CPU retention is approved.

The full 156-request 128/512/1,024-input × 128-output matrix matches all work,
output/state and 4,992 sampled decisions per cohort, including 120 measured
requests. Paired median reductions are 21.3%/20.6%, 25.8%/25.9% and
27.8%/27.9%. All twelve sustained windows pass; no candidate median regresses.
Maximum separate RSS/physical guard charges including reserve are
3.82/3.74/3.74/3.74 GB; differences do not establish allocation savings.
Request-performance gates pass; full CPU retention is approved after quality/state checks.
CPU phase observations show 29.9–30.7% lower prefill medians and 16.2–17.2%
lower decode medians; operator-level attribution remains unqualified. Supplemental
observer metadata/unit errors are corrected with failures retained. The corrected
serial retry passes 55 portable controls per mode. Fresh compiler identity/new
loss build pass; native loss collection completes all 10,000 frozen targets and
ten blocks. Both guards pass; maximum observed outer guard charge is 4.15 GB
including reserve, within 6 GiB. Both loss gates pass; native matches all greedy
loss IDs with maximum NLL delta 3.66e-15 against the modified reference. All 100
tasks and decoding complete, matching all 5,094 sampled decisions and stops.
Isolated grading first stopped on an owned setuid `ps` inspection error. The native
process-group replacement and concurrent-working-set monitor pass 156 shared
tests and nine accounting controls in both Python modes, SDK/live checks and a
260-command inert stress check. All 14 isolation and 28 authored evaluator
controls then pass. The fresh retry completes all 80 isolated program runs,
retains 37/100 task passes for modified reference and native, verifies guest
shutdown and passes both complete quality comparisons. Maximum retry guard
charge is 2.48 GB including the 2 GiB guest reservation; CPU policy remains four
seconds. All 62 state checks and exact event/memory replay pass, including
cancellation/resume to the same 16 IDs/text and all six hidden-stop cases.
Maximum outer guard charge is 3.90 GB, with no survivors; these are functional
checks. Nineteen recovery-pipeline controls pass in both modes.
Astra approves CPU retention and supporting host changes with no actionable
findings. Commit/push references are recorded with publication. Build/assembly
limits and failures remain.
All 441 portable tests across 60 local modules pass in both modes under 512 MiB
after correcting interpreter/module-isolation invocations. A clean publication
worktree preserves root UI/workflow/history edits and adds ownership controls
to CI. Astra approves scoped retention; managed adoption remains open.
Fresh publication formatting and all 59 tests in the changed workflow pass;
all 444 numerical inputs match the 12 required Rust check receipts/logs.
The measured baseline is byte-exactly reproduced by a fresh build from clean
`2da16a9e` after state exit, within the 2 GiB build budget. The original cache
and binary are preserved; the original build's historical commit remains unproven.
The portable record preserves all 208 scalar samples and exact source replay;
initial report power-label/trial-identity assumptions are corrected with failures retained.
[Experiment scope / request measurements](lfm3-finite-widening.md),
[consolidated checks / retained failures](../artifacts/lfm3-finite-guard-2026-11-11/widen-checks-consolidated-1.json),
[primary analysis](../artifacts/lfm3-finite-guard-2026-11-11/native-primary-analysis-2.json),
[matrix execution](../artifacts/lfm3-finite-guard-2026-11-11/compatibility-execution-plan-3.json),
[full matrix analysis](../artifacts/lfm3-finite-guard-2026-11-11/compatibility-analysis-3.json),
[CPU phase observations](../artifacts/lfm3-finite-guard-2026-11-11/compatibility-phase-observations-1.json),
[closed receipts / fresh loss-build binding](../artifacts/lfm3-finite-guard-2026-11-11/compatibility-closure-checks-1.json),
[fresh quality execution](../artifacts/lfm3-finite-guard-2026-11-11/fresh-quality-execution-plan-3.json),
[closed 10,000-target collection](../artifacts/lfm3-finite-guard-2026-11-11/fresh-loss-collection-closure-checks-1.json),
[retained grading failure](../artifacts/lfm3-finite-guard-2026-11-11/fresh-quality-grading-failure-audit-1.json),
[guard stress qualification](../artifacts/lfm3-finite-guard-2026-11-11/grading-inert-guard-1/checks.json),
[fresh complete quality](../artifacts/lfm3-finite-guard-2026-11-11/fresh-quality-recovery-4/original-quality-decision.json),
[host requalification](../artifacts/lfm3-finite-guard-2026-11-11/host-requalification-1/result.json),
[recovery and regression checks](../artifacts/lfm3-finite-guard-2026-11-11/grading-recovery-checks-1.json),
[baseline byte-reproduction preparation](../artifacts/lfm3-finite-guard-2026-11-11/baseline-byte-reproduction-preparation-1.json),
[exact baseline reproduction](../artifacts/lfm3-finite-guard-2026-11-11/baseline-byte-reproduction-result-1.json),
[state execution](../artifacts/lfm3-finite-guard-2026-11-11/state-execution-plan-4.json),
[complete 62-check replay](../artifacts/lfm3-finite-guard-2026-11-11/state-analysis-4.json),
[complete pre-review qualification](../artifacts/lfm3-finite-guard-2026-11-11/finite-qualification-closure-1.json),
[review/commit](../artifacts/lfm3-finite-guard-2026-11-11/finite-publication-checks.json),
[portable measurements](../benchmarks/results/2026-11-12-lfm3-finite-widening-v1.json),
[current experiment](../artifacts/lfm3-finite-guard-2026-11-11/active-experiment-state-1.json).

Granite configuration admission is implemented and under qualification:
canonical default RoPE shares the strict Llama declaration parser; equal supported
dtype aliases are normalized only for Granite. Existing scale, shape, context,
quantization and index checks remain binding. Three pinned raw configs and frozen
CPU/Metal graphs cover the new dialects. Required fmt/both strict Clippy,
737 CPU/883 Metal tests, explicit GPU and 44 CLI cases pass; eight monitor
controls in both modes and an owned live control pass. Six failed attempts
remain retained; compiler-only admission is 3 GiB within the 6 GiB Mac ceiling.
Astra approves, no findings. [Scope](granite-config.md),
[portable checks](../benchmarks/results/2026-11-12-granite-config-v1.json).

Both pinned Granite tensor packages authenticate without model loading or index
repair. Full bounded affine4/8 conversion and saved-byte lineage audit pass:
282 matrices/81 norms, with all 3,659,530,240 matrix values compared exactly to
upstream FP32 during conversion. The offline audit replays hashes and exact
norms; it does not repeat floating-point reconstruction. All 48 CPU/GPU controls,
33 portable tests per Python mode and six actual-artifact CLI cases pass.
Observed conversion charge is 455,131,136 bytes within the 512 MiB audit cap.
Existing Metal metadata accepts 128/512 and rejects 1,024 positions; scalar FP32
exceeds 6 GiB. Earlier memory, index-field and CLI-expectation failures remain
retained. Evidence closure and Astra review pass, no findings. Legacy MiniCPM tools
remain unchanged; full-model fit, inference, quality and performance are open.
[Controls](../artifacts/granite-conversion-2026-11-12/controls-closure-2.json),
[full conversion](../artifacts/granite-conversion-2026-11-12/full-conversion-1/manifest.json),
[saved-byte audit](../artifacts/granite-conversion-2026-11-12/full-audit-1.json),
[CLI cases](../artifacts/granite-conversion-2026-11-12/child-inspection-2.json),
[review/commit](../artifacts/granite-conversion-2026-11-12/publication-checks.json).

The bounded mixed affine4/8 mechanism passes checks and Astra review.
The installed MLX default requests group-64 8-bit output/down projections and
untied head. The existing MiniCPM5 converter remains affine8/6,144-column only;
Granite requires 8,192 columns. No full payload, model inference, quality, physical
fit, request performance or managed adoption is qualified by config admission.
[Conversion constraints](../artifacts/granite-source-planning-2026-11-12/conversion-reuse-inspection-1.json).

Granite conversion `eb14db30` is pushed and both exact-head CI workflows pass.
Both complete tokenizer files authenticate; only the modified source's explicit
BatchLongest left-padding declaration differs from stock. All other tokenizer
fields match, with 100,352 IDs and 96 added-token IDs also in BPE vocabulary.
The first counting assumption fails and remains retained; corrected bounded
inspection passes. Five cold planning cases preserve the existing 512-position
Metal cap and admit the 128/128 request on Apple M1 Pro. The actual raw CLI smoke
passes with all 128 outputs, 255 committed positions and one pending token at
2,899,741,912 bytes observed charge within 6 GiB. Source/binary/input hashes close
unchanged and owned cleanup is empty. Debug clocks do not qualify performance;
full numerical, quality, codec, all-workload fit and adoption gates remain open.
[Codec preparation](../artifacts/granite-codec-2026-11-12/codec-inspection-2.json),
[cold planning](../artifacts/granite-runtime-plan-2026-11-12/runtime-plan-1.json),
[raw native smoke](../artifacts/granite-native-smoke-2026-11-12/native-smoke-1.json).

Granite's complete graph diagnostic rejects the current affine4/8 Metal route:
212/20,268,711,936 profile-coordinate comparisons fail unchanged per-array gates.
All 30 native runs/304 observed state checks pass, with exact schedule, reset and
causal-prefix bytes; the observed peak is 5.03 GB under 6 GiB. Independent
reference capture passes within 512 MiB. The first failure is reproduced from
native Q/K/V, pointing upstream; other failures remain unlocalized. Required
fmt/strict CPU+Metal Clippy, 737 CPU/887 Metal tests, 58 portable controls per mode,
1,120 fixture comparisons, the GPU fixture and admitted numeric controls pass.
Failed reference/launcher attempts are retained. Tooling retention and the
explicit non-adoption decision have Astra follow-up approval, with no remaining
findings. Published as `8a90db86`; both exact-head CI workflows pass. Its two required-
evidence findings are fixed: deletion controls and fresh genuine replay verify
all 304 checks/10 resets/40 receipts with byte-identical numerical records. The
genuine capture remains numerically rejected. Quality, performance, overhead
and managed adoption remain unqualified.
[Scope](granite-graph.md),
[decision](../benchmarks/results/2026-11-13-granite-graph-v1.json),
[strict comparison](../artifacts/granite-graph-2026-11-12/offline-comparison-2/manifest.json).

Ministral source preparation records the existing configuration's independent
3,072 hidden/4,096 query widths and the selected adapter's unsupported text tag.
Its source combines YaRN with absolute-position query scaling cast before SDPA;
independent fixtures must cover the 16,384-token boundary and cache offsets.
The pinned Q4 index rejection is retained; payload, conversion and runtime
qualification remain open. This inspection launches no model or compiler.
[Source inspection and remaining gates](../artifacts/ministral3-source-planning-2026-11-12/static-source-inspection-1.json). Current source pins are
revalidated after retaining/explaining Granite's changed Rust hash. At least
1.15 GB more than observed free space is needed even for complete parents plus a
theoretical all4 text child and disk floor, before format/exclusion overhead;
rebuildable compiler-cache cleanup precedes payload work. No model is launched.
[Current preflight](../artifacts/ministral3-source-planning-2026-11-13/input-revalidation-2.json).

Ministral preparation reclaims 1.66 GB observed disk space from 12,277 retired
Cargo intermediates; measured binaries, current caches and raw evidence remain
available. A frozen 11-case profile now precedes independent Rust/fixture work
on cached CPU Half YaRN and post-rotation query gain. The 1.21 MB independent
fixture, 24 portable controls per mode, six targeted Rust tests, fmt and strict
CPU/Metal Clippy and all 744 CPU/894 Metal-feature tests pass. Six release
primitive tests and zero-allocation key/query/rejection controls pass; optimized
Python regenerates identical bytes. Astra approves scoped retention with no
findings. Published as `7ea011a6`; both exact-head CI workflows pass. This mechanism
admits no model or Metal route; original downstream gates remain. Next work splits
Metal implementation and source/conversion planning, with exclusive file ownership
and one root-owned heavy-check queue.
[Execution plan](../artifacts/ministral3-yarn-2026-11-13/parallel-work-plan-1.json).
[Review/commit](../artifacts/ministral3-yarn-2026-11-13/publication-checks.json).
[Cache receipt](../artifacts/ministral3-source-planning-2026-11-13/cache-cleanup-result-1.json),
[prospective profile](../artifacts/ministral3-yarn-2026-11-13/prospective-primitive-protocol-2.json).

Ministral selected-config binding and test-integrated borrowed Metal YaRN
pass 755 CPU/907 Metal-feature tests, fmt/strict Clippy and three release GPU
diagnostics under unchanged gates. GPU output comparisons, exact replay/chunk/
rebase, numeric status/reuse and zero measured Rust encode allocations pass.
Four 3+10 operator cohorts are diagnostic only. A Cargo-kind receipt-finalization
error is retained and corrected without repeated checks. Astra approves scoped
retention with no findings. Published as `5c0eee96`; both exact-head CI workflows
pass. Model/source/graph/quality/request and O1-overhead qualification remain open.
[Checks/review](../artifacts/ministral3-metal-2026-11-13/publication-checks.json).

Ministral source acquisition authenticates four complete containers within the
512MiB setup policy after6.55GB observed compiler-cache reclamation. Stock's
optional index count remains rejected; modified-only mixed conversion passes
all3,428,843,520 matrix reconstructions exactly. All144 genuine controls and86
portable tests perPythonmode pass. The saved-byte auditor now validates rich
metadata identities and all4,252 tiles; failed checks stayretained. ActualCLI
preservesunknown_architecture. RequiredRust checks reuse4,641 unchangedinputs.
Astra approves scoped retention with no findings. Published as `83a8617a`;
all three exact-head CI workflows pass. Graph/codec/quality/fit/request/O1/
adoption remain open. The shared tokenizer JSON authenticates separately; raw
codec and differing chat profiles remain unqualified.
[Next codec preflight](../artifacts/ministral3-codec-2026-11-13/preflight-1.json).
[Conversion/checks/review](../artifacts/ministral3-conversion-2026-11-13/publication-checks.json).

The [parallel delivery plan](parallel-delivery.md) now assigns the active
Qwen3.5-2B milestone to typed-source, hybrid-graph and independent-reference lanes.
One integrator owns shared APIs, early integration, the single heavy-job queue,
combined Astra review and publication. The
[dispatch receipt](../artifacts/qwen35-2b-source-planning-2026-11-13/parallel-work-plan-1.json)
records file ownership, resource bounds and remaining gates. Track integration,
queue/review wait and repeated-check reasons in one ledger. The typed-source/tiny-graph slice passes formatting,
strict CPU/Metal Clippy, 781 CPU/940 Metal-feature tests, twelve portable controls
per Python mode and its explicit GPU storage check. Astra approves scoped
diagnostic retention with no findings. No workflow speedup is measured;
full-model qualification gates remain open.
[Review/publication](../artifacts/qwen35-2b-source-planning-2026-11-13/publication-checks.json).
[Diagnostic scope](qwen35-2b-typed-source.md).

The next 2B slice implements the pinned raw-ID hybrid CLI, complete phase
capture/reference protocols and bounded source/parent/conversion audits. Early
compilation, strict CPU/Metal Clippy and 786 CPU tests pass; real qualification
and combined Astra review remain pending. Fresh selected-source byte verification
passes after a retained monitor-interpreter failure. Parent history and the full
metadata directory are captured; tensor values are being compared under the
unchanged gates. Pinned-CA TLS retains certificate/hostname verification. Source
and reference controls pass 34/23 tests per explicit optimization mode; prior
mock/fixture failures and journal-reservation correction remain linked in their
receipts. No conversion, model, quality or performance result is inferred.
[Current source evidence](../artifacts/qwen35-2b-payload-2026-11-13/source-setup-qualification-1.json).

Ministral graph/CLI diagnostic code passes fmt, strict CPU/Metal Clippy,
763 CPU/922 Metal-feature tests and all five release GPU controls, including
eight zero-Rust-allocation tiny decode steps. The debug allocation failure is
retained. Full capture completes 22 runs/9,218 arrays and exact state/causal
checks, but 78/244,809,728 profile coordinates fail the unchanged numerical gate.
Console mode/phase/memory, request/error/cancel and custom-stop controls pass;
the private pre-child reader failure and corrected retry are retained. Current
route remains unqualified and does not advance to performance cohorts. Targeted
native-input replay traces the earliest projection failure to propagated inputs;
sampled operators pass their local gates without changing the full rejection.
Astra approves scoped diagnostic retention with no blocking code findings.
[Review/publication](../artifacts/ministral3-graph-2026-11-13/publication-checks.json),
[scope](ministral3-graph.md),
[decision](../artifacts/ministral3-graph-2026-11-13/graph-comparison-1/manifest.json).

Qwen3.5-2B needs a distinct 24-layer/16-value-head source profile. Its MLX
headers change eighteen gated-norm vectors and convolution layouts; normalization
offsets also depend on layout and source key prefixes. A stale historical Rust
hash fails the first static check; the corrected inspection pins current source
without changing existing 4B/9B gates. Declared-parent/payload authentication,
numerical/quality and runtime qualification remain open. No model or compiler
was launched.
[Source inspection and retained failure](../artifacts/qwen35-2b-source-planning-2026-11-12/static-source-inspection-2.json).

Completed LFM3/MiniCPM mechanism reports and prior disk-cleanup receipts
are preserved in the [12 November reviewed snapshot](archive/2026-11-12/README.md).
Exact inverse link reconstruction and local links pass; only 32 reviewed checked
entries and 106 narrative lines move out of working documents, reducing their
combined bytes 38.4%. Pending work, original user history and live PLAN stay intact.
[Validation](../artifacts/lfm3-finite-guard-2026-11-11/working-history-archive-checks-1.json).

## Retained performance decisions

| Change | Measured result and scope | Evidence |
| --- | --- | --- |
| LFM3 validated finite BF16 widening | Retained for CPU BF16: primary median requests fall 29.0%/29.5%, full matrix 20.6–27.9%; full original quality and 62 state checks pass within 6 GiB. Exact source/reduction order preserved; Astra approves with no findings. | [Scope](lfm3-finite-widening.md), [208 samples](../benchmarks/results/2026-11-12-lfm3-finite-widening-v1.json), [review/commit](../artifacts/lfm3-finite-guard-2026-11-11/finite-publication-checks.json) |
| MiniCPM5 exact-reduction RMS stores | Explicit Mac opt-in; two loaded-model CLI comparisons reduce warm requests **21.51% / 21.50%** (500 to 393 ms), TTFT about14%, and raise decode about35%. All 2,614 native arrays remain bitwise equal; required checks, native65 and Astra pass. Extra64MiB allowance, 6GiB and all model/default gates remain explicit. | [Scope](minicpm5-rms-store.md), [review/commit](../artifacts/minicpm5-rms-store-2026-11-10/publication-checks.json) |
| Ordered CPU projection tile | Warm 128/128 TTFT improves about 77.6% and request time about 52% in two pairs. Qualified opt-in; scalar reference preserved. | [Retention](experiments/2026-10-12-cpu-tiled-retention.md), `0abbe60` |
| Exact committed-prefix reuse | On Q4 0.5B, editing position 128 of a 512-token prompt reduces first-token latency about 21.6%; editing only position 511 about 93%, in paired runs. No KV copying; not Q4 quality qualification. | [Continuation](experiments/2026-10-13-prefix-continuation.md), `340c965` |
| Selected Half/Q4 and Online attention | M5/M6 retained only their documented configurations and quality/memory/context scopes. FP16 KV retained after the I8 evaluation. | [M5 release](experiments/2026-10-15-m5-release.md), [M6 release](experiments/2026-10-19-m6-release.md) |
| Four-row native affine8 prefill | Pinned Qwen2.5-0.5B MLX8 on M1 Pro: 1,024-token prefill improves 13.69%, complete fixed-token request 6.22%. Mac opt-in; managed adoption remains separate. | [M9 decision](m9-mac-affine-decision.md) |
| Gemma BF16 G2 last-row prefill | Official BF16 Mac route: both primary runs improve prefill about 61% and request time about 28%; retained in native45. | [Retention](../benchmarks/results/2026-11-06-gemma-bf16-g2-retention-v1.json) |
| Exact guard validation reduction | Same complete guard coverage with automatic integer vectorization. Two Mac BF16/G2 primary runs reduce request time25.0–26.6%;626 sustained requests confirm26.3%,512 deeper decode pairs bit-identical. Required checks/Astra pass ; 6 GiB preserved. No UI, memory-saving or broader-model claim. | [Scope](gemma-guard-validation.md), [checks/review/commit](../artifacts/gemma-guard-validation-2026-11-08/publication-checks.json) |
| Qwen2 prompt lookup | Repeated contract prompt gains about 53% decode, but the code prompt misses the 5% gate. Explicit nonstreaming API retained; default adoption rejected. | [Timing](../benchmarks/results/2026-11-06-m9-qwen2-lookup-timing-v1.json) |

## Existing correctness, memory and observability foundations

LFM3 bias-free BF16 CPU projections match 94,248 source outputs across four
selected rows of all 154 rank-two tensors, 65,280 finite controls and finite
corner/rejection-marker cases. The original FP32 diagnostic passes; no expanded
FP32 weights or numerical allocations. Fresh fmt/both strict Clippy/723 CPU/
869 Metal/nine example, eight optimized source tests and native77/C smokes pass.
All 42 release operator cohorts pass; 128-token representative MLP calls cost
802–810 ms. Component baselines only; actual child CPU/Metal exact correspondence
rejects. Astra approves scoped retention, no findings; attention/full graph/CLI/
quality/fit/adoption remain
open, with original gates unchanged.
[Scope/evidence](lfm3-projection-bf16.md),
[measurements](../benchmarks/results/2026-11-11-lfm3-projection-bf16-v1.json).

LFM3 transactional BF16 state is implemented with 13 six-row histories, six full
and five 256-token KV owners. Independent fixtures and actual declared-parent/
MLX CPU/Metal logical-state comparisons pass. All-owner validation precedes any
publication; stale/foreign proposals and failures retain the committed prefix.
Fresh fmt/both strict Clippy/714 CPU/860 Metal/nine example checks, eight release
state tests, native76 three 46-export slices and CPU/Metal C smokes pass. Debug
and release transitions allocate nothing. All 24 original FP32 5% compatibility
pairs pass; 18 BF16 state cohorts measure 0.338–0.365 / 7.119–7.254 / 27.706–28.644 ms at
1/32/128 tokens. State owns 28,108,984 B at capacity 2,048. Astra approves scoped retention, no findings;
complete LFM3 graph/CLI/fit/quality/adoption remain open. Existing
arithmetic correspondence rejections and original gates remain unchanged.
[Scope/evidence](lfm3-bf16-state.md).

Canonical LFM3 CPU BF16 convolution/binary/head-gate leaves match 8,146,944
convolution words, all 34 binary grids and broadcast source fixtures. Chunk
splits and the original FP32 diagnostic pass; all four operators allocate
nothing. Fresh required/native75 checks and six release component cohorts
pass: medians 1.150–1.168 / 33.547–33.923 / 135.099–135.660 ms at 1/32/128
tokens. Operator baselines only; actual child CPU/Metal exact correspondence
rejects. Astra approves scoped retention, no findings; transactional BF16
history/window/graph and full model fit/quality/adoption remain open.
[Scope/evidence](lfm3-binary-conv-source.md),
[review/commit](../artifacts/lfm3-conv-bf16-2026-11-11/publication-checks.json).

Canonical LFM3 CPU BF16 norms/unary gates match 4,082,688 norm outputs,
5,706 FP32 statistic rows, 16 corner cases and both full finite unary grids.
Both widths have zero Rust allocations; fresh required/native73 checks pass.
Six frozen release component cohorts measure medians 1.285–1.315 / 39.048–39.207 /
152.448–154.071 ms at 1/32/128 tokens. These are operator baselines; actual child
CPU/Metal exact component correspondence rejects. Astra approves scoped
retention, no findings; full LFM3 graph/fit/quality/adoption and remaining
BF16/window composition stay open.
[Scope/evidence](lfm3-bf16-source.md),
[review/commit](../artifacts/lfm3-bf16-2026-11-10/publication-checks.json).

Source-profile LFM3 BF16 rotary passes exact frequency/trig/460,800 output
fixtures, sparse rows and chunk schedules with zero steady-state allocations.
Two shared admitted owners total 1,049,152 B. Fresh required/native72 checks
pass; 18 release operator cohorts on M1 Pro measure eleven-layer Q/K medians
of 0.0665–0.0703 / 2.112–2.226 / 8.437–8.695 ms for chunks 1/32/128. These
are operator baselines; full model execution/fit/quality/adoption remain open.
Astra approves scoped retention, with no findings.
[Scope/evidence](lfm3-rotary-source.md),
[review/commit](../artifacts/lfm3-rotary-2026-11-10/publication-checks.json).

Seven-tap FP32 convolution/history passes independent exact/tensor fixtures,
288 schedules and zero-allocation checks, preserving concrete three-tap APIs.
The first generic public prepare boundary is rejected at its performance screen;
revised concrete entrypoints pass all six original 5% comparisons (−2.33% to
+1.26%). New seven-tap operator baselines are about 0.357/9.3/37 ms for chunks
1/32/128 across 13 width-2,048 owners. Fresh required/native71 checks pass;
Astra approves scoped retention, with no findings. Full LFM3/BF16 activation/
fit/quality/adoption remain open.
[Scope/evidence](lfm3-seven-tap-history.md),
[review/commit](../artifacts/lfm3-seven-tap-2026-11-10/publication-checks.json).

LFM3's complete BF16 child and modified parent authenticate, with exact equality
of all 286 tensors/1.729B parameters, 13 singleton-axis convolution transforms
and 119 norms. Independent child replay covers 1,014 blocks; no parent payload
is retained. Nineteen portable tests and 28 actual offline CLI rejections pass.
Closed stream/replay audits stay below a separate 512 MiB aggregate guard;
maximum conservative final RSS sum is 247.6 MB. Required engine inputs/outputs
are unchanged and fresh fmt passes. Astra's RSS finding is fixed: each final
peak must cover the same PID's sampled RSS, with fresh replay/negative controls;
Astra approves scoped retention. Next qualify
seven-tap scalar convolution/history fixtures and then the remaining graph;
quality, runtime, fit, performance and adoption stay open.
[Scope](lfm3-source-qualification.md),
[review/commit](../artifacts/lfm3-source-2026-11-10/publication-checks.json),
[parent audit](../artifacts/lfm3-source-2026-11-10/parent-checks-1.json),
[offline replay](../artifacts/lfm3-source-2026-11-10/offline-checks-2.json).

The MiniCPM5 native precision audit finds no output-grid obstruction across all
442,863,360paired original CPU coordinates. A complete replay agrees; a separate
constructive method verifies native witnesses everywhere. Both methods preserve
all original bounds and native 50 Half/1 FP32 domains, with 172.4 MB maximum final RSS
under a separate 512 MiB audit guard. Eight new/25 total portable, 11 CLI and one
actual source-byte negative pass. This does not fix the native 82 failures:
current managed adoption remains rejected, with the research route retained.
Next qualify LFM3's pinned BF16 source/payload and exact graph. Astra approves
scoped diagnostic retention, with no findings;
quality, precision, memory and performance gates stay unchanged.
[Scope](minicpm5-precision-audit.md),
[review/commit](../artifacts/minicpm5-representability-2026-11-10/publication-checks.json).

MiniCPM5 numerical isolation rejects the cached rotary model experiment.
All eight sampled projection coordinates match exact dots of their own inputs;
local rotary fidelity improves to zero failures in 7,084,800 original primitive
values, but the original short model suite fails six of 52,896,768 comparisons.
The required original gates remain binding. Candidate required/native69 checks,
two explicit GPU tests, 36 portable, six raw witnesses and 12 resealed negatives pass;
the deeper run is deliberately stopped and remains unqualified. The rejected
engine/CLI/reader patch is archived outside the proposed documentation-only
commit. Astra approves documentation-only retention with no findings; no
performance, quality or adoption claim.
[Scope/next audit](minicpm5-rotary-rejection.md),
[review/commit](../artifacts/minicpm5-projection-isolation-2026-11-10/publication-checks.json).

MiniCPM5 deeper diagnostics complete with a numerical rejection: 82 failures in
1.773B original-gate comparisons, reproduced by a fresh complete cold replay.
All 87,624 arrays/1,718 chunks and recovery controls complete within the 6 GiB
guard. No disjoint CPU bounds occur across 442.9M coordinates; isolate one
projection mismatch next. Fresh required/native68 checks, 17 portable tests,
15 CLI/51 metadata-sample negatives and 48 independent raw witness replays pass.
Astra's clock/ledger findings are fixed; 13 resealed datasets and all former
zero-ledger/repeated-window counterexamples reject. Both full replays preserve
the numerical result; Astra approves scoped retention, no remaining findings.
The ordinary route stays at 512/chunk32; full
quality, context/adoption and overhead gates remain open.
[Scope/results](minicpm5-context1025.md),
[review/commit](../artifacts/minicpm5-context1025-2026-11-10/publication-checks.json).

MiniCPM5 complete quality inputs now preserve all original 10,000 targets/100
tasks and their caps. Cold recipe replay, seven portable tests and 28 altered-
input rejections pass. Astra independently replays all payload hashes and
approves scoped retention, no findings. Original stock metadata is pinned,
without its weight download; all corpus/task IDs and task renders match the
selected child. Full scoring requires 1,025 positions (largest task 559), beyond
the unchanged 512 diagnostic limit. Numerical/quality/adoption gates remain
open. [Scope/evidence](minicpm5-quality-inputs.md),
[review/commit](../artifacts/minicpm5-quality-inputs-2026-11-10/publication-checks.json).

MiniCPM5 matrix reproduction is retained with Astra approval. The frozen
MLX 0.32.3 GPU/BF16 affine8 recipe reproduces all 296 matrices/888 arrays
exactly while authenticating both full parent objects. All 85 copied norms also
match; 4,128 tiles retain no parent payload. The separate 512 MiB audit bound
passes. This proves a reproducible conversion relation, without authenticating
the publisher's exact binary or qualifying inference/quality/adoption. All 35
portable tests, 34 actual-output negatives, exact seal replay and input-based
Rust/native check reuse pass. Astra independently replays the seal, no findings.
[Scope](minicpm5-matrix-lineage.md),
[review/commit](../artifacts/minicpm5-matrix-lineage-2026-11-10/publication-checks.json).

MiniCPM5 bounded basic text/CLI is retained with Astra approval. Exact98source fixtures/66streams,679CPU/821Metal, native67/C smokes,
36portable and83actual-report negatives pass. The unchanged22k-word/660-phase
cache stress stays below 512 MiB after a preserved/fixed ASCII decode-bound bug.
Two14/38-token native histories compare4.18Mlogits bitwise; eight cohorts retain
80measured/24warm requests, with RMS opt-in gains18.85%/18.32% and12.29%/11.79%.
6GiB owners/observations pass in named scopes; final physical peak unavailable.
Default allocator, original quality/deeper numerical,
managed/default adoption and1%summary-overhead gates remain open. Astra's sole
first-text enclosure finding is fixed; unchanged raw cohorts revalidate.
[Scope/evidence](minicpm5-text-codec.md),
[review/commit](../artifacts/minicpm5-text-codec-2026-11-10/publication-checks.json).

MiniCPM5 bounded source streaming verifies both complete BF16 parent hashes
(5.033GB),381tensors and85exact copied norms, retaining1,349block digests and no
payload. Fifteen portable tests and offline witness sealing pass; engine/native
inputs are unchanged. Astra approves scoped source retention. Matrix reproduction
is covered above; original task quality and adoption remain open.
[Scope](minicpm5-parent-auth.md).

- Readable scalar CPU implementations and independent numerical fixtures remain
  correctness references. Real-model, reduced-precision and quality results have
  separate frozen scopes; [model coverage](model-coverage.md).
- Immutable weights and session-owned KV/scratch preserve committed versus pending
  token history, exact-prefix reuse, capacity admission and cancellation/replay;
  [session evidence](../benchmarks/results/2026-10-09-qwen-session.md),
  [Metal engine](experiments/2026-10-13-metal-engine.md).
- Resident weights, streamed loading, explicit layouts and bounded scratch avoid
  redundant full-model ownership. Zero steady-state Rust numerical allocations
  are measured only for their recorded routes; native/application allocations
  remain separately scoped; [Metal engine](experiments/2026-10-13-metal-engine.md).
- Generation reports, completed Metal timings, bounded calibrated GPU profiles,
  Swift callback metrics and runtime planning already exist;
  [current source inventory](observability-performance-proposal.md#evidence-already-available).
- Memory reporting separates model/session owners, allocated KV capacity,
  planning/admission allowances and measured footprint/conditional loading peaks;
  [owner evidence](../benchmarks/results/2026-11-02-session-owner-memory-v1.json),
  [KV limits](../benchmarks/results/2026-11-03-kv-preview-apply-v1.json),
  [load checkpoints](../benchmarks/results/2026-11-03-load-memory-checkpoints-v1.json).
- Managed Mac profiles and official Gemma public download/offline Chat have scoped
  app evidence; [journeys](mac-hub-journeys.md),
  [official Gemma](../benchmarks/results/2026-11-07-gemma-public-resume-mac-v1.json).
  Further iPhone work remains deferred.

## Restrictions that remain binding

| Result | Current decision |
| --- | --- |
| Historical failed quantization/precision trials | Keep failed evidence and gates; a later passing configuration does not rehabilitate an earlier candidate. [Full history](archive/2026-11-07/improvements.md#rejected-or-inconclusive-experiments). |
| Qwen3 fidelity and Gemma3 coupled intermediates | Preserve rejected trajectories and distinguish native conformance from original adoption quality. [Qwen3](qwen3-native-profile.md), [Gemma3](gemma3-native-conformance.md). |
| Split-KV and Llama Online | Do not adopt rejected arithmetic. [Split-KV](../benchmarks/results/2026-11-05-m9-split-kv-session-v1.json), [Llama](todo-implementation-plans.md#llama-online-correctness). |
| Gemma QAT task quality | Full-task loss exceeds policy despite passing loss/state/fidelity checks. [Decision](../benchmarks/results/2026-11-06-gemma-qat-full-quality-v1.json). |
| Gemma G3 assistant | Longer guard regresses about 21%; keep test-only, retain ordinary G2. [Decision](../benchmarks/results/2026-11-06-gemma-assistant-mtp-retention-v1.json). |
| New device/context/default adoption | Requires its own applicable checks. Mac execution, metadata support or admission estimates do not establish unrestricted support. |

## Console reference comparison

The [LFM3 external configuration preflight](lfm3-external-config.md) is reviewed
in its metadata-only scope. Follow-on setup authenticates all 286 BF16 payloads
and four JSON files. A separate Ollama view changes thirteen singleton shape
entries for its pinned loader, preserving payload bytes and the original import.
All four fresh 3+10 external cohorts complete under 6 GiB with owned cleanup and
source authentication. Ollama lacks authoritative prompt IDs; vLLM matches exact
prompt/cache admission but matches native output IDs/text on only 3/13 prompts
per cohort. Client medians and explicit limitations are in the
[collection summary](../artifacts/lfm3-engine-comparison-2026-11-11/external-collection-summary-1.json).
Twenty-six genuine vLLM captures replay through the reviewed optional-timing
CLI with consistent arithmetic. No cross-engine speedup, quality or managed
adoption is qualified; full comparison/Astra review remains open. Setup and
failed/passing guards, source preservation and the removed redundant 3.46 GB
upload copy are retained in the [configuration evidence](lfm3-external-config.md).

The earlier Gemma comparison completes the following separate scope.
All78 requests pass across two3warm+10 cohorts for each of3engines on M1Pro16GiB.
Official BF16/G2 Toklet request medians2.184/2.202s; verified uncensoredMLX4bit
Ollama121/119ms and vLLM-Metal147/149ms. Precision, weights, cache hits and timing
boundaries differ; no engine speedup ratio or adoption qualification. All observed
memory guards and observed external lifetime sums stay below6GiB. Ollama0.40.15 and
local vLLM0.33.0/Metal0.33.0 are installed/verified. Original BF16 external runs
fail tested admission. The ordinary native affine4 inspector remains strict;
the mixed artifact uses the separate diagnostic below.
Offline report/document checks pass; Astra approves scoped publication.
[Comparison / limitations](console-engine-comparison.md),
[full evidence](../artifacts/engine-comparison-2026-11-07/console-comparison-report.json).

## Observability status

[O1 summaries](o1-gemma-request-summaries.md), [O2.A phase reporting](o2-gemma-cost-report.md),
[O2.B1 repeated workloads](o2-gemma-workloads.md), [O2.B2.A publication](o2-ui-publication.md),
[O3 attribution](o3-gemma-runtime.md) and [R1 screening](gemma-regression-job.md)
are delivered in their reviewed scopes. Full counts, failures, raw measurements
and commit references are in the [completed history](archive/2026-11-09/improvements.md#current-observability-work).
Summary collection stays opt-in/default-disabled: the 1% overhead bound remains
unqualified. Controller publication does not establish visible text.

The user has deferred screen-capture qualification to concentrate on engine/CLI
work. Ready9 finishes13exact trials but rejects433/1721future-display callbacks,
with zero occlusion;32numeric samples are retained. The session is closed, and
no further sharing requests are planned. Prior failures, isolated UI code and
all raw evidence remain preserved, unreviewed and uncommitted. First-visible
metrics remain unqualified; console metrics and automated checks lead active work.
[Deferred status](../artifacts/o2-ui-frames-2026-11-07/status-audit-1/current-state.json).

## Selected model decisions

Order remains Gemma 4 E2B, Qwen3.5-4B, Qwen3.5-9B, then the five under-4B
candidates. Prefer authenticated uncensored MLX artifacts, retaining each
artifact's independent numerical, quality, memory and request gates. The 6 GiB
Mac ceiling, 512/32 pilot and 512 MiB codec budget stay fixed; iPhone is deferred.
[Model/metric plan](model-observability-plan.md).

| Scope | Retained result or restriction | Evidence |
| --- | --- | --- |
| Gemma Heretic affine4 | Cold planning, authenticated mixed-format loading and independent fidelity are complete; task-quality loss rejects adoption. Keep official BF16/G2 selected. | [Artifact](gemma-heretic-mlx.md), [quality rejection](gemma-heretic-quality-probes.md) |
| Qwen3.5-4B | Separate pinned owners, scalar/hybrid graph, codec and generation state bridge are delivered. Full 128/256 fidelity fails: 13,505 failures/3,132 disjoint intervals. No managed adoption; unchanged gates bind later work. | [Model plan](model-observability-plan.md), [decision](../artifacts/qwen35-context256-2026-11-09/decision.json) |
| Qwen3.5-9B recipe/store/head | Separate strict source and untied-head support retain original packed owners and authenticate the complete 5.038 GB stream. These mechanisms alone do not qualify execution or fit. | [Recipe](qwen35-9b-cold.md), [store](qwen35-9b-storage.md), [head](qwen35-untied-head.md) |
| 9B one-input oracle/Metal | Both complete dual-profile captures pass. Hybrid Metal has zero retained CPU matrix bytes and 5.667 GB peak; 64.72×/64.42× applies only to the raw one-input baseline. | [Oracle](qwen35-9b-oracle.md), [Metal](qwen35-9b-metal.md) |
| 9B context17 | Both 33.85M-value gates, partitions, nonzero-prefix failure/repair/retry and zero-allocation checks pass. Release medians 4.093/4.121s are descriptive; no overhead, request, tail, sustained or adoption claim. | [Context17](qwen35-9b-context17.md), `41287c6c` |
| Exact recurrence reference | All 67.17M short-context FP32/FP64 values match scalar references bitwise. Two cohorts lower helper step time 60.02/60.25% FP64 and 95.55/95.23% FP32. No engine/deeper-context gain follows. | [Reference helper](qwen35-reference-vector.md), `88950b11` |
| LFM3 CPU BF16 | Authenticated graph/CLI, 1,025-position parity, both complete original quality comparisons and 62 state checks pass; finite widening retained after matched request measurements and Astra approval. Managed/GPU adoption stays open. | [Scope and decision](lfm3-finite-widening.md) |
| Remaining under-4B qualification | Metadata is pinned; Granite, Ministral and Qwen3.5-2B still need payload/graph/quality/fit/request qualification. MiniCPM5's numerical rejection and LFM3's scoped CPU results remain explicit. | [Candidates and receipts](model-observability-plan.md#five-candidates-strictly-below-4b-total-parameters) |

Original individual check counts, source revisions, measured cohorts, rejected
experiments and Astra/commit receipts remain in the [reviewed history](archive/2026-11-09/improvements.md#latest-documentation-work)
and linked scope reports. Metadata and narrowly passing probes do not qualify
whole-GPU execution, default allocator/codec/application fit or managed requests.

## Current implementation

The Qwen3.5-2B real-source/CLI diagnostic passes full prospective parent lineage,
bounded Q4 conversion and saved-byte audit for 187 matrices/133 exact small
owners. Required fmt/strict CPU+Metal Clippy, 786 CPU/946 Metal tests, 70 portable
controls per Python mode, four GPU/state/repair and twelve CLI cases pass.
The short graph passes 159,187,008 comparisons. Context17 rejects three of
680,841,024 comparisons; both native traces match, all schedule/reset/causal
controls and complete journals/failing bytes audit. Conditional fixed255,
quality and performance do not advance. Astra approves scoped diagnostic
retention with no blocking findings. Local-input investigation, codec/stop
witnesses, original full quality, representative fit/performance/overhead and
adoption remain open; no performance gain follows. Failed monitor, reader,
allocator, import, binary-selection and audit-wrapper attempts stay retained.
[Scope](qwen35-2b-real-source.md),
[review/commit](../artifacts/qwen35-2b-payload-2026-11-13/publication-checks.json).

The bounded follow-up reproduces all 34,119 native arrays and recovers all
17,394 staged prefix arrays exactly. None of the 24 precision policies or two
declared FP32 reduction schemes passes both selected cases under both original
profiles. This variant search stops with no engine change; Astra approves
the evidence closure with no findings. [Final local rejection](../artifacts/qwen35-2b-attention-2026-11-13/two-f32-qk-check-1/result.json).

Delivery now overlaps evidence closure, current LFM3 profiling preparation and
independent exactness controls with exclusive file ownership. One integrator
keeps the serial heavy queue, required checks, combined Astra review and push.
LFM3 two-token weight reuse remains a hypothesis pending current profiling;
there is no measured workflow or kernel speedup.
[Delivery plan](parallel-delivery.md),
[checks/review/publication](../artifacts/qwen35-2b-attention-2026-11-13/velocity-publication-checks.json),
[source preflight](../artifacts/qwen35-2b-attention-2026-11-13/lfm3-next-performance-preflight-1.json).
Fresh preparation now completes the cached release-probe build and three full
projection baseline cohorts, with 30 measured samples and six passing power/
thermal endpoints. The sandbox-denied initial launch remains failed; the retry
passes. Whole-request attribution and any candidate implementation/retention
remain open. [Current baseline](../artifacts/qwen35-2b-attention-2026-11-13/lfm3-current-operator-baseline-1.json).
Current request attribution completes with exact output/state and 22 matching
input endpoints under 6 GiB. Projection dot accounts for 801/828 active-main
samples in the initial prefill window; this warrants an exact projection
experiment, with no speedup claim. A report-wrapper line-index correction
requires no model rerun. [Profile closure](../artifacts/lfm3-tokens2-2026-11-13/request-profile-closure-1.json).
Before a new kernel integration, the official Gemma observer-overhead gate is
next: its current clean-source Mac archive build passes and binds 462 input
hashes. Historical archives are not substituted for this artifact.
[Archive binding](../artifacts/lfm3-tokens2-2026-11-13/gemma-current-staticlib-1.json).
The original fresh ten-pair TTFT bounds remain unqualified and retained. Bounded
10..100-pair support now preserves default behavior, the worker ring and timed
boundaries. Forty dependent portable controls pass in both modes; default
real-data statistics replay exactly. Strict Swift/conformance and both fixed
100-pair cohorts pass all original headless gates; independent audit reconciles
412 IDs, 206 raw summaries, state/status/clocks and 489 input endpoints. Combined
Astra approves scoped retention with no findings. App/UI/render qualification
stays separate and opt-in.
[Scope, results and evidence](gemma-summary-overhead.md),
[checks/review/publication](../artifacts/lfm3-tokens2-2026-11-13/gemma-stronger-publication-checks.json).

The bounded two-token LFM3 experiment is **closed, rejected for code retention**.
Correctness/allocation controls, fmt, strict CPU/Metal Clippy and 794 CPU/954 Metal
tests pass. Corrected fixed operator cohorts preserve whole outputs and reduce
large-shape latency 48.99/49.53%, but eight slower one-token pairs across six
configurations leave the original compatibility gate unresolved. Astra approves
evidence-only closure; all five candidate code files are restored exactly to
`cf80a168`. No further variant search or model/request integration follows.
Original negatives, the excluded preparation-overlap screen and full quiet-host
rerun are retained. Exclusive file ownership, one heavy queue and explicit pause
acknowledgements govern [parallel delivery](parallel-delivery.md); no workflow
speedup is measured yet.
[Scope](lfm3-two-token-projection.md),
[review](../artifacts/lfm3-tokens2-2026-11-13/tokens2-astra-review-2.json),
[revert](../artifacts/lfm3-tokens2-2026-11-13/tokens2-code-revert-2.json),
[checks/review/publication](../artifacts/lfm3-tokens2-2026-11-13/tokens2-publication-checks-2.json).

Headless Gemma workload identity/admission has complete fixed-cohort evidence.
Astra approves scoped code retention and exact workload qualification, no findings. Strict Swift, 57 portable
controls per Python mode, fmt, strict CPU/Metal Clippy and 786 CPU/946 Metal tests
pass. All 52 requests/40 measured/26 summaries verify exact384input/128output,
state512/511/1, reset admission, clocks, native work and 6 GiB fit. Both original
1% median-ratio and paired-median 95% upper gates pass for both metrics/cohorts;
independent replay reproduces all statistics and verifies489 current endpoints.
The cold writer bound and launch dependency fixes are independently checked;
original power/refusal failures remain recorded. Only this headless BF16/G2 M1
Pro boundary is qualified; production stays disabled/opt-in. No numerical512-input,
UI/render, tail or default-adoption qualification is transferred.
[Scope](gemma-workload-identity.md),
[checks](../artifacts/gemma-workload-identity-2026-11-13/checks-final-3.json),
[audit](../artifacts/gemma-workload-identity-2026-11-13/independent-workload-audit-3.json),
[review/commit/CI](../artifacts/gemma-workload-identity-2026-11-13/publication-checks-3.json).

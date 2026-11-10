# Toklet improvement index

Updated **10 November 2026**. This is the compact index of retained capabilities,
measured gains and material restrictions. The [full dated history](archive/2026-11-09/improvements.md)
preserves every previous entry, source revision, failed check and evidence link.
Read it only when the current task needs that detail.

Measurements apply to the named hardware, route, workload and timing boundary.
Gains from different experiments cannot be added together. Current task and
qualification scope live in [TODO](../TODO.md); acceptance policy is in
[PLAN](../PLAN.md#validation-and-performance-policy).

The 9 November refresh preserves reviewed `88950b11` snapshots and cuts routine
TODO/improvement bytes about 56%. Exact inverse-link reconstruction, local links,
anchors and whitespace pass; live PLAN and numerical inputs are unchanged.
[Archive validation](../artifacts/qwen35-9b-context256-2026-11-09/working-history-checks.json).

The fourth user-requested disk cleanup removed 7,919 inactive Cargo test
executables, recovering 22.5 GiB net and leaving 80.0 GiB free at completion.
All 22,464 protected hashes and 494,721 retained-cache metadata checks pass.
Benchmark examples, engine outputs, current caches, models, raw evidence and
user work are preserved; engine implementation and qualification are unchanged.
[Receipt](../artifacts/lfm3-source-2026-11-10/disk-cleanup-4/result.json).

The third user-requested disk cleanup removed 23,292 inactive Cargo
support/intermediate files and recovered 5.3 GiB net, leaving 65.8 GiB free.
All 4,054 protected hashes and 502,723 retained-cache metadata checks
pass; models, raw evidence, engine binaries/libraries, current caches and user
work are preserved. Corrected process/schema preflight failures are retained.
This cleanup adds no engine implementation or qualification claim.
[Receipt](../artifacts/minicpm5-projection-isolation-2026-11-10/disk-cleanup-3/result.json).

User-requested disk recovery removed 1,809,290 compiler intermediates from 190
inactive Cargo caches, recovering about 51.1 GiB net and leaving 68.0 GiB free.
All 4,054 protected hashes and retained-cache metadata checks pass; models,
evidence, executables/libraries, current caches, user work and the running
numerical capture are preserved. Compressed manifests retain the exact deletion
list and the cleanup controller's verified interrupt/resume history.
[Receipt](../artifacts/minicpm5-context1025-2026-11-10/disk-cleanup-2/result.json),
[manifest storage](../artifacts/minicpm5-context1025-2026-11-10/disk-cleanup-2/manifest-storage.json).

## Retained performance decisions

| Change | Measured result and scope | Evidence |
| --- | --- | --- |
| MiniCPM5 exact-reduction RMS stores | Explicit Mac opt-in; two loaded-model CLI comparisons reduce warm requests **21.51% / 21.50%** (500 to 393 ms), TTFT about14%, and raise decode about35%. All 2,614 native arrays remain bitwise equal; required checks, native65 and Astra pass. Extra64MiB allowance, 6GiB and all model/default gates remain explicit. | [Scope](minicpm5-rms-store.md), [review/commit](../artifacts/minicpm5-rms-store-2026-11-10/publication-checks.json) |
| Ordered CPU projection tile | Warm 128/128 TTFT improves about 77.6% and request time about 52% in two pairs. Qualified opt-in; scalar reference preserved. | [Retention](experiments/2026-10-12-cpu-tiled-retention.md), `0abbe60` |
| Exact committed-prefix reuse | On Q4 0.5B, editing position 128 of a 512-token prompt reduces first-token latency about 21.6%; editing only position 511 about 93%, in paired runs. No KV copying; not Q4 quality qualification. | [Continuation](experiments/2026-10-13-prefix-continuation.md), `340c965` |
| Selected Half/Q4 and Online attention | M5/M6 retained only their documented configurations and quality/memory/context scopes. FP16 KV retained after the I8 evaluation. | [M5 release](experiments/2026-10-15-m5-release.md), [M6 release](experiments/2026-10-19-m6-release.md) |
| Four-row native affine8 prefill | Pinned Qwen2.5-0.5B MLX8 on M1 Pro: 1,024-token prefill improves 13.69%, complete fixed-token request 6.22%. Mac opt-in; managed adoption remains separate. | [M9 decision](m9-mac-affine-decision.md) |
| Gemma BF16 G2 last-row prefill | Official BF16 Mac route: both primary runs improve prefill about 61% and request time about 28%; retained in native45. | [Retention](../benchmarks/results/2026-11-06-gemma-bf16-g2-retention-v1.json) |
| Exact guard validation reduction | Same complete guard coverage with automatic integer vectorization. Two Mac BF16/G2 primary runs reduce request time25.0–26.6%;626 sustained requests confirm26.3%,512 deeper decode pairs bit-identical. Required checks/Astra pass ; 6 GiB preserved. No UI, memory-saving or broader-model claim. | [Scope](gemma-guard-validation.md), [checks/review/commit](../artifacts/gemma-guard-validation-2026-11-08/publication-checks.json) |
| Qwen2 prompt lookup | Repeated contract prompt gains about 53% decode, but the code prompt misses the 5% gate. Explicit nonstreaming API retained; default adoption rejected. | [Timing](../benchmarks/results/2026-11-06-m9-qwen2-lookup-timing-v1.json) |

## Existing correctness, memory and observability foundations

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

MiniCPM5 S1.A adds explicit default-RoPE declarations for the existing Llama
graph. Required fmt/both strict Clippy, 670 CPU/806 Metal tests and five explicit
GPU tests pass. Three equivalent CLI reports and four rejection controls pass;
after metadata medians are 2.107–2.145 ms/open. No repeated ≥5% legacy regression
appears, but unchanged-before-binary drift was 36.6%; no optimization claim.
Three native slices/46 exports each and CPU/Metal C smokes pass. Astra approves
after the timing-envelope fix:10portable tests and all six unchanged cohorts
revalidate. Checks/review/commit are in the linked publication record. The original 4-bit source
now returns401; the accessible labelled-uncensored 8-bit candidate has consistent
metadata and a declared accessible parent. Whole payload/parent authentication,
model execution, codec, quality, physical fit, requests and adoption stay open.
[Scope](minicpm5-default-rope.md),
[checks/review/commit](../artifacts/minicpm5-default-rope-2026-11-10/publication-checks.json),
[measurements](../artifacts/minicpm5-default-rope-2026-11-10/metadata-comparison-v3.json),
[access/recovery record](../artifacts/minicpm5-payload-2026-11-10/decision.json).

S1.B short console checks pass: complete publisher identity, 698 CPU reference
arrays, 52,896,768 dual-profile native comparisons (zero failures), exact state
checks, fresh 670 CPU/810 Metal tests and native64 three-slice/C-smoke checks.
Forty measured requests plus 12 warmups show token-only readback 0.64–0.70% slower
on the fixed raw 4..7-context workload; keep defaults and both explicit options.
No optimization, full quality, codec, deeper numerical, adoption or 1% overhead
claim follows. Parent whole hashes/lineage remain open; disk/toolchain failures
and protected cache recovery evidence are preserved. Astra timing-enclosure and
owner-ledger findings are fixed with exact offline revalidation; scoped retention
is approved. [Review/commit](../artifacts/minicpm5-heretic8-payload-2026-11-10/publication-checks.json).
[Scope, measurements and limitations](minicpm5-console.md).

The [MiniCPM5 CLI profiler](minicpm5-profile.md) passes 104 matched phase pairs,
74,672 calibrated intervals and 13.58M exact logit comparisons across two runs.
Prefill projections and decode normalization are measured targets; diagnostic
phase inflation is 11.5–19.7%, so no ordinary operator-cost, 1% overhead or speedup
claim follows. Fresh lint/example tests, portable/damaged-output controls and
input-bound core/native reuse pass. Astra approves the counter-enclosure fix and
scoped retention; [review/commit](../artifacts/minicpm5-profile-2026-11-10/publication-checks.json).

## Selected model decisions

Order remains Gemma 4 E2B, Qwen3.5-4B, Qwen3.5-9B, then the five under-4B
candidates. Prefer authenticated uncensored MLX artifacts, retaining each
artifact's independent numerical, quality, memory and request gates. The 6 GiB
Mac ceiling, 512/32 pilot and 512 MiB codec budget stay fixed; iPhone is deferred.
[Model/metric plan](model-observability-plan.md).

| Scope | Retained result or restriction | Evidence |
| --- | --- | --- |
| Gemma Heretic affine4 | Cold planning, authenticated mixed-format loading and independent fidelity are complete; task-quality loss rejects adoption. Keep official BF16/G2 selected. | [Artifact and decision](gemma-heretic-mlx.md) |
| Qwen3.5-4B | Separate pinned owners, scalar/hybrid graph, codec and generation state bridge are delivered. Full 128/256 fidelity fails: 13,505 failures/3,132 disjoint intervals. No managed adoption; unchanged gates bind later work. | [Model plan](model-observability-plan.md), [decision](../artifacts/qwen35-context256-2026-11-09/decision.json) |
| Qwen3.5-9B recipe/store/head | Separate strict source and untied-head support retain original packed owners and authenticate the complete 5.038 GB stream. These mechanisms alone do not qualify execution or fit. | [Recipe](qwen35-9b-cold.md), [store](qwen35-9b-storage.md), [head](qwen35-untied-head.md) |
| 9B one-input oracle/Metal | Both complete dual-profile captures pass. Hybrid Metal has zero retained CPU matrix bytes and 5.667 GB peak; 64.72×/64.42× applies only to the raw one-input baseline. | [Oracle](qwen35-9b-oracle.md), [Metal](qwen35-9b-metal.md) |
| 9B context17 | Both 33.85M-value gates, partitions, nonzero-prefix failure/repair/retry and zero-allocation checks pass. Release medians 4.093/4.121s are descriptive; no overhead, request, tail, sustained or adoption claim. | [Context17](qwen35-9b-context17.md), `41287c6c` |
| Exact recurrence reference | All 67.17M short-context FP32/FP64 values match scalar references bitwise. Two cohorts lower helper step time 60.02/60.25% FP64 and 95.55/95.23% FP32. No engine/deeper-context gain follows. | [Reference helper](qwen35-reference-vector.md), `88950b11` |
| Under-4B metadata | Five candidates have pinned shape/format/lineage evidence and explicit inaccessible-parent, stale-index and oversized-format restrictions. No authenticated payload, new numerical graph, quality or adoption qualification. | [Candidates and receipts](model-observability-plan.md#five-candidates-strictly-below-4b-total-parameters) |

Original individual check counts, source revisions, measured cohorts, rejected
experiments and Astra/commit receipts remain in the [reviewed history](archive/2026-11-09/improvements.md#latest-documentation-work)
and linked scope reports. Metadata and narrowly passing probes do not qualify
whole-GPU execution, default allocator/codec/application fit or managed requests.

## Current implementation

Q2.G's [9B128/256 evaluation](qwen35-9b-context256.md) is in progress. The separate
producer and strict reader preserve both original profiles/gates. Fresh required
Rust checks,190portable Python tests, native build/ABI and two strict release-probe
builds pass; version2 adds session setup timing and10synthetic metric/closure
checks pass. Native capture v2 preserves primary guard/cleanup results before
parsing; strict receipt/raw linkage and2new synthetic failure controls pass.
Both captures now reject45,332/79.35M comparisons each;14,847 saved coordinates
have disjoint original profile tolerances. Native functional/recovery checks pass
at5.667GBpeak and3positive/39damage controls pass. No managed adoption follows.
The first benchmark attempt rejects before samples; retry2 preserves raw
preflight and passes12metric/preflight tests. After power correction,40measured/
12warm samples close at nominal boundaries:128medians32.068/30.848s and256medians
68.399/63.330s. First256 cohort varies widely; all samples retained. Full outputs
match; peak5.676GB conservative charge and1positive/17performance controls pass.
Astra approves scoped diagnostic retention and commit, no findings; no
optimization, overhead or adoption claim.
[Review/commit](../artifacts/qwen35-9b-context256-2026-11-09/publication-checks.json).

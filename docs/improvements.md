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

## Retained performance decisions

| Change | Measured result and scope | Evidence |
| --- | --- | --- |
| Ordered CPU projection tile | Warm 128/128 TTFT improves about 77.6% and request time about 52% in two pairs. Qualified opt-in; scalar reference preserved. | [Retention](experiments/2026-10-12-cpu-tiled-retention.md), `0abbe60` |
| Exact committed-prefix reuse | On Q4 0.5B, editing position 128 of a 512-token prompt reduces first-token latency about 21.6%; editing only position 511 about 93%, in paired runs. No KV copying; not Q4 quality qualification. | [Continuation](experiments/2026-10-13-prefix-continuation.md), `340c965` |
| Selected Half/Q4 and Online attention | M5/M6 retained only their documented configurations and quality/memory/context scopes. FP16 KV retained after the I8 evaluation. | [M5 release](experiments/2026-10-15-m5-release.md), [M6 release](experiments/2026-10-19-m6-release.md) |
| Four-row native affine8 prefill | Pinned Qwen2.5-0.5B MLX8 on M1 Pro: 1,024-token prefill improves 13.69%, complete fixed-token request 6.22%. Mac opt-in; managed adoption remains separate. | [M9 decision](m9-mac-affine-decision.md) |
| Gemma BF16 G2 last-row prefill | Official BF16 Mac route: both primary runs improve prefill about 61% and request time about 28%; retained in native45. | [Retention](../benchmarks/results/2026-11-06-gemma-bf16-g2-retention-v1.json) |
| Exact guard validation reduction | Same complete guard coverage with automatic integer vectorization. Two Mac BF16/G2 primary runs reduce request time25.0–26.6%;626 sustained requests confirm26.3%,512 deeper decode pairs bit-identical. Required checks/Astra pass;6GiB preserved. No UI, memory-saving or broader-model claim. | [Scope](gemma-guard-validation.md), [checks/review/commit](../artifacts/gemma-guard-validation-2026-11-08/publication-checks.json) |
| Qwen2 prompt lookup | Repeated contract prompt gains about 53% decode, but the code prompt misses the 5% gate. Explicit nonstreaming API retained; default adoption rejected. | [Timing](../benchmarks/results/2026-11-06-m9-qwen2-lookup-timing-v1.json) |

## Existing correctness, memory and observability foundations

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

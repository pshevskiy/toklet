# Toklet improvement index

Updated **8 November 2026**. This is the compact index of retained capabilities,
measured gains and material restrictions. The [full dated history](archive/2026-11-07/improvements.md)
preserves every previous entry, source revision, failed check and evidence link.
Read it only when the current task needs that detail.

Measurements apply to the named hardware, route, workload and timing boundary.
Gains from different experiments cannot be added together. Current task and
qualification scope live in [TODO](../TODO.md); acceptance policy is in
[PLAN](../PLAN.md#validation-and-performance-policy).

## Retained performance decisions

| Change | Measured result and scope | Evidence |
| --- | --- | --- |
| Ordered CPU projection tile | Warm 128/128 TTFT improves about 77.6% and request time about 52% in two pairs. Qualified opt-in; scalar reference preserved. | [Retention](experiments/2026-10-12-cpu-tiled-retention.md), `0abbe60` |
| Exact committed-prefix reuse | On Q4 0.5B, editing position 128 of a 512-token prompt reduces first-token latency about 21.6%; editing only position 511 about 93%, in paired runs. No KV copying; not Q4 quality qualification. | [Continuation](experiments/2026-10-13-prefix-continuation.md), `340c965` |
| Selected Half/Q4 and Online attention | M5/M6 retained only their documented configurations and quality/memory/context scopes. FP16 KV retained after the I8 evaluation. | [M5 release](experiments/2026-10-15-m5-release.md), [M6 release](experiments/2026-10-19-m6-release.md) |
| Four-row native affine8 prefill | Pinned Qwen2.5-0.5B MLX8 on M1 Pro: 1,024-token prefill improves 13.69%, complete fixed-token request 6.22%. Mac opt-in; managed adoption remains separate. | [M9 decision](m9-mac-affine-decision.md) |
| Gemma BF16 G2 last-row prefill | Official BF16 Mac route: both primary runs improve prefill about 61% and request time about 28%; retained in native45. | [Retention](../benchmarks/results/2026-11-06-gemma-bf16-g2-retention-v1.json) |
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
fail tested admission; native Q4 format remains strict and rejects this candidate.
Offline report/document checks pass; Astra approves scoped publication.
[Comparison / limitations](console-engine-comparison.md),
[full evidence](../artifacts/engine-comparison-2026-11-07/console-comparison-report.json).

## Current observability work

[O1 Gemma summaries](o1-gemma-request-summaries.md) and
[O2.A native phase/reporting](o2-gemma-cost-report.md) are delivered with Astra
approval and required checks. Native46 preserves the legacy ABI and adds a
completed-report copy; summary schema2 carries phase/work counters. Matched warm
reports identify decode as the dominant phase; no kernel speedup is claimed.
The ≤1% overhead bound remains unqualified, so collection stays opt-in/default
disabled. [O2.B1 numerical coverage](o2-gemma-workloads.md) now adds repeated
load/continuation comparisons and 601 seconds of sustained work. Both batches
pass token/state and counter checks; maximum child-process lifetime peak is
4.924 GB, below 6 GiB. Six new portable tests, strict standalone Swift and Astra
review pass; unchanged production checks are reused by verified source pins.
No optimization gain or UI/whole-app/tail qualification follows.
[O2.B1 evidence](../benchmarks/results/2026-11-07-o2b-gemma-workloads-v1.json).
[O2.B2.A publication metrics](o2-ui-publication.md) add bounded Send preparation,
callback TTFT and controller handling/publication. Six real Gemma controls,
52 UI and 52 headless paired requests, fresh Swift/app builds and 29 portable
checks pass; Astra approves after a typed state/work join correction. Both overhead tracks remain
unqualified/default-disabled. Visible-frame qualification and O3 remain open.
A bounded display-frame lab is now in validation: strict clock/pixel fixtures,
offline corruption checks, actual renderer crop inspection and fresh platform
builds pass. User selected the primary display; first attempt failed strict viewport visibility
before capture. Captured-frame/resource evidence and Astra remain pending; [current evidence](../artifacts/o2-ui-frames-2026-11-07/iteration-notes.json).
O3 continues independently: explicit BF16 CLI/live-owner inventory passes initial
unit/Metal/Clippy and Mac reconciliation (279ops/349tiles, exact5.814GB admission,
4.918GB processpeak). Encode-time capture, final gates and Astra remain;
[evidence](../artifacts/o3-gemma-runtime-2026-11-07/iteration-notes.json).
[Evidence](../benchmarks/results/2026-11-07-o2a-gemma-phase-report-v1.json).

The [R1 offline regression job](gemma-regression-job.md) now rejects incomparable
or reused collections and separates regression/noisy/speed signals over disabled
caller timings. Local named CI checks pass17 tests; Astra approval passes after
three fixes. It never accepts an optimization or qualifies missing quality,
memory, UI, thermal or tail gates.
[Checks/review/commit](../artifacts/r1-gemma-regression-2026-11-07/checks.json).

## Latest documentation work

The [selected model plan](model-observability-plan.md) now makes observability
the first deliverable: defined latency/throughput/memory/state/outcome metrics,
bounded collection and paired overhead checks before optimization. Order:
Gemma 4 E2B, Qwen3.5-4B, Qwen3.5-9B; five strictly under-4B candidates follow.
Uncensored MLX is preferred with per-artifact quality/admission gates; conversion
gaps and an over-budget F32 derivative are explicit. Content/link/whitespace
[checks pass](../artifacts/model-observability-plan-2026-11-07/checks.json), with
binding acceptance policy unchanged. This adds no qualified runtime capability
in those documentation-only commits; later O1/O2 deliveries are listed above.

The user stopped O1 before integration/checks/Astra review and requested
[current model research](models-under-30b-2026-11-07.md). The ten-model shortlist
distinguishes total from active parameters and capability from local memory fit.
Documentation [checks](../artifacts/models-under-30b-2026-11-07/checks.json) pass;
those research commits claimed no new engine capability or device qualification.
O1 was subsequently completed as recorded above.

The [observability/performance proposal](observability-performance-proposal.md)
was committed as `338c4e5b`. It proposes correlated request summaries, a bounded
overhead budget, explainable selected operations and one measured performance
change at a time. Existing code/evidence and primary ClickHouse sources were
reviewed; [documentation checks](../artifacts/observability-performance-proposal-2026-11-07/checks.txt)
pass. It is a proposal, not an implemented capability or a new speedup claim.

The user then requested archiving earlier material to reduce context. Full
snapshots are preserved in the [archive](archive/2026-11-07/README.md), with
rebased links and [hash manifest](archive/2026-11-07/manifest.json). Live indexes
retain active work, acceptance policy and important limitations. Routine document
bytes fall about 97%; all 32 milestone criteria, four binding policy sections and
both open items are preserved. Integrity/link/content/whitespace
[checks pass](../artifacts/context-archive-2026-11-07/checks.json). This is a
documentation-only change, with no new runtime or performance claim.

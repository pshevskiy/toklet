# Toklet progress checklist

Updated **13 November 2026**. **Improvement goal active; earlier Mac milestones
complete only in their recorded scopes.** iPhone runtime work remains deferred.
Read the current task, relevant plan criteria and targeted source. Completed
history is in the [dated archive](docs/archive/2026-11-12/README.md).

## Current and next task

- [x] S1.L external request observations: scoped publication of all
  four fresh Ollama/vLLM 3+10 cohorts and both frozen CPU baseline cohorts.
  Exact vLLM inputs pass, but only 3/13 output-ID/text rows match CPU per cohort;
  Ollama IDs are unavailable. Source/authentication, ownership/runtime, bounded
  lifecycle and shape-only import controls pass; all 26 genuine vLLM timing
  captures replay consistently. No cross-engine speedup, quality or adoption
  qualification. Content/raw-binding checks pass; Astra approves with no findings.
  [Report](docs/lfm3-engine-comparison.md),
  [checks/review/commit](artifacts/lfm3-engine-comparison-2026-11-11/comparison-publication-checks.json).

- [x] S1.L optional vLLM request timing: offline report binds and replays sealed
  captures, distinguishes queue/TTFT/decode/overall-rate boundaries and preserves
  missing or inconsistent fields. Twenty-four portable controls in both modes,
  five pinned-source cases and nine guarded offline CLI cases pass; helper
  namespace/call failures are retained and corrected. Required checks reuse
  5,177 unchanged inputs; both 52-test CI commands pass. Astra's empty-entry
  and stop-metadata findings are fixed and covered by regressions; follow-up
  approves scoped retention with no findings. Actual runtime/model/performance qualification
  remains separate. All 26 genuine vLLM captures also replay with consistent
  timing arithmetic. Both pushed GitHub Actions jobs pass.
  [Actual replay](artifacts/lfm3-engine-comparison-2026-11-11/actual-vllm-timelines-2/checks.json),
  [remote CI](artifacts/lfm3-engine-comparison-2026-11-11/vllm-timeline-remote-ci-1.json).
  [Scope](docs/lfm3-vllm-timeline.md).
  [Review/commit](artifacts/lfm3-engine-comparison-2026-11-11/vllm-timeline-publication-checks.json).

- [x] S1.L HTTP benchmark observations: bounded Ollama/vLLM stream client,
  exact prompt/cache gates and explicit missing/partial metrics. Twenty-eight
  portable controls in both modes and twelve guarded synthetic HTTP/CLI cases
  pass after correcting a malformed negative fixture. All 26 saved genuine
  Ollama replies replay with unchanged counters/text and original event clocks.
  Required checks reuse 5,177 unchanged input files. Astra's rejected-byte,
  counter-overflow and stop-ID findings are fixed; scoped retention is approved.
  [Review/commit](artifacts/lfm3-engine-comparison-2026-11-11/http-observer-publication-checks.json).
  Actual engine
  startup, quality and performance are separate. [Scope](docs/lfm3-http-metrics.md).

Active goal: implement the whole selected observability/performance/model
roadmap. Order remains Gemma 4 E2B, Qwen3.5-4B, Qwen3.5-9B, then the selected
under-4B track; [model/metric plan](docs/model-observability-plan.md).
LFM3 CPU code and portable measurements are published as `612771d7` on
`codex/console-engine-comparison`.
[Publication / CI](artifacts/lfm3-finite-guard-2026-11-11/finite-publication-checks.json).

Current/next task: **continue Ministral qualification after Granite's reviewed
numerical rejection**. LFM3 CPU qualification and finite-widening retention
pass; managed application adoption remains open. Preserve the 6 GiB ceiling,
original numerical/quality gates and prior model decisions. Engine/CLI work
leads; interactive screen capture and iPhone runtime remain deferred.

- [x] S1.L original and fresh candidate quality: both complete 10,000-target/
  100-task comparisons pass unchanged gates. Stock passes 35 tasks; modified
  reference and native pass 37. All greedy loss IDs and 5,094 sampled task
  decisions/stops/decodes match; maximum NLL delta is 3.66e-15. Guest shutdown
  is verified; failed/interrupted output stays unqualified. Astra approves
  scoped result/retention review; managed adoption remains open.
  [Campaign](docs/lfm3-quality.md),
  [review/commit](artifacts/lfm3-finite-guard-2026-11-11/finite-publication-checks.json).

- [x] S1.L external comparison preflight: authenticated metadata-only views
  preserve numerical configuration and source bytes. Installed AutoConfig and
  MLX-VLM built-in config admission pass; the publisher module's incompatible
  interface is recorded. Eight portable controls and the 512 MiB guarded probe
  pass after correcting a synthetic fixture byte count and bounding actual reads
  against concurrent growth. Astra approves with no findings.
  This preflight does not qualify model startup, inference or performance.
  [Scope](docs/lfm3-external-config.md).

- [x] S1.L finite BF16 widening retained for the CPU route after full public
  validation, with unchanged FMA/reduction/store order. Required fmt/strict
  CPU+Metal Clippy, 734 CPU/880 Metal tests, release/allocation/native/C smokes
  pass. All 854 observations/68,054,016 words, 30 wrapped views and two late
  retries match. Primary ABBA: 29.0%/29.5% median request reduction; complete
  156-request matrix: 20.6–27.9%, no regression, all 12 sustained windows pass.
  Fresh full quality and all 62 state/replay checks pass within 6 GiB.
  Baseline bytes reproduce from clean `2da16a9e`; original build history stays
  unproven. The evaluator fix passes SDK/live, 14 isolation/28 authored cases,
  nine accounting controls and a 260-command stress check. All 441 local portable
  tests pass in both modes; fresh publication fmt/59 changed-workflow tests pass.
  Astra approves with no findings. Failed runs remain retained. No tail,
  observer-overhead, GPU/mobile, allocation-saving or managed-adoption claim.
  [Scope](docs/lfm3-finite-widening.md),
  [208 scalar samples](benchmarks/results/2026-11-12-lfm3-finite-widening-v1.json),
  [complete qualification](artifacts/lfm3-finite-guard-2026-11-11/finite-qualification-closure-1.json),
  [review/commit](artifacts/lfm3-finite-guard-2026-11-11/finite-publication-checks.json).

- [x] S1.G strict Granite configuration: canonical default RoPE and identical
  supported dtype aliases preserve existing scales, shapes and context limits.
  Required fmt/strict CPU+Metal Clippy, 737 CPU/883 Metal tests, the explicit GPU
  fixture and 44 CLI cases pass. Three pinned configs and 936 independent CPU
  arrays are checked; six failed attempts stay retained. The exec-aware private
  monitor passes eight controls in both Python modes and an owned live test;
  historical charges remain conservative. Compiler-only cap is 3 GiB, engine
  ceiling 6 GiB and audit/setup cap 512 MiB. Astra approves, no findings.
  [Scope](docs/granite-config.md),
  [portable checks](benchmarks/results/2026-11-12-granite-config-v1.json),
  [full qualification](artifacts/granite-config-2026-11-12/qualification-1.json),
  [review/commit](artifacts/granite-config-2026-11-12/publication-checks.json).

All four Heretic/stock tensor containers authenticate (14,639,034,320 bytes)
with pinned SHA256/header checks within the 512 MiB setup policy. No model is
loaded and source indexes remain unchanged; stock's understated total still
rejects. Codec, full graph/quality/fit/request gates remain open.
[Source authentication](artifacts/granite-payload-2026-11-12/result-1.json).

- [x] S1.G bounded affine4/8 conversion: full producer and saved-byte audit pass
  for 282 matrices/81 norms. All 3,659,530,240 matrix values reconstruct exactly
  against upstream FP32 during conversion; the offline audit replays actual
  source/child hashes and exact norms without recomputing floating-point values.
  All 48 CPU/GPU controls, 33 portable tests per Python mode and six actual-artifact
  CLI cases pass. Observed conversion charge is 455,131,136 bytes within the
  512 MiB audit cap. Metal metadata accepts contexts 128/512; 1,024 correctly
  rejects at the existing 512-position limit. Scalar FP32 exceeds 6 GiB.
  Earlier memory, index-field and CLI-expectation failures remain retained;
  legacy MiniCPM tools are unchanged. Evidence closure and Astra review pass, no findings.
  Full-model fit, inference, quality, performance and adoption are open.
  [Controls](artifacts/granite-conversion-2026-11-12/controls-closure-2.json),
  [full conversion](artifacts/granite-conversion-2026-11-12/full-conversion-1/manifest.json),
  [saved-byte audit](artifacts/granite-conversion-2026-11-12/full-audit-1.json),
  [CLI cases](artifacts/granite-conversion-2026-11-12/child-inspection-2.json),
  [review/commit](artifacts/granite-conversion-2026-11-12/publication-checks.json).

The conversion is published as `eb14db30`; both exact-head CI workflows pass.
Granite text preparation authenticates both complete tokenizer files. Their BPE,
added-token, normalization and decoder definitions match; only the modified
source's BatchLongest left-padding policy differs. All 100,352 IDs are covered,
including 96 added tokens also present in BPE vocabulary. An initial counting
assumption failed and is retained; the corrected bounded source check passes.
No encoding/template or managed-codec qualification follows.
[Codec inspection](artifacts/granite-codec-2026-11-12/codec-inspection-2.json).

Five cold device/session planning cases pass on Apple M1 Pro. The 128-input/
128-output request plans within 6 GiB; 512/1,024-input and capacity-boundary
requests remain unsupported at the existing 512-position Metal limit. No
numerical buffers are allocated by planning. The actual raw 128/128 CLI smoke
passes: 128 outputs, 255 committed positions and one pending token; observed
charge 2,899,741,912 bytes within 6 GiB. Source/binary/input hashes close unchanged
and owned cleanup is empty. Debug clocks remain diagnostic; this one request
does not qualify full-model numerical/quality, all-workload fit or performance.
[Cold planning](artifacts/granite-runtime-plan-2026-11-12/runtime-plan-1.json),
[native smoke](artifacts/granite-native-smoke-2026-11-12/native-smoke-1.json).

- [x] S1.G full-model graph diagnostic: required fmt/strict CPU+Metal Clippy,
  737 CPU/887 Metal tests, 58 portable tests per mode, 1,120 frozen-fixture
  comparisons, the explicit GPU fixture and admitted checker controls pass.
  All 30 native runs/304 observed state checks pass at a 5.03 GB observed peak;
  schedule/reset/causal-prefix bytes match exactly. The complete comparison
  rejects 212/20,268,711,936 profile coordinates under unchanged gates. The first
  failure localizes to propagated native-input differences; other failures remain
  unlocalized. Current route stays unqualified. Astra's completeness/receipt
  findings are fixed: 34 checker tests per mode and fresh numeric controls pass;
  genuine replay verifies 304 checks/10 resets/40 receipts with byte-identical
  numerical records. Astra follow-up approves scoped tooling/rejection retention,
  no remaining findings. Failed checks remain retained; publication is in progress.
  [Scope](docs/granite-graph.md),
  [scalar decision](benchmarks/results/2026-11-13-granite-graph-v1.json),
  [required checks](artifacts/granite-graph-2026-11-12/required-checks-closure-2.json),
  [native capture](artifacts/granite-graph-2026-11-12/native-capture-1.json),
  [strict replay](artifacts/granite-graph-2026-11-12/strict-replay-closure-2.json),
  [localization](artifacts/granite-graph-2026-11-12/attention-localization-result-1.json),
  [review/commit](artifacts/granite-graph-2026-11-12/publication-checks.json).

Next: finish publication and continue the selected Ministral qualification.
Granite's current affine4/8 Metal route does not advance to adoption or quality/
performance qualification. Any future candidate must pass the original graph,
10,000-target/100-task quality, codec, fit and request/overhead gates. The 6 GiB
ceiling and 512-position Metal cap remain unchanged. Continue the selected
Ministral track after publication.

Subsequent Ministral source inspection records 4,096 query width, YaRN and
absolute-position query scaling at the 16,384-token boundary. Its pinned Q4 index
remains rejected; payload, conversion, numerical/quality and runtime gates remain
open. No model or compiler was launched.
[Source inspection](artifacts/ministral3-source-planning-2026-11-12/static-source-inspection-1.json).

Qwen3.5-2B source inspection records a distinct 24-layer/16-value-head profile
and joint normalization/layout requirements. The earlier preflight's stale Rust
hash is retained as a failed check; current source is inspected and pinned
separately. Parent/payload authentication, numerical/quality and runtime gates
remain open. No model or compiler was launched.
[Source inspection](artifacts/qwen35-2b-source-planning-2026-11-12/static-source-inspection-2.json).

## Completed work index

Full check counts, failed attempts, limitations, reviews and commit references
are preserved in the [reviewed history](docs/archive/2026-11-12/TODO.md). Use the
linked scope reports for individual results; a passed narrow gate never expands
model, context, device or adoption qualification.

| Completed scope | Current decision and evidence |
| --- | --- |
| O1, O2.A, O2.B1, O2.B2.A | Bounded summaries, native phases, repeated numerical workloads and controller publication delivered; summary mode stays opt-in because the 1% overhead bound is unqualified. [O1](docs/o1-gemma-request-summaries.md), [phases](docs/o2-gemma-cost-report.md), [workloads](docs/o2-gemma-workloads.md), [publication](docs/o2-ui-publication.md). |
| O3 and R1 | Scoped runtime explanation/profile and comparable offline regression screening delivered. [O3](docs/o3-gemma-runtime.md), [R1](docs/gemma-regression-job.md). |
| Console engines | Ollama/vLLM installed and 78 requests collected; unmatched weights/precision/cache prohibit engine ratios. [Comparison](docs/console-engine-comparison.md), `34db5778`. |
| Gemma guard and preferred MLX artifact | Exact guard optimization retained. Heretic affine4 candidate rejected on task quality; official BF16 remains selected. [Guard](docs/gemma-guard-validation.md), [artifact](docs/gemma-heretic-mlx.md). |
| Q1.A–I: Qwen3.5-4B mechanisms | Pinned source, bounded owners, scalar/hybrid kernels, codec and generation bridge delivered only in their recorded scopes. [Model plan](docs/model-observability-plan.md). |
| Q1.J: 4B 128/256 | Rejected: 13,505 failures and 3,132 disjoint dual-profile intervals; matching greedy IDs do not qualify adoption. [Decision](artifacts/qwen35-context256-2026-11-09/decision.json), `9d270f3a`. |
| Q2.A–D: separate 9B support | Pinned recipe/store, distinct head and independent one-input oracle delivered without loosening 4B APIs or gates. [Recipe](docs/qwen35-9b-cold.md), [store](docs/qwen35-9b-storage.md), [head](docs/qwen35-untied-head.md), [oracle](docs/qwen35-9b-oracle.md). |
| Q2.E: 9B hybrid Metal | Both one-input captures pass; 64.72×/64.42× applies only to the raw one-input baseline. [Scope](docs/qwen35-9b-metal.md), `3ccce99f`. |
| Q2.F: 9B context17 | Both 33.85M-value gates and prefix/repair/zero-allocation witnesses pass; descriptive release medians 4.093/4.121s. [Scope](docs/qwen35-9b-context17.md), `41287c6c`. |
| Q2.G0: exact reference helper | All 67.17M short-context profile values bitwise match scalar references. Helper gains are separate from engine speed or deeper-context qualification. [Scope](docs/qwen35-reference-vector.md), `88950b11`. |
| LFM3 CPU BF16 | Authenticated graph/CLI, 1,025-position parity, complete original quality and 62 state checks pass; finite widening retained after Astra with 29.0–29.5% primary gains. Managed/GPU adoption remains open. [Scope](docs/lfm3-finite-widening.md). |
| Five under-4B candidates | Metadata research complete; payload, new graphs, numerical/quality/fit/request and adoption gates remain open. [Selected order and evidence](docs/model-observability-plan.md#five-candidates-strictly-below-4b-total-parameters). |

## Remaining observability gates

- [ ] Display-frame qualification **deferred by user, engine/CLI priority**.
  Ready9 reuses checked43source/native46 inputs and completes13exact trials;
  fresh viewport/crop visually inspected, new receipt bound to its own UUID.
  Coverage rejects433/1721future-display callbacks, zero occlusion;32numeric
  samples retained. The owned session is closed; no further sharing requests.
  Earlier7/8failures and isolated UI WIP remain preserved/unreviewed/uncommitted.
  [Deferred status](artifacts/o2-ui-frames-2026-11-07/status-audit-1/current-state.json).
- [ ] Keep first-visible/final-answer and ordinary summary-overhead qualification
  explicit. Controller publication is not rendering; fixed-answer frame matches
  provide upper bounds only. iPhone work remains deferred.

## Working method

- Update current/next work, checkboxes, blockers and check/review outcomes after
  every iteration, including research, failed checks and interrupted work.
- Keep entries concise; link full logs in artifacts and evidence/commit records.
- Reviewed `7524640e` history is archived with byte-exact inverse link checks;
  routine TODO/improvement bytes fall 38.4%. Pending work, original user history,
  live PLAN and numerical inputs are preserved.
  [Archive validation](artifacts/lfm3-finite-guard-2026-11-11/working-history-archive-checks-1.json).
- Code completion requires applicable checks and one combined Astra review of
  diff, invariants and results before commit. Keep [improvements](docs/improvements.md)
  current. Documentation-only work needs link/content and whitespace checks.
- Push reviewed milestone commits to `origin/codex/console-engine-comparison`,
  as authorized on 11 November. Reviewed tip `af100419` publishes the finite
  LFM3 retention metadata; exact-head push and both CI jobs pass. Granite config
  checks and Astra review pass; publication is recorded in its own receipt.
  Original UI/workflow work is preserved.
  [Latest recorded push verification](artifacts/lfm3-finite-guard-2026-11-11/finite-publication-checks.json).
- Read [PLAN.md](PLAN.md) for binding criteria, the
  [design rationale](apple_llm_performance_plan.md) when relevant, and targeted
  source before assuming a feature exists. Read archives only on demand.
- Preserve historical failures and scope limits. Mac observations do not qualify
  iPhone; planning/admission estimates do not establish allocation or execution.

## Current delivery and retained decisions

| Area | Current scope and evidence |
| --- | --- |
| Foundation M0–M7 | Complete in recorded scopes; [plan](PLAN.md), [M7 closure](docs/experiments/2026-10-21-m7-release.md). Later iPhone work remains deferred. |
| M8 shared blocks, model breadth and capacity controls | Selected Mac engine/app scopes delivered; [coverage index](docs/model-coverage.md), [shared-block contract](docs/shared-block-model-plan.md), [historical checklist](docs/archive/2026-11-07/TODO.md#m8--architecture-breadth-and-usable-context-controls). No blanket family/full-window qualification. |
| Managed Mac journeys | Six earlier profile journeys reviewed; [report](docs/mac-hub-journeys.md). Qwen1.5B Chat/native41 complete; [evidence](benchmarks/results/2026-11-06-qwen15-mac-app-v1.json). |
| Official Gemma BF16 Mac | Full 10.28 GB public snapshot, six-file SHA, managed publication, Chat/Stop and network-denied reopen pass; [journey](benchmarks/results/2026-11-07-gemma-public-resume-mac-v1.json). Native45 Settings/context/KV checks pass; [evidence](benchmarks/results/2026-11-06-gemma-bf16-live-window-v1.json). |
| G2 prefill | Retained on official BF16 Mac route; both primary runs reduce prefill about 61% and request time about 28%; [evidence](benchmarks/results/2026-11-06-gemma-bf16-g2-retention-v1.json). |
| M9 affine tile | First delivery complete as pinned Mac opt-in, with full quality, state and repeated timing. Managed adoption remains separate; [decision](docs/m9-mac-affine-decision.md). |
| Memory reporting | Owner snapshots, physical KV ceiling in Preview/Apply and conditional load-process peaks complete; [owner](benchmarks/results/2026-11-02-session-owner-memory-v1.json), [KV](benchmarks/results/2026-11-03-kv-preview-apply-v1.json), [loading](benchmarks/results/2026-11-03-load-memory-checkpoints-v1.json). Planned/admitted, owned and measured memory remain distinct. |

## Open or deferred scope

- [ ] Record separate iPhone Settings runtime evidence.
- [ ] Independent physical iPhone14Pro journeys/Settings; latest connection query times out.

The user deferred this device work on 5 November. The paired iPhone 14 Pro is
unavailable; a different Pro Max is not a substitute.
[Availability evidence](artifacts/m8-phone-availability-2026-11-04/availability.json).

Full-Qwen3-window expansion and unselected research techniques remain deferred.
The [detailed plans](docs/todo-implementation-plans.md) preserve task-specific
acceptance; older queue statements and historical pending notes do not override
later accepted decisions. Use the [archived M8 closure record](docs/archive/2026-11-07/TODO.md#m8-closure)
for individual model/adoption limitations.

Preserve these rejected or restricted results:

- Gemma3 coupled-intermediate rejections remain failed under the recorded
  [conformance policy](docs/gemma3-native-conformance.md); full adoption is distinct.
- Qwen3 historical fidelity rejections remain separate from scoped native
  qualification; [native profile](docs/qwen3-native-profile.md).
- Llama Online and split-KV candidates remain rejected/deferred under unchanged
  gates; [Llama decision](docs/todo-implementation-plans.md#llama-online-correctness),
  [split-KV evidence](benchmarks/results/2026-11-05-m9-split-kv-session-v1.json).
- Gemma QAT full task quality fails the two-point loss limit; passing loss/wrap
  fidelity does not override it; [evidence](benchmarks/results/2026-11-06-gemma-qat-full-quality-v1.json).
- G3 assistant remains test-only after the longer guard regressed about 21%;
  [decision](benchmarks/results/2026-11-06-gemma-assistant-mtp-retention-v1.json).
- Prompt lookup remains an explicit nonstreaming API; default adoption rejected.
  [Plan and evidence](docs/todo-implementation-plans.md#p5-strict-greedy-verifier-and-prompt-lookup).

## Latest iteration

| Date | Result and checks | Evidence / next action |
| --- | --- | --- |
| 2026-11-08 | Exact guard reduction retained:25.0–26.6% primary/26.3% sustained request reduction;512 deeper bitwise pairs and626 sustained requests pass. Required CPU/Metal/GPU/native checks and Astra approval pass;6GiB preserved. | [Scope](docs/gemma-guard-validation.md), [checks/review/commit](artifacts/gemma-guard-validation-2026-11-08/publication-checks.json); next preferred Gemma MLX format/quality evaluation. |
| 2026-11-07 | O2.B2.A bounded UI publication complete: six real Gemma controls, 104 paired requests across UI/headless tracks, fresh strict Swift/app checks and 29 portable tests pass. Astra approves state/count join fix; timing captures recompute unchanged. Both 1% overhead bounds remain unqualified/default-disabled. | [Scope](docs/o2-ui-publication.md), [result](benchmarks/results/2026-11-07-o2-ui-publication-v1.json), [checks/commit](artifacts/o2-ui-publication-2026-11-07/checks.json); visible-frame/O3 remain open. |
| 2026-11-07 | O2.B1 numerical coverage complete: strict Swift/new Python and reused production gates pass; two batches plus 601 seconds sustained, max lifetime peak 4.924 GB. Astra approves. Earlier counter-assumption/interference run is superseded by fresh captures; no quality gate changed. | [Scope](docs/o2-gemma-workloads.md), [result](benchmarks/results/2026-11-07-o2b-gemma-workloads-v1.json), [checks/commit](artifacts/o2b-gemma-workloads-2026-11-07/checks.json); next UI/O3. |
| 2026-11-07 | R1 offline regression screening complete: comparable frozen identity/work, independent cohorts, explicit noisy/failure receipts and named portable CI. Local17 tests/Astra pass; no runtime or optimization qualification. O2.B1 corrected suite running. | [Job](docs/gemma-regression-job.md), [checks/commit](artifacts/r1-gemma-regression-2026-11-07/checks.json); next O2.B1 completion, then UI/O3. |
| 2026-11-07 | O2.A native46 API/schema2 phases and strict matched warm reports complete. Required checks and Astra approval pass after parser clock/enclosure fixes. Decode dominates this workload; overhead remains unqualified/default-disabled. | [Report](docs/o2-gemma-cost-report.md), [receipt](benchmarks/results/2026-11-07-o2a-gemma-phase-report-v1.json), [checks/commit](artifacts/o2-gemma-metrics-2026-11-07/checks.json); next O2.B. |
| 2026-11-07 | O1 Gemma opt-in summaries complete; Cargo, strict Swift/Xcode, real output/state/failure/admission controls pass. Corrected full-boundary overhead remains unqualified; Astra approves after three fixes. Earlier archive mismatch, compiler sandbox/type failures and superseded timings are retained in check notes/logs. | [Pilot](docs/o1-gemma-request-summaries.md), [result](benchmarks/results/2026-11-07-o1-gemma-summary-v1.json), [checks/commit](artifacts/o1-gemma-summary-2026-11-07/checks.json); next O2 within full-roadmap goal. |
| 2026-11-07 | Observability-first plan complete; five exact under-4B counts verified. Preferred uncensored MLX formats/lineage and conversion/memory gaps recorded. Initial metadata DNS/Python certificate failures resolved with system curl; receipts retained. Documentation checks pass; binding policy unchanged. No runtime checks/Astra review claimed. | [Plan](docs/model-observability-plan.md), [checks/commit](artifacts/model-observability-plan-2026-11-07/checks.json); next implementation O1 remains stopped. |
| 2026-11-07 | Researched ten current under-30B candidates from primary sources; capability/local-efficiency shortlist, not common-harness qualification. O1 stopped; unfinished code uncommitted and unchecked. Documentation checks pass. | [Research](docs/models-under-30b-2026-11-07.md), [checks](artifacts/models-under-30b-2026-11-07/checks.json); model selection next. |
| 2026-11-07 | Archived full TODO/PLAN/improvement snapshots at `338c4e5b`; routine document bytes reduced about 97%. All snapshot text, 32 milestone criteria, four binding policy sections and both open items preserved; link/content/whitespace checks pass. Documentation-only. | [Archive manifest](docs/archive/2026-11-07/manifest.json), [checks](artifacts/context-archive-2026-11-07/checks.json); next proposed O1 pilot. |

Older iterations and all completed checklists remain in the
[archived TODO](docs/archive/2026-11-07/TODO.md). Add only a concise recent record
here; archive completed history periodically rather than loading it by default.

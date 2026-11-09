# Toklet progress checklist

Updated **10 November 2026**. **Improvement goal active; earlier Mac milestones
complete only in their recorded scopes.** iPhone runtime work remains deferred.
Read the current task, relevant plan criteria and targeted source. Completed
history is in the [dated archive](docs/archive/2026-11-09/README.md).

## Current and next task

Active goal: implement the whole selected observability/performance/model
roadmap. Order remains Gemma 4 E2B, Qwen3.5-4B, Qwen3.5-9B, then the selected
under-4B track; [model/metric plan](docs/model-observability-plan.md).

Current/next task: **engine/CLI GPU profiling and remaining S1.B MiniCPM5 gates**,
after the reviewed short console evaluation and S1.A source-admission change.
The user deferred screen capture; engine/CLI metrics, performance and automated
checks lead work. The full goal stays active; 4B/9B deeper rejections stay binding.

- [x] S1.B short console vertical: authenticated 2.674 GB Heretic MLX8 object;
  parent headers match 381 BF16 tensors, whole-parent/lineage still open. The
  698 CPU arrays and 52,896,768 dual-profile native comparisons pass with zero
  failures/max gate fraction 0.97221; exact schedules, causal prefix and 512-token
  functional state pass. Fresh fmt/both strict Clippy, 670 CPU/810 Metal tests,
  native64 three slices/46 exports each and CPU/Metal C smokes pass. Portable
  7 numerical/10 capture-reader/11 request-reader tests and 32/41/39 resealed
  negative controls pass. Forty measured requests plus 12 warmups close four
  eligible cohorts: token-only readback is
  0.64–0.70% slower; reject a performance/default change. Memory observations
  stay below 6 GiB in named scopes; final physical peaks remain unavailable.
  Failed invocations, toolchain mismatch, disk admission/floor observation and
  verified cache recoveries are retained. Astra timing-enclosure and balanced
  owner-ledger findings are fixed in consumers; Astra approves scoped retention.
  [Review/commit](artifacts/minicpm5-heretic8-payload-2026-11-10/publication-checks.json).
  Next: attribute the dominant GPU work before one bounded CLI experiment.
  Whole-parent/lineage and original codec/quality/deeper-numerical gates stay open.
  No model adoption, visible-text timing or 1% overhead qualification.
  [Scope and all evidence](docs/minicpm5-console.md).

- [x] S1.A: explicit default-RoPE declarations for the existing Llama graph;
  equivalent aliases preserve kernels/state/budgets and cold CLI reports.
  Required fmt/both strict Clippy, 670 CPU/806 Metal tests, five targeted GPU
  tests, three positive/four negative CLI controls and three native slices with
  46 exports each plus CPU/Metal C smokes pass. Six metadata cohorts retain
  60 measured/18 warm batches, after medians 2.107–2.145 ms/open; no repeated
  ≥5% legacy regression, but unchanged-before-binary drift was 36.6%.
  Astra approves after the timing-envelope fix: 10 portable tests and all six
  unchanged cohorts revalidate. No model execution/codec/quality/fit/adoption
  or optimization qualification. Original source-access/probe failures preserved.
  [Scope](docs/minicpm5-default-rope.md),
  [checks/review/commit](artifacts/minicpm5-default-rope-2026-11-10/publication-checks.json).

- [x] Q2.G9B128/256 evaluation retained as a rejection in `8215d96a`:
  both79.35M-value captures reject45,332 values;14,847 saved coordinates have
  disjoint original profile intervals. Functional/recovery, required667CPU/
  803Metal,190Python and two performance runs pass in their scoped senses.
  First256 cohort is noisy; no optimization/adoption claim. Astra approves
  diagnostic retention; feature-integrated, no push/main change, rootWIP kept.
  [Full scope/failed attempts](docs/qwen35-9b-context256.md),
  [checks/review/commit](artifacts/qwen35-9b-context256-2026-11-09/publication-checks.json).

- [x] Refresh completed working-document history from reviewed `88950b11`;
  exact inverse-link reconstruction, local link/anchor and whitespace checks
  pass. Routine TODO/improvement bytes fall about 56%; live PLAN, 15 reference
  producers, 3,967 Cargo/native inputs and root user work remain unchanged.
  [Archive](docs/archive/2026-11-09/README.md),
  [validation](artifacts/qwen35-9b-context256-2026-11-09/working-history-checks.json).

## Completed work index

Full check counts, failed attempts, limitations, reviews and commit references
are preserved in the [reviewed history](docs/archive/2026-11-09/TODO.md). Use the
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
- Code completion requires applicable checks and one combined Astra review of
  diff, invariants and results before commit. Keep [improvements](docs/improvements.md)
  current. Documentation-only work needs link/content and whitespace checks.
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

# Toklet progress checklist

Updated **8 November 2026**. **Prior Mac milestones complete in recorded scopes; improvement goal active.**
iPhone runtime work remains deferred by user.
Read this file first, then the relevant plan criteria and targeted code/diff.
Completed history is in the [dated archive](docs/archive/2026-11-07/README.md).

## Current and next task

- [x] Research and write the [observability/performance proposal](docs/observability-performance-proposal.md).
  Current code, retained experiments and primary ClickHouse sources inspected;
  documentation checks pass. Commit `338c4e5b`;
  [checks](artifacts/observability-performance-proposal-2026-11-07/checks.txt).
- [x] Archive previous working-document history and replace live records with
  compact current indexes; preserve all evidence and binding acceptance policy.
- [x] Pass archive integrity, link/anchor, content-preservation and whitespace
  checks; [validation record](artifacts/context-archive-2026-11-07/checks.json).

Active goal: implement the whole selected observability/performance/model
roadmap. Order remains Gemma 4 E2B, Qwen3.5-4B, Qwen3.5-9B, then the selected
under-4B track; [model/metric plan](docs/model-observability-plan.md).

Current task: **Qwen3.5-4B independent actual-source fidelity**, then GPU,
codec/quality and the same request metrics before optimization.
The full selected goal stays active.

- [x] G1.Ca independent mixed-source fidelity and parent pinning: all1,094
  consumed components reproduce from authenticated modified BF16 parent;22 full
  rows/5.77M values through512 pass unchanged FP64/staged-FP32 gates with no
  greedy mismatches and exact repeated prefixes.610CPU/742Metal,18Python,
  fmt/strictClippy and native512/state controls pass; peak2.942GB for512capture.
  Astra approves scoped prerequisite with no findings. No full quality/adoption.
  [Scope](docs/gemma-heretic-fidelity.md),
  [checks/review/commit](artifacts/gemma-heretic-quality-2026-11-08/publication-checks.json).
- [x] G1.Cb quality probes and rejection decision complete; Astra approves.
  Fresh611CPU/745Metal,27Python, fmt/strictClippy, nine GPU controls, native52
  three slices/46exports and both Mac C smokes pass. Both1,025fidelity gates
  reject; all16selected greedy IDs match. Both10kNLL gates pass; tasks score
  original52/modified50/quantized43 of100. Quantized quality rejects9pp loss
  versus original and7pp versus modified. Raw replays4,244/4,378decisions match;
  isolated graders restore stopped VM. Public512/6GiB and all gates unchanged.
  Typed aggregate-budget rejection fixed; historical failures/pins preserved.
  [Scope](docs/gemma-heretic-quality-probes.md),
  [checks/review/commit](artifacts/gemma-heretic-quality-probes-2026-11-08/publication-checks.json),
  [quality/rejection](artifacts/gemma-heretic-quality-probes-2026-11-08/quality-summary.json).

- [x] Q1.A cold Qwen3.5-4B recipe: pinned119,829B header,
  249affine matrices/153BF16 vectors/24four-tap convolutions, distinct24delta/
  8KV owner indices and tied embedding/head. Fresh618CPU/750Metal tests,
  fmt/strict CPU+Metal Clippy and two3warm+10metadata probes pass (about2.53ms,
  zero payload reads). Astra approves with no findings. No numerical execution,
  6GiB model fit or managed adoption is claimed.
  [Scope](docs/qwen35-4b-cold.md),
  [checks/review/commit](artifacts/qwen35-4b-cold-2026-11-08/publication-checks.json).

- [x] G1.B authenticated Mac mixed-format diagnostic:277 original affine4
  matrices plus one BF16 projection, immutable owners/session scratch, matched
  byte proof and explicit load/memory/request metrics. Cap512/chunk32; no managed
  adoption. Final fmt/strict CPU+Metal Clippy,610CPU/742Metal (33/294 ignored),
  seven GPU regressions, three native slices/46exports and Mac C smokes pass.
  Paired caller time falls26.2%, eventTTFT61.4–61.5%;512 deeper logit pairs match.
  Sustained322/434requests over600s each confirm26.18%, peaks below2.947GB;
  two-session fault/replay control peak3.104GB. No decode speedup claimed.
  Astra approves scoped implementation/retention with no findings.
  [Scope](docs/gemma-heretic-native.md),
  [checks/review/commit](artifacts/gemma-heretic-runtime-2026-11-08/publication-checks.json).
  Initial import/Clippy failures and premature report probe remain in linked logs;
  [next quality preparation](artifacts/gemma-heretic-runtime-2026-11-08/next-quality-preparation.json).

- [x] G1.A pinned cold mixed-format plan complete:277affine4 matrices and the
  original BF16 PLE projection remain explicit;140unused-KV/1415media exclusions.
  Fresh610CPU/735Metal tests, strict Clippy/fmt and two3warm+10 local header probes
  pass (about6.2ms, zero payload reads). Astra approves structural scope only.
  [Scope](docs/gemma-heretic-mlx.md),
  [checks/review/commit](artifacts/gemma-heretic-mlx-2026-11-08/publication-checks.json).
  Native diagnostic is recorded above; full fidelity/quality and adoption remain.

- [x] Exact guard reduction retained on recorded Mac BF16/G2 cases: two primary
  request reductions25.0–26.6%;512 matched deeper decode-state pairs bit-identical.
  Sustained268/358 requests pass over ten minutes each (26.3% lower median).
  Fresh604CPU/729Metal tests, strict Clippy, six GPU controls, three native slices/
  46exports and Mac C smoke pass. Astra approves; all guard words remain checked,
  with no numerical/shader/allocation change or tolerance relaxation.
  [Scope](docs/gemma-guard-validation.md),
  [checks/review/commit](artifacts/gemma-guard-validation-2026-11-08/publication-checks.json).

- [x] O3 selected BF16/G2 explanation and bounded profile complete: cold/live
  inventory, full-logit/state/fault/overflow controls,604CPU/726Metal tests,
  both strict Clippy configurations and three native slices/46exports pass.
  Final lifetime fix passes512constructor/16,384export repetitions on a plain
  Rust worker with zero observed postwarm growth. Astra approves afterP2/P3 fixes.
  Refreshed104request overhead tracks pass; capture changes−0.28%/+0.98%
  (ordinal2), default None+0.22%/−0.09%. No1%bound; diagnostic remains opt-in.
  [Scope](docs/o3-gemma-runtime.md),
  [checks/review/commit](artifacts/o3-gemma-runtime-2026-11-07/publication-checks.json).
- [x] Console comparison published as`34db5778`; [report](docs/console-engine-comparison.md).
  Externalquantized/nativeBF16 precision/cache/timing differences remain explicit.

Earlier release-cache reuse and an incorrect pending-logit test assumption were
caught, retained and corrected before final checks/measurements. No native model
format or numerical tolerance was relaxed. O3 identified validation/copy cost;
the exact guard reduction above is now retained in its measured scope.

UI lab remains paused: third capture has13 exact trials but25 invalid callbacks;
fourth interrupted before reference verification. It remains unqualified with
capture/Astra gates open; [UI evidence](artifacts/o2-ui-frames-2026-11-07).
The first measured performance change is retained. Continue preferred Gemma
MLX artifact evaluation before Qwen3.5 breadth; iPhone runtime remains deferred.

O2.B2.A bounded Send/preparation/publication correlation is reviewed and delivered
in the existing summary ring; no lifecycle/render conflation.
[Iteration evidence](artifacts/o2-ui-publication-2026-11-07/iteration-notes.json).
- [x] O2.B2.A publication: strict fixtures, six real controller controls,
  52 UI +52 headless paired requests, fresh Swift/Mac/simulator builds and
  29 portable checks pass. Astra approves after typed state/work join fix;
  strict offline recomputation preserves captures. Both overhead bounds remain
  unqualified/default-disabled. [Checks/review/commit](artifacts/o2-ui-publication-2026-11-07/checks.json).
- [ ] Correlate request IDs with UI publication and qualified first-visible/
  final-answer observations; preserve explicit unknowns and bounded retention.
- [ ] Validate a bounded diagnostic display-frame instrument, clock conversion,
  capture coverage and actual Mac behavior; lifecycle callbacks precede rendering.
  [Research](artifacts/o2-ui-observation-2026-11-07/research.md).
- [ ] For the visible-frame instrument run applicable checks, resolve Astra
  review and commit.
  iPhone remains deferred; no render or device qualification from headless runs.

Completed observability stages (full logs/review/commit in linked receipts):
- [x] [O1 opt-in summaries](docs/o1-gemma-request-summaries.md), `4e160566`:
  bounded request outcomes/state/memory; full caller-boundary overhead trials.
- [x] [O2.A native phases/report](docs/o2-gemma-cost-report.md), `f9477e99`:
  additive ABI, typed phases and strict matched warm report; required checks/Astra pass.
- [x] [R1 regression job](docs/gemma-regression-job.md), `de7ee673`:
  comparable independent bundles, noisy/failure receipts and named portable CI.
- [x] [O2.B1 numerical workloads](docs/o2-gemma-workloads.md): two load/continuation
  batches, 854 raw records and 601.060 seconds/554 sustained requests pass. Strict
  Swift, six new tests, local CI 14+9 and Astra pass; 352 native/42 Swift inputs
  match prior required checks. Report-only guards recompute unchanged data.
  [Checks/review/commit](artifacts/o2b-gemma-workloads-2026-11-07/checks.json).

Summary mode stays opt-in/default-disabled: the 1% overhead bound is unqualified.
Existing single-turn decode dominates; no new optimization gain is claimed.
O3 must cover the actual official BF16 tiled projection/CPU graph and file-backed
PLE, not reuse dense graph event ordering.
[Source pointers](artifacts/o2b-gemma-workloads-2026-11-07/o3-scope.txt).
After observability, screen exact-prefix use at the Gemma Chat caller, which
currently resets every request; [hypothesis/constraints](artifacts/o2b-gemma-workloads-2026-11-07/prefix-candidate.json).
Uncensored MLX adoption and larger/new model quality/memory admission remain
separate. Qwen2.5-1.5B remains a regression reference; under-30B research options
do not expand scope or the 6 GiB ceiling.

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

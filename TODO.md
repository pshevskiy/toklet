# Toklet progress checklist

Updated **9 November 2026**. **Prior Mac milestones complete in recorded scopes; improvement goal active.**
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

Current/next task: **Q2.B pinned Qwen3.5-9B CPU payload qualification** after
reviewed Q2.A. The4B managed route remains ineligible under its unchanged gate.
The full selected goal stays active.

- [ ] Q2.B separate9B payload ownership: authenticate the full stream before
  publication, preserve original affine bytes and exact converted small values,
  measure actual capacity/physical loading peak within6GiB. Keep4B APIs/source
  strict; no numerical head binding, GPU/codec/request or managed route in this
  bounded step. Freeze its contract before edits;
  [next scope and inspected gaps](artifacts/qwen35-9b-cold-2026-11-09/next-scope-analysis.json).

- [x] Q2.A pinned9B cold recipe: distinct head/embedding,250affine matrices/
  153BF16 vectors/24convolutions; exact source/shape/dtype/span checks. Strict4B
  APIs unchanged. Required fmt/both strictClippy,652CPU/786Metal(42/320ignored),
  actual4B loads/independent924component/747row checks and corruption rejection
  pass. Native59 three slices/46exports each and Mac CPU/Metal Csmokes pass.
  All14release cohorts retained:9B metadata2.488/2.554ms; initial4B+13% does not
  repeat in four confirming pairs, pooled60/version+2.31%. No1% bound, request
  gain or model-fit claim. Astra approves scoped retention with no findings.
  Finalizer recovery, cache target shortfall and corrected link/stash checkers
  are retained; all3,946Rust/406package inputs and57protected outputs unchanged.
  [Scope](docs/qwen35-9b-cold.md),
  [checks/review/commit](artifacts/qwen35-9b-cold-2026-11-09/publication-checks.json),
  [contract](artifacts/qwen35-9b-cold-2026-11-09/acceptance-contract.json).

- [x] Q1.J full128/256-input evaluation: closed4,204-array/76.98M comparison
  rejects numerical qualification (13,505failures; maxratio34.066). The original
  references have3,132disjoint tolerance intervals; greedy IDs still match.
  Keep512capacity/32chunk/6GiB and the original dual-profile gate unchanged.
  Native replay/recovery/zero-allocation controls pass at2.765GB; reference peak
  810MB. Required fmt/bothClippy,644CPU/778Metal,26Python/24native/30reference
  evidence controls pass; native58 reuse is audited. Prior failed attempts remain
  recorded. The separate prototype matches all38.49M originalFP32 values bitwise;
  8unit/32actual evidence controls pass, with no collector promotion or model-speed claim.
  Persistent-codec singleton stress passes4processes/15controls within512MiB;
  repeated-construction and aggregate fit restrictions remain. Astra approves
  scoped retention with no findings; full requests/overhead/quality/adoption
  do not inherit qualification.
  [Scope](docs/qwen35-context256.md),
  [checks/review/commit](artifacts/qwen35-context256-2026-11-09/publication-checks.json),
  [decision](artifacts/qwen35-context256-2026-11-09/decision.json),
  [codec memory scope](docs/qwen35-text-codec.md),
  [continuation/evidence](artifacts/qwen35-context256-2026-11-09/continuation-state.json).

- [x] Under-4B metadata research: MiniCPM5 header/index cover296packed/85small
  tensors and match the2.517B declaration. Declared BF16-parent API access returns401;
  fullFP32 expansion exceeds6GiB. No payload/execution/adoption or order change.
  [Preflight and open gates](artifacts/minicpm5-preflight-2026-11-09/preflight-summary.json).
  A separate LFM3 BF16 candidate has286tensors/3.458GB and a manifest linked to
  public pinned BF16-parent metadata. Header checks pass; new graph, payload,
  conversion/quality and physical admission remain open; old F32 rejection stays.
  [LFM3 preflight](artifacts/lfm3-preflight-2026-11-09/preflight-summary.json).
  [Granite/Ministral/Qwen2B headers](artifacts/small-model-metadata-2026-11-09/preflight-summary.json)
  expose config/YaRN, stale-index/size and norm-precision gaps;53files/12headers
  pass metadata checks with rejected declarations retained. Runtime gates stay open.

- [x] Authenticate the selected9B source bytes while Q1.J closes: all5.038GB,
  927header tensors and config/index hashes pass an independent read-only check.
  No9B parser/engine/fit/quality or route qualification follows. Verified cache
  cleanup frees3.14GB with retained outputs unchanged;2GiB disk guard holds and
  recorded start/live headroom covers1GiB pending capture/evidence growth.
  [Byte verification](artifacts/qwen35-9b-payload-2026-11-09/verified.json),
  [Source contract](artifacts/qwen35-9b-payload-2026-11-09/contract.json),
  [growth audit](artifacts/qwen35-9b-payload-2026-11-09/growth-audit.json),
  [cache preservation](artifacts/qwen35-context256-2026-11-09/cache-space-reclaimed-9b.json).

- [x] Q1.I bounded generation state bridge: separate pending history, exact whole
  committed-prefix reuse and reset/replay for changed or shortened prompts.
  Detailed hybrid/wrapper memory and last numerical work snapshots; no route or
  sampling-mask change.644CPU/778Metal, fmt/both strictClippy, native58/Csmokes
  and tiny Metal repair/budget controls pass. Actual17/offset37 state/logits/retry
  are exact; peak2.765GB. Initial context-name/unused-helper errors, superseded
  source-changing checks and footer-collector rejection are retained; corrected
  source/collector pass. Both release cohorts pass: generation2.985/2.950s,
  direct2.968/2.961s, peaks2.765GB;8Python/36resealed controls pass.
  Astra approves with no findings; request summaries/overhead and quality remain open.
  [Checks/review/commit](artifacts/qwen35-generation-state-2026-11-09/publication-checks.json),
  [next-scope constraints](artifacts/qwen35-generation-state-2026-11-09/next-scope-analysis.json).
  [Contract](artifacts/qwen35-generation-state-2026-11-09/acceptance-contract.json),
  [scope](docs/qwen35-generation-state.md),
  [corrections](artifacts/qwen35-generation-state-2026-11-09/source-corrections.json).

- [x] Q1.H pinned bounded text codec: two frozen witnesses and native100records/
  165stream schedules pass, including exact whitespace, bothEOS IDs and padding
  rejection. Controlled638CPU/772Metal, fmt/bothClippy and native56/Csmokes pass.
  Default13load baseline rejects512MiB cap at686MB; explicit child-only
  `MallocLargeCache=0` baselines pass at261MB, construction360/358ms.
  InheritedNano0 retry now records/sanitizes only child settings. Limits and
  app/library defaults unchanged; nonblocking/no-follow opens now qualified.
  Astra approves scoped retention with no findings;
  [checks/review/commit](artifacts/qwen35-text-codec-2026-11-08/publication-checks.json).
  Controller/request metrics, deeper contexts, quality and adoption remain next gates;
  [next-scope analysis](artifacts/qwen35-text-codec-2026-11-08/next-scope-analysis.json).
  [Scope](docs/qwen35-text-codec.md),
  [contract](artifacts/qwen35-text-codec-2026-11-08/acceptance-contract.json),
  [investigation](artifacts/qwen35-text-codec-2026-11-08/memory-investigation.json).

- [x] Q1.G static Metal projection/gather with consumed CPU matrix ownership:
  independent one/context17 pass32.135M/32.667M comparisons, maxratio0.447;
  physical peak3.425GB, full state/bit recovery and late-head repair pass.
  CPU631/Metal765, fmt/both strict Clippy, native54/Csmokes and18Python/
  20coupled+6performance evidence controls pass. Reader timeline coverage fixed;
  earlier compile/reader/Clippy corrections retained, parity-fix binary identical.
  Both matched forward cohorts pass:138.03s→3.31s/138.02s→3.10s,97.6–97.8%
  lower; physicalpeak5.307GB. Astra approves scoped retention, no findings.
  [Scope](docs/qwen35-metal-executor.md),
  [checks/review/commit](artifacts/qwen35-metal-executor-2026-11-08/publication-checks.json).
  Next-scope metadata: [4B codec](artifacts/qwen35-codec-preflight-2026-11-08/preflight-summary.json)
  authenticated tokenizer exposes248044/248046EOS disagreement; Q1.H qualifies
  the selected union for bounded text, while controller qualification stays open;
  [9B header](artifacts/qwen35-9b-preflight-2026-11-08/preflight-summary.json)
  has an untied output head and a 5.038 GB container. Neither admits a route.

- [x] Q1.F all-249 FP32 Metal projection diagnostic: both batches pass
  24.846M independent comparisons in each of two owned processes; max ratio
  0.000580, native peak 5.299 GB. Four cohorts: scalar 8.047–8.149 s,
  Metal 0.191–0.202 s; retain isolated operator candidate. CPU 631/Metal 765,
  fmt/both strict Clippy, 20Python/20evidence controls and GPU transactions pass;
  Astra approves after reader-admission P2 fix; no remaining findings. Original native
  receipts are unchanged. Initial zero-test discovery was corrected and retained.
  [Scope](docs/qwen35-gpu-projections.md),
  [checks/review/commit](artifacts/qwen35-gpu-projections-2026-11-08/publication-checks.json).

Combined reviewed G1.Cb/Q1.A–C source passes631CPU/765Metal, strict Clippy/fmt,
native53 three slices/46exports and Mac C smokes; [integration checks](artifacts/model-integration-2026-11-08/checks.json).
The next [one-input independent Qwen oracle contract](artifacts/qwen35-real-oracle-2026-11-08/acceptance-contract.json)
is frozen; source collection and both full-array comparisons pass.
Contextual/GPU/quality gates remain open; Astra approves the scoped oracle.

- [x] Q1.E actual-source17input/offset37 context replay: complete final-stage,
  recurrent/history/KV/logit oracle plus chunk/reset/failure/cancel/retry and
  authoritative completed-work checks.27Python/12negative controls and fresh
  631CPU/765Metal, fmt/both strictClippy pass. Final native bit/recovery replay
  passes;32.667M comparisons pass original gates, maxratio0.447, no greedy
  mismatch. Three actual warmed resets allocate0; reference/native peaks0.668/
  2.655GB. Final counters describe the
  last rejection; retry12 counters are asserted before it. Earlier setup,
  report and value-equality corrections stay scoped. Astra approves with no findings;
  no128/512/GPU/quality/adoption qualification. [Scope](docs/qwen35-context-fidelity.md),
  [checks/review/commit](artifacts/qwen35-context-2026-11-08/publication-checks.json).

- [x] Q1.D independent actual-source oracle: bounded packed rows/unaligned BF16,
  parameterized streaming scalar math and unchanged tiny fixture bytes.
  Nineteen Python/nine damaged-evidence controls pass; closed reference uses0.540GBpeak.
  Both native captures pass32.135M comparisons each, max gate ratio0.239, no
  greedy mismatches.3,923Rust inputs unchanged; required/native checks reused.
  Astra approves with no findings; no contextual/GPU/quality/adoption qualification.
  Initial mock-path and premature-freeze errors are retained; failed freeze was
  never executed. [Scope](docs/qwen35-real-fidelity.md),
  [checks/review/commit](artifacts/qwen35-real-oracle-2026-11-08/publication-checks.json).

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

- [x] Q1.C concrete packed scalar hybrid model/session:32-layer source binding,
  distinct recurrent/history/KV owners, transactional chunk rollback and work/
  memory counters. Frozen ideal/staged fixtures pass3.675Mstage comparisons
  (max gate ratio0.153),107observer failures, chunk/cancel/reset/admission and
  zero-allocation controls. Fresh630CPU/762Metal, strict Clippy/fmt,14Python
  and five damaged-capture controls pass. Two actual32-layer one-input CPU
  captures match1051arrays each, zero warmed allocations, physicalpeak2.638GB.
  Astra approves with no findings; real coupled fidelity/GPU/codec/quality remain open.
  First fixture-number representation assertion and mixed Clippy-toolchain
  preflight failures retained; no tolerance/fixture changes.
  [Scope](docs/qwen35-hybrid.md),
  [checks/review/commit](artifacts/qwen35-hybrid-2026-11-08/publication-checks.json).

- [x] Q1.B bounded authenticated CPU payload owner: preserve249packed matrices,
  decode153BF16 vectors/24four-tap weight tensors, hash the full source before publish
  and expose explicit load/capacity accounting.624CPU/756Metal, fmt/strict
  Clippy, five Python/six damaged-evidence controls and two full source proofs
  pass;2.382GB physical peaks,747rows/2.70Mvalues bit-exact per capture.
  Initial type error and stale model label fixed/retained; Astra approves.
  No hybrid graph, codec, runtime fit or adoption qualification.
  [Scope](docs/qwen35-4b-payload.md),
  [checks/review/commit](artifacts/qwen35-4b-payload-2026-11-08/publication-checks.json).

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

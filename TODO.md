# Toklet progress checklist

Last updated: **28 October 2026**. Active milestone: **M8**.
Next implementation task: **B3 — select a compact Gemma text checkpoint and compose its local/global graph**.
No implementation task is currently blocked on user input.

This is the live progress index for [PLAN.md](PLAN.md), the
[M8 shared-block plan](docs/shared-block-model-plan.md) and
[M9 performance plan](docs/next-performance-milestone.md). Those documents define
acceptance; checked items here require evidence. Broad architecture coverage is
the goal, not a claim that arbitrary Hugging Face models already run.

## Update after every iteration

- Update the affected checkboxes, current task and any blockers before closing
  an iteration, including failed, interrupted or research-only iterations.
- Append a dated log entry with the result, checks, Astra review where relevant,
  commit/evidence links and next action. Preserve unresolved failures.
- Mark a code task complete only after required checks and Astra review pass.
  Commit each completed task and update [the improvement record](docs/improvements.md).
- Keep Mac and iPhone results separate. Pending hardware checks stay pending;
  planning, parsing and memory estimates do not qualify numerical execution.
- Split a task into concrete subitems when its implementation scope is known.
  Record scope changes explicitly; do not silently remove open requirements.

## Completed foundation

- [x] M0–M7 completed in their recorded, qualified scopes; see
  [PLAN.md](PLAN.md) and [M7 closure](docs/experiments/2026-10-21-m7-release.md).
- [x] Replace full-Qwen3-window priority with shared blocks, model breadth and
  context/KV controls; preserve the interrupted 8K experiment as unqualified.
- [x] Review the shared architecture with Astra and schedule performance as M9.
  Commit: `1d673e0`.
- [x] B0: common checked physical KV-owner sizing and pure capacity search.
- [x] B0: reuse constructor equations and resolve fixed Metal layouts once.
- [x] B0: default/Metal checks, three tiny GPU controls and independent Astra
  review. Commit: `ea874d1`; [report](docs/capacity-core.md),
  [evidence](benchmarks/results/2026-10-28-capacity-core-v1.json).

## M8 — Architecture breadth and usable context controls

### B1 — Llama3 scaled RoPE (complete in selected Mac512 engine scope)

- [x] Pin a fitting MLX artifact, revision, config, packing and tokenizer bytes:
  [Llama 3.2 1B inventory](docs/llama3-rotary-core.md). Metadata only; no weights.
- [x] B1a: validated scalar scaling/rotation, independent boundary/high-offset
  fixtures, standard checks and Astra approval; [evidence](benchmarks/results/2026-10-28-llama3-rotary-core-v1.json).
- [x] B1b: strictly parse and propagate a source/model-owned positional plan;
  ordinary constructors preserved, independent tiny graph/session checks and
  explicit pre-GPU rejection pass. [Report](docs/llama3-position-plan.md).
- [x] Define structural block/kernel admission instead of adding a model-name
  or geometry tuple as the sole compatibility rule.
- [x] B1c1: isolated resident Half rotary and frozen dual-gate GPU checks pass
  on Mac; ordinary shader and loader guard remain. [Report](docs/llama3-metal-rotary.md).
- [x] B1c2a: production composite UInt32 index contracts and boundary tests,
  with existing strided projection/gather GPU regression. [Report](docs/metal-index-contracts.md).
- [x] B1c2b: compose the proven Metal operation with exact shared memory accounting,
  structural Llama admission and independent tiny Half graph/session checks.
  [Report](docs/llama3-metal-composition.md); 512-position implementation scope,
  metadata untested until separate real-checkpoint qualification.
- [x] B1d: freeze a bounded plain-text chat/BOS/stop contract and independent
  model/source comparisons; tokenizer metadata alone does not establish Chat.
- [x] B1d artifact: verify the complete pinned 1B payload, metadata/header and
  source identity; preserve bounded-transfer failure and exact-range recovery.
- [x] B1d reference: review/export the fixed dual CPU profile and plain-text
  fixtures under the source/native contract; no app admission yet.
- [x] B1d runner: check the explicit pinned diagnostic and independent complete
  source comparisons, native schedule/prefix/readback/state and raw generation.
- [x] B1d screen: freeze corpus/IDs and run the separate 1,024-target consistency
  screen; retain full adoption quality as a distinct requirement.
- [x] Pass short complete source/native, cache/session, bounded real generation
  and measured memory checks on Mac; obtain Astra review and commit.

B1 closes only the selected native Mac engine-adapter scope. The composed
frequency owner, structural limits and zero-omitted Swift memory fields pass;
[short real-model checks](docs/llama3-real-model-contract.md) and the unchanged
[1,024-target screen](docs/llama3-loss1024.md) pass below 6 GiB. Declared131072,
implemented512, scored257-row blocks and512 state checks remain distinct.
Production Llama Chat, original unquantized10k/100 adoption quality and iPhone
remain explicit B5/M8 requirements. This is not whole-family or full-window
qualification. Proceed to B2 as Astra's acceptance review requires.

### B2 — Fused projection composition

- [x] Select/pin a compact Phi-style candidate and exact metadata/header:
  public Phi-3-mini-4k-instruct-4bit at `5b3819ed6317784fb20eddeae9bed984f778d0d0`.
  Initial selection was metadata only; acquisition is now verified separately.
  Bounded source/native checks now pass; app and full adoption remain separate.
- [x] B2a: checked original QKV/gate-up row spans and source range loading;
  preserve packing, original row order and safe unaligned-header byte handling.
  [Report](docs/fused-row-source.md); audited ranges reject and poison identity.
- [x] B2b1: shared cold logical row mapping and CPU/Metal loading/accounting;
  original physical payloads and audited whole readers preserved.
  [Report](docs/logical-projection-loading.md); no Phi admission yet.
- [x] B2b2: strict Phi3 config/composition and tiny head96 graph checks; retain
  declared sliding2047 and enforce bounded full/sliding equivalence on CPU/Metal.
  LongRoPE variants remain rejected until actually implemented.
  [Report](docs/phi3-dense-composition.md); tiny CPU/Metal checks and Astra review pass.
- [x] Implement checked fused QKV/gate-up mapping using shared projection blocks.
- [x] Validate packed row spans/alignment, an independent tiny graph and actual
  LongRoPE when required; reject unimplemented variants.
- [x] B2c artifact: acquire and independently verify the complete pinned2.15GB
  payload/header/index; acquisition only, no model execution.
- [x] B2c1: reviewed independent lazy real-model reader/graph, frozen source
  probes and tokenizer/template contract under measured Mac process limits.
  [Report](docs/phi3-real-reference.md);349 pairs, exact text replay and independent Astra verification.
- [x] B2c2: short real native comparisons, raw generation/cache state and
  measured memory against the frozen source; keep app/full quality separate.
- [x] Pass short real-model/session/text/memory checks; Astra review and commit.
  [Report](docs/phi3-real-model.md);23,946,432 comparisons, zero failures and measured native fit below6GiB.

B2 closes only the pinned native Mac512 adapter. Production Phi text/Chat,
iPhone, full declared context and adoption quality remain B5/M8 work.

### B3 — Local/global composition

- [ ] Select and inventory a compact Gemma text checkpoint and its exact graph.
- [ ] Compose its norm, residual, activation, rotary and local/global mechanisms,
  reusing existing window/shared state only where semantics match.
- [ ] Validate an independent tiny graph and a meaningful cache-wrap boundary.
- [ ] Resolve selected-checkpoint numerical/quality gaps and pass representative
  generation/resource checks; Astra review and commit before claiming support.

### B4 — Additional concrete mechanisms

- [ ] Choose and document the next fitting model need: routed MoE,
  recurrent/hybrid state or MLA. This selection is pending, not a user blocker.
- [ ] Split the selected mechanism into implementation tasks with state/layout,
  routing/update/rollback fixtures and complete physical-owner accounting.
- [ ] Qualify a fitting real representative and obtain Astra review before a
  family support claim. Add further admissions only as explicitly scoped tasks.

### B5 — Context/KV controls and Hugging Face integration

- [ ] Report declared context, implemented backend range, budget/device fit,
  validation scope and app/text limits separately for each model.
- [ ] Include actual weights, KV owners, scratch, loading peaks and application
  allowances; distinguish numerical fit from whole-process fit.
- [ ] Expose supported context, reply limit, memory budget and advanced chunk/
  KV-byte ceilings in the existing app; show current fixed KV storage honestly.
- [ ] Persist by artifact/plan identity; apply changes atomically with safe
  session recreation/replay and preserve rejected settings/drafts.
- [ ] Check actual rendered prompt plus reply boundaries, stale reports,
  cancellation/recovery and supported search/download/load/offline chat journeys.
- [ ] Finish Mac app checks and record separate iPhone 14 Pro validation or
  explicit remaining device gaps; Astra review and commit each code task.

- [ ] Implement the pinned Llama3 plain renderer/codec in the app path, checking
  single BOS, all three hidden stops, trim/decoding, cancellation and actual
  rendered prompt/reply boundaries. Independent fixtures alone do not enable Chat.

### M8 closure

- [ ] Reconcile completed adapters and exact supported configurations with the
  capability report, app discovery and documentation.
- [ ] Complete required full adoption quality for newly adopted artifacts,
  including Llama3's original unquantized 10,000-target / 100-task gates. The
  1,024-target same-artifact screen is not a substitute or app qualification.
- [ ] Publish supported context/KV settings, evidence and remaining limitations;
  make explicit decisions on open/conditional admissions before milestone closure.
- [ ] Confirm required checks/reviews/commits and update this checklist and
  the improvement record. Do not infer support from a family name alone.

## M9 — Next performance milestone

First delivery: **P0 → one selected P1 candidate → P2**.

- [ ] P0: rebaseline the pinned Qwen2.5 MLX8 workload on the final M8 revision;
  collect bounded, correctly labelled profiles and uninstrumented timings.
- [ ] P0: rank normalization/projection costs and freeze a primary metric and
  comparison contract. Record Mac and iPhone evidence independently.
- [ ] P1: implement one useful candidate: parallel Compensated RMSNorm or a
  four-token native MLX prefill tile; retain the qualified fallback.
- [ ] P1: pass targeted numerical/state and usefulness screens with Astra review.
- [ ] P2: pass unchanged quality, state, memory and app/device gates; collect
  independent performance confirmation and adopt or document rejection.
- [ ] Publish the M9 decision and update the checklist/improvement record.

Conditional follow-ons require their own decision; they do not all block M9:

- [ ] P3: split-KV attention after measured demand and qualified Online support.
- [ ] R1: rotary reuse if profiling justifies its cost and storage.
- [ ] P4: integrated I8 KV for a defined capacity requirement, with an actual codec.
- [ ] P5: strict-greedy transactional verifier, then measured prompt lookup.
- [ ] G2/G3: checkpoint-defined shared-KV prefill/MTP after selected Gemma support.
- [ ] Separate correctness task: reproduce and resolve frozen Llama Online failure.

Full-Qwen3-window expansion, old streaming prototypes and unselected techniques
remain deferred, with historical evidence preserved. They are not silently
completed or prerequisites for the active breadth-first queue.

## Iteration log

| Date | Iteration | Result, checks and review | Commit / next action |
| --- | --- | --- | --- |
| 2026-10-28 | Active plan and M9 architecture | Astra approved shared-block and performance boundaries; local links/whitespace passed. | `1d673e0`; B0 core. |
| 2026-10-28 | B0 capacity core | 375 default / 254 Metal-library tests (overlap), fmt, strict Clippy and three tiny GPU checks passed; independent Astra approval. | `ea874d1`; B1 artifact and positional-block research. |
| 2026-10-28 | Progress tracking | Created this live checklist and persistent iteration-update instruction; documentation link/content and whitespace checks. | `4602baf`; continue B1 with Astra advice/review. |
| 2026-10-28 | B1 candidate research | Astra pinned public Llama 3.2 1B metadata/tokenizer/header; all 13 source pins, Git blobs and header extents verified. No tensor payload or model run. | [Inventory](docs/llama3-rotary-core.md); bounded positional work first. |
| 2026-10-28 | B1a positional core | Six targeted / 381 default / 260 Metal-library tests (overlap), fmt, strict Clippy and deterministic fixture replay pass; independent Astra approval. | `c5b1a1d`; B1b positional-plan propagation. |
| 2026-10-28 | B1b initial check | Tiny graph/session tests passed; lint test compilation found a synthetic Source initializer missing the new field. Fixed its Ordinary default; failed receipt preserved. | [Evidence](benchmarks/results/2026-10-28-llama3-position-plan-v1.json); complete rerun. |
| 2026-10-28 | B1b positional propagation | Independent four-graph fixture/replay, 383 default / 260 Metal-library / targeted 9/10 Llama tests (overlap), fmt and strict Clippy pass; Astra approved. | `429e400`; B1c scaled Metal and structural admission. |
| 2026-10-28 | B1c1 numerical policy | Astra froze dual per-element gates and optional-operation ownership before GPU outcomes; fixture export/replay exact. | [Policy and results](docs/llama3-metal-rotary.md); isolated collection only. |
| 2026-10-28 | B1c1 check launcher | Private driver rewrite raised NameError before child launch; retained and corrected with exact replacement and AST preflight. No numerical outcome or gate change. | [Evidence](benchmarks/results/2026-10-28-llama3-metal-rotary-v1.json); full successful check driver v2. |
| 2026-10-28 | B1c1 isolated Metal rotary | 383 default / 261 Metal-library tests (overlap), strict checks and both explicit GPU tests pass; 17,408 scaled and 13,824 ordinary comparisons. Astra final approval. | `f5ab3be`; B1c2 structural model integration. |
| 2026-10-28 | B1c2a index contracts | UInt32/strided and synthetic planner boundaries, 383 default / 263 Metal-library tests (overlap), strict checks and existing strided MLX GPU regression pass; Astra approval. | `1a4621b`; B1c2b model residency/composition. |
| 2026-10-28 | B1c2b integration decisions | Astra resolved separate positional accounting, optional Swift decoder fields, allocation-free inspection and the unchanged tiny Half graph policy. Design only; model integration remains unchecked. | Commit containing this entry; implement the explicit residency contract before wiring the model. |
| 2026-10-28 | B1c2b reference export | Independent Half graph frozen and replay-identical; four portable controls pass. Astra verified all 312 stage gates. Initial controls wrapper used relative log paths and failed formatting after successful cleanup; absolute-path export rerun passes. | [Composition scope](docs/llama3-metal-composition.md); target outcomes pending. |
| 2026-10-28 | B1c2b preflight corrections | First Clippy run rejected manual saturating arithmetic; spelling corrected. Astra found a theta-below-1 native angle/FTZ admission gap; explicit native domain restricted before GPU results. Tiny source/CPU semantics remain broader. | Final cold checks and prospective review pending. |
| 2026-10-28 | B1c2b shared Swift checks | Legacy/scaled positional wire checks, current CPU and Metal resource-worker checks and unsigned iOS typecheck pass. Initial private wrapper omitted existing NativeContext.swift dependency; corrected source list rerun passes. No physical device run. | [Composition scope](docs/llama3-metal-composition.md); complete numerical collection next. |
| 2026-10-28 | B1c2b Metal composition | 384 default / 265 Metal-library tests, strict checks, current CPU/Metal Swift workers and unsigned iOS typecheck pass. Seven explicit GPU checks pass; 1,680 scaled snapshots / 110,208 values, exact numeric schedule/prefix/resume and accounting controls. Astra verified final evidence; reviewed task complete. | [Report](docs/llama3-metal-composition.md); B1d real source/text qualification next. |
| 2026-10-28 | B1d artifact and contract | Astra reviewed bounded acquisition. Initial 600s transfer timed out near completion; separate exact 206 range plus full SHA/header/index verification passes. No model execution or app admission. | [Contract](docs/llama3-real-model-contract.md); independent references and runner checks in progress. |
| 2026-10-28 | B1d reference controls | Seven synthetic reader/graph and two renderer tests pass. Initial cleanup test inspected a deliberately cleared owner; retained its mock before cleanup and checked both cleared fields. Producer arithmetic and gates unchanged. | Source-only export follows the reviewed runtime/controller contract. |
| 2026-10-28 | B1d frozen source export | CPU dual-profile 349-array export, independent saved-bit verification and exact text replay pass; four owned groups close. Observed source peak RSS 2,875,883,520 bytes and lifetime physical peak 1,210,075,872 bytes, below 6 GiB. Native comparison still pending. | [Contract](docs/llama3-real-model-contract.md); final prospective native review. |
| 2026-10-28 | B1d short native vertical | Fixed source/consumer policy; 28,791,552 value comparisons pass with zero failures, exact logit schedules/causal prefix/readback/state and actual 512-token boundary pass. Native observed physical lifetime peak 1,587,398,264 bytes. Astra independently rescored all values and verified final evidence; reviewed task complete. | [Results](benchmarks/results/2026-10-28-llama3-real-short-vertical-v1.json); 1,024-target screen, scoped B1 closure and B2 next; text/app and adoption quality remain open. |
| 2026-10-28 | B1 execution-order clarification | Astra checked the actual B1 acceptance row: after the fixed consistency screen and scoped review, move to B2 architecture work. Production Llama Chat and full adoption quality stay explicit in B5/M8 closure; no gate waiver. | Screen remains next; B1 is not yet closed. |
| 2026-10-28 | B1d consistency screen | Fixed 1,024-target paired screen passes for chunks32/128: native mean2.9947039895, paired delta−0.0000441146691 ≤0.05; exact numeric row hashes match. 384 default /265 Metal-library /15 example and nine Python controls pass. Astra independently verified final statistics and evidence; scoped B1 closure approved. | [Report](docs/llama3-loss1024.md); close selected Mac512 B1 and move B2 after review. |
| 2026-10-28 | B2 candidate research | Astra pinned a fitting public Phi3 four-bit metadata/header with fused QKV/gate-up, ordinary head96 RoPE, sliding2047 constraint and text/padded-vocabulary limits. No payload or model run. | Private pinned research; B2a is next, preserving all scope limits. |
| 2026-10-28 | B2a row-span source core | 397 default /278 Metal-library tests (overlap), targeted row18/range4/Metal Llama9, fmt and strict Clippy pass. Exact source spans, unaligned reads, discard/retry and sticky audit controls pass; Astra approved final source and evidence. No Phi model run. | [Report](docs/fused-row-source.md); B2b shared cold mapping and tiny graph next. |
| 2026-10-28 | B2b1 logical projection loading | 403 default /285 Metal-library /8 logical tests (overlap), strict checks and existing tiny tied/untied Metal graph pass. Initial sandbox GPU access failed; hardware-only retry with identical source passes. Astra approved final source/evidence. | [Report](docs/logical-projection-loading.md); B2b2 strict Phi/window and independent tiny graph next. |
| 2026-10-28 | B2c artifact acquisition | Bounded public transfer, full SHA/header/index/455 contiguous tensor extents and endpoint checks pass; Astra independently rehashed the entire payload. No model execution or app admission. | [Contract](docs/phi3-fused-source.md); B2b2 remains next before real qualification. |
| 2026-10-28 | B2b2 frozen tiny reference | Strict parser tests and Metal compilation pass. Five independent head96 source cases export/replay identically; Astra verifies195 stage groups/585 finite arrays and fixed gates, original fused rows and unaligned storage. Six fixture files promoted exactly. No target numerical or GPU result yet. | [Source contract](docs/phi3-fused-source.md); CPU/native consumer checks remain in progress. |
| 2026-10-28 | B2b2 initial CPU check | New test failed compilation on a slice/reference equality assertion before graph execution. Added the required borrow; engine and frozen gates unchanged. Failed receipt retained; fresh v2 targeted run in progress. | Finish CPU and native checks; no qualification upgrade. |
| 2026-10-28 | B2b2 CPU policy spelling control | Three metadata/window/accounting tests pass; the graph test initially stops on arbitrary-precision JSON exponent spelling. Compare authenticated policy numerically; fixed gate unchanged. Failed v2 receipt retained. | Fresh v3 run follows. |
| 2026-10-28 | B2b2 CPU vertical | All four targeted tests pass:195 independent stages/513,704 values, exact cached schedules/causal prefix/reset, unique source/scalar peak and window/native-domain controls. | GPU consumers remain under prospective review; no real model execution yet. |
| 2026-10-28 | B2b2 cold checks and tiny GPU | 410 default /288 Metal-library tests, strict checks and two frozen-binary GPU tests pass. 19 schedules/3,815 snapshots/1,996,224 main values pass unchanged policy; wide diagnostics exceedances0. State/readback/budget controls pass. Real metadata rejects15.28GB FP32 expansion; real execution remains pending. | Final Astra review and task commit pending; B2c real protocol follows. |
| 2026-10-28 | B2b2 reviewed completion | Astra approves final source/evidence, frozen policy and tiny composition. 17,984 retained logits independently rescored; full intermediate coverage remains checked in the Rust consumer. Real Phi execution, app/phone, full-window and adoption quality stay open. | [Report](docs/phi3-dense-composition.md); reviewed task commit, then B2c. |
| 2026-10-28 | B2c protocol and portable controls | Astra approves fixed short/native-profile protocol and4.49GB source ledger. Ten synthetic reader/ownership/corruption and five plain-text guard tests pass; independent reviews completed. No real graph/text export yet. | Freeze actual runtime/controller before source-only export; no native outcome. |
| 2026-10-28 | B2c pre-export whitespace | Staged preflight caught a new producer blank line at EOF before real outcomes. Immutable proposal v2 removes only that LF; ten controls rerun pass. Old B1 producers and all numerical gates remain unchanged. | Source/controller identities renewed before real export. |
| 2026-10-28 | B2c1 real source/text reference |349 main/wide pairs and full text replay pass; five closed groups, OS peak RSS2,846,588,928 bytes below6GiB. Astra independently authenticates all saved values/envelopes and evidence. Native execution remains pending. | [Report](docs/phi3-real-reference.md); reviewed source task commit, then B2c2. |
| 2026-10-28 | B2c2 native preparation | Fixed capture/checker and real exact-budget controls compile; fmt, strict default/Metal Clippy,410 default /288 Metal-library /3 runner /4 checker controls pass. Astra prospective review corrected copied IDs and evidence retention before target results. | Native binary/controller freeze; actual Metal comparison pending. |
| 2026-10-28 | B2c2 real native vertical |23,946,432 main comparisons pass unchanged gates; exact schedules/causal prefix/readbacks/raw generation/state and actual load/session budgets pass. Observed native lifetime physical peak2,517,715,224 bytes below6GiB; Astra independently rescores the complete capture and approves retention. | [Report](docs/phi3-real-model.md); reviewed task commit, then B3. |

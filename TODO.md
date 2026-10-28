# Toklet progress checklist

Last updated: **28 October 2026**. Active milestone: **M8**.
Next implementation task: **B1b — propagate the positional plan into models**.
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

### B1 — Llama3 scaled RoPE (in progress)

- [x] Pin a fitting MLX artifact, revision, config, packing and tokenizer bytes:
  [Llama 3.2 1B inventory](docs/llama3-rotary-core.md). Metadata only; no weights.
- [x] B1a: validated scalar scaling/rotation, independent boundary/high-offset
  fixtures, standard checks and Astra approval; [evidence](benchmarks/results/2026-10-28-llama3-rotary-core-v1.json).
- [ ] B1b: strictly parse and propagate a source/model-owned positional plan;
  preserve ordinary constructors and reject unsupported/conflicting scaling fields.
- [ ] Define structural block/kernel admission instead of adding a model-name
  or geometry tuple as the sole compatibility rule.
- [ ] B1c: implement scaled Metal rotation with explicit arithmetic/resource
  rules and consistent absolute positions; preserve existing ordinary outputs.
- [ ] B1d: freeze a bounded plain-text chat/BOS/stop contract and independent
  model/source comparisons; tokenizer metadata alone does not establish Chat.
- [ ] Pass short complete source/native, cache/session, bounded real generation
  and measured memory checks on Mac; obtain Astra review and commit.

### B2 — Fused projection composition

- [ ] Select and pin a compact Phi-style model with explicit positional rules.
- [ ] Implement checked fused QKV/gate-up mapping using shared projection blocks.
- [ ] Validate packed row spans/alignment, an independent tiny graph and actual
  LongRoPE when required; reject unimplemented variants.
- [ ] Pass short real-model/session/text/memory checks; Astra review and commit.

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

### M8 closure

- [ ] Reconcile completed adapters and exact supported configurations with the
  capability report, app discovery and documentation.
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
| 2026-10-28 | B1a positional core | Six targeted / 381 default / 260 Metal-library tests (overlap), fmt, strict Clippy and deterministic fixture replay pass; independent Astra approval. | Commit containing this entry; B1b positional-plan propagation next. |

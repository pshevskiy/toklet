# Toklet progress checklist

Last updated: **29 October 2026**. Active milestone: **M8**.
Next implementation task: **B4d4b — independent tiny Half mixed-graph source**.
No implementation task is currently blocked on user input.

Current comparison policy: [Gemma3 native conformance](docs/gemma3-native-conformance.md).
Both historical coupled-intermediate rejections remain failed; all selected
values retain a fatal semantic or coupled check. Full adoption quality stays open.

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

### B3 — Local/global composition (complete in selected Mac544 native scope)

- [x] Select and inventory public Gemma3 270M four-bit text at
  `ff1143e3a10547c9f2129e94ca37059b096b23f4`; independent Astra metadata/source
  review passes. Full payload/tokenizer acquisition is verified separately below.
- [x] B3a: strict cold plan, raw BF16/tied ownership, exact236 logical/490
  physical tensors and shared Gemma3/Gemma4 metadata helpers; checks and Astra review pass.
  [Design and scope](docs/gemma3-local-global.md).
- [x] B3b1: independent tiny CPU graph/operator fixtures, frozen gate and
  byte-identical replay;192 arrays/108,816 values and Astra review pass.
  [Reference](docs/gemma3-cpu-reference.md).
- [x] B3b2: actual CPU composition
  for four hidden norms, Q/K norm, embedding scale, GELU, residual order and
  both rotary/mask branches. Include meaningful window visibility and transactions.
- [x] B3c1: optional Half RMS with model-owned FP32 effective scales; frozen
  independent main/wide/rounding fixtures, GPU guards/status and precise encoding.
  [Report](docs/gemma3-half-rms.md); standalone release Mac qualification only.
- [x] B3c2: checked Half tanh-GELU and embedding scaling with explicit store boundaries.
  [Report](docs/gemma3-half-pointwise.md); standalone release Mac qualification only.
- [x] B3c3: resident old-ring/new-chunk attention and two-phase cache commit;
  uncertain commit failure requires reset/replay.
- [x] B3c3a: shared row-span/copy descriptors used by CPU cache views and commit;
  independent small-ring and multihead chunk-partition checks.
  [Report](docs/window-row-geometry.md); no GPU execution claim.
- [x] B3c3b: frozen independent Half old/new attention fixtures and optional
  native reference operation, including seeded512→513→514 and chunk-wrap cases.
- [x] B3c3b1: reviewed independent reference, exact export/replay and frozen
  local/full numerical and expected-copy arrays. [Report](docs/gemma3-staged-attention-reference.md).
- [x] B3c3b2: checked borrowed native reference attention and complete saved
  source comparisons; keep expected-copy obligations separate until B3c3c.
  [Report](docs/gemma3-staged-attention.md); no native cache transaction/model claim.
- [x] B3c3c: resident all-owner two-phase copy/publication and poison/reset/replay
  checks; no in-place overwrite during attention.
  [Report](docs/gemma3-resident-cache.md); standalone cache/state qualification only.
- [x] B3c4: compose the complete resident packed/Half tiny graph, accounting
  local/full staging/commit; source comparisons and native state/readback checks.
- [x] B3c4a: shared checked original-byte matrix upload, preserving existing
  audited Source loading and packed projection/gather behavior.
  [Report](docs/original-matrix-upload.md); seven release hardware checks pass.
- [x] B3c4b: independent Half graph and D32/256 ordinary-RoPE source fixtures,
  with fixed stage inventory, store boundaries and unchanged gates.
  [Report](docs/gemma3-half-reference.md); source-only exact export/replay and independent reconstruction pass.
- [x] B3c4c: concrete resident Model/Session, consumed optional operations,
  full tiny stage/state/failure/allocation/accounting checks.
  [Report](docs/gemma3-resident-graph.md); independent saved-byte audit and all three release GPU jobs pass.
- [x] B3d: verified real payload/text and fixed short/513-plus-window/resource
  journey on Mac; selected544 scope passes its frozen native conformance protocol.
- [x] B3d1: acquire/authenticate original selected payload and tokenizer files,
  source identity and complete layout; no numerical/text admission.
  [Report](docs/gemma3-artifact-acquisition.md).
- [x] B3d2: freeze independent selected real Half arithmetic/text/source protocol
  and selected544-position probes within the Mac process budget.
  [Report](docs/gemma3-real-reference.md); exact export/replay and independent saved-source audit pass.
- [x] B3d3: checked original-byte real loader/native runner, source comparisons,
  two-partition window journey and bounded text/resource/state controls.
- [x] B3d3a: original-byte CPU store, explicit raw-gamma preparation and resident
  upload lifetime; actual original-range/hash checks pass.
  [Report](docs/gemma3-original-loader.md).
- [x] B3d3b: bounded real native runner, frozen conformance comparisons and
  two-partition544 window/text/state/resource journey.
  [Report](docs/gemma3-real-native.md); historical trajectory screens remain rejected.
- [x] Investigate both coupled-intermediate rejections without changing the
  engine: all84 source3 failures map to42 paired values matching own-input
  independent checks; strict final/teacher outputs pass. Astra reviews the proof.
- [x] Review/commit the [universal native conformance policy](docs/gemma3-native-conformance.md)
  before checker implementation; preserve both historical failed screens.
- [x] Implement/review the generic actual-input checker, analytic corruption
  controls and bounded resources; no coordinate whitelist or unsupported skip.
  [Checked diagnostic](docs/gemma3-conformance-checker.md); subsequent fresh result above.
- [x] Complete posthoc diagnostics, then a separately frozen fresh capture,
  independent full checking and Astra review before closing B3d3b.
- [x] B3d3 prerequisite: qualify the source-supported FP16 clamped residual at
  both Gemma3 branch additions, preserving generic residual behavior and all
  finite/numerical gates; retain the failed strict-Half real-source evidence.
  [Report](docs/gemma3-half-residual.md); independent primitive/tiny raw audits pass.
- [x] Resolve selected bounded native conformance gaps and obtain Astra review
  for the selected engine adapter. Original adoption quality/app/Chat/phone and
  full-window work remain explicitly outside this B3 closure in B5/M8.

B3 closes the selected Mac544 arithmetic/state/text-codec engine contract only.
All18,537,288 fatal-route values pass; original92/84 trajectory rejections remain.
It does not admit ordinary app/Chat or close original-unquantized10k/100 quality.

### B4 — Additional concrete mechanisms

- [x] Choose and document the next fitting mechanism: LFM2-350M's two-row
  short-convolution history composed with full attention. Astra checks cached
  metadata/source and planning fit; [plan](docs/lfm2-hybrid-plan.md).
- [x] Split into strict adapter/source plan, scalar history, tiny hybrid graph,
  resident block/composition and bounded real qualification; preserve unknown
  historical converter identity and separate text/adoption-quality requirements.
- [x] B4a: authenticate the declared same-artifact loader/norm profile and strict
  config/header plan before a real source reference; verify payload separately.
  [Cold plan report](docs/lfm2-cold-plan.md); full default/Metal checks pass, no runtime admission.
- [x] B4b: independent scalar short-convolution and transactional state controls.
  [History report](docs/short-conv3-history.md); unchanged scalar leaf, six state
  controls, zero warmed Rust allocations and full default/Metal checks pass.
- [x] B4c: independent complete tiny mixed attention/convolution graph.
  - [x] B4c1: freeze independent original-packed tiny scalar reference and state fixtures.
    [Reference report](docs/lfm2-scalar-reference.md); eight controls, exact
    export/replay and independent complete saved-array/state audit pass.
  - [x] B4c2: compose shared blocks with whole-step KV/history/token publication.
    [Scalar hybrid](docs/lfm2-scalar-hybrid.md); frozen source/state, rollback,
    partitions, warmed allocations and full450/531 default/Metal checks pass.
- [ ] B4d: resident block/composition, allocation/ownership and recovery checks.
  - [x] B4d1: [direct-gain Half RMS](docs/lfm2-half-rms.md) at epsilon1e-5,
    head64 up to512 rows and hidden256/1024; source/release Mac/legacy checks pass.
  - [x] B4d2: [borrowed three-tap Half convolution](docs/causal-conv3-half.md)
    with immutable old history and exact separate candidates; full source/GPU checks.
    Model-wide transaction publication remains B4d4.
  - [x] B4d3: [multi-KV staged attention/cache](docs/staged-gqa-half.md) views,
    checked physical accounting and all-query/source/state/legacy Mac qualification.
  - [ ] B4d4: resident tiny mixed graph, whole-step status/publication/recovery.
    - [x] B4d4a: [strided B*x/C*z products](docs/strided-half-product.md);
      independent source, borrowed native wrapper and Mac release checks pass.
    - [ ] B4d4b: independent tiny Half mixed-graph source with accurate
      original-FP32 affine rounding and complete state/stage fixtures.
    - [ ] B4d4c: concrete resident graph and all-owner publication/recovery.
- [ ] B4e: acquire and qualify the fitting representative on Mac; numerical,
  text/session and actual process-memory checks before any support claim.
  - [x] B4e1: authenticate complete original weights and all eight cached metadata
    files; [acquisition](docs/lfm2-artifact-acquisition.md). No decoder admission.
  - [ ] B4e2: freeze bounded independent real source/text/reference protocol.
  - [ ] B4e3: pass bounded native real-model/session/text/memory qualification.
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
| 2026-10-28 | B2c1 real source/text reference |349 main/wide pairs and full text replay pass; five closed groups, OS peak RSS2,846,588,928 bytes below6GiB. Astra independently authenticates all saved values/envelopes and evidence. Native execution remains pending. | [Report](docs/phi3-real-reference.md); `d592ac7`; continue B2c2. |
| 2026-10-28 | B2c2 native preparation | Fixed capture/checker and real exact-budget controls compile; fmt, strict default/Metal Clippy,410 default /288 Metal-library /3 runner /4 checker controls pass. Astra prospective review corrected copied IDs and evidence retention before target results. | Native binary/controller freeze; actual Metal comparison pending. |
| 2026-10-28 | B2c2 real native vertical |23,946,432 main comparisons pass unchanged gates; exact schedules/causal prefix/readbacks/raw generation/state and actual load/session budgets pass. Observed native lifetime physical peak2,517,715,224 bytes below6GiB; Astra independently rescores the complete capture and approves retention. | [Report](docs/phi3-real-model.md); `b68b399`; continue B3. |
| 2026-10-28 | B3 selection and cold plan | Astra pins/reviews public Gemma3 270M metadata and source; no payload or execution. Strict236/490 cold inventory and shared Gemma metadata implementation are in checks; runtime eligibility unchanged. | [Design](docs/gemma3-local-global.md); finish B3a checks/review, then CPU composition. |
| 2026-10-28 | B3a initial metadata check | Full tests found the candidate index also has exact total_parameters268098176, absent from the older Gemma4 index schema. Strict rejection caught the omitted known field; v2 validates it and adds changed/missing-count controls. Failed v1 retained, numerical policy/source fixtures unchanged. | Full v2 rerun in progress. |
| 2026-10-28 | B3a cold checks | Strict v2 schema passes415 default /293 Metal-library /5 Gemma3 /14 Gemma4 regression tests (overlap), fmt and strict Clippy. Seven groups close; no payload/model/GPU work. | [Evidence](benchmarks/results/2026-10-28-gemma3-cold-plan-v1.json); `bfd8d4a`; continue B3b. |
| 2026-10-28 | B3b1 independent CPU fixtures | Six portable controls,192 source arrays/108,816 finite values and exact six-file replay pass. Astra verifies source packing/causality/components and independently rescores1,856 tied-head logits. No Rust target result yet. | [Reference](docs/gemma3-cpu-reference.md); `b7f01a6`; continue B3b2. |
| 2026-10-28 | B3b2 initial compile | Formatting passes; strict Clippy catches unused default-build Metal compatibility import and two staging-size error conversions before numerical tests. Explicit feature guard and hoisted typed conversion fix both; failed receipt preserved. | Corrected v3 source, fresh CPU-check v2 in progress; frozen fixtures/gates unchanged. |
| 2026-10-28 | B3b2 concrete CPU graph | All428,832 frozen comparisons/2,400 observations pass; exact schedules, causal prefix and ordinary/TRACE agree. Two-/18-layer fault/retry/reset, budget-minus-one and D256/W512CPU-ring controls pass with zero measured steady-state allocations.420 default /298 Metal-library and regressions pass. | [Report](docs/gemma3-cpu-composition.md); `cdf5f7a`; continue B3c. |
| 2026-10-28 | B3c1 Half/F32-scale RMS preparation | Astra selects the missing operator; four independent portable controls pass with exact0x3fa7 vs0x3fa6 scale witness. Private borrowed-encode kernel/plan draft; default Metal paths unchanged. No export/GPU outcome yet. | Guarded source export/replay, then reviewed GPU consumer. |
| 2026-10-29 | B3c1 cold checks and binary selection | Exact reference export/replay and Astra verification pass. Formatting, strict Clippy,420 default /300 Metal-library tests and regressions pass. Collector then rejects Cargo's rlib/staticlib artifact kind; preserved failed receipt and authenticated metadata-only continuation freeze the existing binary. No source/gate change or GPU result. | Three explicit reviewed GPU checks next. |
| 2026-10-29 | B3c1 debug allocation rejection | First GPU case retains64 exact reference values and intact guards/status, but fails unchanged zero-Rust-allocation assertions at5/5. Astra source inspection predicts debug-only Objective-C signature parsing; failure preserved. Same source/gates compile in release and both cold controls pass. | Frozen release GPU run pending; allocation claim will be release-only. |
| 2026-10-29 | B3c1 release RMS qualification | Three explicit GPU jobs pass:108,544 source values bit-exact,192 recovery and64 subnormal words, guards/status/immutability and12 zero-Rust-allocation encodes. Astra independently authenticates all saved results and closed groups. Debug failure retained. | [Report](docs/gemma3-half-rms.md); `9b68a22`; continue B3c2. |
| 2026-10-29 | B3c2 pointwise reference preparation | Astra reviews the independent Half GELU/product and scalar-scale source; four portable controls pass, including exact intermediate-store and subnormal witnesses. Two optional borrowed kernel drafts remain private; no export or GPU result. | Freeze/export/replay source, then reviewed native consumers. |
| 2026-10-29 | B3c2 frozen reference and cold preflight | Export/replay and independent152,512-value main/wide verification pass. First strict Metal Clippy stops on redundant field-name spelling before target tests; corrected to field shorthand, failed receipt preserved. Numerical source/gates unchanged. | Fresh cold checks, then reviewed release GPU execution. |
| 2026-10-29 | B3c2 release pointwise qualification |420 default /302 Metal-library tests, strict checks and two explicit GPU jobs pass. Astra independently verifies218,432 bit-exact MAIN comparisons, zero WIDE gate exceedances, physical Half/gating/store/subnormal controls,8 zero-Rust-allocation encodes and5,888 recovery words. | [Report](docs/gemma3-half-pointwise.md); `a954146`; continue B3c3. |
| 2026-10-29 | B3c3 shared window geometry | Astra resolves physical retentionW versus old attention viewW−1 and recommends small prevalidated row helpers. CPU view/contiguous-copy extraction and independent row/chunk-partition controls are in checks; GPU attention/state remain unimplemented. | Finish B3c3a checks/review, then B3c3b. |
| 2026-10-29 | B3c3a checked CPU row composition | Formatting/strict checks,424 default /306 Metal-library /11 targeted tests pass. Independent678 old-span and9,772 copy-plan cases plus64 multihead/partition combinations/576 commits preserve exact bits, spans and addresses. Existing frozen graph/failure/allocation regressions pass. | [Report](docs/window-row-geometry.md); `7635606`; continue B3c3b. |
| 2026-10-29 | B3c3b1 frozen attention reference | Five portable controls and exact export/replay pass. Astra independently verifies722,896 numerical values,104 arrays,24 gates and exact physical/mask/chained expectations. Source-only seeded primitive states; no engine/GPU transaction outcome. | [Report](docs/gemma3-staged-attention-reference.md); `2ba2a42`; continue native operator. |
| 2026-10-29 | B3c3b2 native attention preparation | Borrowed serial-Reference host/shader and raw-first consumer are in cold checks. Frozen eight-case comparisons, W1/c0 analytical identities, short/alias errors and nonfinite/status recovery are prospectively fixed. No GPU result yet. | Complete strict checks, Astra review and explicit release GPU collection. |
| 2026-10-29 | B3c3b2 checked resident attention leaf | Strict checks,424 default /308 Metal-library tests and both explicit GPU jobs pass. Astra rescored361,448 source values (one tiny probability difference, zero failures),16,016 exact recovery values and all27 raw records/identities/guards/masks. Eight release encodes allocate zero Rust objects. | [Report](docs/gemma3-staged-attention.md); `229eb8b`; continue two-phase cache set. |
| 2026-10-29 | B3c3c transaction design | Astra advises and independently checks a concrete set-owned command/opaque readiness API, all-owner Half-bit validation after NaN staging fill, and whole-set poison after submitted copy uncertainty. Private host/shader/tests proposal is under preparation; no build/GPU outcome. | Freeze/review candidate, then cold and actual copy/state checks. |
| 2026-10-29 | B3c3c initial cold preflight | Frozen candidate passes formatting/default Clippy; strict Metal Clippy catches an unnecessary unsafe block around a safe API and a collapsible readback guard. Mechanical fixes preserve transaction semantics; failed receipt retained. Prospective allocation reporting distinguishes27 observed boundaries from12 required-zero source boundaries. | Fresh cold checks and reviewed three-job GPU controller next. |
| 2026-10-29 | B3c3c checked cache transaction | Strict checks,424 default /310 Metal-library tests and three explicit GPU jobs pass. Astra reconstructs67 snapshots/6,210,796 Half words, exact copies/chained513→514, all-owner readiness, real partial-copy poison and terminal zero-prefix GPU replay. All27 observed boundary allocations zero;12 are required-zero. | [Report](docs/gemma3-resident-cache.md); `8f0b092`; continue B3c4. |
| 2026-10-29 | B3c4 composition advice | Astra inspects actual helpers: packed product/gather already support unequal hidden/query widths. Choose a shared original-byte upload helper first, then frozen independent Half graph and concrete resident Model/Session. Existing Source upload stays unchanged; actual270M norm interpretation remains B3d. | Private bounded upload proposal under preparation; no new build/GPU outcome. |
| 2026-10-29 | B3c4a upload review/application | Astra independently approves exact original-byte copy, unaligned/nested views, rejection-before-write and seven unchanged existing function bodies. Guarded two-file application matches the private proposal; cold checks next. | Prospective review `2d9df9858004668b8ceedad7634060821ddf5b9c6b5971e47e533a81e33d7949`; no new GPU result. |
| 2026-10-29 | B3c4a regression checks | Formatting, strict default/Metal Clippy, three new cold controls,424 default tests across library/integration suites and313 Metal library tests pass. Context integration release binary is frozen; release upload controls/libtest freeze finishing. | Private cold receipts; GPU bytes/numerical regressions next. |
| 2026-10-29 | B3c4b prospective source review | Separate Astra approves the exact new48-stage Half graph and eight ordinary-RoPE components. Fifteen dependency pins,384 graph plus40 rotary arrays and fixed main-only gates authenticated. Guarded application passes; portable controls next. | Review `2330080446243d0872985d725c9beaf8804c146da9004a28e3ab869d41c7b630`; source export/native comparison still pending. |
| 2026-10-29 | B3c4a hardware collection / B3c4b controls | All eight cold commands and separate Context binary freeze pass; reviewed seven-job release Metal collection starts. Seven new Half reference portable controls pass. Controller scope wording is corrected before execution to include the existing tiny Qwen session regression; no executable AST change or failed run. | Upload launch approval `bb554e9fac807597b099a49c6ece585ca0a3ca8b29476912915f74b6d6856284`; source export review next. |
| 2026-10-29 | B3c4a checked original upload | All strict/default/Metal/cold checks and seven release hardware jobs pass. Astra independently reconstructs18 original/resident/rejected byte cases;2352 resident bytes remain exact after rejection. Existing Context390096 exact value assertions pass under unchanged gates. | [Report](docs/original-matrix-upload.md); `11bcc36`; continue B3c4b/c. |
| 2026-10-29 | B3c4b source execution | Seven portable controls and two fresh exact exports pass; all three output files match byte-for-byte. The frozen candidate contains384 graph plus40 rotary arrays. Astra independently reconstructs numerical profiles before fixture publication. | `11bcc36` upload commit complete; B3c4b source audit pending, no new native graph claim. |
| 2026-10-29 | B3c4b frozen Half graph source | Astra independently reconstructs all424 arrays/715648 bytes,192 graph groups and eight rotary components, both causal prefixes, all main-only gates and23040 rotary envelope comparisons. Exact original manifest/arrays promoted; seven portable controls and fresh replay pass. | [Report](docs/gemma3-half-reference.md); `1ecff33`; native graph still open. |
| 2026-10-29 | B3c4c resident core proposal | Separate Astra drafts concrete immutable Model/session-owned cache/scratch, existing packed operations, one all-layer/head math phase followed by cache publication,48 stage snapshots and checked accounting. Static formatting only; independent code review starts before application. | Private proposal `0dce0702cb36ea7b3d67ced41001bfeeed6f80a80ecf52d80baf3ca68a7f7487`; full native tests remain required. |
| 2026-10-29 | B3c4c first independent review | Astra finds two bounded draft issues before application: include hidden width in the shared zero-bias maximum and report session planned peak at least its admitted bytes. Full core/forward/snapshot review finds no other blocker; author prepares v2 with mixed-shape and peak controls. | Review `6c36623eb8720b5fe99ad271a47b855807acb10f9559ccca60a0c7d6dacc313d`; no live Rust change/build/GPU. |
| 2026-10-29 | B3c4c corrected core application | Astra confirms both sizing/reporting findings closed by shared helpers and five cold controls; all forward/encode/snapshot bytes remain unchanged. Exact eight-file guarded application matches v2. Required strict/default/Metal/debug/release checks start. | Review `df5bdc13815c74589ec71953f658456b2fa6d36cc82182bdc541612c6a069a9d`; complete native qualification remains pending. |
| 2026-10-29 | B3c4c initial compile | First cold receipt passes formatting/default Clippy, then Metal compile catches a missing MTLResource trait import and Config ownership captured across an outer borrow. Minimal import/reborrow fixes preserve all arithmetic, state and gates; failed receipt retained. | Root frozen successor v1; corrected cold checks next, native consumer still pending. |
| 2026-10-29 | B3c4c corrected compile/control checks | Formatting, strict default/Metal Clippy and all five cold controls pass. Astra independently approves exactly the trait import and inner Config reborrow; no clone, arithmetic or publication change. Full regressions remain running. | Root review `065c88115cb52e090a54ffebcf0ea2bc753c7060b70168bab09e898f17ba8978`; GPU consumer under preparation. |
| 2026-10-29 | B3c4c core cold regression completion | All eight corrected cold commands pass: formatting, strict default/Metal Clippy, five debug/release controls,424 default tests,318 Metal library tests and release libtest freeze. No hardware decoder execution yet; checked GPU stage/state/fault consumer remains required before task completion/commit. | Cold receipt `artifacts/new-goal-b3c4c-cold-checks-v2/checks.json`; next independent consumer review. |
| 2026-10-29 | B3c4c native consumer first review | Separate Astra-authored consumer proposes16 schedules/2400 stage snapshots/428832 main comparisons, state/fault/replay/ownership and32 rotary component commands. Checker requires nonzero guards, complete prior history/published-state assertions and per-call allocation logs before unwrap; no core arithmetic or gate finding. | Consumer remains unapplied; author correction/review then final Metal checks and explicit hardware collection. |
| 2026-10-29 | B3d bounded provenance advice | Astra pins converter0.26.3 dispatch/sanitize/cast/quantize/save code: positive support for raw gamma; original Google revision is separately unknown and does not alone block same-artifact qualification. Eleven-file190208344-byte acquisition and two valid544-token partitions are proposed; payload/input IDs/source/native protocol still unexecuted. | Private advice manifest `e372f30e4f1e93175e52ae952aa2749876ed8650aa3a0baba907da37c971adbc`; finish B3c4c before real model execution. |
| 2026-10-29 | B3c4c corrected consumer application | Astra confirms all three evidence findings closed; nonzero guards seed only cold/quiescent storage, failure checks preserve complete old publication, and allocation counts emit before unwrap. All five applied files match v2; final feature-Metal checks start, reusing the checked unchanged default configuration. | Review `e298ec1e1daf2a0734e731b5abdd5a889f4fa8be65d1d9c34d9ba0136e3dfcd1`; GPU launch remains pending binary/controller review. |
| 2026-10-29 | B3c4c consumer strict-lint preflight | Final Metal v1 stops on one needless-range-loop lint in the independent cache oracle; no tests/GPU ran. Mechanical mutable enumerate plus an explicit two-owner assertion preserves expected words. Failed receipt/controller drafts remain unchanged; successors are frozen. | Final Metal v2 checks next; no tolerance or core change. |
| 2026-10-29 | B3c4c final checks / hardware launch | Final six feature-Metal commands pass: strict checks, seven debug/release controls,320 Metal-library tests and82,869,088-byte release libtest freeze. Astra verifies5871 identities/1871 runtime files and all nine prior source/cold groups closed, then approves exactly three bounded native jobs. | GPU-v2 approval `ce6f4af4151f03148375103507c9a41bbb268ac9030be8401c68b38cc1ea7019`; collection started, results not yet qualified. |
| 2026-10-29 | B3c4c native collection passes | All three reviewed release GPU jobs pass:2400 snapshots/428832 main values (max gate fraction0.41270), exact schedules/state/recovery, twelve warmed zero-allocation forward calls, and11520 rotary values (main0.80659/math0.84023 max fractions). Complete raw arrays/state/status retained; independent Astra audit starts before qualification or commit. | GPU-v2 receipt; source gates unchanged, no real-checkpoint/app/performance claim. |
| 2026-10-29 | B3c4c independent native rescore | Astra independently rescored all428832 main values/2400 stages with zero failures, reconstructed170 physical snapshots and122 successful commits, exact partial-copy mutation/recovery, full publication/history and immutable original tensors; all12 allocations and rotary gates pass. Reader-only indexing/swapped-role mistakes were preserved/corrected without target/gate changes. | Final masked-zero/provenance audit still finishing before publication/commit. |
| 2026-10-29 | B3c4c checked complete tiny resident graph | Independent Astra saved-byte audit passes all428832 main values/2400 stages,170 physical snapshots/122 commits, exact recovery/publication, guards/originals, twelve zero-allocation forwards and11520 rotary values. Frozen gates unchanged. Budget/getter controls are reviewed Rust assertions; sparse memory samples are not peaks. | [Report](docs/gemma3-resident-graph.md); `73416b0`; continue B3d acquisition/source/window journey. |
| 2026-10-29 | B3d1 verified artifact acquisition | Eleven files/190208344 bytes authenticate under200MiB transport cap. Eight cached metadata plus three public immutable LFS transfers pass hashes, exact53490-byte header/490 spans/index,109 finite BF16 norm vectors and atomic exclusive publication. Astra independently rehashes all inputs/payloads/outputs and confirms three closed groups plus retained hardlinks. | [Report](docs/gemma3-artifact-acquisition.md); `f6015b4`; real source and inference unqualified. |
| 2026-10-29 | B3d2/B3d3 source and loader design | Astra freezes a real NativeHalf NumPy BLAS profile with explicit stores/all18 layers at eight selected rows, raw FP32 effective norms and bounded selected-head output; tokenizer-only IDs are being replayed with a fresh cache prefix before numerical proposal. Another Astra prepares one immutable original-payload owner with small typed gamma conversion and reused cold I/O helpers. | No numerical source/model run yet; keep gates fixed and B3d2/B3d3 open. |
| 2026-10-29 | B3d3 loader guarded application | Astra approves one original-payload owner, explicit raw-BF16 preparation/temporary overlap, requested capacity1..544 and three shared cold helpers with unchanged Gemma4 API/semantics. All seven applied files match frozen proposal; default/Metal strict and cold checks start. | Review `b938b3a561ac5547c82b434642d2d3c1887adac97bac3e2b582097ae054411c0`; no real loader/model execution yet. |
| 2026-10-29 | B3d2 diagnostic-contract review correction | Reading actual Metal and frozen scalar attention corrects the parent assumption: raw dot scores remain for all materialized keys, masked probabilities are zero. Preserve the source raw-score contract and native absolute-column subset; no target, source or gate change is needed for this assumption correction. Fresh-cache tokenizer replay matches IDs; numerical source review continues. | Reference proposal remains unexecuted, pending independent full review. |
| 2026-10-29 | B3d2 guarded source application | Independent Astra validates unchanged raw-score/materialized-column contract, all400 literal stages,3200 row-stage groups, bounded97MB maximum arrays and4.931GB prospective ledger. Exact three source/test files applied; ten portable controls start with isolated fresh bytecode prefix and frozen NumPy/tokenizer runtime. | Review `2365c8cb391eb4f5c738fce37f53260071a080cd6f5959f1872485ec36b1c25a`; no numerical real-source export or native real-model run yet. |
| 2026-10-29 | B3d source/loader cold completion | All eight loader checks pass:431 default tests,327 Metal-library tests,26 targeted Gemma3/Gemma4 controls in debug/release, strict formatting/Clippy and release freeze. All ten real-reference portable controls pass separately. A standard pinned virtualenv launcher is being reviewed without synthetic prefix mutation; exact source export remains unexecuted. | Loader/source receipts retained; actual original-byte upload controls and real numerical journey remain pending. |
| 2026-10-29 | B3d3 original-byte consumer application | Astra independently approves two ignored controls for actual authenticated file storage and resident original-byte upload only:236 source logical records,237 resident records across491 physical buffers, analytical109 FP32 norm hashes and source-store drop lifetime. Exact cfg(test)-only consumer applied; final feature-Metal checks start. | Review `7df34e12f471a71bb09895dec49d4fca77cc2c7136f30cb40891541597e49c40`; execution waits until reviewed source export/replay groups close. |
| 2026-10-29 | B3d2 real source launch / loader final compile | Final loader Metal checks all pass with327 library tests,26 overlapping debug/release controls and two new ignored real-storage/upload tests compiled. Astra authenticates23261 source/runtime pins and truthful standard virtualenv startup, approving exactly two isolated export/replay CPU jobs at6GiB/900s bounds. Source collection starts; native models wait for both source groups to close. | Source approval `e24ea37fd90dc6c858374071459ee69eb98d5800ffd232ea3c5f30485a056ca2`; no completed source or native real-model result yet. |
| 2026-10-29 | B3d first real-source failure | Export stops cleanly at plain-France BOS layer8 FF residual: Half64864 +14520 gives FP3279384 outside±65504. All398 partial groups/669568 bytes, memory trace and failure remain; no replay/native inference occurred. Astra verifies pinned MLX0.26.3 clamps both Half residuals; BF16 gamma708/effective709 is authenticated and unchanged. | Primary/saved proof `a4d00e7cdc94cd7ae7505fd09ddd8c254be64decd9607687ec49a9ad0060c3ad`; add explicit clamp/profile successor rather than relax gates. |
| 2026-10-29 | B3d Half-profile correction preparation | Astra approves clipped source-v2 arithmetic with12 controls and unchanged gates/IDs. A native-author service-capacity failure repeats twice; parent implements the already advised minimal kernel/test draft locally, with old generic residual unchanged and only Gemma3 selecting the new operation. | Root clamp proposal frozen for Astra review; uncompiled/unqualified, source-v1 still failed. |
| 2026-10-29 | B3d explicit clamped profile checks start | Astra independently approves the root kernel/tag/width/two-site integration and16 exact word controls, preserving generic residual shader unchanged. Reviewed clipped Python source-v2 is applied through metadata-only v3 current-kernel pin;12 portable source controls and final feature-Metal checks start. | Native review `3b1a26e0606030cab3bb30ec6b6d76e61122efe44a216bdebf1000bff740fbf8`; source review `7a47c433d94227f7653f87d9b60240503023ef9aabb0a466f7bf239e4238a835`; no source-v2 export/native real inference yet. |

| 2026-10-29 | B3d clamped cold checks pass | All twelve portable source controls and six feature-Metal commands pass, including strict formatting/Clippy, library regressions and current release freeze. Separate Astra review approves the minimal Gemma3-only operation; actual primitive plus three tiny GPU checks are next. | Current binary `047e437173b5e101cfdfbe224203b8ce8da18d13dbeec05293e4ca7843976351`; real source-v2 and real native inference remain pending. |

| 2026-10-29 | B3d clipped Half GPU regression | Four literal release GPU jobs pass:20,962 exact primitive values, sticky invalid/status/legacy/store-boundary controls,428,832 unchanged tiny graph comparisons, cache recovery and11,520 rotary comparisons. Eight primitive encodes and twelve complete forward calls report zero Rust allocations in release. | Raw evidence retained; Astra saved-output review and separate correction commit next. Real source-v2/native real inference remain open. |

| 2026-10-29 | B3d clipped correction independent qualification | Astra independently verifies every20,962 primitive word, store/invalid/guard/status controls and all428,832 tiny graph values/2,400 snapshots,170 physical states, cache recovery, release allocations and rotary gates. | [Report](docs/gemma3-half-residual.md); `8528b5b`; correction committed. Corrected real source export/replay runs separately; no real native model yet. |

| 2026-10-29 | B3d corrected real reference succeeds | Both finite CPU jobs pass3,232 stage groups/13,473,568 values per profile; output replay is exact and both669,568-byte/398-descriptor historical prefixes remain exact. OS RSS peaks2,806,284,288 /2,735,538,176 bytes below6GiB; all owned groups close. | Source-v2 receipt retained; independent saved-source review and reference task commit next. |
| 2026-10-29 | B3d loader controller placement preflight | Parent approval retained the v1 inventory path after copying the controller to v2; runtime_inventory identity guard rejects before any child/model/GPU job. Failed receipt preserved. v3 keeps controller/test bytes exact and corrects only approval placement metadata. | Astra review of metadata successor, then storage/upload collection; no numerical change or failed model execution. |

| 2026-10-29 | B3d original storage/upload collection succeeds | Corrected placement v3 launches two literal release jobs; actual original storage and491 resident buffers pass before/after CPU-store drop. Root independently reconstructs all236 source logical records per job,127 matrices and55,936 effective norm words from original file ranges. | Reader-only capacity assumptions corrected in preserved v2 reader; no target/gate changes. Real inference remains pending; separate loader task publication next. |

| 2026-10-29 | B3d2 independently frozen real source | Astra authenticates all3,232 groups/6,464 finite arrays and unchanged gates, exact replay/prefixes, original109 norms,368,640 residual stores/40 saturations,589,824 gating stores and10,240 embedding-scale words. | [Report](docs/gemma3-real-reference.md); reference-only final review/commit, then loader/native journey. |

| 2026-10-29 | B3d2 reviewed source commit | Exact seven-file Astra staged review passes; source-only reference task committed as`ba88fcc`. Both successful jobs and the failed unclipped predecessor remain recorded. | B3d3a original-byte loader task final review next; B3d3b real native runner remains in preparation. |
| 2026-10-29 | B3d3a original-byte ownership verified | Storage/upload tests and independent complete original-range hashes pass, including all491 physical owners after source-store drop;431 default/327 Metal checks and strict review already pass. | [Report](docs/gemma3-original-loader.md); loader-only staged review/commit next. No inference qualification. |

| 2026-10-29 | B3d3a reviewed original loader commit | Exact12-file Astra staged review passes; original-byte loader/storage/upload task committed as`a38142d`. Working tree retains only the pre-existing unrelated apple/.DS_Store. | B3d3b private native runner proposal is under peer review; independent checker is being prepared before target results. |

| 2026-10-29 | B3d3b native runner guarded application | Astra approves the concrete public-API diagnostic runner:6,400 selected stage captures, two544-token partitions, bounded teacher/free generation, raw-first evidence and public session/codec/resource controls. Exact two-file proposal applied; strict/build/control checks next. | Review`57c99ce9586334e5bce65c64598cc7db5a77190a2182af5237861baba8b238be`; no native model execution yet. Separate checker remains a prerequisite. |

| 2026-10-29 | B3d3b first native compile | Formatting/default strict Clippy pass; feature-Metal strict Clippy stops on one collapsible nested-if spelling in diagnostic exact-row comparison. Preserved failed receipt; mechanical let-chain successor changes no capture/control/numerical behavior. | Root frozen correction for Astra confirmation; fresh cold checks next, no model/GPU execution. |

| 2026-10-29 | B3d3b native cold checks pass | All eight corrected checks pass: formatting, strict default/Metal Clippy,431 default and327 Metal library tests, three runner controls in debug/release and optimized example freeze. Astra confirms the mechanical lint-only correction. Reviewed separate checker applies exactly; six synthetic controls start before target results. | Native binary`8c4890f54bd9155055e920b592d1f0136db09a61436ffe3c4233ad3348b85ec3`; no native model run yet. |

| 2026-10-29 | B3d3b bounded native capture completes | Single reviewed real Mac process completes6,400 stage records/141,258,016 raw bytes,2,351 committed tokens and187 forwards under fixed protocols. Both544-token partitions, causal/logical row equality and public state/codec controls pass in the runner.693 direct observations show812,025,152-byte lifetime physical maximum; owned group closes. | Capture is structurally complete, not numerical qualification. Frozen separate saved-output checker is next; no app/phone/performance claim. |

| 2026-10-29 | B3d3b real numerical comparison rejected | Frozen saved-output checker completes all18,537,288 comparisons and rejects92 MAIN values (98 WIDE diagnostic exceedances), max error/gate3.16603. First failure is plain-France row5/layer3/post-attention norm. All raw arrays and clean closed failed-reader receipt remain; state/codec/ledger audit completes. | Astra investigates saved operands and precise reductions; no gate/source/target change or native retry. B3d3b stays open. |

| 2026-10-29 | B3d3b independent rejection diagnosis | Separate Astra scalar audit exactly reproduces92 MAIN/98 WIDE failures. First failing RMS matches all640 own-input oracle words; original gamma73/effective74 and the preceding projection channel match independent reconstruction. First divergence lies upstream in attention context; exact native partitions exclude schedule/cache drift. | Saved-failure review`6a2e2107f56a60765c58a0968ef9198373b46b7e43d9852045cc92338b187063`; bounded first-layer attention investigation continues, no source/gate/target mutation. |

| 2026-10-29 | B3d3b native-store contract correction | Direct six-token trace confirms frozen row3 FP32 RMS/V predictions; Astra verifies all42,528 words/13 captures and11,520 own-input RMS words. Independent review approves uniform certified-projection/FP32-reference source successor with exact legacy3,232 coefficients, diagnostic new envelopes and unchanged WIDE hashes. Exact three-file application passes;18 controls start. | Reference center/relative element bounds change explicitly; original92 rejection remains. No engine change or full-window retry. |

| 2026-10-29 | B3d native-store source export/replay | Both corrected-profile CPU jobs and independent saved-source audit pass all3,232 groups/6,464 arrays, exact legacy gate dictionaries/WIDE hashes/teacher prefixes,143,984,128 projection stores and all32 exact-rounding witnesses (31 retained-input dots reconstructed). Raw97,110,272 bytes replay exactly. | [Contract](docs/gemma3-native-store-reference.md); reference-only staged review/commit next. Native reuse/fresh qualification remain separate; old92 rejection stays rejected. |

| 2026-10-29 | B3d native-store reference committed / consumer checks | Reference-only task committed as`0ef34f6` after exact8-file Astra review. Separate consumer v2 closes an endpoint-failure eligibility-label bug before execution, then exactly applies3 metadata/checker files. Four Rust and ten Python controls plus strict checks/release freeze start; numerical engine remains unchanged. | Historical rescore is explicitly diagnostic only; fresh preregistered capture remains required. |

| 2026-10-29 | B3d native-store consumer checks pass | Strict Metal Clippy/formatting, four Rust controls in debug/release, optimized release freeze and ten Python controls pass. Historical posthoc launcher then rejects a truncated historical-array SHA literal before any child (commands[]); data/checker/source remain unchanged. | Failed owner v2 preserved; exact provenance-only successor pending. Fresh native qualification still required after diagnostic success. |

| 2026-10-29 | B3d source3 trajectory rejection investigated | Corrected provenance-only reader completes all18,537,288 values and rejects84 MAIN intermediates/98 WIDE diagnostics. All84 map42 paired values that match independent own-input checks;33 whole groups/24,984 values pass old local gates. Strict final/teacher12,582,912 values pass. Astra checks the proof and fixed provider contracts. | Both historical screens remain rejected; no kernel/reference/gate change or fresh target run. |
| 2026-10-29 | B3d prospective semantic policy | Astra approves universal coverage design:624 coupled groups/13,185,444 values plus5,808 own-input groups/5,351,844 values. Faithful exact-dot Half brackets, exact basic/RMS stores and independently propagated provider contracts supplement unchanged local coefficients; unsupported proof is fatal. | [Policy](docs/gemma3-native-conformance.md), plan correction and doc checks/review before commit; checker implementation follows. Full adoption quality/app/phone remain open. |
| 2026-10-29 | B3d policy committed | Exact five-document Astra review,1,003 existing local-link checks and staged whitespace/content checks pass; prospective policy committed as`ae7a457`. Both rejected screens and every original selected value remain recorded. | Generic checker/corruption implementation is now authorized under that frozen policy; no new target run or engine change. |
| 2026-10-29 | B4 concrete mechanism selected | Astra verifies cached LFM2-350M metadata/current primary source:6 attention plus10 short-convolution layers, actual FF4608 and two-row per-channel state. B4a–e now separate strict source/adapter, scalar state, tiny hybrid, resident composition and bounded real qualification. Estimated fit is not measured; historical converter identity is unknown. | [Plan](docs/lfm2-hybrid-plan.md); documentation review/commit. B3 remains active; no LFM2 payload or model execution. |
| 2026-10-29 | B3d generic conformance checker applied | After policy`ae7a457`, Astra approves exact five-file generic checker proposal. All five private synthetic attempts pass; final16 operator plus12 checker controls pass again from live files under23,224 authenticated bindings. All237 Rust/Metal/Cargo files still match the passing native cold checks. | Code review`513ff607da5d7e408f9e5afd9f60f26c76a5d5213e3a9eba790e5db071e579a2`; controls`86fb5c32961f119dd2946dfa34ecdbd9523d8c1e728e6504c27bf841831542b7`. Complete saved-byte diagnostic next, then separately frozen fresh capture; no qualification yet. |
| 2026-10-29 | B3d complete historical conformance diagnostic | All624 strict coupled groups/13,185,444 values and5,808 actual-input groups/5,351,844 values pass fatal checks;8 projection cases use exact fallback. Every18,537,288 original trajectory value is still reported, retaining84 MAIN/98 WIDE rejections. Single252.36s process closes cleanly, lifetime physical368,150,712 bytes and observed RSS743,981,056 bytes below6GiB. | Diagnostic owner`bd35a1db769d596a3d55cdf4cab87380fb43dd4e97a1f2d6f7a9be5095221057`; eligibility/qualification remain false. Astra saved-result and fresh-controller review next. |
| 2026-10-29 | B3d checker code task complete | Astra independently audits all6,496 raw records,508 original ranges,9,132 masked+0 words,2,378 memory observations and closed group, preserving84/98 diagnostics. Sixteen operator/twelve reader controls and unchanged source checks pass. | [Report](docs/gemma3-conformance-checker.md); exact staged review and separate checker commit. Eight fallback branches are aggregate reports, not independently replayed retained certificates. Fresh native runner task stays open. |
| 2026-10-29 | B3d checker committed / fresh native launch | Exact nine-file staged Astra review passes; checker task committed as`3cfd38a`. Astra then verifies final fresh placement, all23,573 identity-union entries and13 closed prior groups, approving one900s/6GiB native capture with unchanged binary/workload. | Fresh capture starts under policy`ae7a457`; structural completion and independent fresh numerical checking are still pending. |
| 2026-10-29 | B3d fresh native and complete checker pass | Fresh native completes187 forwards/2,351 tokens in73.03s, lifetime physical815,334,720 bytes; full raw141,258,016 bytes match history exactly. Astra verifies15,003 events/6,496 records and688 observations. Separate fresh reader completes261.07s/2,445 samples with zero fatal failures across18,537,288 values, eligibility true after endpoints, original84/98 rejected diagnostics preserved. | Fresh owner`fed52da1fbfc16947b8c4dc5f2c0fc7d9f0c2aa08f2f6c67a7a2e365ea8dba5a`; checker owner`e3bf7eb8b0442f39ee023cd97381901158040623e5d5eee724b542c9be47ba7e`. Final independent bounded-scope decision and runner commit next. |
| 2026-10-29 | B3 selected Mac native closure | Astra independently replays all13,185,444 fatal coupled gates, authenticates5,351,844 reviewed local results, complete raw/mask/weight/resource coverage and closed endpoints, approving selected Mac544 native conformance. Both historical rejections and all original quality/app/phone/full-window limits remain explicit. | [Release report](docs/gemma3-real-native.md); qualification`96e0cdd820ad4a86d7f0f4d3950d334196b2ecb7994e2f375fedd6726a06966e`. Exact runner/report staged review+commit, then B4a cold adapter; no extra model campaign. |
| 2026-10-29 | B3 committed / B4a source profile resolved | Exact seven-file Astra review passes; bounded native task committed as`aed0a56`, with only pre-existing apple/.DS_Store outside commits. Independent B4a source review resolves current MLX-LM/MLX0.32.3 direct gain and convolution orientation, preserving unknown historical converter. Native RMS single-store choice is explicit; epsilon1e-5/head64+hidden1024 and GQA16:8/D64 remain future qualification. | [B4 profile](docs/lfm2-hybrid-plan.md); private cold-adapter implementation starts, no payload/model/GPU or backend admission. |
| 2026-10-29 | B4a cold adapter applied / broader check catches stale control | Astra approves exact cold plan,12 source/fixture actions and unchanged shared helper bodies. Private six-command checks pass; parent formatting/strict default+Metal Clippy and437 full default tests pass. Full Metal run exposes an older integration test still rejecting theta12345 for Llama despite B1 structural admission. | Failed owner v1 retained; independent history/source diagnosis confirms no production regression or changed qualification gate. |
| 2026-10-29 | B4a compatibility control correction | Astra reviews a test-only mini-patch: true unsupported theta0.5 still rejects all four families before GPU creation; separate Llama12345 metadata-only control asserts CompatibleUntested, planning still required and no text/quality qualification. Production code/limits remain unchanged. | Reviewed correction applied; fresh five-command full default/Metal retry v2 running. Cold adapter is not committed/qualified yet. |
| 2026-10-29 | B4a full checks pass | All five parent commands pass: formatting, strict default/Metal all-target Clippy,438 default and519 Metal-feature tests. Test-only structural-Llama regression repair is independently reviewed; failed predecessor retained. | [Cold plan](docs/lfm2-cold-plan.md); Astra verifies all3,985 bindings and closed jobs; exact staged review precedes commit. B4b reuses the existing scalar primitive rather than adding a duplicate. |
| 2026-10-29 | B4a reviewed commit / B4b starts | Exact17-file Astra review and all checks pass; cold adapter committed as`a3c3a6f`. Existing scalar convolution already covers tap orientation/raw-input history and independent fixtures; next adds bounded prepared history ownership and batch publication. | B4b design preserves replay-from-zero for changed earlier prefixes; no dense-session or backend admission from the standalone state block. |
| 2026-10-29 | B4b scalar history checks pass | Reused convolution unchanged; six state tests and separate warmed allocation test pass. All five full parent checks pass:445 default/526 Metal-feature tests, strict all-target Clippy and formatting. Astra verifies code, original fixtures and all private evidence with no blocker. | [History report](docs/short-conv3-history.md); exact staged review and task commit, then independent tiny hybrid reference. |
| 2026-10-29 | B4b reviewed commit / B4c1 starts | Exact10-file Astra approval and all parent checks pass; standalone scalar history committed as`9c18a4f`. Next freezes a three-layer independent packed hybrid source and complete physical-state fixtures before Rust graph work. | Geometry H256/I256/Q4/KV2/D64/V64/C12; original BF16 direct gains, no initial embedding norm, no SiLU in convolution, no Half or real-model claim. |
| 2026-10-29 | B4c1 source fixture approved | Eight portable controls and two finite source-only exports pass; all6 files match exactly. Astra independently verifies252 groups/548,944 finite values, original packed ranges, complete history/KV states and causal prefixes. | [Reference](docs/lfm2-scalar-reference.md); source-only exact staged review/commit, then Rust scalar hybrid compound publication. No target/GPU/real-model qualification. |
| 2026-10-29 | B4c1 reviewed source commit | Exact15-file Astra staged review passes; source fixture committed as`97d8b4d`. All252 groups and source/state invariants are frozen before Rust target outcomes. | B4c2 privately implements the shared scalar graph and compound KV/history/token/logit validation/publication boundary. |
| 2026-10-29 | B4c2 private compile correction | First compile catches four Rust test literal spellings and one nested error conversion in sizing after formatting passes. Failed attempt is retained; mechanical corrections enter attempt2 before any numerical target run. | No reference, gate or numerical operator changed; private checks and Astra review remain pending. |
| 2026-10-29 | B4c2 scalar hybrid checks pass | All57 mathematical stages plus actual prefix-state observations pass unchanged gates across43 chunks/1,651,856 comparisons. Compound publication/reset, rollback, exact final partition state and zero warmed allocations pass; full450 default/531 Metal-feature checks pass. | [Report](docs/lfm2-scalar-hybrid.md); Astra verifies code/private evidence and independent504,656 unique compared source entries; final staged review/commit next. No native/real-model admission. |
| 2026-10-29 | B4c2 reviewed commit / B4d1 starts | Exact13-file Astra approval and all full checks pass; tiny scalar hybrid committed as`cb24079`. B4c closes only the synthetic CPU source/state/publication scope. | Next qualifies direct-gain Half RMS at epsilon1e-5/useful actual shapes; separately prepared bounded LFM2 acquisition grants no execution support. |
| 2026-10-29 | B4e1 independent artifact acquisition | Reviewed bounded controller downloads one exact199,527,820-byte model, reuses eight authenticated cached files and atomically publishes9files/204,378,119 bytes. Astra independently checks full LFS/hash/header/index and clean closed curl group. | [Artifact report](docs/lfm2-artifact-acquisition.md); bytes only. B4d1 source qualification proceeds independently; no model/tokenizer/GPU or app admission. |
| 2026-10-29 | B4e1 reviewed artifact commit / D1 source pass | Exact5-file Astra publication review passes; original artifact record committed as`a9a53bf`. D1 six controls and two four-file exports pass; Astra exactly reconstructs all75,008 outputs in each RMS profile, direct gains and epsilon/underflow witnesses. | Native RMS consumer/cold checks/release GPU qualification remain before D1 closure; model bytes alone grant no execution support. |
| 2026-10-29 | B4d1 cold/release pass and GPU launch | All6 cold/build jobs pass450 default/534 Metal-feature tests, strict Clippy/formatting and optimized release freeze. Astra reviews v3 launch with consumed-byte JSON identity, bounded logs and exact frozen binary; seven reader controls pass. Five literal GPU jobs start. | Earlier unrun owner drafts/AST failure retained. Same numerical source, fixtures and gates; GPU raw-data qualification still pending. |
| 2026-10-29 | D1 first GPU wrapper failure / pipe correction | First v3 job exitsSIGXFSZ before raw RMS/gate output;116B stdout/0stderr and closed group retained. Both Astra agents identify process-wide file-cap scope; exact incidental write remains unknown. V4 caps only two pipes with unchanged binary/data/gates, and four actual CPU transport controls pass. | V3 failure preserved; v4 hardware launch awaits reviewed current package. No numerical outcome or tolerance change. |
| 2026-10-29 | B4d1 standalone Mac RMS qualified | All5 reviewed release jobs pass; Astra exactly rescores150,016 normal+640 recovery LFM values,108,544 legacy normal+192 recovery+64 subnormal words, guards/status and26 zero-allocation encodes. Cold450/534 and source gates unchanged. | [RMS report](docs/lfm2-half-rms.md); exact staged review/commit, then borrowed convolution. Sparse observations are not process peaks; real model/GQA/app remain open. |
| 2026-10-29 | B4d1 reviewed native RMS commit / D2 starts | Exact17-file Astra review passes; standalone native RMS task committed as`7ef4cab`. Next raw conv3 source fixes original BF16 coefficients, explicit FP32 tap order/Half store and exact two-row raw-u candidates. | Peer catches the T1 dispatch requirement: each channel writes both candidate rows. No convolution implementation or hardware outcome yet. |
| 2026-10-29 | B4d2 independent raw Conv3 source qualified | Eight portable controls and exact four-file export/replay pass. Astra reconstructs46,571 MAIN/WIDE outputs,7,238 exact candidate words,10,857 original BF16 coefficients and233 boundaries across35 schedules. | Frozen fixture at tests/fixtures/causal-conv3-half-v1; native operator/consumer and GPU/status/ownership qualification remain pending. |
| 2026-10-29 | B4d2 raw native Conv3 qualified | Source8controls/35schedules and full450/536 cold checks pass. Astra independently reconstructs312,670 exact valid outputs across274commands/286captures, all233chunk boundaries, candidate bits, guards/status and24zero-allocation encodes. | [Conv3 report](docs/causal-conv3-half.md); exact staged review/commit, then all-query multi-KV source. Unrun missing-log-cap draft retained; no numerical changes or model transaction claim. |
| 2026-10-29 | B4d2 reviewed commit / D3 source starts | Exact18-file Astra review passes; raw native Conv3 task committed as`782fd0b`. D3 applies the prospectively reviewed full-query source, with600s/1GiB source-worker bounds and fixed head-major32-row staging. | Eleven cases/143arrays include512→513 and all32 deep queries; no native GQA change or outcome yet. |
| 2026-10-29 | D3 complete all-query source qualified | Eightcontrols/two3-fileexports pass600s/1GiBworkers. Astra reconstructs all143arrays/22,405,366B,8,358,427physicalwords and bothfullqueryprofiles, all36,232masked+0 and per-head copy/validation intervals. | Frozenfixture tests/fixtures/staged-gqa-half-v1; native GQA/cache and legacy regression qualification remain pending. |
| 2026-10-29 | D3 reviewed native extension and full cold checks | Astra approves the nine-file successor after terminal raw capture is moved before metadata rejection assertions. All six parent jobs pass: formatting, strict default/Metal all-target Clippy,450default/539Metal tests and optimized release freeze. | Separate new4-job16MiB and legacy7-job64MiB packages retain complete known legacy logs; no pruning. Mac GPU qualification remains pending. |
| 2026-10-29 | D3 all eleven release GPU jobs pass | Four new all-query GQA/cache and seven existing attention/cache/tiny-Gemma jobs pass with closed groups and unchanged binary/source/gates. New raw logs total39,103,956B; full legacy logs96,364,008B. Exact tested executable is retained before future Cargo rebuilds. | Independent saved-raw numerical/physical-state qualification and exact staged review remain before task closure/commit. |
| 2026-10-29 | B4d3 standalone GQA/cache qualified | Astra independently gates1,978,360 new MAIN values,119 raw records/98 submissions, complete per-head physical/cache/recovery state and seven legacy jobs. All100 observed release allocation boundaries are zero. | [GQA/cache report](docs/staged-gqa-half.md); source/target/gates unchanged. Six-owner fault workload is64, single-owner cases reach544; resident mixed graph and real model remain open. |
| 2026-10-29 | B4d3 reviewed commit / D4a starts | Exact22-file Astra approval and all parent checks pass; multi-KV attention/cache committed as`0effda1`. Next independent strided-product source covers both B*x/C*z views into one physical BCx owner. | Prospective five cases/67,072 values per profile, fixed store witnesses and120s/256MiB source bounds. No new product/kernel or mixed-graph qualification yet. |
| 2026-10-29 | D4a product source controls/replay pass | Reviewed source and bounded owner pass eight controls plus exact four-file export/replay: five cases/17arrays/738,248B. Explicit same-parent readonly views, offsets/guards, signed/subnormal/RNE/overflow and invalid-zero-counterpart controls remain fixed. | Astra saved-array audit pending before fixture/native qualification; no graph or performance claim. |
| 2026-10-29 | D4a strided-product source qualified | Astra's independent integer-dyadic/RNE audit exactly reconstructs67,072 words in each profile,234,980 physical input words,67,240 physical output words,96partition boundaries and all store/invalid controls. Exact six-file fixture promoted. | tests/fixtures/strided-half-product-v1; native wrapper/release checks remain. B4d4 is split into product, independent Half tiny source and concrete resident graph to track actual remaining work. |
| 2026-10-29 | B4d4a standalone strided product qualified | Reviewed four-file native leaf, full450/541 checks and both optimized GPU jobs pass. Astra independently reconstructs347,384 finite outputs exactly, including342,784 MAIN/WIDE comparisons,133raw/128submissions, guards/readonly parents,96boundaries and10zeroencodes. | [Product report](docs/strided-half-product.md); no existing shader changes. Exact staged review/commit follows, then approved independent Half graph source. No model transaction or performance claim. |

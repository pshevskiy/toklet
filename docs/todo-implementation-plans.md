# Implementation plans for open TODO items

These plans map to the 19 items open in [TODO.md](../TODO.md) on 5 November 2026. Start with MoE wiring and stock/native validation. M9 follows the new-model priority, with its first delivery in order. The phone track retains its deferment. Follow-on work requires the stated prerequisites. Preserve failed gates and attach receipts to the relevant checkbox.

For each code task, freeze the acceptance criteria, check the affected path and required Rust configurations, send Astra one diff and evidence review, then commit. Reuse unchanged evidence by identity. Store full logs in artifacts and record only the decision and evidence link in the tracker.

### Phone Settings runtime

Build the current approved package, install on the physical iPhone 14 Pro, and exercise the rendered context and KV controls. Capture Preview, Apply, rejection and restoration with the bundle and device identities. Close on observed runtime evidence. Keep the existing hardware deferment until the device is available for testing.

### Physical iPhone journeys

Use the physical iPhone 14 Pro and each eligible profile. Exercise discovery, verified download, Chat, Stop, restart, and network-denied reuse. Record process limits and exact profile eligibility independently from Mac results. Close when all eligible journeys have device receipts.

### Sparse MoE resident graph

The shared scratch and compensated Half router projection pass same-command GPU checks. Reuse the dense attention, KV and transaction path with a layer-level expert branch. Stream pinned BF16 weights into their final Half buffers, then allocate router and expert scratch from a consumed combined plan. Validate numerical stages, actual bytes, budget rejection, chunk equivalence, reset and failed-candidate state with a small independent fixture.

### Sparse MoE real execution

After the resident graph passes, acquire the pinned original artifact and verify its digest. Freeze the prompt, stop and stock trajectory. Compare complete decision rows and routing sets independently, exercise the admitted 512-position boundary, and collect process peaks in separate owned processes.

### Gemma4 real 1,025 numerical qualification

The [same-input V](gemma-v-boundary.md), [head](../benchmarks/results/2026-11-05-gemma-head-witness-v1.json)
and [RMSNorm](../benchmarks/results/2026-11-05-gemma-norm-witness-v1.json)
witnesses pass locally. The [layer 12 attention witness](../benchmarks/results/2026-11-06-gemma-attention-boundary-v1.json)
binds position 433 K/V to the archived capture and passes both independent
context and output-projection replays. The full 1,025 rejection remains.
The [layer-12 tail replay](../benchmarks/results/2026-11-06-gemma-layer12-tail-v1.json)
passes both arithmetic contracts at position 433. Its attention output and
next hidden match the archived boundary rows exactly. Astra ends the local
layer walk. Further graph work needs a concrete hypothesis about the source
or propagated precision differences.
The [archived band intersection](../benchmarks/results/2026-11-06-gemma-reference-intersection-v1.json)
rules out mutually exclusive bands for the retained coordinates. It leaves
the joint graph computation and the original failures unresolved.
The [authenticated prefix64 replay](../benchmarks/results/2026-11-06-gemma-first-staged-v1.json)
localizes the first observed position48 propagation to layer3. Attention
and FF up are exact. The first captured difference is the per-layer gate.
Replay the remaining FF stages using the same operands and 32-row shape.
The [FF replay](../benchmarks/results/2026-11-06-gemma-layer3-ff-v1.json)
finds identical inputs and one misrounded gate projection word. The exact
dot lies above the midpoint. Replay the lane/tree, then test an isolated
three-component correction on that row and midpoint/cancellation controls.
Recheck frozen affine fixtures before any prefix or full-graph experiment.
The [isolated CPU and Metal probe](../benchmarks/results/2026-11-06-gemma-affine-expansion-v1.json)
recovers the exact lost term and corrects the captured word. Use a
source-isolated shader swap for the prefix64 A/B. Require the corrected
gate and inspect every recorded stage and K/V endpoint. Carry useful
results to the unchanged full 1,025 comparison and resource checks.
The [isolated prefix result](../benchmarks/results/2026-11-06-gemma-affine-prefix-v1.json)
matches all 25 staged comparisons bitwise. The full run retains the historical
32-row digest requirement alongside the numerical and state gates.
The [full comparison](../benchmarks/results/2026-11-06-gemma-affine-full-v1.json)
preserves state and the historical digest, but fails 5 mathematical and
4 staged observations. The frozen staged oracle itself exceeds the
mathematical tolerance at two final-logit coordinates. Defer the expansion
as a broad correction and preserve both contracts.

### Gemma4 adoption task gate

Use the failed 100-task collection and the stock decisions to distinguish artifact quality from engine errors. Screen a concrete candidate for combined memory fit before payload acquisition. Freeze candidate, baseline and grading policy. Repeat the unchanged 10,000-target loss and 100-task adoption gates before enabling the route.

The [Q6 checkpoint](gemma6-feasibility.md) fits the stock worker and matches
original BF16 at the four disputed decisions. The complete trajectory gate
still rejects after bounded recovery and an independent causal-prefix
diagnostic. Defer this checkpoint under the current policy. A different
candidate needs a new immutable source and unchanged full quality gates.

The [pinned OptiQ stock preflight](../benchmarks/results/2026-11-06-gemma-optiq-stock-v1.json)
fits the 6 GiB process policy. One of four saved decisions matches original
BF16 and three match Q4. The [HumanEval/37 trajectory](../benchmarks/results/2026-11-06-gemma-optiq-trajectory37-v1.json)
passes all 122 decisions with zero repairs. The [complete stock evaluation](../benchmarks/results/2026-11-06-gemma-optiq-task-quality-v1.json)
reproduces all 100 trajectories, but 49/100 versus original 52/100 exceeds
the two-point limit. Astra advises selecting another candidate only after
a concrete precision hypothesis and conservative memory fit.

The [distinct oQ6 screen](../benchmarks/results/2026-11-06-gemma-oq6-header-v1.json)
finds 244 affine6 and 34 affine8 text matrices, with an estimated 927 MB
margin under 6 GiB. Verify actual strict loading with the assistant absent,
the known redundant KV projections discarded, and exact per-matrix modes.
Run the frozen four decisions and HumanEval/37 before any full collection.

The [oQ6 run](../benchmarks/results/2026-11-06-gemma-oq6-stock-v1.json)
matches all four original decisions. HumanEval/37 rejects on a proposal
cycle after three repairs, so defer the full collection. A separate Q8
stock estimate leaves 205 MB under 6 GiB with no live tokenizer or app.
Authenticate both shards, verify the strict affine8 text inventory and
enforce the live memory limit. Run four decisions first and condition the
frozen single trajectory on that result. The native/app Q8 ledger still fails.
The [bounded Q8 run](../benchmarks/results/2026-11-06-gemma-q8-stock-v1.json)
matches all four original decisions and all77 samples on HumanEval/37,
without repairs. Peak physical memory is5.89GB. Run the unchanged cold and
output-limit controls and full100 collection under the same live cap.
Grade only after every trajectory verifies and the owner exits.
The [Q8 memory decision](../benchmarks/results/2026-11-06-gemma-q8-stock-cap-v1.json)
retains 32 completed cases, of which 31 verify. HumanEval/24 reaches a
proposal cycle. A hardened HumanEval/32 retry exceeds6GiB by14.8MB and
is stopped. A later group query confirms exit. Grading is withheld.
The [native chunk-pool change](../benchmarks/results/2026-11-06-gemma-chunk-pools-v1.json)
reduces Mac512 resident admission by130.7MB at chunk3 and139.7MB at chunk1.
Applying those Q4 savings to the historical Q8 estimate leaves 28MB and
37MB before Q8-specific costs. The stock trajectory and memory rejections
remain in force.
The [QAT header](../benchmarks/results/2026-11-06-gemma-qat-header-v1.json)
contains 173 affine4 and 105 affine8 matrices and leaves 1.74GB of estimated
stock headroom. Authenticate the single shard and run strict stock loading
and frozen raw-token checks. Compare the four saved decisions
informationally. Run full quality only after the bounded trajectory verifies.
The changed template and unknown conversion lineage require separate scope.
The [QAT text witness](../benchmarks/results/2026-11-06-gemma-qat-text-v1.json)
matches all 97 frozen basic-text records and the installed loader's parsed
text arguments. It leaves native identity and live mixed loading separate.
The [bounded QAT stock result](../benchmarks/results/2026-11-06-gemma-qat-stock-v1.json)
verifies strict mixed loading and all 144 HumanEval/37 samples after one
repair, at 4.47GB peak. Run the frozen100 cases first. If the grade passes,
score the exact ten reset loss windows against the unchanged original mean.
The [full stock trajectory](../benchmarks/results/2026-11-06-gemma-qat-full-tasks-v1.json)
verifies 98 cases and rejects cycles in HumanEval/4 and lambada/row-29.
Grading and loss are withheld. The [two-worker shape witness](../benchmarks/results/2026-11-06-gemma-qat-shape-v1.json)
reproduces both stock flips by changing only replay length. Fixed-shape
future variants match bitwise. Astra approves a distinct native diagnostic
while the stock rejection stays recorded.

**Independent affine8 leaf.** Qualify the missing projection mechanism on
authored original-code fixtures. Keep group64/BF16 storage and the current
FP32 compensation. Validate the full byte codebook, guarded budgets,
residual-sensitive dots and affine4 parity. Keep the resident loader and
catalog out of this task.
[Standalone affine8 evidence](../benchmarks/results/2026-11-06-gemma-affine8-leaf-v1.json).

**Pinned native QAT diagnostic.** Verify the mixed cold tensor plan and
consumed byte identity. Reuse the affine layout as the pool key, admit
both projection pipelines with charged owners and keep Q4 gather and the
existing graph blocks. Freeze an artifact-owned numerical reference,
chunk1/32 and causal-invariance witnesses, state/reset and actual process
peaks before running the real model. Compare the original 10k/100 gates
only after native trajectory and fidelity qualify. Chat and G2/G3 still
require their separate acceptance evidence.
The [exact cold plan](../benchmarks/results/2026-11-06-gemma-qat-cold-v1.json)
passes for the pinned metadata. [Authenticated resident storage](../benchmarks/results/2026-11-06-gemma-qat-resident-v1.json)
passes the exact budget and original-byte checks. The [independent prefix oracle](../benchmarks/results/2026-11-06-gemma-qat-native-reference-v1.json)
is frozen. The [mixed diagnostic session](../benchmarks/results/2026-11-06-gemma-qat-session-v1.json)
passes exact short state and legacy controls. The [two-prefix fidelity gate](../benchmarks/results/2026-11-06-gemma-qat-prefix-parity-v1.json)
passes both numerical contracts, chunk sizes1/3/32 and the causal future
extension. The frozen1,025-token oracle and quality-only wrap state checks
are separate. Preserve the public diagnostic ceiling at512.

If the native prefix gate passes, reuse the existing raw Gemma loss and
task loops with QAT-bound source identity and measured owner plans. Admit
the loss context at1,025 and tasks at567 separately under6GiB. Verify the
three-category pilot and reset, capacity and failure behavior before the
full run. Collect the exact10,000 targets against the authenticated original
loss baseline. Recheck every sampled task decision with a fresh native
teacher-forced replay, including hidden EOS and lowest-ID greedy ties,
then grade all100 new completions against the existing original52/100.

### Qwen2.5-1.5B MLX8 quality and managed Mac Chat

The [short alignment](../benchmarks/results/2026-11-06-qwen15-original-alignment-v1.json)
binds the original FP32 baseline and matches native chunk schedules. The
[full native loss](../benchmarks/results/2026-11-06-qwen15-full-loss-v1.json)
passes all 10,000 causal targets with an increase of 0.001528 nats. The
[raw task pilot](../benchmarks/results/2026-11-06-qwen15-task-pilot-v1.json)
passes after isolating numerical assets and decoding after native exit.

1. The [frozen literal task grade](../benchmarks/results/2026-11-06-qwen15-task-quality-v1.json)
   passes at 12/100 versus matched original FP32 11/100, with all 100 native
   responses replayed and the isolated VM restored stopped.
2. After the grade passes, pin the six-file dense artifact and its exact plain
   renderer/codec. Compare no-tools, defaults, Unicode and stop handling. The
   distinct tool-template literal stays outside this route.
3. Bind the larger profile to Reference attention for initial sessions,
   memory plans and replacements. Preserve the existing smaller Qwen policy.
   Verify actual mode, exact and below-budget admission, owner mismatch,
   mutated assets and generation state through the additive native boundary.
4. The [managed Mac controller and package](../benchmarks/results/2026-11-06-qwen15-mac-app-v1.json)
   verify fresh download, open, Send/Stop, context Apply and network-denied
   reopen under the 3 GiB process allowance. The reviewed native41 selection
   preserves native40. Physical iPhone evidence remains deferred.

### M8 closure audit

Join exact artifact, backend, device, context and app receipts to the coverage table. Check commits, reviews and linked verification for each prerequisite. Close only after the remaining M8 gates have their required evidence.

### P0 baseline collection

Use the pinned Mac MLX8 workload and release build in the performance plan. Separate completed uninstrumented timings from bounded profile events. Label each chunk and sampled decode position, verify parity and retain raw samples and memory facts.

### P0 metric selection

Rank projection and normalization costs from the completed work. Choose one primary end-to-end metric, comparison recipe, fallback and minimum useful effect before coding. Record the device-specific evidence and contract.

### P1 implementation

Implement the single kernel candidate selected by P0. Dispatch only for its admitted shapes and keep the existing qualified fallback. Measure representative projection or normalization shapes and retain the change only if the candidate passes the next screen.

### P1 numerical and usefulness screen

Compare fixed intermediates, full decision rows and transactional state against the baseline. Run the frozen timing comparison. Close with Astra's decision and raw results showing the candidate meets the screen, or record rejection and select the next measured candidate.

### P2 confirmation and adoption

Reproduce the effect in an independent run. Execute unchanged quality, state, memory and app/device gates for the proposed deployment scope. Bind the adoption or rejection decision to the exact configuration and evidence.

### M9 publication

Record the baseline, candidate, independent result, scope and decision after P2. Link the evidence and update the checklist and improvement record.

### P3 split KV

Confirm a decode attention bottleneck and qualified Online mode. Implement partitioned accumulation and stable merging with a checked workspace plan. Verify masking, GQA, boundaries and end-to-end improvement before enabling the selected route.

The [paired command screen](../benchmarks/results/2026-11-05-m9-split-kv-timing-v1.json)
supports an explicit Qwen2 experiment at valid depths 1,025–1,152 for
single-token decode. Admit and reuse guarded partials, keep existing Online
outside the selected range, and qualify complete numerical decisions, state,
quality and uninstrumented requests before considering adoption.

The [first real session experiment](../benchmarks/results/2026-11-05-m9-split-kv-session-v1.json)
failed all 128 complete rows and one greedy decision. Replay identical
layer-0 operands at depth 1,025, compare both paths and the partition packets
to FP64, and identify the first Half rounding boundary. Defer this route if
the witness shows ordinary reassociation without an implementation defect.


### R1 rotary reuse

Measure repeated rotary work and size the retained phase data. If the benefit justifies the memory, implement reuse keyed by positional semantics and compare offsets, resets, and end-to-end latency. Close with a measured retain or defer decision.

### P4 I8 KV

Specify the target capacity and codec, including scale and storage layout. Implement writes and reads with fused reconstruction, then bind the solver to real allocation. Qualify quality, state, memory peaks and device behavior before exposing the option.

The current Mac Qwen3 Chat limit is 512 positions. Its profile allows 1.5 GiB
numerical and 2.25 GiB whole-process memory. The 4,096-position result is a
separate diagnostic. At the declared 40,960 positions, applying the existing
[M6 I8 format](experiments/2026-10-15-m6-kv-component.md) to the planner's FP16
KV payload would save 2,202,009,600 raw bytes. The numerical projection after
this substitution is still 3,386,567,826 bytes, before I8-specific buffers
and guards. This prospective window is outside the current Mac profile.
Astra recommends deferring integration until a selected larger profile shows
an FP16 capacity rejection under its own complete budget. Then reuse the M6
format, implement cache writes first, and qualify the chosen attention path
before raising capacity. [Existing projection](qwen3-context-admission.md).

### P5 strict greedy verifier and prompt lookup

Qualify tentative state, exact target decisions and rollback first. Then add the separate lookup proposer and measure accepted tokens, latency and memory. Require identical greedy trajectories across rejection, cancellation and replay cases.
The [nonstreaming CPU path](../benchmarks/results/2026-11-05-m9-cpu-lookup-v1.json)
passes authored tie and near-tie controls. The [real Qwen2 Metal witness](../benchmarks/results/2026-11-05-m9-qwen2-lookup-parity-v1.json)
passes exact target parity and rollback. The [atomic Metal round](../benchmarks/results/2026-11-05-m9-qwen2-atomic-round-v1.json)
passes using the admitted all-row workspace and retains Completion replay.
The [bounded Metal generation path](../benchmarks/results/2026-11-06-m9-qwen2-metal-generation-v1.json)
passes pinned transcript, state, pre-cancel and hidden-stop checks. In
[both release orders](../benchmarks/results/2026-11-06-m9-qwen2-lookup-timing-v1.json),
the repeated contract prompt saves 53.33%/53.76% decode time. The code
continuation misses the 5% gate. Retain the explicit API for demonstrated
favorable prefixes. Default and streaming adoption stay outside the tested scope.

### G2/G3 shared KV and MTP

**G2 shared-KV prefill.** After a selected Gemma graph passes its gates,
freeze its checkpoint producer/consumer map. Validate retained local/global
rows, exact token decisions and transactional rollback across chunk and ring
boundaries. Compare completed prefill timings and physical owners on the
same pinned workloads before retaining a scheduling change.

**G3 MTP.** Pin the matching assistant and its target inputs. Keep draft
state separate and verify proposals using the target's strict greedy
semantics. Test rejection, partial acceptance, hidden stops, cancellation
and ring rollback before measuring acceptance rate, latency and memory.
The target-only path remains the reference for every workload.

### Llama Online correctness

The [same-operand witness](../benchmarks/results/2026-11-05-llama-online-witness-v1.json)
reproduces the layer 1 projection failure on the tied graph at offset 5.
Online prefill and decode agree and padding poison has no effect. The Online
quotient crosses a Half midpoint relative to Reference's normalized-probability
sum. The [ablation](../benchmarks/results/2026-11-05-llama-online-ablation-v1.json)
isolates a compensated alternative with a host FP64 quotient. Astra recommends
retaining the raw Online rejection because a GPU compensated quotient needs
independent numerical, quality and performance qualification.

[The depth-32 staged screen](../benchmarks/results/2026-11-05-llama-online-staged-v1.json) passes the frozen graphs. The next candidate uses separately admitted FP32 scores and an ascending normalized-V pass. Check depth boundaries, state and the real Mac512 checkpoint against Reference before retention.

The [deeper staged experiment](../benchmarks/results/2026-11-05-llama-staged-depth-v1.json)
matches ten greedy IDs through depth 512 on one reused-prefix schedule.
The [guarded leaf](../benchmarks/results/2026-11-05-llama-staged-leaf-v1.json)
passes finite/status, poisoned KV, GQA and single-row/batch scratch-reset controls.
The scratch stores unnormalized exponentials.
The [pinned corpus screen](../benchmarks/results/2026-11-05-llama-staged-corpus-v1.json)
passes chunk1/chunk32 greedy agreement. Before full quality collection,
the [uninstrumented 128-input/128-decode comparison](../benchmarks/results/2026-11-05-llama-staged-timing-v1.json)
passed in both worker orders. Preserve the original 10,000-target
0.05-nat limit and 100-task two-point limit, with separate diagnostic
capacity for those workloads. Both isolated gates now pass
([loss](../benchmarks/results/2026-11-05-llama-staged-loss-v1.json),
[tasks](../benchmarks/results/2026-11-05-llama-staged-tasks-v1.json)).
The [explicit audited Mac512 selector](../benchmarks/results/2026-11-05-llama-staged-session-v1.json)
passes exact workspace accounting, lifecycle controls and isolated-row replay.
The [opt-in Mac app route](../benchmarks/results/2026-11-06-llama-staged-app-v1.json)
proves the staged dispatch from Chat, with Stop, New Chat, rejected capacity
replacement, unload/reopen and observed process memory. Raw Online remains
rejected. Managed-default staged adoption has a separate decision.

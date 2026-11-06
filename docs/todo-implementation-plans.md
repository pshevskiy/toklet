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
extension. The [quality-only wrap](../benchmarks/results/2026-11-06-gemma-qat-wrap-state-v1.json)
passes both schedules and exact state. The [guarded math and staged references](../benchmarks/results/2026-11-06-gemma-qat-wrap-parity-v1.json)
pass every retained comparison. The public diagnostic remains capped at512.

If the native prefix gate passes, reuse the existing raw Gemma loss and
task loops with QAT-bound source identity and measured owner plans. Admit
the loss context at1,025 and tasks at567 separately under6GiB. Verify the
three-category pilot and reset, capacity and failure behavior before the
full run. Collect the exact10,000 targets against the authenticated original
loss baseline. Recheck every sampled task decision with a fresh native
teacher-forced replay, including hidden EOS and lowest-ID greedy ties,
then grade all100 new completions against the existing original52/100.
The [quality pilots](../benchmarks/results/2026-11-06-gemma-qat-quality-pilot-v1.json)
pass. The [full loss gate](../benchmarks/results/2026-11-06-gemma-qat-full-loss-v1.json)
passes all10,000 targets. The full task score rejects QAT for adoption,
so the owned application route remains conditional on another target.

If all gates pass, add a private owned QAT adapter to the existing native
model/session dispatch. Preserve the drop order from session through graph,
norms and authenticated weights. Bind one exact QAT artifact and a distinct
basic-text codec receipt, including the combined codec/engine budget.
After owner and codec checks, add the managed Mac profile and verify fresh
download, offline Chat, cancellation and state preservation before selecting
a source-matched app package.

### Official Gemma BF16 fallback

The [QAT task gate](../benchmarks/results/2026-11-06-gemma-qat-full-quality-v1.json)
rejects46/100 versus52/100. QAT remains diagnostic. Use the exact official
source with BF16 coefficients and a file-backed PLE lookup. Full text
requires9,257,138,758 bytes. The PLE table accounts for4,697,620,480.
The other text tensors require4,559,518,278 raw bytes. [Official source inventory](../docs/gemma-bf16-reader.md).

1. Admit the pinned BF16 config and header, classify selected and excluded
   ranges, and prove the exact PLE and resident byte totals.
   [Cold plan result](../benchmarks/results/2026-11-06-gemma-bf16-cold-v1.json).
2. Authenticate the whole source descriptor and hash PLE rows while
   streaming. Verify each bounded row on demand before publishing output.
   Count integrity metadata, gather scratch and concurrent source buffers.
   [Authenticated PLE source result](../benchmarks/results/2026-11-06-gemma-bf16-ple-source-v1.json).
3. The [BF16 Metal projection leaf](../benchmarks/results/2026-11-06-gemma-bf16-projection-v1.json)
   passes. Stream the pinned descriptor once in physical order. Fill private
   resident BF16 buffers from the same bytes included in the whole-file SHA,
   hash PLE rows and publish only after exact extent and endpoint checks.
   Reuse role bindings and add BF16 embedding lookup before graph integration.
   [Full storage result](../benchmarks/results/2026-11-06-gemma-bf16-resident-v1.json).
4. The [official short graph](../benchmarks/results/2026-11-06-gemma-bf16-short-graph-v1.json)
   passes both contracts and late-failure checks. The [long state witness](../benchmarks/results/2026-11-06-gemma-bf16-wrap-state-v1.json)
   passes both1,025-token schedules. Complete both independent wrap
   references, raw trajectories and original-relative
   quality before factory or app work.
   The [official basic-text codec](../benchmarks/results/2026-11-06-gemma-bf16-codec-v1.json)
   passes its standalone gates. Measure combined codec/engine peak under6GiB.
   The [quality pilots](../benchmarks/results/2026-11-06-gemma-bf16-quality-pilot-v1.json)
   pass the first1,024 loss targets and fresh replay of66 sampled decisions.
   Finish all10,000 targets, all100 fresh trajectories and isolated grading
   against the original source before the adoption decision.
   The [sealed comparison and full loss](../benchmarks/results/2026-11-06-gemma-bf16-long-decision-v1.json)
   preserve the dual numerical rejection. The frozen bands have no common
   FP32 value at61 coordinates. All10,000 loss targets pass separately.
   Replay complete V projection and norm on the observed native operands at
   positions508 and826, with adjacent passing controls and exact cache transfer.
   Use this attribution to specify a complete, independently propagated native
   arithmetic reference at the unchanged tolerance.
   Both [original-quality gates](../benchmarks/results/2026-11-06-gemma-bf16-original-quality-v1.json)
   pass for the same binary. Fresh replay verifies4,149 decisions and
   isolated grading is51/100 against52/100. Numerical fidelity stays separate.
5. After numerical and quality acceptance, measure the combined owner and
   codec, then exercise the managed Mac profile. The six pinned files total
   10,278,816,137 bytes, exceeding the current Hub file and snapshot limits.
   Scope larger transfer bounds to this exact Mac manifest and resolved shard
   URL. Keep 64-bit range, cancellation, disk admission and publication checks.
   - Add the Mac-only audited owner with stable graph and weight allocations.
     Reuse generation dispatch and preserve drop order through the source
     descriptor. Bind codec and model identity, handle fallible poisoned reset,
     and check exact load, session and transition budgets before publication.
     The [safe Rust owner](../benchmarks/results/2026-11-06-gemma-bf16-official-mac-owner-v1.json)
     pairs both assets with borrowed graph and session lifetimes. The additive
     [raw native bridge](../benchmarks/results/2026-11-06-gemma-bf16-native-raw-v1.json)
     retains them after external release and reports combined admission separately.
     [Audited C Chat](../benchmarks/results/2026-11-06-gemma-bf16-native-chat-v1.json)
     reuses the owner codec and records the exact six-file identity.
   - Decode the exact Swift identity and use the official worker route. Check
     Chat, Stop, suspend/replay and unload on the verified local source. Report
     the declared window, loaded limit, remaining output, exact KV bytes and
     combined admission without counting the codec twice.
     [Worker evidence](../benchmarks/results/2026-11-06-gemma-bf16-mac-worker-v1.json).
   - Scope Hub discovery and large transfer to the pinned Mac profile. Check
     search, SHA-verified publication, cancel, disk limits and network-denied
     reopen. Keep generic transfer bounds and iPhone eligibility. The
     [Mac Hub checks](../benchmarks/results/2026-11-06-gemma-bf16-mac-hub-v1.json)
     qualify exact live ranges and cached publication. A full public transfer
     timed out. Repeat it independently of the qualified offline journey.
   - Expose exact cold FP32 KV buffer bytes from the shared layout for the
     supported Mac capacity and chunk. Check the prediction against real
     owners. [KV planner evidence](../benchmarks/results/2026-11-06-gemma-bf16-kv-plan-v1.json).
     Qualify next-load controls, rejected settings and late
     PLE corruption.
     Build the source-matched Mac package and measure the app process. Keep
     append-only text delivery as a separate gate from visible-prefix Chat.

#### Prospective ordered BF16 arithmetic gate

Astra reviewed this policy before the full exploratory comparison. The prior
dual-reference rejection stands. Qualification under this separately named
profile requires a fresh owned reference and a distinct holdout. Existing
exploratory output is diagnostic evidence.

- Use the pinned original BF16 text weights. Center projection comparisons
  on independently rounded BF16×FP32 dots. Exact fallback covers ambiguous
  nearest-even boundaries. Bind the binary64 reduction assumptions, outward
  intervals, exact fallback, source closure and provider before collection.
- Propagate the complete graph from raw token IDs. Fix FP32 storage points,
  chronological FP64 attention and norm reductions, binary64 rotary with
  FP32 trigonometric values, and binary64 softcap with one FP32 store. This
  reference defines projection arithmetic independently of Metal's compensated
  lane reduction. Preserve provider distinctions and test their effects.
- Compare all frozen1,025-token retained K/V values and eight full logits
  in both schedules at the original `1e-5 + 1e-4 * abs(reference)` gate.
  Require every greedy decision, exact state and no omitted coordinates.
  Preserve mutation, source, ownership, allocation and6GiB process gates.
- Freeze the holdout to 651 raw IDs from `benchmarks/gemma-quality-v1/held_out_loss.u32le`
  at indices `[2048, 2699)`. The slice SHA-256 is
  `64ff1c8f61b4546ab26bfffee1a23a50cc1ff4bba73e400a06ce77e3a277839c`.
  Compare anchors0/31/127/433/511/512/649/650, complete producer chronology
  through the window boundary, both native schedules and exact greedy/state.
- Retain original-relative loss and isolated task grading for the same
  numerical engine and require both before managed Mac adoption. Any
  exploratory rejection calls for localization on identical operands.

The [ordered exploratory run](../benchmarks/results/2026-11-06-gemma-bf16-ordered-exploratory-v1.json)
sealed and reproduced the same26 V failures. The first captured cache drift
appears at producer1, position70. Preserve the original chunk64–95 and
trace producers0–1 at69/70/71, including normalized input, rotated Q,
context, O projection, PLE input and layer output. Bind causal K/V to the
authenticated wrap and localize the first divergent operation before changing
any engine arithmetic. Keep the651-token holdout unused.

The [exact witness](../benchmarks/results/2026-11-06-gemma-bf16-first-drift-v1.json)
locates the first difference at producer0 output projection column1093. The
exact dot is `17/2^53` above the midpoint. Add a BF16-only third expansion term
and midpoint-aware store. Keep affine4/8 separate. Verify authored widths,
cancellation and subnormals, this dot and the frozen96-token prefix first.
Retain a passing candidate only after full fidelity, state and quality reruns.

The [fresh numerical evidence](../benchmarks/results/2026-11-06-gemma-bf16-rounding-numerical-v1.json)
passes all45,000,704 values and32 greedy checks across the wrap and distinct
holdout. The original dual-reference rejection remains. The [corrected quality
gate](../benchmarks/results/2026-11-06-gemma-bf16-rounding-quality-v1.json)
also passes all10,000 loss targets and51/100 versus52/100. Retain the numerical
repair and proceed to the audited owner. Managed Mac acceptance stays separate.

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
freeze its producer/consumer map and test a last-row generation path.
Execute all rows through the final KV producer. In the consumer tail,
preserve the staged KV view and absolute query position, and carry the last
hidden row with its token-major PLE slice. Publish every admitted token and
producer proposal only after the tail and head succeed. Keep full-row
forward for scoring and fresh trajectory replay.

Start with exact tiny-graph last logits, retained KV, overflow, tail/head
failure and cancellation across chunk and ring boundaries. Repeat with the
selected checkpoint. Compare completed prefill timing and physical owners
on the same pinned workloads before retaining the change.

**G3 MTP.** The prospective QAT assistant has24 affine4 matrices and a
262,144-entry ordering tensor. The target and draft share every base piece
ID. The target adds a video control. Pairing to this target revision needs
an interface and proposal check. [Header evidence](../benchmarks/results/2026-11-06-gemma-qat-assistant-header-v1.json)
and [pinned assistant](https://huggingface.co/mlx-community/gemma-4-E2B-it-qat-assistant-4bit/tree/7acb8d02364777be103dbf8c8dcb007f55f717e6).

Pin the installed interface, target embedding/hidden state and shared
local/global KV inputs. Authenticate the draft and independently check the
complete token permutation, centroid top32 ties and output IDs. Use the
target tokenizer and keep draft state separate. Verify exact target greedy,
rejection, partial acceptance, hidden stops, cancellation and ring
rollback before measuring acceptance rate, latency and memory. [Reference interface](https://huggingface.co/docs/transformers/model_doc/gemma4_assistant).

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

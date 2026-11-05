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
Astra advises requiring a concrete source/precision hypothesis before further
whole-graph work.

### Gemma4 adoption task gate

Use the failed 100-task collection and the stock decisions to distinguish artifact quality from engine errors. Screen a concrete candidate for combined memory fit before payload acquisition. Freeze candidate, baseline and grading policy. Repeat the unchanged 10,000-target loss and 100-task adoption gates before enabling the route.

The [Q6 checkpoint](gemma6-feasibility.md) fits the stock worker and matches
original BF16 at the four disputed decisions. The complete trajectory gate
still rejects after bounded recovery and an independent causal-prefix
diagnostic. Defer this checkpoint under the current policy. A different
candidate needs a new immutable source and unchanged full quality gates.

### Qwen2.5-1.5B MLX8 original-weight quality

The [short alignment](../benchmarks/results/2026-11-06-qwen15-original-alignment-v1.json)
binds the original FP32 baseline and shows bit equality between native chunk
schedules. Score the unchanged 10,000 causal targets in 1,025-position blocks,
then replay and grade the frozen 100 tasks against the matched original
responses. Keep each owner's observed peak below 6 GiB. Managed Chat requires
both gates, audited artifact/text admission and a separate Mac journey.

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

Wait for the selected Gemma graph to pass its required gates. Validate checkpoint-defined shared layers and output semantics separately for prefill and MTP. Compare committed state and emitted decisions, then qualify any measured benefit with complete owner accounting.

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

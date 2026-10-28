# Qwen3 full-window delivery after native-profile qualification

**Deferred historical track:** the user changed the goal on28 October to
[shared-block architecture coverage and per-model capacity/KV tuning](shared-block-model-plan.md).
The8192 run was [interrupted and remains unqualified](qwen3-context8192-experiment.md).
The work below is preserved; it is not the current execution queue.

This track followed the [Qwen3 queue](qwen3-mac-delivery-plan.md). Q4's frozen current-native
qualification is complete. This Astra investigation prepares Q6; it grants no
larger app limit or full-window qualification by itself.

## F1 — Scoped diagnostic admission and resource planning

**Mechanism complete on28 October2026.** [Admission and evidence](qwen3-context-admission.md)
cover the scoped route, pure resource controls and authenticated successful source
opening. Long-context numerical execution and app admission remain unqualified.

Add an explicit Qwen3-only long-context diagnostic route through its same-document
declared40960 positions. Preserve ordinary512, historical quality-diagnostic1025,
other-model and scalar CPU limits. Do not globally widen the32768 ceiling merely
to make this model pass. Keep any validated longer-bound authority private to the
diagnostic model/session path. Test exact40961 rejection, old-path compatibility,
checked allocation arithmetic, budget equality/underflow and per-buffer bounds
without allocating full numerical buffers. C/Swift ordinary admission follows
qualification, rather than becoming an implicit diagnostic bypass.

Use the existing Half KV and Reference attention. A new kernel, Online attention
or cache quantization is not a prerequisite. Optimize only after measuring a
delivery bottleneck.

## F2 — Boundary qualification and a bounded source reference

The [4096-position diagnostic](qwen3-context4096.md) now passes its finite native
state/schedule/generation, independent controls,1024-target BF16 quality and
whole-process memory gates. The scoped result leaves ordinary app512 unchanged.
The measured scheduling experiment below is retained for its scoped4096 endpoint.
Proceed to the separate8192 diagnostic; later
boundaries and the full model window remain open.

Use1025 as the existing anchor, then4096,8192,16384,32768 and40960. Freeze input
IDs, candidate/source profiles and criteria before execution. Require deep-offset
rotary/operator controls under the original primitive gates, long causal/GQA
controls, exact native schedules/prefix/state/replay, selected-capacity+1 rejection
and whole-process memory. Old512 hashes establish continuity, not deep accuracy.
Measure completed chunk latency at4096 before estimating the remaining duration.

The current pure planner projects the following Reference/chunk32 allocations,
including native metadata allowance, using633366062 resident weight bytes:

| Capacity | Predicted numerical allocation |
| ---: | ---: |
|4096|1.127GiB|
|8192|1.580GiB|
|16384|2.486GiB|
|32768|4.299GiB|
|40960|5.205GiB|

At40960, KV payload is4697620480 bytes; allocated KV including guards is
4697621264 bytes. Chunk32 score/probability payload adds160MiB plus56 guard bytes.
Chunk128 instead needs640MiB for these buffers and predicts5.732GiB before
text/application costs. Use chunk32 for the proposed full-window Mac profile.
A candidate partition is5.25GiB numerical,256MiB tokenizer/text and512MiB app
allowance within6GiB. This is a hypothesis; actual planner and lifetime peaks must
pass. Do not transfer that budget to iPhone.

Preserve the original full10000-target/100-task quality policy for final current
candidate binding. Add prospectively frozen held-out teacher-forced suffixes at
the named depths, with the same0.05-nat mean-loss allowance at each depth. Keep
fixed long-context task probes as separate observations; do not derive success
thresholds or repair outputs after seeing them. Intermediate admission checks do
not automatically require rerunning the complete100-task benchmark every time.

The unquantized BF16 reference needs a separate checked long execution profile.
Pinned MLX-LM's ordinary KVCache can preallocate once with instance `step=capacity`;
guard offset+rows before updates and return only the valid prefix. Keep original
BF16 modules and full history. Rotating/trimming or quantized caches are unsuitable
substitutes. Do not assume slice assignment donates storage or that a lazy graph
retains only one old/new cache pair.

A small reference-only adapter can evaluate/synchronize the full cache state after
the original update, discard temporary views, then return a fresh valid-prefix
view. Evaluate/synchronize each original transformer block as well. Freeze chunk8;
skip output projection on wholly unscored prefix chunks, but project the whole
eight-row batch when scoring and select target rows afterward. Slicing hidden
rows before projection changes its batch shape and source arithmetic.

First require exact same-chunk logits and all-layer valid BF16 K/V equivalence
against stock source execution, including255/256/257 and1024/1025. This does not
promise equality to the old cache=None/full-block baseline. Then run a bounded
single-layer40960-key workspace probe before a full-cache allocation. Resident
weights, full KV, one layer's old/new overlap,256MiB framework allowance and64MiB
workspace provisionally total6393036800 bytes, just49414144 below6GiB; workspace
is not proven. Reject admission if the checked ledger fails. A Metal memory-limit
hint cannot replace that proof or owned external lifetime measurement.

A subsequent Astra ledger audit identifies a missing separate host-reader reserve
in that provisional total. Keeping the current 512MiB framework and 160MiB host
allowances does not admit a full40960 source run. A proposed reader bounded by
construction to32MiB, with a separately qualified256MiB non-buffer allowance,
would raise the provisional total to6426591232 bytes, only15859712 below6GiB.
Neither smaller allowance is established by a successful smaller-context peak.
Require a concrete bounded-reader/lifetime design, unique backing-allocation
accounting and the finite workspace/process probe before using that candidate
ledger. The current4096 experiment keeps its original conservative reserves.

### Measured Reference bottleneck before8192

**Closed, retained on28 October:** the [exact-order parallel schedule](qwen3-reference-schedule.md)
passes complete F2/current-quality bridges and two fixed opposite-order3+10
performance runs. Primary medians improve6.8801→1.2322s and6.8830→1.2324s;
minimum reduction82.08996% exceeds maximum between-run variation0.04164%, with
no required-metric regressions and maximum physical footprint1669318336 bytes.
This is cached-tail/short-request evidence; Serial remains default and app512
unchanged. Next independently qualify8192 using a separate API/protocol, unchanged
4096 rejection and historicalA prefix, fresh disjoint deep quality and bounded
whole-process checks. Prepared private proposals alone grant no larger window.

The frozen4096 experiment's completed chunk32 diagnostic pass records463.1623
seconds of completed model calls, excluding load and diagnostic host processing.
These are individual planning observations, not a performance benchmark. The
current shader gives one query/head lane serial QK and PV reductions; call time
grows with valid cache depth. No per-layer host wait or whole-cache copy explains
this observed cost.

After F2 closes successfully, Astra recommends one bounded scheduling experiment
before8192. Preserve the serial Reference implementation as oracle and default.
An explicit experimental Qwen3 diagnostic schedule may split score, stable
softmax and context into three GPU dispatches, parallelizing independent outputs
while preserving every reduction order, division, mask and Half rounding. Reuse
the same score/probability buffers and command dependency chain. Bound capacity
and absolute positions to the admitted4096 endpoint; this is not a new arithmetic
profile or an automatic larger-window rollout.

Require exact independent controls, the complete frozen F2 row/state/quality
bridge and current Q4's complete10000-target/100-task output bridge. Retain the
experiment only after representative completed whole-model calls and short
generation requests satisfy the original warmup/repetition/confirmation policy,
memory accounting and practical gain criteria. Deep prefixes may be prepared
once and safely replayed to measure named tail calls; record that replay cost.
Such a benchmark cannot claim full4096-request throughput. Revert complexity
that fails identity or useful measured benefit. Do not switch schedules during
the current frozen F2 execution.

## F3 — Bounded real streaming and Mac application admission

Connect the existing Rust incremental response stream to the real native worker
and Swift delivery path. The current worker's collected-token/whole-decode path,
Chat512-input/256-output/64KiB text,16KiB draft and Expert128-output policies do
not deliver the model window. Derive token budgets from the admitted shared
window; separately reserve decoder carry, response/event batches, retained token
IDs, UI snapshots and serialized history. Keep explicit byte ceilings and atomic
overflow/draft/prefix behavior. Do not merely increase the256 constant.

After the corresponding boundary qualifies, expose that admitted partial bound
truthfully in Chat/Expert and context facts. Complete full40960 input/output,
normal stop, cancellation/resume, persistence/reopen and rejected40961 journeys
with whole-process memory before declaring Q6 complete. Preserve the historical
fidelity rejections and weak absolute task-quality limitation throughout.

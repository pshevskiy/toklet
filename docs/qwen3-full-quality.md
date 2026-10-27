# Qwen3 Mac full quality comparison — 27 October 2026

**Q3 completed its full workload. Retain native Half as an experimental candidate
under the unchanged relative regression policy.** Numerical C512 qualification
remains rejected and app limits remain 512 tokens.

**Absolute strict task quality is weak: native 3/100, baseline 5/100. The two-case
loss is exactly the allowed limit.** This provides little evidence of useful
instruction-following quality and no margin below the task threshold.

## Complete measured results

The frozen [contract](qwen3-quality-contract.md) supplies all 10,000 WikiText loss
targets and 100 equally weighted tasks per profile. No input, target block or
requested output was clipped to the 512-position app window.

| Measure | Original BF16 baseline | Native eight-bit/Half | Acceptance |
| --- | ---: | ---: | --- |
| Mean loss, nats/token |3.248645013938721|3.252649075779492|Increase 0.004004061840770845 ≤0.05: pass|
| HumanEval literal completion |5/40|3/40|Included in100-case aggregate|
| TruthfulQA exact letter |0/30|0/30|Included in100-case aggregate|
| LAMBADA exact first word |0/30|0/30|Included in100-case aggregate|
| Total strict task score |5/100|3/100|Loss 2 percentage points ≤2: pass at boundary|

All 200 outcomes resolve: 80 code programs through the existing isolated Linux
evaluator and 120 text outcomes through its unchanged literal rules. Native
passes HumanEval/27,/30,/34; baseline also passes/2 and/15. Original source
tests are invoked. No generated code executes on the Mac.

Formatting contributes heavily to the low score. Code responses begin with
Markdown fences in 25/40 baseline and 31/40 native cases despite requests for
indented completions. All 30 prose responses in each profile contain more than
one uppercase letter. These observations introduce no fence repair, answer
extraction or regrading. Complete visible token sequences agree on 60/100 cases.

## Frozen execution and resources

The native candidate uses frozen Rust/shader/Cargo/worker bytes at revision
`477a536`, the pinned Qwen3-0.6B eight-bit checkpoint, MlxAffine projection,
Compensated normalization, Reference attention and Half activations/KV. The
baseline is the verified official BF16 checkpoint described in the contract.
Configs match excluding quantization and both consume identical frozen IDs.
The converter's omission of a source revision remains explicit.

The unchanged MLX-LM Qwen3 graph is a reference only: MLX 0.32.3, MLX-LM 0.32.0,
NumPy 2.3.2 in the authenticated MLX-only environment. Static strict loading uses
no dtype override, adapters or remote code. Its 596,049,920 tied BF16 parameters
occupy 1,192,099,840 bytes; original sanitization removes the duplicate file-level
output head and the graph uses the embedding as its output matrix.

The [expected plan](../benchmarks/qwen3-quality-v1/reference-plan.json) and
[actual probe profile](../benchmarks/qwen3-quality-v1/reference-actual-profile.json)
are frozen before held-out execution. Source loss uses batch1/cacheNone/full
blocks; native loss uses 128-row chunks and all-row snapshots. Every target is
scored once, positions reset by block, and the final input commits without
scoring its next distribution. Per-target IDs, losses, target bits and distribution
hashes remain in authenticated journals.

Tasks reset for each request and prefill 128 tokens at a time. Source BF16 logits
are widened bit-exactly before stable FP64 statistics and lowest-ID greedy
selection from raw logits. This avoids adding rounding ties through a reduced-
precision log-softmax wrapper. Original BF16 KVCache grows in 256-position steps;
its admission includes rounded capacity and growth overlap. EOS 151645 is hidden
but consumes a sampled/pending slot. Caps stay 256/32/16; the largest request
needs 545 shared positions. Native weights remain immutable and cache/scratch
remain session-owned. The production engine stays Rust/Metal.

Runs are sequential on Apple M1 Pro/MacBookPro18,3/16GiB,
macOS 26.6.2(25G83), with release Rust 1.90/Metal for the native candidate.

| Completed process | Peak physical bytes | Maximum RSS bytes |
| --- | ---: | ---: |
| BF16 source full loss |2257569544|1376714752|
| Native full loss |1032733896|743538688|
| BF16 source 100 tasks |1681360816|1335476224|
| Native 100 tasks |864453760|605700096|

Every model process fits 6 GiB. Source admission reserves 5,927,671,248 bytes with
separate load/inference phases and conditional copies distinguished from actual
physical allocations. Native loss/task admission including separate application
allowance is 1,144,712,236/987,556,630 bytes.

All seven collection processes/groups exit before decoding/grading. Source,
model, runtime, input and journal endpoints pass. The qualified rootless image,
client/server versions, interpreter, containment, literal construction and limits
remain unchanged. All 80 containers are removed. The initially stopped 2 GiB VM is
restored to stopped; worker helper lifetime and independent outer verification
finish before publishing the decision. Grader-host peak physical/RSS is
200,000,736/286,031,872 bytes; guest reservation is separate. A whole grading-phase
physical peak is not measured. These observations are quality/resource measurements; latency remains separate.

## Checks and remaining scope

Nineteen cold regression groups cover stable reference math, finite/shape/full
coverage admission, early EOS, partial writes, complete replay, exact limits,
unresolved outcomes, file growth, post-decision failures and VM-aware supervision.
Cancellation requests TERM and retains the owner through cleanup without
forcible killing. Final CPU suite, formatting and strict default/Metal Clippy pass.

The first authored probe caught a loader callback keyword mismatch before model
construction or held-out scoring. Its failed evidence remains. The callback
parameter now matches the unchanged loader API; both probes then pass. No
numerical kernel, quality threshold or frozen input changed.

Astra accepts Q3 closure as **relative-quality acceptance of an experimental
candidate**. C512 numerical fidelity, useful instruction-following quality,
40,960-position delivery, app expansion, latency, other checkpoints and iPhone
qualification remain open. The historical stage contract stays rejected.
Next follow the bounded [Q4 queue](qwen3-mac-delivery-plan.md). See
[public evidence](../benchmarks/results/2026-10-27-qwen3-full-quality-v1.json).

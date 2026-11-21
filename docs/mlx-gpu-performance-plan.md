# Beat the MLX GPU baselines

Updated 22 November 2026. The competitive goal is active. Released LFM3
`4b50550a` improves warmed completion by 21–22% against its Toklet predecessor.
The controlled external comparison shows a 9.63× first-ID and 6.02× completion
gap to vLLM Metal. Nanbeige4.2-3B is the next model admission target. Prior
performance and adoption decisions retain their original scope.
[Released LFM3 evidence](improvements.md#lfm3-metal-sampled-decode) ·
[Controlled comparison](../artifacts/mlx-gpu-gap-2026-11-15/sol62-lfm3-trial5-pushed-controlled-clock-2/join-normal.json).

G1 and the bounded G2 producer pilot are delivered.
Astra approves the three-layer numerical, state and required checks. The
[cost map](gemma-current-cost-map.md) identifies
CPU attention and completion waits as the next targets. The
[reusable attention building block](gemma-gpu-attention.md) prepares shared-buffer
block composition; [Apple hardware priorities](apple-silicon-gpu-goal.md) focus on
GPU residency, unified-memory reuse and bounded ANE feasibility. Astra research
and Sol implementation proceed in parallel. Hardware measurements stay serial.
The LFM3 placement improves internal warmed latency. External wins, model
adoption and overhead qualification remain open.
[Delivery outcomes](improvement-outcomes.md).

## Rust and Metal architecture proposal

### What caused the measured gap

On the 16 GB M1 Pro, the controlled BF16 LFM3 trial uses 512 uncached prompt
tokens and 31 matched decode forwards. The vLLM clock includes a forced token
at sample eight. One owner per engine and different final unconsumed samples
limit this to a controlled diagnostic. The ordinary confirmation is open.

| Client completion boundary | Toklet | vLLM Metal | Toklet elapsed time ratio |
| --- | ---: | ---: | ---: |
| First token ID | 4.660 s | 0.484 s | 9.63× |
| First ID through Done | 2.500 s | 0.707 s | 3.54× |
| Request completion | 7.170 s | 1.191 s | 6.02× |
| Guarded peak under each owner's measurement scope | 3.48 GiB | 5.27 GiB | — |

About 70% of the wall gap accumulates before the first ID. On a separate
released decode trace, 2.504 s of engine time decomposes into 1.419 s of GPU
execution, 0.703 s of host wait outside GPU execution, 0.210 s of CPU attention
and 0.172 s elsewhere. Eliminating every nondevice decode cost would still
leave 1.419 s. The prefill diagnostic assigns 2.58–2.59 s to measured
device intervals, 0.27–0.29 s to wait outside the device and 1.75–1.77 s
outside captured waits, including unclocked stages. These diagnostics have
separate scope and cannot be joined into one
request breakdown. [Decode trace](../artifacts/mlx-gpu-gap-2026-11-15/sol62-lfm3-release4b-dec-readback-1/normal.json) ·
[Prefill trace and bounds](../artifacts/mlx-gpu-gap-2026-11-15/sol62-lfm3-prefill-first-token-budget-1/normal.json).

Toklet's strict BF16 route preserves a particular CPU accumulation order and
validates results at several producer/consumer boundaries. In the isolated
T128 gate/up test, MLX takes 2.28 ms for the same operands. Toklet takes
9.29 ms on-device and 11.98 ms through validation. The results differ in
480 of 1,966,080 BF16 words. The MLX engine selects tiled GEMM for batch
shapes and evaluates dependent work together. vLLM Metal uses MLX, paged
attention and ShortConv for this LFM3 route. This explains the measured
opportunity in arithmetic and scheduling. Public vLLM request clocks do not
attribute its time to individual kernels. [MLX projection experiment](../artifacts/mlx-gpu-gap-2026-11-15/sol62-lfm3-mlx-dense-t4-pair-2/run-1/normal.json) ·
[Toklet Panel84 verification](../artifacts/mlx-gpu-gap-2026-11-15/astra-lfm3-panel84-pair-actual-review-1/REVIEW.md) ·
[MLX shape dispatch](https://github.com/ml-explore/mlx/blob/v0.32.4/mlx/backend/metal/matmul.cpp#L1449-L1576) ·
[LFM3 in vLLM Metal](https://github.com/vllm-project/vllm-metal/pull/1212).

### Production execution on Apple Silicon

Extend the existing family-adapter and capacity plan with a prepared Metal
execution path. [Shared blocks and capacity](shared-block-model-plan.md).

| Component | Proposed boundary |
| --- | --- |
| Model adapters | Parse and validate pinned checkpoint config, tokenizer and tensor layouts once. Lower concrete blocks with explicit activation and weight precision, RoPE and state semantics. Keep optional image and audio encoders and their preprocessing behind separate adapters that produce bounded decoder inputs. |
| Execution plan | Construct immutable prefill and decode plans from model blocks, shape, precision and supported Metal capabilities. Resolve buffer ranges, scratch lifetimes and pipeline handles during preparation. Select concrete kernels outside their inner loops. Begin with the blocks required by the pilot. |
| Prefill | Use tiled GEMM and fused unpacking of packed weights, resident GQA/SDPA, and fused pointwise operations where measured. Encode projection, attention and FFN together within each admitted chunk while keeping state private. Increase the internal prefill tile window only after checking causal offsets, cancellation, cache publication and peak memory. Keep required logits at existing public checkpoints available. |
| Decode | Use separate packed GEMV/vector kernels. Keep K/V and recurrent state resident. Encode the token through the sampler and read back the selected ID and status at a checked completion. Keep cancellation, error and retry eligible for exact prefix recovery. Full logits and stages remain available in diagnostics. |
| Unified memory | Use one immutable backing allocation per bank. Borrow or map it directly when platform alignment, lifetime and residency permit it. Otherwise stage it once within the load budget. Reuse session storage and activation scratch by lifetime. Maintain explicit GPU dependencies and keep state proposals private until the owning command completes successfully. |
| Apple devices | Share the Rust planning, state and Metal core across arm64 macOS and iOS. Query device and OS capabilities before selecting pipelines. Layer a small lifecycle integration over it for iOS cancellation and memory pressure. Admit iPhone context, quantization and scratch against an independently tested device budget. Measure sustained speed, memory and thermal behavior on physical devices before claiming support there. |

Dense and sliding-window decoders share attention, projection, norms,
activation and residual kernels. LFM adds causal convolution state. Nanbeige
repeats the same weight-backed blocks with separate K/V in each loop. Hybrid
linear attention and MoE can add their own typed state and grouped execution
when a model under qualification requires them. Preserve distinct cache
positions per sequence and per logical layer, and tag reusable prefixes with
model, precision and position semantics. Leave use of the Neural Engine
conditional on an independently instrumented Core ML prototype and a
measured latency/memory benefit. [Metal synchronization](https://developer.apple.com/documentation/metal/resource-synchronization) ·
[Metal memory recommendation](https://developer.apple.com/documentation/metal/mtldevice/recommendedmaxworkingsetsize) ·
[iOS process budget](https://developer.apple.com/documentation/os/os_proc_available_memory) ·
[Core ML device placement](https://developer.apple.com/documentation/coreml/mlcomputeplan-1w21n).

### Implementation order and decision points

1. **Measure a direct MLX reference before changing the engine.** Replay the
   exact LFM3 diagnostic tape using a local MLX owner without the HTTP serving
   loop. Freeze arithmetic, phases, output and work contracts first. Compare
   through the same first-ID and Done boundaries. Attribute HTTP and serving
   overhead only in a same-worker comparison that preserves the vLLM forward
   path, logits coverage, effective shapes, state and intervention. Record engine
   completion, client/output overhead, commands, readbacks and peak memory.
   Use the frozen three warmups and ten observations in both owner orders.
   A 5% edge over the controlled vLLM result would take about 1.131 s, roughly
   60 ms below that measurement. If direct MLX leaves little margin, the Rust
   implementation needs a measured kernel or memory-traffic advantage.
   The actual same-worker runs verify all 52 requests. HTTP and direct median
   completion measure 1.289/1.284 s and 1.288/1.286 s in opposite orders.
   Paired 95% intervals include zero and their upper ends are below 5%.
   Keep the server path as a benchmark boundary and work on GPU execution.
   [Completed replay](../artifacts/mlx-gpu-gap-2026-11-15/root-lfm3-same-worker-replay-1/normal.json).
2. **Implement one connected Rust/Metal block.** Price matmul, attention and
   FFN on the same authentic operands at T128 and T1 with scoped completion
   clocks. Compare against the existing route and the independent MLX path.
   Use both reversed orders and scale disjoint savings by actual shape and
   layer counts before committing to an end-to-end backend. A shorter submit
   clock alone does not justify retention.
3. **Qualify a separate production arithmetic route.** Keep the current
   exact CPU/Metal references and original gates for existing routes. Freeze
   the new packed format, accumulation and activation rules before running
   the reference comparisons. Require independent numerical, loss and task
   checks, exact token/cache correspondence where specified, and cancellation,
   retry, reset and memory-pressure recovery. An unsupported or failed route
   stays out of the default. Unsupported precision must be explicit.
4. **Confirm full request behavior.** Measure initial load separately. Measure
   TTFT, first visible/answer latency, decode throughput, completion and peak
   memory on the same weights, tokenizer, token work, precision and cache
   policy. Confirm the existing 5% requirement in both cohorts. Keep request
   summaries bounded and record stage completions at existing synchronization
   boundaries. Keep high-resolution GPU tracing opt-in. Apply both original
   marginal and paired 95% overhead bounds of 1% to TTFT and completion.
   The historical LFM3 marginal first-ID upper bound is 1.065%. It remains
   open across this architecture change. [Overhead audit](../artifacts/mlx-gpu-gap-2026-11-15/sol62-lfm3-o4-pre-first-audit-1/RESULT.md).

### Next model admission

**Nanbeige4.2-3B** is the intelligence-first text target for this Mac. The
model card reports 87.4 GPQA Diamond, 72.5 LiveCodeBench v6 and 63.6
SWE-bench Verified in its thinking-mode setup. In Artificial Analysis's
separate 16K evaluation of phone-scale quantized models it shares the lead
with LFM2.5-2.6B at 63 and scores 65 versus 64 at the 64K limit.
LFM2.5-2.6B answers substantially faster in the separate device test. This
selection weights reasoning and coding alongside memory headroom. Those
evaluations use other runtimes and quantizations. Preserve a fixed-token
engine benchmark and a separate quality test with bounded answer time.
[Model card and evaluation setup](https://huggingface.co/Nanbeige/Nanbeige4.2-3B) ·
[Independent quantized evaluation](https://artificialanalysis.ai/articles/mobile-phone-intelligence-inference).

Start from `MercuriusDream/Nanbeige4.2-3B-mlx-4bit` at
`c21f850c4096b67a1b53a2c8eb4f8c8652f89375`. The repository lists 2.35 GB
of weights. The model has about 4.17B total parameters, including its
embeddings and head. Its 22 physical layers run twice. The 44 independent
KV sets with eight heads of width 128 require 352 MiB of raw BF16 K/V
at 2,048 positions. Preserve the 6,144-wide Q projection and each pass's
final norm. Compare both loop-boundary residuals, distinct cache slots,
and cached decode against full recomputation under two prompt partitions.
Validate allocator and load transients within the 6 GiB ceiling. Merged
upstream MLX-LM support provides an independent reference. The exact build
still needs loaded parity and support checks in Toklet, vLLM Metal and
Ollama MLX. Use current LFM3 evidence to calibrate the engine. Compare
Nanbeige on external engines only after the same artifact and arithmetic
are verified. The modified Heretic MLX variant can qualify separately if
it preserves reasoning, coding and tool-call quality on the same tasks.
The pinned model and tokenizer are installed locally. The experimental
Rust and Metal CLI checks both greedy anchors and all 44 independent cache
owners at two prompt partitions. Native logits differ from MLX. The next
operator test isolates the first layer-zero V divergence using authentic
split-K operands. App adoption and external engine speed tests are pending.
[Native admission results](../artifacts/mlx-gpu-gap-2026-11-15/root-nanbeige-mlx-q4-admission-1/RUN.md).

Test the resident layer-zero V matrix with identical BF16-normalized inputs.
Accumulate four contiguous 768-wide products at T128 and eight 384-wide
products at T64. Round each partial to BF16 and round each ordered addition,
initially inside the existing output tile. Check the 64-channel witness, the
full V output against MLX at both shapes and the separate T1 control.
The implementation matches every frozen 0.32.3 word and costs GPU time.
The newer installed MLX 0.32.4 stack keeps FP32 subtotals. On identical
operands its T128 and T64 output panels differ in 11 of 131,072 words.
The existing unsplit Toklet route stays much closer to these modern panels.
Keep the previous BF16 split arithmetic as a reference diagnostic.
[Timed operator and reference versions](../artifacts/mlx-gpu-gap-2026-11-15/root-nanbeige-prefill-work-1/RUN.md).

Gate, up and down account for 69% of per-block projection multiplies.
The retained 32-row tile keeps the same accumulation order for each output.
At T128 it halves their tile-weight reconstruction count across all 44 block
visits. The exact T64 and T128 operator checks and the connected 512/32
workload pass. M1 Pro completion improves 10.10% in both orders. The full
request runs use four T128 chunks at 2.47 GiB peak and preserve token IDs,
committed KV and terminal logits. Assess additional shapes through the
same numerical, latency and memory gates.
[Pinned FFN operands and tile work](../artifacts/mlx-gpu-gap-2026-11-15/root-nanbeige-prefill-work-1/RUN.md).
[Connected results](../artifacts/mlx-gpu-gap-2026-11-15/root-nanbeige-ffn-connected-1/normal.json).

The subsequent direct MLX graph reproduces all 32 sampled IDs in
two warmed owners. It completes in 3.315–3.316 s beside the released
Toklet result of 5.841–5.842 s. Of the 2.53 s difference, about 92%
accumulates in prefill. The T128 gate and up projection references
match all 2,752,512 BF16 words with completed times of 3.21–3.22 ms
in MLX and 5.36–5.37 ms of device time in Toklet. The retained
compact gate/up tile uses a smaller threadgroup and lane-local output.
A signed Q4 basis qualifies the compiled Mac pipeline before it
can select T128. Both connected orders finish about 7% sooner
on the M1 Pro and match IDs, committed KV and terminal logits.
Qualify further shapes with the same gates.
[Direct evidence](../artifacts/mlx-gpu-gap-2026-11-15/root-nanbeige-modern-decode-1/RUN.md) ·
[Connected compact result](../artifacts/mlx-gpu-gap-2026-11-15/root-nanbeige-ffn-compact-connected-1/RUN.md).

For request-summary qualification, compare the same CLI output mode with
summary collection enabled and disabled. Keep counters at existing completion
boundaries and write one bounded summary after the last token. Sample command
and GPU timings in a separate diagnostic run. Verify the paired and marginal
95% upper bounds for first-token and completion overhead in both owner orders.
[Pinned MLX package](https://huggingface.co/MercuriusDream/Nanbeige4.2-3B-mlx-4bit/tree/c21f850c4096b67a1b53a2c8eb4f8c8652f89375) ·
[Upstream config](https://huggingface.co/Nanbeige/Nanbeige4.2-3B/blob/main/config.json) ·
[Publisher execution order](https://huggingface.co/Nanbeige/Nanbeige4.2-3B/blob/main/modeling_nanbeige.py) ·
[Merged MLX reference](https://github.com/ml-explore/mlx-lm/pull/1597) ·
[Heretic MLX package](https://huggingface.co/WaveCut/Nanbeige4.2-3B-heretic-MLX-DWQ-4bit).

## Goal and completion rule

Make Toklet faster than both Ollama MLX GPU and vLLM MLX GPU on the same named
Mac, beginning with Gemma 4 E2B and then LFM3-1.7B. These are distinct model
tracks; success on one does not qualify the other.

Freeze the exact shared active model tensors, tokenizer, precision, positions,
cache policy, workload and timing boundaries before taking acceptance samples.
For each track, require at least 5% lower median completed-request latency than
each external engine in both independent confirmation cohorts, with the paired
95% lower reduction bound above 5%. Report TTFT, prefill and decode separately;
do not conceal a regression by pooling workloads or changing denominators.
Any unexplained required-workload regression prevents retention under
[PLAN.md](../PLAN.md#validation-and-performance-policy). Record cold loading
separately. A failed comparison or model rejection leaves this goal open.

Request summaries must separately meet the original 1% limit: both marginal
overhead and the paired 95% upper endpoint at most 1%, for TTFT and complete
requests, in both cohorts and each retained submission policy. Preserve the
existing 16 KiB fixed collector and 1 MiB total retained-storage bounds, explicit
overflow/errors, and no extra GPU submissions, waits or numerical allocations.
The 6 GiB engine ceiling includes weights, KV, scratch, codec, telemetry,
temporary resources and the existing helper reserve. Memory saving is a
separate measured claim, not a requirement inferred from faster execution.

## Apple Silicon execution sequence

Prioritize the GPU and shared physical owners on the measured M1 Pro. First
qualify the coupled local/global/shared-KV chain; then move the complete Gemma
execution into the ordinary CLI with one checked publication boundary per
chunk/token. Keep immutable weights, committed KV, private proposals and scratch
resident, pass checked buffer ranges between encoders, and count actual commands,
waits and host bytes rather than treating shared memory as automatic zero-copy.

Use reusable layer pools before extending the diagnostic layout to all 35
layers. Static accounting finds 1,101,144,264 bytes for 35 separate layer arenas
and CPU state shadows, excluding weights, committed KV and graph I/O. Proposed
separate one-row and 32-row pools use 843,440 and 26,778,288 bytes; proposals, KV,
transaction receipts and bounded CPU replay storage remain additional owners.
These are source-derived allocation estimates, not measured memory savings.
Audit each node on the GPU before its storage is overwritten, retain persistent
per-node fault receipts, and publish both donor caches only after all checks.
Reusing storage must preserve the original numerical, guard and failure gates.
The bounded 13→14→15 pool now passes numerical/state/allocation and required
checks; Astra approves scoped retention. Full 35-layer/ordinary CLI remain open.
[Qualified pool scope](gemma-gpu-pool.md).
[Reproducible pool/layout research](../artifacts/mlx-gpu-gap-2026-11-15/astra-resident-scratch-reuse-research-1.md).

The 22-operation own-input RMS attribution identified a scheduling bottleneck.
The two-phase artifact now passes 336 original/own-input/derived cases, exact
current-kernel output/status/reason parity, fresh scalar gates, 14 GPU faults and
normal/optimized readers. Two reversed 3-warmup/10-sample runs measure:

| Aggregate GPU time (ms) | Reference run 1/run 2 | Two-phase run 1/run 2 |
| --- | --- | --- |
| Prefill 22 norms |147.73 /149.12 |37.93 /37.77 |
| Decode 22 norms |131.67 /131.43 |38.08 /37.50 |

Both use one completed command/wait; the candidate has 44 dispatches versus 22.
This is an isolated aggregate result; no chain/request cost share or adoption
is established. Astra approves artifact retention with no required fixes. Before graph composition,
qualify persistent per-operation context/header banks, checked nonwrapping epoch
admission, complete paired preflight, partial-encoding failure/drain and late
publication. Preserve original reduction/output arithmetic, whole-chain replay
and global guard/fatal precedence. Sol's source4 private integration now passes
16 numerical cells, 28 lifecycle controls, 64 warm zero-Rust-allocation operations
and all 12 guarded independent readers. Both three-layer completed GPU-command
runs show about 69% lower prefill /68% lower decode time, adding 14,232 session
bytes. These timings cover this bounded graph only. All old regressions and
32 final required stages pass, including 1,050 Metal tests (403 ignored), strict
CPU/Metal Clippy and release builds. Astra approves private graph retention;
full-model, request and external gates remain open.
[Qualified graph/check/review scope](gemma-gpu-rms-scheduling.md).
[Actual graph replay](../artifacts/mlx-gpu-gap-2026-11-15/g2-rms-two-phase-graph-readers-run-1/result.json),
[hardware/lifecycle](../artifacts/mlx-gpu-gap-2026-11-15/g2-rms-two-phase-graph-hardware-progress-3.json).
[Actual candidate/check packet](../artifacts/mlx-gpu-gap-2026-11-15/g2-rms-two-phase-artifact-review-checks-1.json),
[Astra artifact review](../artifacts/mlx-gpu-gap-2026-11-15/astra-rms-two-phase-artifact-review-1/verdict.json),
[conditional integration](../artifacts/mlx-gpu-gap-2026-11-15/g2-rms-two-phase-integration-design-1/handoff.json).

Bridge the authenticated common-Q4 owner into resident execution before full
35-layer composition. Borrow its existing packed chunks and resident PLE; retain
Q4's float2 compensated dot,16,384 tile size and actual16-value codebook bound.
Do not reuse BF16's float3/8,192-tile proof by rounding Q4 bounds. Build fresh
Q4 own-input fixtures and preserve synchronous packed replay without a second
full model copy. The private borrowed-Q4 leaf now passes four build checks,
35 targeted GPU/reader/regression stages, 1,052 Metal tests (409 ignored) and
release CLI; combined Astra review approves scoped retention, no required
fixes. This covers existing immutable owners and bounded real/synthetic fixtures, without model/graph/route
or speed qualification. Full original common-Q4 quality remains a separate gate.
[Qualified leaf scope/checks/review](gemma-gpu-q4-projection.md).
[Concrete Q4 bridge](../artifacts/mlx-gpu-gap-2026-11-15/astra-common-q4-resident-bridge-research-1.md),
[first bounded encoding mechanism](../artifacts/mlx-gpu-gap-2026-11-15/astra-common-q4-first-encode-mechanism-1.md).

After full-graph correctness, profile the resident request again. Select prefill
weight reuse, GPU output selection or further fusion from that measurement;
retain each only when it improves completed requests at the original bounds.
Record CPU encode/validation time, completed GPU time, dispatch/command/wait
counts, host staging/readback bytes and physical owner sizes separately. Export
summaries outside timed numerical work and rerun the unchanged overhead campaign.

Neural Engine experiments remain conditional on actual public-API placement,
qualified precision and a measured benefit after conversion and CPU/GPU crossings.
FP32 graph residency on Metal is the immediate path; do not infer ANE execution
from a compute-unit setting or successful Core ML compilation.

Keep the current numerical repair bounded: native one-key replay identifies
GELU table interpolation independently of attention. Preserve the approved parent
table/proof and evaluate one finer high-curvature core before composing more
layers. The selected refinement appends 90,114 bins (2,162,736 bytes) over
the symmetric core; unchanged outside coverage is inherited with an explicit
composition proof. Native/core target validation, four independent raw readers and all corruption
controls pass. Astra approves the artifact composition; source8 integration,
eight-cell coupled/state validation and required checks pass. Astra approves
the bounded combined engine change, including the failure-count correction.
[Publication/source/CI](../artifacts/mlx-gpu-gap-2026-11-15/g2-deep-chain-publication-1.json).
[Core review/checks](../artifacts/mlx-gpu-gap-2026-11-15/astra-gelu-core-review-1/verdict.json),
[frozen selection](../artifacts/mlx-gpu-gap-2026-11-15/astra-gelu-core-refinement-research-1.md).
A Mac hostname change affects provenance text, not the mathematical platform;
keep kernel/OS/compiler/device checks and original sealed material.
[Native evidence](../artifacts/mlx-gpu-gap-2026-11-15/g2-deep-native-reader-progress-1.json),
[actual host binding](../artifacts/mlx-gpu-gap-2026-11-15/g2-deep-native-host-binding-2.json).

[Hardware/API constraints](apple-silicon-gpu-goal.md#neural-engine-feasibility).

## What currently explains the gap

The following are ranked engineering priorities, not an additive attribution
of the full cross-engine timing difference.

| Finding | Evidence | Consequence and limit |
| --- | --- | --- |
| LFM3's qualified graph is CPU execution | [CLI](lfm3-cli.md), [forward](../src/model/lfm3/forward.rs), [projection](../src/cpu/lfm3_projection.rs) | Its retained 512/32 request still takes 246.60/248.15 s; the external GPU runs take about 1.1 s. More CPU tuning alone is not a credible route to GPU parity. |
| Gemma is a hybrid CPU/Metal graph | [Forward](../src/model/gemma_packed/forward.rs), [execution recipe](../src/mlx/gemma4/bf16/execution.rs) | Only projections use this Metal backend. Attention, RMSNorm, rotary, GELU, residuals, softcap and graph state work return to CPU slices. The reference attention materializes scores/probabilities and uses FP64 arithmetic. |
| Gemma has hundreds of synchronization boundaries per evaluation | [Completed profiles](gemma-profile-cli.md), [projection adapter](../src/metal/gemma_affine/bf16/resident/graph.rs) | Legacy decode has 349 commands/waits. The retained opt-in lowers this to 277 while keeping 349 dispatches. Each projection still stages input, waits, validates and copies results to CPU before continuing. |
| The projection is a correctness-oriented dot kernel, also used for prefill | [Shader](../src/metal/gemma_affine.metal), [dispatch](../src/metal/gemma_affine/project.rs) | One SIMD group computes one token/output-row dot using a three-term compensated accumulator and product-error recovery. There is no cross-token matrix tile reuse in this shader. It does not exploit prefill as a tiled GEMM. Numerical replacement must pass the original bounds. |
| Validation, staging and full-logit work are repeated | [Projection path](../src/metal/gemma_affine/project/bf16_batch.rs), [LFM3 leaf](../src/cpu/lfm3_projection.rs), [loader](../src/weights/lfm3.rs) | Gemma fills and scans guarded scratch and copies each projection. LFM3's public leaf scans finite immutable coefficients again after loader validation. Gemma publishes 262,144 FP32 logits, about 1 MiB, per evaluation. Their individual time shares need measurement. |
| Some apparent gap is benchmark mismatch | [Gemma comparison](console-engine-comparison.md), [LFM3 comparison](lfm3-engine-comparison.md) | Old Gemma compares official BF16 against modified Q4; Ollama caches 14/15 prompt tokens. LFM3 shares source weights, but only 3/13 vLLM output-ID rows match CPU per cohort; Ollama authoritative IDs are unavailable. These records cannot establish a speedup factor. |

The two old Gemma selected frames take 226.079/235.896 ms, with GPU encoder
unions 52.048/51.906 ms and host waits 134.909/127.566 ms. Host waits include
GPU execution; GPU time must not be added to or subtracted from them to invent
queue time or physical GPU idleness. Unattributed host time is 64.813/84.337 ms,
not a measured CPU-operator breakdown. These are legacy selected frames, not
whole-request totals or current batched-path percentages.

The [single-row change](gemma-single-row-submission.md) proves that reducing
some waits helps: complete latency improves 7.4005%/7.1650%, and decode callback
rate rises from 4.254/4.272 to 4.694/4.714 tokens/s. It leaves most architecture
boundaries intact. No further speed multiplier is established yet.

## Observability cost: established and suspected

The [unchanged CLI overhead campaign](gemma-cli.md#fixed-overhead-measurement)
passes five of eight comparisons. Per-tile cohort 1 observes +3.9002% TTFT and
+10.7172% complete time; batched cohort 2 observes +1.1991% TTFT. All failures
remain binding. Large within-campaign drift is recorded; the experiment does
not isolate how much time belongs to collection, serialization or unrelated
run variation. Do not attribute seconds of regression to clock reads without
measuring them.

Static inspection finds two actionable costs in
[the callback](../src/bin/cli/gemma4.rs) and
[the collector](../src/bin/cli/gemma4/summary.rs):

- Summary mode adds token-event/interval bookkeeping and separate clock pairs
  around output decoding and publication on every callback.
- `Collector::record` validates, serializes JSON into scratch and copies the
  serialized row into a byte ring before the done event. File export is already
  deferred until inference ends; moving file I/O is not a new optimization.

Measure empty callbacks, clocks, decoder/publication accounting, record
validation, serialization and ring insertion independently, plus end-to-end
disabled/summary A/B and A/A controls. Microbenchmarks do not qualify the
request overhead bound. Primary numerical generation already supplies phase
counts/times; reuse those values rather than add numerical-loop timers.

The [completed component probe](gemma-current-cost-map.md#summary-component-measurements)
finds complete insertion at 3.569/3.555 us and token accounting at about 133 ns
per token in the confirming process. These standalone observations do not
explain the earlier full-request failures. Defer the typed-ring rewrite until
a useful request-level effect is demonstrated; prioritize accurate resident GPU
attention. Preserve bounded telemetry and run A/A stability diagnostics before
the unchanged fixed-N overhead campaign on a retained candidate.

## Ordered implementation tasks

| Task | Concrete change | Exit and next dependency |
| --- | --- | --- |
| G0: freeze fair competitors | Admit fresh Ollama/vLLM runs with exact active tensors/tokenizer, precision, token work, offsets, capacity and cache policy. Separate numerical fixed-token work from ordinary CLI/HTTP requests. Add authoritative untimed input/output-ID preflight where missing. | Comparable or explicitly incomparable per metric; no old timing reused as a new baseline. Required before any claim of beating an engine. |
| G1: current-path cost map | Reuse the bounded profiler for early/late prefill and decode on current per-tile/batched paths. Add only missing CPU-stage attribution and observed copy/command/wait counts. Run uninstrumented acceptance timings separately. Measure the observability components above. | Ranked costs with coverage, source/shape/depth identities, completed work and uncertainty; no more whole-model tuning on assumptions. First code task. |
| O4: cheaper summaries, conditional | After G1, retain the existing collector unless request-level evidence justifies a change. If justified, store bounded owned typed terminal records; validate on insertion, serialize at explicit export. Retain immutable owner/manifest identity once and only changing request/state fields per record. Select disabled/summary callback specialization outside the token loop; reuse timestamps where boundaries coincide. | Original schema/field meanings and all failure/cancel/replay/output-correction cases pass; zero steady-state collector allocation; no new commands/waits; original fixed overhead protocol passes all eight gates or default remains disabled. Depends on G1 for causal choice; the measured microsecond cost currently defers the typed-ring rewrite. |
| G2: resident Gemma block | Keep activations and proposed KV on GPU; port normalization, rotary, residual/GELU and attention one mechanism at a time. First qualify one complete block, then compose layers. Batch dependent encoders without CPU waits between each projection; retain one checked token/chunk publication boundary. | Primitive, block and full graph numerical/state/failure parity at original bounds; observed command/wait counts fall toward one per evaluation, with extra boundaries only for measured device/resource constraints. Depends on G1 and small numerical feasibility probes. |
| G3: separate prefill and decode matrices | Prefill: compare an available Apple matrix primitive with bounded SIMD-group tiled GEMM on real 32-row shapes, then larger chunks only under admitted memory/context. Decode: coalesced row tiles, reusable input and measured Q/K/V or gate/up grouping. Retain accurate reconstruction/accumulation fallback. | Primitive and real-model numerics pass; matrix speed reaches request timing, both cohorts exceed the original 5% retain gate, no unexplained compatibility regression. Depends on G2's resident interfaces; numerical feasibility may be tested earlier. |
| G4: reduce output and weight traffic | GPU greedy selection returns token ID plus status for ordinary greedy generation; full logits remain a diagnostic/quality mode. Then test a common authenticated Q4 artifact only if its full original quality gates pass. Fuse reconstruction with matmul, never expand the full model each token. | Exact tie/stop/pending state semantics and original numerical/full quality gates; explicit memory and traffic deltas. Depends on G2/G3. Do not reuse the rejected Heretic route as qualified. |
| G5: qualified LFM3 GPU graph | Port the actual BF16 source semantics: fixed projection partial/reduction/store order, normalization/unary rounding, seven-tap history and wrapped attention/KV proposals. Qualify leaves, a mixed conv/attention block, then the complete graph/CLI. Keep the current CPU oracle. | Original component/source, 1,025-position, complete quality and state gates pass before request comparisons. Existing generic MLX component correspondence failures remain failures; importing MLX is not a shortcut around them. Reuse G2/G3 infrastructure only where shapes/semantics match. |
| G6: external wins and regression jobs | Fresh counterbalanced Toklet/Ollama/vLLM cohorts on G0's frozen workloads; extend existing regression reporting with exact engine/backend and resource identities. Confirm warmed and cached tracks separately; sustained runs remain separate. | Every goal completion condition passes; Astra review, checks, commits/push and raw samples linked. Until then this goal remains active, including after any rejected experiment. |

For G2, proposed KV remains private until every required encoder and status
check completes. GPU failure or cancellation must preserve the exact valid
prefix or explicitly require replay. Shared command resources and scratch
remain alive through completion; no per-layer host validation is simply deleted.
Preserve the scalar/reference path. Immutable weights can use load-time proof
inside a private validated graph entry, while public slice APIs keep validation.
Whether bypassing redundant LFM3 scans matters is a G1 measurement, not a reason
to remove boundary checks.

For G3, ordinary matrix/attention arithmetic is not guaranteed to meet Toklet's
original accuracy contract. Test sampled real matrices, cancellation-sensitive
rows, reduction tails and wrapped attention before building a full adapter.
If a fast primitive fails, keep the accurate kernel and optimize residency,
tiling or error correction without loosening tolerances. No unbounded search
over artifacts, precisions or reduction variants is authorized by this plan.

## Shared-model admission and fixed benchmark matrix

Gemma official BF16 already fits Toklet, but the previous external BF16 attempts
failed the 6 GiB guard. G0 must first inspect whether an authenticated text-only
view of the same active tensors fits all three engines without numerical edits.
If not, qualify one common Q4 conversion of the selected source under the
original complete quality gates. A derived artifact has its own identity and
conversion proof. Do not compare qualified BF16 against an unrelated rejected
Q4 and label the difference an engine win. LFM3's existing shared BF16 source
fits both external engines, but its GPU execution must still qualify in Toklet.
No Qwen variant hunt is reopened.

The common precision candidate stores exactly widened FP32 scale/bias/vector
values from the same BF16-source affine4 quantizer, with unchanged U32 codes.
The actual conversion and independent normal/optimized validators pass for both
external-compatible views, with identical active component bytes. Conversion
peaks at 311,705,600 observed bytes including helper; this does not qualify
runtime model memory, quality or speed. [Conversion and validation](../artifacts/mlx-gpu-gap-2026-11-15/g0-q4-full-progress-1.json).

Toklet's independently authenticated third disk view uses its existing strict
U32/BF16 format. Conversion and both independent validators pass 723,455,267
code/value words; the observed conversion peak is 112,197,632 bytes including
helper. [Third-view checks](../artifacts/mlx-gpu-gap-2026-11-15/g0-third-view-progress-1.json). Integer stripping of the zero low bits recovers every
canonical BF16 component; codes, tokenizer and exact decoded FP32 values must
match the external views. Toklet's stored bytes/dtypes differ losslessly, so the
three-engine identity proof is canonical component digests plus decoded values,
not identical container bytes. This prospective protocol is frozen before any
acceptance samples; earlier incomparable timings remain incomparable. No new
runtime F32 format API is needed. An explicit common-Q4 owner/CLI selection and
full original quality/state/precision qualification remain required.
[Third-view and CLI recipe](../artifacts/mlx-gpu-gap-2026-11-15/astra-third-view-q4-cli-integration-research-1.md).

The bounded stock reload probe passes component and 46-parameter tiny Gemma
loader/forward/KV checks; independent readers verify 4,608 reconstructions and
15 corruption controls. This proves synthetic saved-dtype behavior only;
full-model and actual Ollama/vLLM worker precision evidence remains required.
[Tiny proof](../artifacts/mlx-gpu-gap-2026-11-15/g0-stored-fp32-reader-progress-1.json).
Full original numerical, quality/state and 6 GiB gates still apply.
[Stored-FP32 research](../artifacts/mlx-gpu-gap-2026-11-15/astra-common-q4-stored-fp32-research-1.md).

Freeze these cells before collecting candidate acceptance data:

- Gemma: reset 15/8 short case with explicit stopping, fixed 128/128 and
  384/128 token work at capacity 512; exact append, fully cached, pending and
  changed-prefix cases. Do not silently enlarge context or truncate work.
- LFM3: fixed 512/32 primary, and the original 128/512/1,024 inputs with 128
  outputs at capacity 2,048. Keep sliding/conv state and zero-reuse policy exact.
- Use teacher-forced IDs for controlled numerical work where the actual engine
  adapter supports it. Keep those clocks separate from generation/transport.
  Unsupported authoritative work is incomparable, not an invented rate.
- For ordinary generation, freeze raw-text/template mode, greedy/stop policy,
  output cap and actual authoritative IDs/counts. Keep EOS/task-quality runs
  separate from fixed-work runs. No post-hoc prompt selection to obtain a match.


For a prospective Gemma warm zero-reuse comparison, raw prompts may have
predeclared distinct first token IDs across every retained prompt, including
warmups. Freeze the complete prompt set and all three engines' identical
per-sample input IDs/counts before any acceptance data. Verify actual stock
input tokenization and cached-count zero; a mismatch or early stop leaves that
cell unqualified, without prompt replacement or sample exclusion. Keep the
15/8, 128/128 and 384/128 work and the original native reset/state cells. This
cache-miss strategy does not flush the server or rescue historical fixed-prompt
results; LFM3 requires its own cache/tokenizer qualification.

Count-defined ordinary generation may use authoritative output counts and stop
reason when sampled IDs are unavailable. Record missing IDs as null; never infer
them from text re-tokenization. Explicit ID-dependent quality/state/teacher-force
gates remain unchanged. Report hidden one-ahead work separately. Freeze public
Done versus runner-EOF/cleanup clocks without mixing them, and do not infer
allocated KV capacity from Ollama's soft num_ctx option. [Source/control limits](../artifacts/mlx-gpu-gap-2026-11-15/astra-ollama-work-cache-controls-research-1.md),
[prospective zero-reuse/count-work clarification](../artifacts/mlx-gpu-gap-2026-11-15/astra-ollama-work-cache-controls-addendum-1.md).

Use release binaries, exact source/runtime/model hashes, three warmups and ten
measured samples for initial screening, independently repeated in reversed
order. Run actual competitors serially with AC/LPM/thermal admission and no
competing build/model job. Declare the owner/block pairing and bootstrap unit
before collection; retain raw data, MAD, uncertainty and all failed attempts.
Keep the original fixed-N overhead protocol unchanged for its acceptance test;
supplementary A/A and micro-cost experiments cannot replace a failing cohort.
No p95/p99 claim follows from ten samples. Required sustained arms retain at
least 600 completed measured seconds.

Record completed request/TTFT, prefill rows/s, decode evaluations/s and callback
rate with separate denominators; command/dispatch/wait counts, active copy bytes,
operator coverage, allocated KV/scratch/weights/telemetry, observed RSS/physical
peak and memory pressure. Shared memory is counted once where ownership proves
it. External conservative sums and single-process peaks must retain their scopes;
do not infer an exact memory ratio. Unsupported metrics remain null with reasons.

## Checks, review and working order

Current/next work is [TODO.md](../TODO.md). G0 metadata preparation and O4 static
design can overlap G1 source analysis, but all hardware benchmarks and large
build/evaluation jobs run serially. Stop a failed primitive before an expensive
full-model/quality campaign. Use exact unchanged-input check reuse where valid.

Each code task needs fmt, strict all-target Clippy and tests for supported
CPU/Metal features, plus the affected numerical/model/session/GPU/allocation
checks and representative complete-request measurements. Submit one combined
diff/invariants/check-results packet to Astra, fix findings, then commit and
push. Full original quality campaigns apply when numerical behavior changes;
never substitute a short smoke or relaxed tolerance. Documentation-only updates
need link/content/whitespace checks. UI/screenshare/iPhone remain deferred.

## External mechanisms worth evaluating

MLX provides [quantized matmul](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.quantized_matmul.html)
and [fast grouped-query attention](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.fast.scaled_dot_product_attention.html).
Its attention softmax uses FP32 and does not require expanding KV heads.
These are design references, not proof that every recorded external request
used a particular kernel or that those numerical results pass Toklet's gates.
Apple supplies [MPS matrix multiplication](https://developer.apple.com/documentation/metalperformanceshaders/mpsmatrixmultiplication)
and [SIMD-group matrix operations](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf).
Check M1 device/dtype/API support before choosing one; newer TensorOps hardware
is not an assumed capability of this host.

[Diagnosis, checks and review receipt](../artifacts/mlx-gpu-gap-2026-11-15/planning-checks-1.json).

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

One freshly built ordinary Gemma BF16 pair on 384/128 reproduces the
batched T32 prefill gain, with 11.930 to 6.997 s first token and 39.274
to 34.402 s completion. Restage the longer-quality diagnostic with actual
old producer stride and bounded 512-visible-key Metal scratch. Verify
active early T32 and original later fallback on the complete original
1,025-position windows before retention.
[Guarded Gemma replay and Astra review](../artifacts/mlx-gpu-gap-2026-11-15/sol62-gemma-bf16-t32-current-1/ACTUAL-REVIEW.md).

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
owners at two prompt partitions. Native logits differ from MLX. The
private vLLM dense-attention replay matches the direct MLX cache and all
31 decode logits. On the same retained prefill hidden row, selecting
before the head explains its first-score difference. The publisher's
private Ollama fork verifies the imported weights and all output IDs.
The untimed four-T128, 31-T1 Ollama replay scores 32 heads and
reproduces the earlier Ollama terminal state using its own sampled IDs.
All 32 scored-head digests differ from matched MLX. Numerical
qualification remains open. [Ollama work and scores](../artifacts/mlx-gpu-gap-2026-11-15/sol62-ollama-nanbeige-raw-worker-1/matched32/run-1/read-normal.json).
The first logical owner differs from position zero in both K and V,
which locates the discrepancy at or before the first cache write.
Embedding and RMSNorm now match exactly. The shared T128 input locates
both differences in affine K/V projection before RoPE. Ollama's V
matches all 131,072 archived MLX 0.32.3 words. Test a detached MLX-C
bridge against the pinned MLX 0.32.4 wheel with the same Go worker and
verified store. The guarded configure passes. The pinned bridge stops
at the new compile-cache handle arguments for clear and erase.
Preserve cache lifetime and clear/erase semantics in any compatible
bridge before linking. The detached adapter keeps the C ABI and
uses the caller's cache plus the current SDPA and gather-QMM defaults.
The verified wheel and unchanged Go executable join the full 32-head
state with direct MLX and private vLLM. Prefill comparison reads the
512-row prefixes of the terminal cache. Require the sealed positive
numerical join to launch fresh warm timing cohorts with the same
per-request work and original acceptance gates.
[Timing admission](../artifacts/mlx-gpu-gap-2026-11-15/root-nanbeige-ollama-mlx324-1/timing-admission.json).
[Modern Ollama replay](../artifacts/mlx-gpu-gap-2026-11-15/root-nanbeige-ollama-mlx324-1/normal.json).
[Bridge preflight](../artifacts/mlx-gpu-gap-2026-11-15/root-nanbeige-ollama-layer0-1/mlxc-wheel-preflight-1/receipt.json).
[Layer-zero probe](../artifacts/mlx-gpu-gap-2026-11-15/root-nanbeige-ollama-layer0-1/normal.json).
[First prefill difference](../artifacts/mlx-gpu-gap-2026-11-15/root-nanbeige-ollama-prefill-cut-1/check-normal.json).

The direct MLX and private vLLM
witnesses join all 32 one-row scores, their own sampled IDs and both
44-owner cache snapshots. They use four T128 bodies, suppress the first
three heads, gather the last prefill row before projecting and execute
31 T1 evaluations. Both swapped cohorts retain these boundaries and
time the private vLLM dense route with its inherited head observer
removed. The current private vLLM median is 3.631–3.640 s and direct
MLX is 3.312–3.316 s. Use this workload to qualify the current native
32-head route and reproduce the ordinary comparison after integration.

The fresh Ollama/vLLM orders set a tighter 512/32 performance target.
Ollama completes in 3.267 and 3.291 s against 3.621 and 3.622 s for
vLLM. Ollama saves 290–310 ms after the first ID. The measured medians
imply Toklet completion ceilings of 3.104 and 3.127 s for a 5% win
against the faster private engine. Recompute these ceilings with
fresh peers in Toklet's joined timing cohorts. Use the TTFT and decode
split to guide T128 and T1 work on the admitted native schedule.
[Matched private clocks](../artifacts/mlx-gpu-gap-2026-11-15/root-nanbeige-ollama-timing-1/normal.json).

The reversed three-engine timing diagnostic measures Toklet completion
at 4.506–4.546 s, Ollama at 3.738–3.745 s and vLLM at 4.081 s.
Toklet's extra 574–596 ms to first ID explains about 75% of its
completion gap to Ollama. Decode takes 192–205 ms longer than
Ollama and 90–98 ms less than vLLM. This session's 5% threshold against
Ollama is 3.551–3.558 s. Use the faster prior peers as a sensitivity
check. Prioritize T128 prefill and match early cache and logit arithmetic
against the joined peers. The diagnostic joins all IDs and none of the
native terminal K/V or logit digests. It leaves qualification open.
The first native T128 embedding and RMSNorm match the peers bitwise.
The ordinary tiled K/V stage differs by 14/26 BF16 words before RoPE
on identical normalized operands. Match the MLX 0.32.4 projection
dispatcher first, then compare later boundaries on shared inputs.
The pinned affine split-K and ordered merge reproduce both T128
panels exactly on shared operands. M=384 selects the ordinary
non-split path. Repeating the authentic T128 activation three times
then reproduces native Toklet K and V, including the 14 and 26 words
that differ from the T128 oracle. Preserve the four partitions and
single final BF16 rounding in any batching test. Confirm connected
cache and residual state before using a speed result for admission.
Connected split-K matches the first live pre-RoPE Q/K/V and the first
cache pair. The full first logical owner's 512-token prefix also
matches. Five rotated Q words differ. On 123 rows with equal query
and causal cache inputs, native attention differs in 234 BF16
words. O matches all rows whose input agrees. Qualify the rotary
and causal reductions on these common operands before timing.
Test reuse of a single loaded weight tile across two 32-row
activation tiles inside T128 first. Require exact FP32 partials
and charge all intermediate traffic, finite checks and merge.
[Connected comparison](../artifacts/mlx-gpu-gap-2026-11-15/root-nanbeige-connected-splitk-1/normal.json).

The matched rotary FMA order fixes all five Q differences on the
frozen cut. On fully identical inputs the GQA replay still differs
in 237 words. Test the pinned Steel tile with the corrected rotary
and validate later cuts before any public route change. Sharing one
dense Gate/Up weight tile across two activation tiles preserves
the result and loses all 40 measured pairs. Threadgroup storage
grows from 4,608 to 17,408 bytes. Retain the incumbent for this
workload and record the result before further reuse work.
[Attention and rotary](../artifacts/mlx-gpu-gap-2026-11-15/root-nanbeige-attention-replay-1/RUN.md) ·
[Gate/Up measurement](../artifacts/mlx-gpu-gap-2026-11-15/root-nanbeige-dense-tile-reuse-1/RUN.md).

The connected Steel and rotary witness matches every first-cut panel
and slot-zero K/V at all four native prefill boundaries. The remaining
43 prefill owners, all scored logits and all terminal owners differ
from MLX. Check the remaining layer-zero residual and FFN operations
before widening timing. The two proof GPU clocks sum to 2.443 s for
Steel and 2.580 s for split-K with native attention. A separate MLX
shape test uses three different authentic FFN inputs and charges the
stacking operation. Gate/Up and SwiGLU take about 18.4 ms in both
shapes, with zero differences and both paired intervals including zero.
[Steel and FFN shape results](../artifacts/mlx-gpu-gap-2026-11-15/root-nanbeige-steel-connected-1/normal.json).

The incumbent SwiGLU matches its GPU replay on direct Gate and Up
operands and differs from MLX in 451,662 words. The connected T128
candidate preserves the BF16 sigmoid boundaries in both FFN paths
and joins the full first-cut trace through the next layer's raw K/V.
The cache frontier joins slots 0–9 at 128 but only slot 0 from 256
onward. Diagnose the second cut before widening the runtime route.
[FFN and cache frontier](../artifacts/mlx-gpu-gap-2026-11-15/root-nanbeige-ffn-connection-1/RUN.md).

The second cut first differs in FFN RMSNorm on identical residual
input. The pinned MLX reduction matches all 19 traced panels and
176 ordered prefill cache digests across the four cuts. The full
score and terminal gates remain open. Compare the final prefill
hidden row before changing the head and T1 path. The RMS proof
commands take 2.800 s versus 2.721 s in the one-owner control.
[RMS and prefill evidence](../artifacts/mlx-gpu-gap-2026-11-15/root-nanbeige-second-cut-1/RUN.md).

The final prefill hidden row joins MLX. Native tiled scoring joins
the archived batched row and differs in 36,106 words from the live
selective reference. Pinned QMV matches the selective score exactly.
First T1 embedding and RMSNorm match. Current Q/K/V differ in 4,010,
655 and 775 words. FP32 rather than BF16 four-term bias sums reproduce
all three MLX panels offline on the common input. Verify an in-kernel
precision change before pricing or changing the next decode operations.
[First score and T1 trace](../artifacts/mlx-gpu-gap-2026-11-15/root-nanbeige-head-decode-1/RUN.md).

The connected FP32 bias experiment leaves one Q word. A same-input CPU
replay identifies the halves reduction. Native and pinned MLX attention
match all 6,144 words when their visible inputs match. Verify the halves
reduction in Metal and trace the second cache owner at the first decode.
[Decode bias and attention replay](../artifacts/mlx-gpu-gap-2026-11-15/root-nanbeige-t1-bias-1/RUN.md).

The halves reduction makes first-decode Q/K/V exact through gate and up.
Rounded T1 SwiGLU makes all 10,752 down inputs exact. One down output word
differs and the residual is exact. First append divergence moves to owner 5.
Screen pinned `qmv_fast` on every eligible T1 projection before another
operator substitution. Preserve prefill scoring, finite checks and rollback.
[Halves](../artifacts/mlx-gpu-gap-2026-11-15/root-nanbeige-t1-halves-1/native/RUN.md) ·
[SwiGLU](../artifacts/mlx-gpu-gap-2026-11-15/root-nanbeige-t1-swiglu-1/native/RUN.md).

The pinned QMV replay moves the first divergent append to owner 8.
Capture raw operator boundaries at position 512 in one sweep through
owner 8, including owner 1 attention and outgoing residual. Distinguish
RMS, rotary and attention differences on identical operands before
changing another kernel. Reprice after exact connected decode.
[Wheel replay](../artifacts/mlx-gpu-gap-2026-11-15/root-nanbeige-t1-qmv-wheel-1/native/RUN.md).

Owners 1 and 7 differ in one native attention word each on identical
operands. The pinned vector SDPA closes both and matches scores and appends
for the first two decode steps. The first mismatch moves to owner 4 at position
514. Use one runtime-selectable trace build to verify each selected row
against its tape, committed prefix and append hashes. Timed cohorts remain open.
[Owner trace](../artifacts/mlx-gpu-gap-2026-11-15/root-nanbeige-t1-owner-trace-1/RUN.md) ·
[Vector SDPA](../artifacts/mlx-gpu-gap-2026-11-15/root-nanbeige-t1-sdpa-wheel-1/native/RUN.md).

Precise rotary matches the first eight decode steps. At position 520,
the first mismatched append is owner 13. Its input RMSNorm differs in
five words on identical residuals. Test the pinned RMSNorm reduction in
the shared T1 norm path with finite output scans and post-commit counts.
Use the runtime position and owner window for subsequent localization.
[Position trace](../artifacts/mlx-gpu-gap-2026-11-15/root-nanbeige-t1-position-trace-1/RUN.md) ·
[Precise rotary](../artifacts/mlx-gpu-gap-2026-11-15/root-nanbeige-t1-rotary-1/RUN.md) ·
[Owner window](../artifacts/mlx-gpu-gap-2026-11-15/root-nanbeige-t1-owner-window-1/RUN.md).

Pinned T1 RMSNorm completes fixed-tape parity in the detached witness.
Enable the same admitted kernels in an ordinary session with snapshots
disabled. Verify cache and rollback invariants, screen native latency, and
then compare fresh peers under the original warmup and reversed-order rules.
[Complete decode comparison](../artifacts/mlx-gpu-gap-2026-11-15/root-nanbeige-t1-rms-wheel-1/RUN.md).

The ordinary fixed lane passes the ID and cancellation smoke and preserves
the numerical witness. A native-only owner measures 2.539 s TTFT, 0.851 s
decode and 3.391 s completion. Earlier peer clocks are historical context.
Qualify the route against fresh joined Ollama and vLLM cohorts before
retaining it on latency. Measure summary overhead independently.
[Ordinary route checks](../artifacts/mlx-gpu-gap-2026-11-15/root-nanbeige-exact-cli-prep-1/RUN.md) ·
[Timing screen](../artifacts/mlx-gpu-gap-2026-11-15/root-nanbeige-exact-cli-prep-1/SCREEN.md).

The fresh private MLX cohorts pass the 5% gate against vLLM. Against Ollama,
Toklet adds 139–141 ms to TTFT and saves 11–14 ms in decode. A 5% completion
win needs about 290 ms off Toklet's current 3.414–3.415 s medians. Profile
completed T128 prefill first and test final-chunk fused gate/up as an
isolated change. Preserve the fixed tape and qualify summary overhead separately.
[Matched cohorts](../artifacts/mlx-gpu-gap-2026-11-15/root-nanbeige-exact-cohorts-1/RESULT.md).

The exact dense-tail candidate completes in 3.184–3.186 s across the native
orders. It improves prefill and adds 11–19 ms in paired decode. The earlier
Ollama 5% point threshold needs another 60–62 ms. Profile completed T128
work and the decode carry, then rerun fresh peers and the original overhead gate.
[Tail A/B](../artifacts/mlx-gpu-gap-2026-11-15/root-nanbeige-exact-tail-dense-ordinary-1/RUN.md).

The dense route uses about 2.28 s of prefill GPU time. First-T1 launch
delay of 12–25 ms plus 10–12 ms of prefill launch residual cannot close
the 60–62 ms gap. Test reducing GPU work with rotary-owned T128 Q/K/V
finite scans. M1 Pro reports stage counters with dispatch sampling unavailable.
[Command-level profile](../artifacts/mlx-gpu-gap-2026-11-15/root-nanbeige-exact-profile-1/RUN.md).

The checked T128 rotary specialization reads and checks every raw Q/K/V
word. Fault and retry probes pass with sticky status intact. Run the
frozen full-tape witness, then price ordinary completions in both orders
once AC power is connected and Low Power Mode is off.
[Scan fusion](../artifacts/mlx-gpu-gap-2026-11-15/root-nanbeige-qkv-scan-fusion-prep-1/RUN.md).

Archived disabled owners still show native prefill drift after matching
ordinals. Neither ordinal nor previous decode predicts it consistently.
Keep the original warmup count and paired and marginal 1% bounds.
[Summary diagnostic](../artifacts/mlx-gpu-gap-2026-11-15/root-o4-owner-conditioning-1/RUN.md).

Both ordinary tail modes pass untimed cancellation, status failure and
synthetic completion-error reset and replay. The full scan-fusion witness
and timing still require AC with Low Power Mode off. The original summary
run also retains its 5 GiB free-space preflight.
[Untimed rollback evidence](../artifacts/mlx-gpu-gap-2026-11-15/root-nanbeige-exact-fault-prep-1/RUN.md).

The separately guarded scan-fusion replay now passes full-tape numerical
and session checks. Use the unchanged frozen driver for the paired witness
and latency on AC. Preserve both external completion gates and both summary
confidence bounds.
[Untimed scan-fusion result](../artifacts/mlx-gpu-gap-2026-11-15/root-nanbeige-qkv-scan-fusion-prep-1/untimed-full-1/report.json).

The frozen scan-fusion witness passes, and ordinary latency shows decode
regression and inconclusive completion. Keep the dense-tail control and
seek a larger reduction in completed prefill or decode work. Both external
completion gates and the original summary-overhead bounds remain binding.
[Scan-fusion A/B](../artifacts/mlx-gpu-gap-2026-11-15/root-nanbeige-qkv-scan-fusion-prep-1/RUN.md).

Token-first Gate/Up traversal passes all numerical joins. The stable
reverse order's upper completion saving is 22.60 ms, short of 60–62 ms.
Hold the dense-tail control. Test one-time packing of the original A
fragments separately to price the producer and consumer together.
[Traversal A/B](../artifacts/mlx-gpu-gap-2026-11-15/root-nanbeige-dense-traversal-1/RUN.md).

Packing original A fragments into the existing scratch buffer clears the
historical 60–62 ms screen in both native orders. Keep dense tail and carry
the measured reverse decode cost into qualification against fresh peers.
[Prepack A/B](../artifacts/mlx-gpu-gap-2026-11-15/root-nanbeige-dense-prepacked-1/RUN.md).

The fresh prepack cohorts pass the fixed 512/32 completion target against
both pinned MLX peers. Keep dense tail and the prepack candidate for this
lane. Qualify broader CLI integration and the original 1% summary bounds
before claiming the overall goal.
[Prepack matched cohorts](../artifacts/mlx-gpu-gap-2026-11-15/root-nanbeige-prepacked-cohorts-1/RESULT.md).

The pinned fixed route now runs through an experimental CLI opt-in. Normal
and readback-only builds pass the session and numerical checks. Use a
fresh same-boundary worker comparison for integrated performance and keep
the original O4 summary estimator and disk admission.
[Normal CLI checks](../artifacts/mlx-gpu-gap-2026-11-15/root-nanbeige-prepacked-cli-1/RUN.md).

The integrated worker confirms the private fixed 512/32 result in two
reversed orders at 5.58 GiB guarded peak. Prepacking cleared the historical
60–62 ms deficit to the Ollama gate. The integrated lead is 199–208 ms
in completion. The remaining decode difference is +3.73 ms in one order
and −7.71 ms in reverse. Keep dense tail and the prepacked activations.
Price additional decode headroom by moving seven standalone T1 affine
finite checks per visit into their immediate consumers. Preserve exact
BF16 outputs, sticky status, rollback and the independent original
paired and marginal 1% summary gates.
[Integrated timing](../artifacts/mlx-gpu-gap-2026-11-15/root-nanbeige-integrated-cohorts-1/RESULT.md).

The T1 consumer-check trial preserves all four dense prefill cuts and the
exact tape. Eliminating 9,548 standalone finite scans yields upper paired
completion savings of 8.57 and 20.91 ms, below the 60 ms headroom screen
in both orders. The decode intervals include zero. Retain standalone T1
checks and the integrated dense-tail route. The original 1% observations
remain a separate qualification.
[T1 finite-check A/B](../artifacts/mlx-gpu-gap-2026-11-15/sol62-nanbeige-t1-consumer-scan-1/RUN.md).

The LFM3 control measures baseline resolution at the original 512/32
work and unchanged public clocks. All 12 timing checks pass across eight
disabled owners. The tightest marginal TTFT upper is 0.96339%, and the
paired upper in that cell is 0.95948%. The original actual 1.06524% bound
still misses. Require independently evidenced improvement and a fresh
real enabled/disabled comparison with the original tape, exports and gates.
[LFM3 control review](../artifacts/mlx-gpu-gap-2026-11-15/root-lfm3-o4-sham-readiness-1/ACTUAL-REVIEW.md).

The Gemma T32 candidate passes the original block 9 with exact scores and
K/V. Its final 17-token cut uses the CPU fallback. Blocks 0 and 9 cover
1,808 original targets. Continue blocks 1–8 at the frozen reset boundaries.
For the original 100 tasks, use the verified hybrid generation constructor
and a C567 test probe that checks actual per-cut backing and route counts
while preserving prompt, greedy and stop behavior.
[Gemma block 9 proof](../artifacts/mlx-gpu-gap-2026-11-15/sol62-gemma-bf16-t32-quality-block9-1/RESULT.md).

Gemma block 2 exposes a CPU/Metal score rounding mismatch in T32. Preserve
the bitwise score gate for every cell. The ordered CPU scorer with Metal
context passes the exact two-cut witness and rollback checks. Measure the
detached public 384/128 route against CPU attention before repeating the
remaining loss blocks. Rebind the repaired source for the original C567
task pilot, teacher replay and grading after the timing screen.
[C567 conditional review](../artifacts/mlx-gpu-gap-2026-11-15/sol62-gemma-bf16-t32-task-quality-1/REVIEW.md) ·
[Repair proof](../artifacts/mlx-gpu-gap-2026-11-15/sol62-gemma-bf16-t32-quality-block2-repair-1/trial-1/normal.json).

[K/V and shape evidence](../artifacts/mlx-gpu-gap-2026-11-15/root-nanbeige-kv-parity-1/RUN.md).
[First native divergence](../artifacts/mlx-gpu-gap-2026-11-15/root-nanbeige-native-first-cut-1/normal.json).
[Three-engine diagnostic](../artifacts/mlx-gpu-gap-2026-11-15/root-nanbeige-toklet-clocks-1/normal.json).
[Native admission results](../artifacts/mlx-gpu-gap-2026-11-15/root-nanbeige-mlx-q4-admission-1/RUN.md) ·
[External numerical review](../artifacts/mlx-gpu-gap-2026-11-15/sol62-nanbeige-qmm-connected-1/ACTUAL-REVIEW.md) ·
[First head witness and Astra review](../artifacts/mlx-gpu-gap-2026-11-15/sol62-nanbeige-vllm-first-head-shape-1/ACTUAL-REVIEW.md) ·
[Matched work and Astra review](../artifacts/mlx-gpu-gap-2026-11-15/sol62-nanbeige-mlx-matched-32-1/ACTUAL-REVIEW.md) ·
[Matched private clocks and Astra review](../artifacts/mlx-gpu-gap-2026-11-15/sol62-nanbeige-mlx-matched-32-timing-1/ACTUAL-REVIEW.md).

The official unquantized parent contains 4,169,800,704 stored BF16
parameters. Its independent embedding and head contribute about
1.02 GB each. The 8.34 GB weight payload exceeds the 6 GiB guard
before cache or scratch. Reuse the existing verified shard reader
and pinned MLX primitives for a separate streamed reference. Verify
the real memory fit before applying the original held-out and task gates.
[Official parent inventory and Astra proposal](../artifacts/mlx-gpu-gap-2026-11-15/astra-nanbeige-bf16-quality-1/PROPOSAL.md).

The layerwise Q4 validation matches every checked stock MLX score and
terminal head and K/V hash, including the 1,025-position window. The
guarded BF16 transfer authenticates both complete origin streams and all
201 tensor spans. The held-out loader will now read two contiguous
spans per physical block and recheck each tensor against the verified
hashes. Score the original corpus and independent smoke in one traversal.
[Guarded schedule review](../artifacts/mlx-gpu-gap-2026-11-15/root-nanbeige-layerwise-oracle-1/REVIEW.md) ·
[Parent acquisition review](../artifacts/mlx-gpu-gap-2026-11-15/root-nanbeige-layerwise-oracle-1/ACQUISITION-REVIEW.md) ·
[Parent range plan](../artifacts/mlx-gpu-gap-2026-11-15/root-nanbeige-layerwise-oracle-1/parent-scoring-layout.json).

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

The T128 K64 trial targets the FFN down projection. Direct MLX 0.32.4
matches all 393,216 BF16 outputs at about 3.28 ms through completion.
In Toklet, a 64-column tile keeps the original ascending FP32 K8 steps
and moves four adjacent BF16 activation words at a time. The tile
loads affine metadata once per G64 group and halves the number of
threadgroup barriers over 10,752 columns. Its two paired lower savings
of 1.747 and 1.796 ms clear the isolated 1.544 ms screen. The T128
512/32 request join passes with both paired lower completion gains
above 5%, full terminal state and 2.47 GiB peak guarded charge.
The separate compact Q trial saves 1.31–1.32 ms per visit with
bitwise output parity, below the isolated request screen. Measure
Q and attention output on one authentic layer-zero visit next. Use
the captured output projection operands and original causal row-zero
and post-attention FFN normalization witnesses.
[Direct MLX down projection](../artifacts/mlx-gpu-gap-2026-11-15/root-nanbeige-down-reference-1/RUN.md) ·
[Down operator](../artifacts/mlx-gpu-gap-2026-11-15/root-nanbeige-prefill-work-1/sol-down-k64-1/normal.json) ·
[Connected trial](../artifacts/mlx-gpu-gap-2026-11-15/root-nanbeige-down-connected-1/RUN.md) ·
[Query operator](../artifacts/mlx-gpu-gap-2026-11-15/sol62-nanbeige-query-compact-1/RUN.md) ·
[MLX attention operands](../artifacts/mlx-gpu-gap-2026-11-15/root-nanbeige-attention-input-1/normal.json).

One completed T128 layer-zero attention command uses the existing
compact K32 query and K64 output tiles. The two native arms share
K/V projection, RoPE, cache layout and causal GQA. Both replay all
3,276,800 BF16 words exactly. The paired lower GPU savings are
2.702 and 2.727 ms. The independently captured MLX O input yields
393,216 exact output words with both native projection tiles.
Astra approves a connected trial that checks the aligned attention
input separately from the FFN down input.
The 52 matched requests pass all terminal state joins. Completion
falls to 4.61–4.62 s. Compared with the archived direct MLX reference,
prefill accounts for about 84% of the remaining median latency gap.
[Joined attention command](../artifacts/mlx-gpu-gap-2026-11-15/sol62-nanbeige-attention-pair-1/normal.json) ·
[Connected attention trial](../artifacts/mlx-gpu-gap-2026-11-15/root-nanbeige-attention-connected-1/RUN.md).

Nanbeige's four-cut T128 layer-zero GQA baseline measures 6.203
and 4.654 ms in opposite orders. The reverse result misses the
5.250 ms absolute screen, so keep the current GQA kernel. The
five-cut resident LFM3 convolution screen gives conditional lower
savings of 0.112–0.121 s against a 0.452 s request-level screen.
Keep the current convolution handoff.
[GQA baseline](../artifacts/mlx-gpu-gap-2026-11-15/sol62-nanbeige-gqa-offsets-1/RUN.md) ·
[Convolution screen](../artifacts/mlx-gpu-gap-2026-11-15/sol62-lfm3-connected-conv-chain-2/RUN.md).

The four-cut Nanbeige attribution profile measures 2.16 s in
FFN projections and 0.85 s in query plus attention output. GQA's
0.205 s across all 176 visits falls below the 0.229 s development
threshold at the observed 4.589 s control completion. The paired
one-encoder command spends roughly 3.50 s on the GPU with less
than 3 ms in host encoding and wait outside GPU work. Prioritize
matrix throughput with authentic operands. The exact direct MLX
Q/O comparison yields an isolated 0.167 s extrapolation, below
the development screen. A 64-token gate/up tile with the same
K32 arithmetic slows both orders. The dual-bank fused K64 M32
tile shares activation staging and applies the existing SwiGLU
calculation after both BF16 roundings. It saves over 1.53 ms
per visit at the paired lower bound in both isolated orders and
reproduces the exact released Down input. Both connected orders
clear 5% completion with exact terminal state. The reverse-order
decode interval observes a small slowdown. The same-owner host
and prefill factors both preserve state. All four decode intervals
include zero. The compiled outer specialization eliminates the
per-visit fusion selector and passes both ordinary completion and
exact-state checks. Its reverse order still observes 1.05 ms slower
decode. Keep both candidates detached under the original rule.
[Stage costs and direct projection clocks](../artifacts/mlx-gpu-gap-2026-11-15/root-nanbeige-prefill-profile-1/RUN.md) ·
[Connected trial review](../artifacts/mlx-gpu-gap-2026-11-15/sol62-nanbeige-gate-up-fused-connected-1/REVIEW.md) ·
[Same-owner clocks](../artifacts/mlx-gpu-gap-2026-11-15/sol62-nanbeige-fused-t1-crossover-1/RUN.md) ·
[Outer specialization](../artifacts/mlx-gpu-gap-2026-11-15/sol62-nanbeige-fused-outer-dispatch-1/RUN.md).

The local vLLM diagnostic still has a 0.82–0.83 s completion lead.
The archive puts three omitted prefill heads at about 38 ms of GPU
time. Guarded real CLI controls confirm the same 32 scored rows with
exact output and state. Public timing and attention arithmetic still
need to match. The packed G64 shuffle test reuses sixteen exact affine
values inside each SIMD half across the T128 FFN. It reproduces both
native output panels and adds about 4.7 ms per visit in both orders,
closing the trial.
[Astra proposal and controls](../artifacts/mlx-gpu-gap-2026-11-15/astra-nanbeige-g64-shuffle-1/PROPOSAL.md) ·
[Isolated FFN result](../artifacts/mlx-gpu-gap-2026-11-15/sol62-nanbeige-affine-codebook-1/normal-1.json) ·
[Detached 32-row CLI path](../artifacts/mlx-gpu-gap-2026-11-15/sol62-nanbeige-head-cli-1/RUN.md) ·
[Real CLI validation](../artifacts/mlx-gpu-gap-2026-11-15/sol62-nanbeige-head-cli-validation-1/REVIEW.md).

One Gate/Up BF16 pair in the existing Apple7 fragment order reduces
completed GPU time from about 10.75–10.78 to 9.00 ms for the exact
T128 FFN chain. Its 132,120,576-byte allocation takes 7.84 ms, and
preparation completes in 7.44 ms wall time including 4.43 ms on the
device. All 20 measured pairs improve. The connected design would
add 2,906,652,672 bytes across 22 physical pairs while retaining packed
Q4 for decode and the same FP32 K8 accumulation. The guarded fit observes
that exact allocation at 5.06 GiB peak. Allocation and prepare take 0.658 s
in total in that fit, with terminal and fallback state matching exactly.
Ordinary completion improves 11.72% and 11.38% at 5.16 GiB peak.
Both paired decode intervals show a slowdown, with median increases of
16.47 and 13.23 ms. Hold the route. The 104-request same-binary study
finds a decode slowdown of 16.92 and 20.58 ms at fixed residency when
dense prefill runs. The residence-only effect changes sign. Hold 22
pairs resident and test the first three chunks dense with the fourth
packed. This tests whether the work immediately preceding decode is
enough to account for the slowdown. Preserve the existing release gates.
The final packed cut confirms the three-plus-one route and matches exact
terminal and fallback state. Both ordinary completion lower bounds
exceed 6%. Median decode improves in both orders. Replace the fixed
prompt recognizer in a separate opt-in with actual remaining-prefill
topology, qualify the compiled prepare and consume pipelines on load,
and measure independent prompt and complete cold CLI behavior.
[Detached layout and numerical checks](../artifacts/mlx-gpu-gap-2026-11-15/sol62-nanbeige-dense-gate-up-1/RUN.md) ·
[Isolated timings](../artifacts/mlx-gpu-gap-2026-11-15/sol62-nanbeige-dense-gate-up-1/normal-2.json) ·
[Guarded model fit](../artifacts/mlx-gpu-gap-2026-11-15/sol62-nanbeige-dense-connected-1/fit-normal.json) ·
[Ordinary timings and Astra review](../artifacts/mlx-gpu-gap-2026-11-15/sol62-nanbeige-dense-connected-1/REVIEW.md) ·
[Same-binary contrasts and Astra review](../artifacts/mlx-gpu-gap-2026-11-15/sol62-nanbeige-dense-causal-1/REVIEW.md) ·
[Final-chunk ordinary review](../artifacts/mlx-gpu-gap-2026-11-15/sol62-nanbeige-dense-tail-packed-1/REVIEW.md).

The generic remaining-prefill selector reproduces the warm gain in both
orders. All 52 requests retain exact tokens and terminal state.
Completion falls about 7.1% with both paired lower bounds above 6.55%.
First token arrives about 313–314 ms sooner and both decode medians
improve at 5.16 GiB peak. Four alternate lengths, signed startup and
the unchanged default CLI schema pass. Both post-completion controls,
the repository suites and Astra review qualify the bounded M1 Pro opt-in.
[Generic timings and Astra review](../artifacts/mlx-gpu-gap-2026-11-15/sol62-nanbeige-dense-option-1/REVIEW.md) ·
[Astra release review](../artifacts/mlx-gpu-gap-2026-11-15/root-nanbeige-dense-release-checks-1/RELEASE-REVIEW.md).

Keeping dense Down for all 22 banks would exceed 6 GiB.
Astra defers layer-major execution until a current connected trace shows
enough removable GPU time. The earlier trace added 101–104 ms by
splitting every dispatch. Measure targeted splits and the incremental
timestamp cost separately before using current stage durations. Retain
the complete final packed chunk in any later scheduling experiment.
[Architecture review](../artifacts/mlx-gpu-gap-2026-11-15/sol62-nanbeige-next-prefill-brief-1/REVIEW.md) ·
[Prospective attribution design](../artifacts/mlx-gpu-gap-2026-11-15/sol62-nanbeige-dense-attribution-1/PLAN.md).

The sparse stage trial also exceeds its calibration budget. Encoder
splits shift TTFT by up to 92.23 ms with matching numerical state.
Record the measured ordinary request and command durations and judge
future performance changes with those clocks. M1 Pro reports stage
timestamp support. Dispatch-boundary timestamps are unavailable.
[Guarded capture and Astra review](../artifacts/mlx-gpu-gap-2026-11-15/sol62-nanbeige-dense-attribution-1/CAPTURE-REVIEW.md).

The pinned installed MLX library exposes a single Apple7 affine Q4
T128 pipeline compatible with the Q, O and Down shapes. Earlier isolated
prices imply roughly 247 ms over 176 visits before the required status
scans. Check all three on authentic Toklet inputs and include each scan
in one ordinary command. Require bitwise outputs and both paired lower
savings above 1.219 ms per combined visit before a connected trial.
Both guarded orders pass bitwise and fault controls and yield paired
lower savings of 2.009 and 2.004 ms. Inside the ordinary encoder, the
same three scans and existing fallback yield paired completion lower
gains of 8.40% and 8.19% against the released dense opt-in. Four fit
and fault owners verify the terminal bits and cap. The linked T1 route
bypasses QMM. Build the same single affine BF16 Q4/G64 instantiation
from the pinned MLX source with its MIT notice. The compiled 13,472-byte
image matches all authentic and independent operator panels and fault
controls. The connected Mac opt-in passes fit, fallback, console and
linked T1 checks. Both warm completion intervals clear 5%. The first
order has wholly slower decode and the released baseline drifts across
owners. The same-binary resident crossover preserves the completion gain
and bounds the earlier decode penalty under its fixed residency. The
fixed independent repeat clears both completion thresholds. Both decode
intervals include zero. Retain the exact packaged Mac kernel as an
explicit warm opt-in. The single cold CLI remains descriptive.
Request-summary overhead remains a separate measured gate.
The separate Gate/Up substitution misses its charged latency budget.
[MLX function and inputs](../artifacts/mlx-gpu-gap-2026-11-15/astra-nanbeige-qmm-three-roles-1/PROPOSAL.md) ·
[Guarded result and Astra review](../artifacts/mlx-gpu-gap-2026-11-15/sol62-nanbeige-qmm-three-roles-1/ACTUAL-REVIEW.md) ·
[Ordinary cohorts and Astra review](../artifacts/mlx-gpu-gap-2026-11-15/sol62-nanbeige-qmm-connected-1/ACTUAL-REVIEW.md) ·
[Gate/Up latency screen](../artifacts/mlx-gpu-gap-2026-11-15/sol62-nanbeige-qmm-gate-up-screen-1/SCREEN.md) ·
[Single-kernel parity and Astra review](../artifacts/mlx-gpu-gap-2026-11-15/sol62-nanbeige-qmm-one-symbol-1/ACTUAL-REVIEW.md) ·
[Connected compact-kernel checks and review](../artifacts/mlx-gpu-gap-2026-11-15/sol62-nanbeige-qmm-product-1/ACTUAL-REVIEW.md) ·
[Crossed timings and Astra review](../artifacts/mlx-gpu-gap-2026-11-15/sol62-nanbeige-qmm-resident-causal-1/ACTUAL-REVIEW.md) ·
[Final confirmation and Astra approval](../artifacts/mlx-gpu-gap-2026-11-15/sol62-nanbeige-qmm-product-confirmation-1/ACTUAL-REVIEW.md) ·
[Packaged image and usage](../assets/nanbeige-qmm/README.md) ·
[Pinned MLX release](https://github.com/ml-explore/mlx/releases/tag/v0.32.4) ·
[MLX license](https://github.com/ml-explore/mlx/blob/v0.32.4/LICENSE).

Astra's T384 screen prices coalescing the first three projection panels
through Q/K/V, O, Gate/Up and Down. Preserve the original attention
offsets, cut publication, final packed T128 and T1. The archived clocks
give an estimated 39.36 ms per three-panel block. Closing the conditional
gap to the historical vLLM diagnostic would require a paired lower saving
above 12.97 ms per block. Charge packing, conversion, repair and status in
one exact operator test before considering schedule changes. Timing and
parity remain unmeasured for this layout. For the newer Ollama TTFT
gap, require at least 13.55 ms per visit at the paired lower bound in
both orders. Preserve the final packed T128, decode and cut publication.
[T384 feasibility screen](../artifacts/mlx-gpu-gap-2026-11-15/astra62-nanbeige-t384-normal-work-screen-1/normal.json).

For request-summary qualification, compare the same CLI output mode with
summary collection enabled and disabled. Keep counters at existing completion
boundaries and write one bounded summary after the last token. Sample command
and GPU timings in a separate diagnostic run. Verify the paired and marginal
95% upper bounds for first-token and completion overhead in both owner orders.
The LFM3 fixed-N campaign's failing marginal TTFT upper is 1.06524%.
The eight-owner crossover counterbalances observation per request in
resident owners with the same preallocated recorder. The submit
shift under active collection changes sign between orders. Mean
completion rises by 8.03 and 16.02 ms. After the first token, nearly
all observed time differences occur inside native decode evaluation,
including 14.01 and 10.37 ms in completed split FFN GPU work.
Investigate native decode and keep the original fixed-N workload,
public clocks and uncertainty limits for qualification.
[LFM3 timing diagnostic](../artifacts/mlx-gpu-gap-2026-11-15/root-lfm3-o4-brackets-1/RUN.md) ·
[Crossover and decode attribution](../artifacts/mlx-gpu-gap-2026-11-15/sol62-lfm3-o4-observer-crossover-1/run-1/post-first-normal.json).

Astra's provisional LFM3 design allocates 0.460 s to first ID
and 0.672 s to the 31 completed decode steps against the controlled
vLLM clock. This allows 4.79 ms per T128 block and 21.67 ms per
completed T1 step before shared work. Fresh confirmation against
both engines determines the external target. The archived fast
same-weight Gate/Up outputs leave 418 changed Down inputs even
after canonical pointwise rounding. The first affected row
differs at 27 of 2,048 residual words, rejecting this leaf
substitution. A separate ConvIn decode trial keeps the original
arithmetic and reuses each activation across four output chains.
Exact two-order confirmation shows about 15% less decode
time and full-request paired lower savings of 4.48% and
4.05%. Those are below the 5% retention threshold. Preserve
this result when pricing a more complete GPU decode block.
Keep the original numerical, quality, state, memory and
telemetry gates.
[Astra performance and numerical proposal](../artifacts/mlx-gpu-gap-2026-11-15/astra-lfm3-external-arithmetic-budget-1/PROPOSAL.md) ·
[Exact row result](../artifacts/mlx-gpu-gap-2026-11-15/astra-lfm3-external-arithmetic-row-1/RESULT.md) ·
[ConvIn screen](../artifacts/mlx-gpu-gap-2026-11-15/root-lfm3-t1-convin-fma4-1/RUN.md) ·
[Connected ConvIn results](../artifacts/mlx-gpu-gap-2026-11-15/sol62-lfm3-t1-convin-connected-1/RUN.md).

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

The Gemma per-tile disabled/disabled control exceeds the 1% timing
threshold with summaries disabled. The largest shift is inside native
prefill and decode. Hold the full both-policy campaign until a
repeatable baseline is established.
[Control results](../artifacts/mlx-gpu-gap-2026-11-15/sol62-gemma-bf16-per-tile-o4-readiness-1/run-1/read-normal.json) ·
[Native phase split](../artifacts/mlx-gpu-gap-2026-11-15/sol62-gemma-bf16-per-tile-o4-readiness-1/run-1/phase-localization.json).

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

# Beat the MLX GPU baselines

Updated 15 November 2026. Status: **active goal; G1 diagnosis and diagnostic
implementation underway**. No new performance win, model adoption or overhead
pass is claimed. The [current cost map](gemma-current-cost-map.md) identifies CPU
attention and completion waits as the next targets; summary-component costing
remains open. Previous [delivery outcomes](improvement-outcomes.md) remain valid.

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

## Ordered implementation tasks

| Task | Concrete change | Exit and next dependency |
| --- | --- | --- |
| G0: freeze fair competitors | Admit fresh Ollama/vLLM runs with exact active tensors/tokenizer, precision, token work, offsets, capacity and cache policy. Separate numerical fixed-token work from ordinary CLI/HTTP requests. Add authoritative untimed input/output-ID preflight where missing. | Comparable or explicitly incomparable per metric; no old timing reused as a new baseline. Required before any claim of beating an engine. |
| G1: current-path cost map | Reuse the bounded profiler for early/late prefill and decode on current per-tile/batched paths. Add only missing CPU-stage attribution and observed copy/command/wait counts. Run uninstrumented acceptance timings separately. Measure the observability components above. | Ranked costs with coverage, source/shape/depth identities, completed work and uncertainty; no more whole-model tuning on assumptions. First code task. |
| O4: cheaper summaries | Store bounded owned typed terminal records; validate on insertion, serialize at explicit export. Retain immutable owner/manifest identity once and only changing request/state fields per record. Select disabled/summary callback specialization outside the token loop; reuse timestamps where boundaries coincide. | Original schema/field meanings and all failure/cancel/replay/output-correction cases pass; zero steady-state collector allocation; no new commands/waits; original fixed overhead protocol passes all eight gates or default remains disabled. Depends on G1 for causal choice. |
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

# Beat the MLX GPU baselines

Updated 16 November 2026. Status: **active goal; G1 and bounded G2 producer
pilot delivered; bounded three-layer numerical/state and required checks pass, Astra approves**. The [cost map](gemma-current-cost-map.md) identifies
CPU attention and completion waits as the next targets. The
[reusable attention building block](gemma-gpu-attention.md) prepares shared-buffer
block composition; [Apple hardware priorities](apple-silicon-gpu-goal.md) focus on
GPU residency, unified-memory reuse and bounded ANE feasibility. Astra research
and Sol implementation proceed in parallel; hardware measurements stay serial.
No new engine speedup, model adoption or overhead pass is claimed. Previous
[delivery outcomes](improvement-outcomes.md) remain valid.

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
and global guard/fatal precedence. Sol prepares the conditional integration in
parallel; frozen model/check sources remain unchanged.
[Actual candidate/check packet](../artifacts/mlx-gpu-gap-2026-11-15/g2-rms-two-phase-artifact-review-checks-1.json),
[Astra artifact review](../artifacts/mlx-gpu-gap-2026-11-15/astra-rms-two-phase-artifact-review-1/verdict.json),
[conditional integration](../artifacts/mlx-gpu-gap-2026-11-15/g2-rms-two-phase-integration-design-1/handoff.json).

Bridge the authenticated common-Q4 owner into resident execution before full
35-layer composition. Borrow its existing packed chunks and resident PLE; retain
Q4's float2 compensated dot,16,384 tile size and actual16-value codebook bound.
Do not reuse BF16's float3/8,192-tile proof by rounding Q4 bounds. Build fresh
Q4 own-input fixtures and preserve synchronous packed replay without a second
full model copy. Full original common-Q4 quality remains a separate gate.
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

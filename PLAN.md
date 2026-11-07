# Toklet implementation plan

Updated 7 November 2026. Read [TODO.md](TODO.md) first for the current task.
This live plan retains architecture, session invariants, milestone task criteria
and the complete validation/performance policy. Detailed chronology and original
execution orders are in the [dated archive](docs/archive/2026-11-07/PLAN.md).
Read a linked historical section only when the current task needs it.

M0–M7 are complete in their recorded, qualified scopes. Active Mac delivery is
complete; further physical iPhone work is deferred by the user. M8 support is
model/path/device-specific, and M9's first delivery is a qualified Mac opt-in.
Neither status grants blanket model, context, quality or device support.

Current extension contracts:

- [M8 shared blocks and capacity](docs/shared-block-model-plan.md),
  [model coverage](docs/model-coverage.md), and
  [detailed implementation plans](docs/todo-implementation-plans.md).
- [M9 measured performance](docs/next-performance-milestone.md) and
  [Mac affine decision](docs/m9-mac-affine-decision.md).
- [Observability/performance proposal](docs/observability-performance-proposal.md):
  design rationale; [O1 opt-in pilot](docs/o1-gemma-request-summaries.md) now implements
  scoped available observations with explicit native-phase/UI coverage gaps.
- [Selected model and metric plan](docs/model-observability-plan.md): observability
  first, then measured performance on Gemma 4 E2B, Qwen3.5-4B and Qwen3.5-9B.
- [Gemma3 native conformance](docs/gemma3-native-conformance.md): preserve historic
  fidelity rejections and original adoption-quality gates.

## Current model priority and observability gates

1. **Gemma 4 E2B first:** O1 request-summary metrics and O2 matched baseline on
   the existing qualified BF16 Mac route; then one measured optimization and
   independent evaluation of the preferred uncensored MLX artifact.
2. **Qwen3.5-4B next, then Qwen3.5-9B:** prefer qualified uncensored MLX affine
   4-bit artifacts. These replace Qwen2.5-1.5B as forward adoption targets;
   preserve its historical regression evidence. Inspect real graph/packing and
   qualify each artifact before ordinary app adoption.
3. **Under-4B track:** MiniCPM5-2B, LFM3-1.7B, Granite 4.2 3B, Ministral 3 3B
   Reasoning and Qwen3.5-2B. Exact total counts, ranking rationale, inspected MLX
   variants and conversion gaps are in the [shortlist](docs/model-observability-plan.md#five-candidates-strictly-below-4b-total-parameters).

The [metric contract](docs/model-observability-plan.md#metric-contract) is the
first implementation deliverable: request/phase timings, event/visible/answer
TTFT, prefill/decode throughput, state/cache work, owned/observed memory,
outcomes/cancellation, execution counters and telemetry completeness/overhead.
Collect baselines before optimization; unknown observations stay explicit.
Qualify summary overhead at ≤1% median TTFT/request time or keep it opt-in.
Keep the existing ≥5% default performance retain gate and 6 GiB working-set
ceiling. No model download, conversion or implementation is claimed by this
planning update. O1 now has an [opt-in implementation](docs/o1-gemma-request-summaries.md);
its measured overhead bound remains unqualified and missing native/UI probes
remain explicit follow-on work. O2.A now exposes completed native phases;
[O2.B2.A](docs/o2-ui-publication.md) adds Send/controller publication correlation.
Actual visible-frame timing remains unqualified.

The initial settings below describe the foundation. Later selected models and
routes use their own authenticated configuration and scoped acceptance evidence;
initial settings do not expand or replace those contracts.

## Scope and decisions

| Area | Initial decision |
| --- | --- |
| Workload | One interactive session, batch one, bounded output and context |
| Learning model | Vocabulary 64; 2 layers; hidden size 32; 4 query heads; 2 KV heads; head dimension 8; MLP size 64; capacity 32 |
| First pretrained model | A pinned revision of Qwen/Qwen2.5-0.5B-Instruct |
| CPU execution | Scalar operations over slices with FP32 storage; qualified FP64 projection/reduction accumulation; independent Python reference |
| Model import | Local safetensors, config, and tokenizer assets; explicit dtype conversion |
| Initial context | Short fixtures first, then at most 2,048 prompt plus output tokens |
| Metal execution | FP16 storage with qualified accumulation; compatible kernels first |
| First quantization | Versioned grouped affine Q4, group size 64, FP16 scale and bias; FP16 activations and KV |
| Mac memory | Initial 6 GiB engine working-set ceiling, including transient loading allocations |
| Applications | CLI first; Swift bridge and minimal native app after the Mac engine |

The original M0–M7 scope deferred Gemma, GGUF import, model discovery and downloads, speculative decoding, multi-session serving, paged KV, disk snapshots, and broad UI work. Gemma and model discovery/downloads are now active M8 work under the linked extension plans. TensorOps is an optional implementation choice within the Metal milestones after capability and shape measurements. Core AI and newer GPU features are separate experiments; they are not prerequisites.

The published Qwen config specifies hidden size 896, intermediate size 4,864, 24 layers, 14 query heads, 2 KV heads, vocabulary 151,936, BF16, and tied embeddings. Head dimension is therefore 64. Validate the pinned artifact rather than assuming every family member shares these values. The initial 2,048-token cap is an engine qualification limit. [Qwen configuration](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/blob/main/config.json).

## Architecture and ownership

Start with one Cargo package containing a library and CLI. Add each module when its first implementation lands. Keep CPU tests usable without Metal dependencies; make Apple bindings target-specific and Metal support a feature. Lock Rust and Python dependencies when introduced.

| Proposed location | Responsibility | Introduced in |
| --- | --- | --- |
| `src/cpu/` | Linear algebra, normalization, positions, attention, and activations | M1 |
| `src/model/` | Validated configuration, immutable weights, decoder order | M1 |
| `src/weights/` | Tensor import, dtype conversion, validation, later packing | M2 |
| `src/session.rs` | Token history, committed KV length, capacity, reusable scratch | M2 |
| `src/sampling.rs` | Greedy selection, later reproducible stochastic sampling | M1 |
| `src/generation.rs` | Bounded generation, stop rules, cancellation, token events | M2 |
| `src/bin/toklet.rs` | Model inspection, generation, benchmark entry points | M0 onward |
| `src/bench/` | Workloads, completed-work timing, result records | M2 |
| `src/metal/` and `src/metal/kernels/` | Buffers, pipelines, shaders, command submission | M4 |
| `tools/reference/` | Independent tiny oracle and pinned Transformers exporter | M0 onward |
| `tools/bench/` | Baseline adapters and comparison reports | M0 onward |
| `tests/fixtures/` | Small versioned inputs, weights, expected values, manifests | M0 onward |
| `benchmarks/` | Workload definitions, quality corpus manifests, result schema | M0 onward |
| `docs/experiments/` | Measured optimization decisions and reproduction commands | M3 onward |
| `apple/` | Swift bridge, native harness, deployment configuration | M7 |

Use contiguous row-major weights `[out_features, in_features]`, activations `[tokens, hidden]`, and separate per-layer K and V buffers `[kv_heads, capacity, head_dim]`. Keep head dimension contiguous and map query heads to their shared KV head without copying KV heads. Tiny fixtures use the same conventions as Qwen, including its rotary pairing convention and projection biases.

CPU operations initially remain ordinary functions. Introduce a narrow execution boundary during M4, after both implementations expose the necessary operations. Avoid a general tensor framework, graph compiler, or backend trait hierarchy in M0–M2.

The eventual public operations are load model, create bounded session, append token IDs, generate token events, cancel, reset, and read statistics. Model weights are immutable and shared; each session owns its state and scratch. The public API must not require returning the full logits tensor every step. Keep diagnostic tensor access separate from ordinary generation. Tokenization and chat formatting stay outside the numerical decoder.

## Session invariants

Implement these before optimizing state reuse:

1. Validate configuration, token IDs, dimensions, capacity, and checked allocation sizes before mutating a session. Reject requests whose prompt plus requested output exceeds the admitted limit.
2. Store token history separately from `evaluated_len`. KV is valid only for tokens before `evaluated_len`; a sampled token can be in history without having been evaluated. Define how stop tokens affect history and visible output.
3. Advance the committed cache position only after all layers complete successfully. On a partial failure, invalidate the affected suffix and replay before continuing. GPU buffer lifetimes extend through command completion.
4. Reuse state only for an exact token prefix with the same model, representation, and position semantics. M2 supports exact append; changed history can reset and replay. M5 may reuse the longest valid prefix.
5. Keep final logits or final hidden state when the whole prompt is already evaluated. A session with no pending token still needs a next-token distribution; KV alone is insufficient.
6. Cancellation stops submission at a documented boundary and leaves either a committed prefix or an explicit replay requirement. Reset clears positions and pending output state; a later request must not see stale KV.
7. Preallocate or grow in coarse bounded increments. Include allocated capacity, scratch, and retained conversion buffers in memory admission. Count shared CPU/GPU allocations once.

For the current requirement-to-exit-evidence map and remaining-limit classification,
see the [full-plan audit](docs/experiments/2026-10-21-full-plan-audit.md).
Dated evidence notes below retain the status known at collection; their pending
statements do not override subsequent accepted milestone closures.

## Milestones and dependencies

M0 → M1 → M2 → M3 → M4 → M5 → M6 → M7. Completed task criteria below
remain reference requirements; chronology, estimates and detailed results are
archived. M8/M9 follow the extension contracts above. Completion applies only
to the configurations and limits in the accepted evidence.

## M0 Establish the project and measurement contract

- [x] **M0.1 Create the package.** Add `Cargo.toml`, library, CLI shell, locked dependencies, and formatting/lint/test checks. Preserve the license and existing documents. Add artifact ignore rules before fetching weights. Exit: a clean checkout builds and tests without model downloads or Apple frameworks.
- [x] **M0.2 Freeze the tiny reference.** Use an independent NumPy or PyTorch implementation to export actual FP32 weights, token inputs, and named intermediate values. Record shapes, layout, epsilon, rotary settings, tool versions, and hashes. Include at least one manually checked linear example. Exit: Rust reads the saved arrays; correctness never depends on matching RNG seeds across languages.
- [x] **M0.3 Define workloads and quality gates.** Pin the source model/tokenizer revision and baseline software versions. Store token arrays, continuation tokens, prompt hashes, stopping policies, quality corpus identity, and the result schema. Freeze the numerical and quality policy below before lossy work.
- [x] **M0.4 Capture one baseline.** Inventory the actual Mac, power mode, OS, and build tools. Run MLX-LM on the 512-input/128-output workload and retain commands and raw samples. Prepare the llama.cpp comparison adapter now or during M2. Record timing or artifact differences explicitly.

[Detailed criteria, scoped closure and evidence](docs/archive/2026-11-07/PLAN.md#m0-establish-the-project-and-measurement-contract).

## M1 Build the tiny FP32 decoder

- [x] **M1.1 Linear operations.** Implement matrix-vector and simple matrix-matrix multiplication, optional bias, embedding lookup, and residual addition. Validate dimensions at operation boundaries. Cover rectangular shapes, bias handling, and invalid input sizes.
- [x] **M1.2 Stable elementwise and reduction operations.** Implement RMSNorm, max-subtracted softmax, SiLU, and gated MLP. Cover zero inputs, extreme finite values, and the policy for invalid or entirely masked rows.
- [x] **M1.3 Positions and causal attention.** Implement rotary positions with explicit pairing, scale, and absolute offsets; causal scaled attention; and grouped-query head mapping. Compare scores, probabilities, and attention outputs at several positions. Changing a future token must not affect an earlier output.
- [x] **M1.4 Complete the decoder.** Compose both layers, final normalization, tied output projection, and greedy selection. Initially recompute the complete prefix for each generated token. Define deterministic tie-breaking. Exit: saved intermediates, final logits, and a short greedy sequence agree with the independent reference within the frozen tolerances.

[Detailed criteria, scoped closure and evidence](docs/archive/2026-11-07/PLAN.md#m1-build-the-tiny-fp32-decoder).

## M2 Run Qwen with persistent state

- [x] **M2.1 Import and validate Qwen weights.** Reuse safetensors/config parsers and a half-precision conversion library. Inspect actual dtypes; explicitly convert BF16/FP16 to FP32. Validate expected names, shapes, Q/K/V bias presence, normalization epsilon, rotary parameters, and shared embedding/output storage. Support an index manifest if the selected artifact is sharded. Reject unsupported configurations with a useful error. Convert tensor by tensor to bound peak memory.
- [x] **M2.2 Compare the real forward pass.** Export selected intermediates and logits from pinned Transformers in FP32 with dropout disabled. Compare short fixed token sequences, including more than one layer and nonzero positions. Retain small selected probes in Git and regenerate larger outputs locally. Do this before judging decoded prose.
- [x] **M2.3 Add the KV session.** Implement bounded per-layer K/V and reusable scratch, evaluated versus sampled positions, and exact append. Compare one-shot, irregularly chunked, and token-by-token execution with full-prefix recomputation. Prove position offsets and capacity checks at the 2,048-token boundary.
- [x] **M2.4 Add bounded CLI generation.** Reuse the tokenizer parser; reproduce the pinned template and special-token behavior with golden token IDs. Accept local model paths, prompt or token input, maximum output, context capacity, and backend selection. Add greedy streaming, EOS/stop handling, reset, continuation, cancellation, and statistics. Keep template rendering in the CLI layer.
- [x] **M2.5 Add engine profiling and benchmark records.** Record model load/conversion, tokenization, prefill, first token, decode, sampling, and peak memory separately. Run the workload matrix and compare with the external baselines under the documented comparison track. CPU slowness is acceptable here; correctness, state behavior, and timing boundaries are the gate.

Exit: a local Qwen session generates, appends a user/tool-result suffix, cancels, resumes or replays explicitly, and resets without cache corruption. The 1,920-input/128-output workload succeeds; a request exceeding capacity fails before evaluation. Loading and execution stay within the admitted memory budget.

[Detailed criteria, scoped closure and evidence](docs/archive/2026-11-07/PLAN.md#m2-run-qwen-with-persistent-state).

## M3 Bound CPU optimization and ANE research

- [x] **M3.1 Improve one measured CPU bottleneck.** Spend approximately 4–6 hours on a profile-selected change such as contiguous traversal, vectorization, or bounded threading. Preserve the scalar oracle. Retain it only if full workloads improve without correctness or memory regressions.
- [x] **M3.2 Evaluate public CPU and ANE execution.** Spend the remaining 8–14 hours converting the tiny decoder or a supported slice, comparing CPU-only and CPU-and-Neural-Engine policies, and attempting persistent state. Capture actual placement where tooling permits, fallback, conversion cost, completed-work latency, and memory.
- [x] **M3.3 Close the experiment.** Record useful, uncompetitive, unsupported, or inconclusive, with commands and evidence. A slice result applies only to that slice. Adopt an ANE engine path only after complete stateful decode is measured; otherwise continue to M4 when the time budget ends.

[Detailed criteria, scoped closure and evidence](docs/archive/2026-11-07/PLAN.md#m3-bound-cpu-optimization-and-ane-research).

## M4 Implement complete FP16 Metal execution

- [x] **M4.1 Establish the Apple backend.** Qualify the Xcode/Metal toolchain, deployment target, device capability checks, buffer ownership, and command completion. Use a small `objc2`/`objc2-metal` boundary, following the [metal-rs migration recommendation](https://github.com/gfx-rs/metal-rs). Add a minimal kernel dispatch and CPU-versus-GPU comparison before porting the model.
- [x] **M4.2 Port the forward pass in slices.** Implement embedding, projections, RMSNorm, rotary/cache writes, causal GQA, gated MLP, residuals, final normalization, and output projection. Begin with one layer and then the full tiny model, followed by Qwen. Qualify reduction accumulation and FP16 ranges using intermediate checks. CPU greedy sampling is acceptable initially.
- [x] **M4.3 Keep model and session buffers resident.** Preallocate scratch for admitted shapes; cache pipeline objects. Load FP16 buffers without retaining an unnecessary full FP32 copy. Run CPU/GPU comparisons in separate processes when needed to respect peak memory. Introduce the narrow backend interface here.
- [x] **M4.4 Validate scheduling and failure boundaries.** Remove debugging waits after individual operators once parity passes. Preserve dependencies, wait at required request/token boundaries, and maintain committed state on cancellation or GPU failure. Capture a GPU trace that separates kernel time, host encoding, allocations, and idle gaps.

Exit: the M2 session suite passes with FP16-specific numerical gates, complete Qwen generation runs on Metal, allocations remain bounded across repeated tokens, and the trace explains remaining costs. Speed parity with mature engines is measured separately from functional completion.


[Detailed criteria, scoped closure and evidence](docs/archive/2026-11-07/PLAN.md#m4-implement-complete-fp16-metal-execution).

## M5 Optimize decode and prefill

- [x] **M5.1 Freeze and test Q4 format version 1.** Specify row/group order, nibble order, rounding/clipping, constant groups, padding, alignment, endianness, FP16 metadata rounding, and excluded tensors. Start with groups of 64 along each linear row; reconstruct as `weight = scale * code + bias`, with codes 0–15. Keep norms, biases, and initially the tied embedding/output matrix in FP16. Implement a Python packer and simple Rust dequantization oracle. Test constants, extreme values, row tails, and exact packed-byte round trips. This is an engine format, not an implied claim of MLX compatibility.
- [x] **M5.2 Implement fused Q4 decode.** Dequantize tiles or values during matrix-vector execution; do not expand the whole model per token. Compare reconstructed-weight math with the CPU oracle and run the frozen quality suite against the unquantized checkpoint. Preserve the FP16 path if Q4 misses its quality gate.
- [x] **M5.3 Implement separate prefill.** Add tiled matrix-matrix execution and reuse dequantized weights across prompt rows. Sweep chunks of 128, 256, 512, and 1,024 within the memory ceiling. Compute vocabulary logits only for required positions. Verify irregular final chunks and nonzero cached offsets; retain simple attention until M6.
- [x] **M5.4 Reduce measured dispatch and traffic costs.** Evaluate GPU greedy selection returning a token ID, residual/normalization fusion, rotary/cache-write fusion, and MLP gating fusion individually. Add stochastic sampling only when required, with explicit RNG state and all filters/penalties in the measured path. Retain a small shape/device dispatch table rather than a general autotuner.
- [x] **M5.5 Optimize conversation continuation.** Find the longest exact shared token prefix, truncate only representable state, and replay from the required point. Handle an entirely cached prompt, a pending sampled token, and edits near the start or end. Compare continuation latency with reset-and-replay.
- [x] **M5.6 Qualify the performance release.** Repeat controlled and application comparisons, sustained runs, and quality checks. Select one larger checkpoint in the implemented Qwen family using actual tensor payload and memory admission; record its exact artifact before claiming behavior at larger scale. Reject configurations that exceed 6 GiB, including high-precision exclusions and load-time scratch.

Exit: Q4 passes quality checks and improves at least one named workload beyond measured noise; other accepted optimizations satisfy their declared tradeoffs. Report baseline gaps if they remain. Claim an external speed lead only when the same quality, memory, and comparison conditions support it.

[Detailed criteria, scoped closure and evidence](docs/archive/2026-11-07/PLAN.md#m5-optimize-decode-and-prefill).

## M6 Qualify longer context and attention

- [x] **M6.1 Implement tiled online-softmax attention.** Build separate decode and prefill paths without materializing the full attention-score matrix. Preserve causal boundaries, absolute positions, and GQA sharing. Compare irregular chunk sizes and substantial cached depths with the reference.
- [x] **M6.2 Extend context admission and profiling.** Measure 4K, 8K, 16K, and 32K total-token workloads only where the pinned model permits them. Reserve output capacity: at a validated limit of 32,768, a 128-output boundary run has 32,640 prompt tokens. Account for allocated KV, workspace, and model limits before dispatch.
- [x] **M6.3 Evaluate optional KV quantization.** Only if measured KV traffic or capacity warrants it, try an eight-bit cache with explicit K/V grouping and dequantization inside attention tiles. Repeat loss, long-context behavior, and latency checks. Retain FP16 KV if the tradeoff is unfavorable.

[Detailed criteria, scoped closure and evidence](docs/archive/2026-11-07/PLAN.md#m6-qualify-longer-context-and-attention).

## M7 Integrate and qualify a physical iPhone

- [x] **M7.1 Add the native boundary.** Build a thin C ABI or other explicitly chosen Swift bridge with opaque model/session handles, ownership rules, structured errors, cancellation, and token events. Contain Rust panics at the boundary. Package libraries for supported Apple targets and set deployment requirements from qualified APIs.
- [x] **M7.2 Add a minimal Swift harness.** Exercise load, prompt, token streaming, continuation, reset, and cancellation. Run inference away from the UI thread and define callback lifetime and serialization. Reuse the same Rust engine and local artifact format.
- [x] **M7.3 Qualify the physical device.** Measure an independent iPhone 14 Pro memory budget, thermal behavior, sustained throughput, and energy where instrumentation is available. Test inactive/background transitions, memory warnings, interruption during generation, resume/replay, and orderly resource release. Add iPad and newer Macs as separate device qualifications.

Exit: publish a reproducible device report and supported model/context limits. If physical hardware is unavailable, leave M7 pending; simulator results establish integration behavior only.


[Detailed criteria, scoped closure and evidence](docs/archive/2026-11-07/PLAN.md#m7-integrate-and-qualify-a-physical-iphone).

## Validation and performance policy

### Correctness and quality

Fast offline checks cover primitives, tiny intermediate values, causal masking, head mapping, rotary offsets, cache equivalence, tied weights, invalid shapes/tokens, empty input policy, exact capacity, reset, stop behavior, and cancellation recovery. Regression fixtures include a fully cached prompt and one whose last sampled token is not yet in KV.

Start tiny FP32 elementwise comparisons with `abs(actual - expected) <= 1e-5 + 1e-4 * abs(expected)`. In M0, record the measured reference envelope and freeze per-operation limits. Establish separate limits for real FP32, FP16, and quantized execution before accepting those implementations. Track absolute/relative error and teacher-forced loss; a near-tied argmax change requires investigation rather than automatic failure or relaxed thresholds.

Proposed initial lossy-quality gates are mean teacher-forced negative-log-likelihood increase no greater than 0.05 nats/token and task pass-rate loss no greater than two percentage points relative to the unquantized reference. Freeze the corpus, metric aggregation, and thresholds in M0; use at least 10,000 held-out scored tokens and 100 fixed task cases spanning prose, code, and continuation. These are project acceptance choices, not model quality claims. Keep calibration/quantizer tuning data separate from held-out checks. Once evaluation begins, changing a gate is an explicit scope change, not a way to accept a failing optimization.

### Workloads and comparison rules

| Workload | Initial definition |
| --- | --- |
| Short, medium, long prompt | 128, 512, and 1,024 input tokens, each with 128 output steps |
| Capacity boundary | 1,920 input plus 128 output tokens; also test one token beyond capacity |
| Continuation | Exact-prefix append, fully cached prompt, pending sampled token, changed-prefix replay |
| Cold and warm | Separate new-process loading/compilation from loaded-model empty-session requests |
| Sustained | A documented sequence of repeated requests for at least 10 minutes, with cache resets and thermal/power conditions recorded |
| Application trace | Fixed multi-turn coding/tool-result transcript, reported separately from raw throughput |

Controlled comparisons require matching source checkpoint, token IDs, output step count, compatible weight/KV representation, and cache semantics. Use teacher-forced continuations when matching generated tokens is necessary. Application comparisons may use each runtime's practical quantization only after applying the same quality and memory policy. Do not combine the two tracks in one unlabeled ranking.

Use pinned Transformers for numerical checks, MLX-LM for Apple single-session/cache behavior, and llama.cpp Metal for an independent throughput comparison. MLX-LM exposes prompt-cache support. [MLX-LM documentation](https://github.com/ml-explore/mlx-lm). The llama-bench core excludes tokenization and sampling, so compare matching engine timings or add application timers. [llama-bench documentation](https://github.com/ggml-org/llama.cpp/blob/master/tools/llama-bench/README.md).

### Measurements and acceptance

Use release builds, three warmups, and ten measured repetitions for initial short comparisons. Retain raw samples; report median and dispersion, without inferring reliable tail percentiles from ten requests. Alternate candidates when thermal or power drift matters, and repeat a promising change in a second run. A practical default retain threshold is at least 5% improvement in the declared primary metric, exceeding observed run-to-run variation, with no unexplained regression in the other required workloads. Improvements smaller than that need stronger measurement before acceptance.

Measure warm TTFT from request submission to the first token event; report first-visible-text latency separately. For `N > 1`, decode rate is `(N - 1) / (last_token_time - first_token_time)`. Also report prefill tokens/s, per-token decode latency at named depths, complete request time, continuation time, and process footprint/system memory pressure. Time completed GPU work, not command submission. State the instrument and scope for energy readings; leave unavailable data unset.

Each result records device/OS/power details, engine and baseline commits, compiler/build flags, dependency versions, model/tokenizer revisions and hashes, packed-format version, weight/KV dtype and groups, capacity and used context, token hashes, stop policy, emitted count, cache state, timing boundaries, raw samples, and memory measurements. Packed-weight caches additionally key on source hash and relevant backend properties.

Each optimization record contains its hypothesis, code/config difference, correctness and quality results, workload, timings, memory impact, interpretation, and retain/revert decision.

### Checks at each change

Once M0 creates the package, run formatting, linting, and the fast CPU suite on ordinary changes. Run model parity checks for model/loader changes, the session suite for state changes, and GPU validation plus a small representative workload for shader changes. Real-model downloads and physical-device tests are explicit suites; unavailable hardware is reported as unverified. Broaden performance runs only when the affected behavior warrants them.

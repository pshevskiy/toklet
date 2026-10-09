# Toklet implementation plan

Prepared 9 October 2026. Status: M0 and M1 are complete; M2–M7 remain pending.

Build a correct Rust decoder, run Qwen2.5-0.5B-Instruct with persistent session state, and use that reference to develop a measured Metal implementation. The first useful release is a CPU CLI with reliable continuation. The performance release adds FP16 Metal execution, Q4 weights, and separate prefill and decode paths. Physical iPhone qualification follows the Mac engine.

This plan translates the [Apple LLM performance proposal](apple_llm_performance_plan.md) into repository work. The proposal supplies design context; its references to an existing linear layer and earlier plan do not establish completed work in this checkout.

## Starting point

Before implementation, the repository contained a short README, license, ignore file, and the supplied performance proposal. There was no Cargo package, engine code, fixture suite, or benchmark harness. M0 establishes the package, frozen independent fixtures, workloads, quality gates, and the first native MLX baseline. M1 implements the complete scalar tiny decoder and greedy trace. Pretrained loading and cached sessions remain pending.

The environment reports Rust and Cargo 1.90.0, macOS 26.6.2 (25G83), and `/Library/Developer/CommandLineTools` as the active developer directory. Read-only inventory confirms Apple M1 Pro, MacBookPro18,3, and 16 GiB physical memory. AC power is attached with AC low-power mode disabled. `xcrun --find metal` fails; qualify an Xcode and Metal toolchain before engine GPU work. A hardware inventory does not establish numerical or performance qualification.

## Scope and decisions

| Area | Initial decision |
| --- | --- |
| Workload | One interactive session, batch one, bounded output and context |
| Learning model | Vocabulary 64; 2 layers; hidden size 32; 4 query heads; 2 KV heads; head dimension 8; MLP size 64; capacity 32 |
| First pretrained model | A pinned revision of Qwen/Qwen2.5-0.5B-Instruct |
| CPU execution | Scalar FP32 operations over slices; independent Python reference |
| Model import | Local safetensors, config, and tokenizer assets; explicit dtype conversion |
| Initial context | Short fixtures first, then at most 2,048 prompt plus output tokens |
| Metal execution | FP16 storage with qualified accumulation; compatible kernels first |
| First quantization | Versioned grouped affine Q4, group size 64, FP16 scale and bias; FP16 activations and KV |
| Mac memory | Initial 6 GiB engine working-set ceiling, including transient loading allocations |
| Applications | CLI first; Swift bridge and minimal native app after the Mac engine |

Defer Gemma, GGUF import, model discovery and downloads, speculative decoding, multi-session serving, paged KV, disk snapshots, and broad UI work. TensorOps is an optional implementation choice within the Metal milestones after capability and shape measurements. Core AI and newer GPU features are separate experiments; they are not prerequisites.

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

## Milestones and dependencies

The main dependency chain is M0 → M1 → M2 → M3 → M4 → M5 → M6 → M7. M3 ends when its time limit expires even if ANE qualification fails. Mobile build preparation can begin after M4, but final device qualification uses the selected M5/M6 configuration.

| Milestone | Deliverable and exit gate | Focused hours |
| --- | --- | ---: |
| M0 | Buildable package, immutable fixtures, workload contract, first external baseline | 6–10 |
| M1 | Tiny FP32 decoder agrees with independent intermediate values and logits | 20–35 |
| M2 | Qwen CPU CLI, persistent KV, reliable continuation and cancellation through 2,048 total tokens | 35–60 |
| M3 | One bounded CPU optimization and a documented CPU/ANE feasibility result | 12–20 |
| M4 | Full FP16 Metal forward pass, correct generation, memory accounting, explained GPU trace | 60–100 |
| M5 | Qualified Q4 decode, efficient prefill, resident state, repeated performance comparisons | 80–140 |
| M6 | Tiled attention and selected longer contexts meet memory and correctness gates | 45–80 |
| M7 | Physical iPhone sessions pass lifecycle, memory, and sustained-run checks | 40–70 |

The summed estimate is 298–515 focused hours, approximately 300–520. M0–M2 provide the useful CPU engine in 61–105 hours. At 10–14 hours per week, the complete scope is roughly five to twelve months. Re-estimate after M2 and M4 using actual effort. These are planning ranges; exceeding mature runtimes is a research outcome, not a delivery guarantee.

## M0 Establish the project and measurement contract

- [x] **M0.1 Create the package.** Add `Cargo.toml`, library, CLI shell, locked dependencies, and formatting/lint/test checks. Preserve the license and existing documents. Add artifact ignore rules before fetching weights. Exit: a clean checkout builds and tests without model downloads or Apple frameworks.
- [x] **M0.2 Freeze the tiny reference.** Use an independent NumPy or PyTorch implementation to export actual FP32 weights, token inputs, and named intermediate values. Record shapes, layout, epsilon, rotary settings, tool versions, and hashes. Include at least one manually checked linear example. Exit: Rust reads the saved arrays; correctness never depends on matching RNG seeds across languages.
- [x] **M0.3 Define workloads and quality gates.** Pin the source model/tokenizer revision and baseline software versions. Store token arrays, continuation tokens, prompt hashes, stopping policies, quality corpus identity, and the result schema. Freeze the numerical and quality policy below before lossy work.
- [x] **M0.4 Capture one baseline.** Inventory the actual Mac, power mode, OS, and build tools. Run MLX-LM on the 512-input/128-output workload and retain commands and raw samples. Prepare the llama.cpp comparison adapter now or during M2. Record timing or artifact differences explicitly.

M0.1 and M0.2 can proceed independently. M0.4 depends on M0.3. Keep large source models and generated benchmark data outside Git; commit small fixtures, manifests, scripts, and compact result summaries.

M0.1 evidence, 9 October 2026: the dependency-free library and CLI shell passed `cargo fmt --check`, `cargo clippy --locked --offline --all-targets -- -D warnings`, `cargo test --locked --offline` (four CLI integration tests), and `cargo build --locked --offline --release` with Rust/Cargo 1.90.0 on macOS arm64. Release build and tests also passed in an isolated source copy with initially empty Cargo cache and target directories. Artifact ignore checks preserve small fixtures and reports; README links and package-file whitespace passed. Astra reviewed the completed code task and reported no findings. The GitHub Actions workflow is configured but has not run remotely; no numerical, GPU, or performance qualification is claimed.

M0.2 evidence, 9 October 2026: [tiny-v1](tests/fixtures/tiny-v1/manifest.json) freezes 242 arrays with 383,720 payload bytes, exported by the [independent NumPy oracle](tools/reference/README.md) using Python 3.12.8 and NumPy 2.3.2. The per-operation FP32 gate remains `1e-5 + 1e-4 * abs(expected)`; the largest measured FP32/FP64 discrepancy uses 7.8058% of that tolerance. Both layers' intermediates, logits, single-token/offset/causal/capacity cases, stable probes, manual linear outputs, and greedy trace are saved with shapes and SHA-256 hashes. `cargo fmt --check`, Clippy with warnings denied, all eight offline Rust tests, release build, and `python3 tools/reference/tiny.py --check tests/fixtures/tiny-v1` passed. A second export reproduced every file exactly in the same environment; overwrite refusal and corrupted-array rejection passed. The Rust tests use locked development dependencies, fetched once before offline checks. Astra reported no findings after independently checking masking, probabilities, rotary norms, scalar GQA contractions, and recorded FP64 envelopes. No Rust decoder or real-model parity is claimed.

Review and commit policy: Astra reviews each completed task that adds or changes code before the next code task begins. Run the applicable checks and commit each completed task after review passes. Individual edits and documentation-only updates do not require separate reviews. Continue autonomously within the plan and report unavailable external or hardware checks accurately.

M0.3 evidence, 9 October 2026: [source identities](benchmarks/sources.json), [contract v1](benchmarks/contract-v1/manifest.json), [quality rules](benchmarks/README.md), and [result schema](benchmarks/result.schema.json) are frozen. Qwen/tokenizer revision is `7ae557604adf67be50417f59c2c2f167def9a775`; baseline versions and all preparation dependencies are locked. Throughput and continuation token arrays, 10,000 held-out loss targets, and 100 fixed quality task identities are retained with hashes and an offline preparation recipe. Original text, tokenizer assets, and prepared task payloads remain ignored local artifacts. Formatting, Clippy with warnings denied, ten Rust tests, offline contract verification, local quality-payload verification, and three Python integrity/schema tests passed. Astra independently verified source hashes, tokenizations and admission bounds, and accepted the task after the schema was corrected to represent native BF16 baseline KV. No baseline timings or quality scores are claimed yet. The next code task in the initial queue is M1.1; M0.4 remains pending.

M0.4 evidence, 9 October 2026: [adapter](tools/bench/mlx_baseline.py), [raw record](benchmarks/results/2026-10-09-mlx-bf16-512-128.json), and [report](benchmarks/results/2026-10-09-mlx-bf16-512-128.md) retain the native BF16 MLX-LM 512-input/128-output workload on Apple M1 Pro with 16 GiB. Three warmups and ten measured requests gave median warm TTFT 134.531 ms, decode 115.29 steps/s, and total 1.237924 s; own-process physical peak was 1,941,932,168 bytes. Timing excludes tokenization, sampling, detokenization, and visible-text latency. Source verification warms the file cache, so loading is not a cold-storage measurement. Formatting, Clippy with warnings denied, all sixteen Rust tests, seven Python checks, schema and whitespace checks passed. Astra accepted after finding and correcting unnecessary prefix vocabulary evaluation, followed by fresh measurements and exact adapter-hash verification. No quality ranking or Toklet performance qualification is claimed. The llama.cpp adapter remains deferred to M2; M1.2 is next.

## M1 Build the tiny FP32 decoder

Each ticket depends on the previous numerical building blocks and M0.2 fixtures.

- [x] **M1.1 Linear operations.** Implement matrix-vector and simple matrix-matrix multiplication, optional bias, embedding lookup, and residual addition. Validate dimensions at operation boundaries. Cover rectangular shapes, bias handling, and invalid input sizes.
- [x] **M1.2 Stable elementwise and reduction operations.** Implement RMSNorm, max-subtracted softmax, SiLU, and gated MLP. Cover zero inputs, extreme finite values, and the policy for invalid or entirely masked rows.
- [x] **M1.3 Positions and causal attention.** Implement rotary positions with explicit pairing, scale, and absolute offsets; causal scaled attention; and grouped-query head mapping. Compare scores, probabilities, and attention outputs at several positions. Changing a future token must not affect an earlier output.
- [x] **M1.4 Complete the decoder.** Compose both layers, final normalization, tied output projection, and greedy selection. Initially recompute the complete prefix for each generated token. Define deterministic tie-breaking. Exit: saved intermediates, final logits, and a short greedy sequence agree with the independent reference within the frozen tolerances.

Keep this full-prefix CPU path as the oracle for the cached implementation. It need not be fast.

M1.1 evidence, 9 October 2026: [scalar CPU operations](src/cpu/mod.rs) implement row-major FP32 matrix-vector/matrix-matrix projection, optional bias, embedding lookup, and in-place residual addition without allocating buffers. Shape arithmetic, lengths, token bounds, and finite inputs are checked before writing; arithmetic errors require discarding any partial output. [Six targeted tests](tests/cpu_linear.rs) compare every saved linear projection across both layers and all five tiny cases, plus tied output, embeddings, residual composition, the manual example, and invalid-input behavior. Formatting, Clippy with warnings denied, all sixteen Rust tests, and whitespace checks passed. Astra independently reviewed bounds, ownership, accumulation order, and failure semantics and reported no findings. These are correctness references, with no performance qualification. M0.4 is next in the initial queue.

M1.2 evidence, 9 October 2026: [stable CPU operations](src/cpu/reductions.rs) add RMSNorm, boolean-masked softmax, stable SiLU, elementwise gating, and the complete bias-free gated MLP. Borrowed weights and caller-owned output/scratch preserve explicit ownership without allocations. Normalization and softmax use documented FP64 intermediate arithmetic with FP32 storage; linear and activation operations remain FP32. Invalid epsilon, shapes, nonfinite inputs, and entirely masked rows fail before writes; arithmetic failures require discarding partial buffers. [Six targeted tests](tests/cpu_reductions.rs) cover every saved normalization/probability/MLP across both layers and all five cases, frozen edge probes, FP32 extremes, masks, and failure behavior under the unchanged tolerances. Formatting, Clippy with warnings denied, all twenty-two Rust tests, independent oracle verification, and whitespace checks passed. Astra independently accepted without findings. No performance qualification is claimed. M1.3 is next.

M1.3 evidence, 9 October 2026: [rotary and attention](src/cpu/attention.rs) implement unscaled absolute positions with Qwen split-half pairing, causal scaled dot products, and grouped-query mapping without expanded KV storage. Rotary frequencies/angles/trigonometry use FP64 with FP32 rotation coefficients and output; attention dots and context accumulate in FP32, using the existing stable softmax. Layouts and numerical/error policies are documented. [Six tests](tests/cpu_attention.rs) compare rotary values, unmasked scores, causal probabilities, and contexts across all five cases and both layers, plus manual pairing/head mapping, chunk-offset identity, norm preservation, future-token invariance, and invalid/overflow behavior. Formatting, Clippy with warnings denied, all twenty-eight Rust tests, and whitespace checks passed. Astra independently accepted without findings. No performance qualification is claimed. M1.4 is next.

M1.4 evidence, 9 October 2026: [immutable decoder and scratch](src/model/mod.rs), [diagnostic trace](src/model/trace.rs), and [greedy selection](src/sampling.rs) compose the complete scalar full-prefix path. Ordinary calls reuse bounded scratch and return final-token logits; diagnostic calls allocate all-position arrays separately with checked size arithmetic and fallible reserves. The embedding table is the tied output projection. [Seven tests](tests/cpu_decoder.rs) match every named intermediate and final logits across both layers and all five cases, plus four saved greedy steps, lowest-ID ties, exact causal-prefix invariance, tied weights, configuration/weight validation, capacity rejection, and scratch reuse after rejected requests. Formatting, Clippy with warnings denied, all thirty-five Rust tests, release build, and whitespace checks passed. Astra independently accepted without findings and confirmed ordinary execution has no allocation sites by inspection; no allocation-count or performance measurement is claimed. M1's exit gate is complete. M2.1 is next.

## M2 Run Qwen with persistent state

- [ ] **M2.1 Import and validate Qwen weights.** Reuse safetensors/config parsers and a half-precision conversion library. Inspect actual dtypes; explicitly convert BF16/FP16 to FP32. Validate expected names, shapes, Q/K/V bias presence, normalization epsilon, rotary parameters, and shared embedding/output storage. Support an index manifest if the selected artifact is sharded. Reject unsupported configurations with a useful error. Convert tensor by tensor to bound peak memory.
- [ ] **M2.2 Compare the real forward pass.** Export selected intermediates and logits from pinned Transformers in FP32 with dropout disabled. Compare short fixed token sequences, including more than one layer and nonzero positions. Retain small selected probes in Git and regenerate larger outputs locally. Do this before judging decoded prose.
- [ ] **M2.3 Add the KV session.** Implement bounded per-layer K/V and reusable scratch, evaluated versus sampled positions, and exact append. Compare one-shot, irregularly chunked, and token-by-token execution with full-prefix recomputation. Prove position offsets and capacity checks at the 2,048-token boundary.
- [ ] **M2.4 Add bounded CLI generation.** Reuse the tokenizer parser; reproduce the pinned template and special-token behavior with golden token IDs. Accept local model paths, prompt or token input, maximum output, context capacity, and backend selection. Add greedy streaming, EOS/stop handling, reset, continuation, cancellation, and statistics. Keep template rendering in the CLI layer.
- [ ] **M2.5 Add engine profiling and benchmark records.** Record model load/conversion, tokenization, prefill, first token, decode, sampling, and peak memory separately. Run the workload matrix and compare with the external baselines under the documented comparison track. CPU slowness is acceptable here; correctness, state behavior, and timing boundaries are the gate.

M2.1 follows M1.4; M2.2 follows M2.1. M2.3 can start on the tiny model after M1.4 and must pass real-model checks before M2 closes. M2.4 depends on M2.3; M2.5 completes after the generation path is stable.

Exit: a local Qwen session generates, appends a user/tool-result suffix, cancels, resumes or replays explicitly, and resets without cache corruption. The 1,920-input/128-output workload succeeds; a request exceeding capacity fails before evaluation. Loading and execution stay within the admitted memory budget.

## M3 Bound CPU optimization and ANE research

- [ ] **M3.1 Improve one measured CPU bottleneck.** Spend approximately 4–6 hours on a profile-selected change such as contiguous traversal, vectorization, or bounded threading. Preserve the scalar oracle. Retain it only if full workloads improve without correctness or memory regressions.
- [ ] **M3.2 Evaluate public CPU and ANE execution.** Spend the remaining 8–14 hours converting the tiny decoder or a supported slice, comparing CPU-only and CPU-and-Neural-Engine policies, and attempting persistent state. Capture actual placement where tooling permits, fallback, conversion cost, completed-work latency, and memory.
- [ ] **M3.3 Close the experiment.** Record useful, uncompetitive, unsupported, or inconclusive, with commands and evidence. A slice result applies only to that slice. Adopt an ANE engine path only after complete stateful decode is measured; otherwise continue to M4 when the time budget ends.

Use the public [CPU-and-Neural-Engine compute policy](https://developer.apple.com/documentation/coreml/mlcomputeunits/cpuandneuralengine). Core ML supports stateful buffers from iOS 18/macOS 15; conversion success alone does not demonstrate ANE placement. [Core ML stateful models](https://apple.github.io/coremltools/docs-guides/source/stateful-models.html).

## M4 Implement complete FP16 Metal execution

- [ ] **M4.1 Establish the Apple backend.** Qualify the Xcode/Metal toolchain, deployment target, device capability checks, buffer ownership, and command completion. Use a small `objc2`/`objc2-metal` boundary, following the [metal-rs migration recommendation](https://github.com/gfx-rs/metal-rs). Add a minimal kernel dispatch and CPU-versus-GPU comparison before porting the model.
- [ ] **M4.2 Port the forward pass in slices.** Implement embedding, projections, RMSNorm, rotary/cache writes, causal GQA, gated MLP, residuals, final normalization, and output projection. Begin with one layer and then the full tiny model, followed by Qwen. Qualify reduction accumulation and FP16 ranges using intermediate checks. CPU greedy sampling is acceptable initially.
- [ ] **M4.3 Keep model and session buffers resident.** Preallocate scratch for admitted shapes; cache pipeline objects. Load FP16 buffers without retaining an unnecessary full FP32 copy. Run CPU/GPU comparisons in separate processes when needed to respect peak memory. Introduce the narrow backend interface here.
- [ ] **M4.4 Validate scheduling and failure boundaries.** Remove debugging waits after individual operators once parity passes. Preserve dependencies, wait at required request/token boundaries, and maintain committed state on cancellation or GPU failure. Capture a GPU trace that separates kernel time, host encoding, allocations, and idle gaps.

Before writing a substantial custom GEMM, compare an available Apple matrix primitive or TensorOps implementation at the actual projection shapes. Retain compatible kernels when the API is absent or slower; record the chosen device/OS requirements.

Exit: the M2 session suite passes with FP16-specific numerical gates, complete Qwen generation runs on Metal, allocations remain bounded across repeated tokens, and the trace explains remaining costs. Speed parity with mature engines is measured separately from functional completion.

## M5 Optimize decode and prefill

- [ ] **M5.1 Freeze and test Q4 format version 1.** Specify row/group order, nibble order, rounding/clipping, constant groups, padding, alignment, endianness, FP16 metadata rounding, and excluded tensors. Start with groups of 64 along each linear row; reconstruct as `weight = scale * code + bias`, with codes 0–15. Keep norms, biases, and initially the tied embedding/output matrix in FP16. Implement a Python packer and simple Rust dequantization oracle. Test constants, extreme values, row tails, and exact packed-byte round trips. This is an engine format, not an implied claim of MLX compatibility.
- [ ] **M5.2 Implement fused Q4 decode.** Dequantize tiles or values during matrix-vector execution; do not expand the whole model per token. Compare reconstructed-weight math with the CPU oracle and run the frozen quality suite against the unquantized checkpoint. Preserve the FP16 path if Q4 misses its quality gate.
- [ ] **M5.3 Implement separate prefill.** Add tiled matrix-matrix execution and reuse dequantized weights across prompt rows. Sweep chunks of 128, 256, 512, and 1,024 within the memory ceiling. Compute vocabulary logits only for required positions. Verify irregular final chunks and nonzero cached offsets; retain simple attention until M6.
- [ ] **M5.4 Reduce measured dispatch and traffic costs.** Evaluate GPU greedy selection returning a token ID, residual/normalization fusion, rotary/cache-write fusion, and MLP gating fusion individually. Add stochastic sampling only when required, with explicit RNG state and all filters/penalties in the measured path. Retain a small shape/device dispatch table rather than a general autotuner.
- [ ] **M5.5 Optimize conversation continuation.** Find the longest exact shared token prefix, truncate only representable state, and replay from the required point. Handle an entirely cached prompt, a pending sampled token, and edits near the start or end. Compare continuation latency with reset-and-replay.
- [ ] **M5.6 Qualify the performance release.** Repeat controlled and application comparisons, sustained runs, and quality checks. Select one larger checkpoint in the implemented Qwen family using actual tensor payload and memory admission; record its exact artifact before claiming behavior at larger scale. Reject configurations that exceed 6 GiB, including high-precision exclusions and load-time scratch.

M5.2 depends on M5.1. M5.3 and M5.5 can start from M4 independently of Q4; M5.4 follows the relevant profile. M5.6 requires all accepted changes. Grouped affine quantization is a practical candidate, but container names alone do not establish format compatibility. [MLX quantization API](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.quantize.html).

Exit: Q4 passes quality checks and improves at least one named workload beyond measured noise; other accepted optimizations satisfy their declared tradeoffs. Report baseline gaps if they remain. Claim an external speed lead only when the same quality, memory, and comparison conditions support it.

## M6 Qualify longer context and attention

- [ ] **M6.1 Implement tiled online-softmax attention.** Build separate decode and prefill paths without materializing the full attention-score matrix. Preserve causal boundaries, absolute positions, and GQA sharing. Compare irregular chunk sizes and substantial cached depths with the reference.
- [ ] **M6.2 Extend context admission and profiling.** Measure 4K, 8K, 16K, and 32K total-token workloads only where the pinned model permits them. Reserve output capacity: at a validated limit of 32,768, a 128-output boundary run has 32,640 prompt tokens. Account for allocated KV, workspace, and model limits before dispatch.
- [ ] **M6.3 Evaluate optional KV quantization.** Only if measured KV traffic or capacity warrants it, try an eight-bit cache with explicit K/V grouping and dequantization inside attention tiles. Repeat loss, long-context behavior, and latency checks. Retain FP16 KV if the tradeoff is unfavorable.

For the first Qwen config, calculated KV payload is `2 × 24 × tokens × 2 × 64 × bytes_per_element`: 48 MiB at 2,048 tokens in FP32, or 24 MiB in FP16. This is logical KV only; use measured footprint for admission validation. Selected depths must fit the budget and pass long-context quality checks before becoming supported configurations.

## M7 Integrate and qualify a physical iPhone

- [ ] **M7.1 Add the native boundary.** Build a thin C ABI or other explicitly chosen Swift bridge with opaque model/session handles, ownership rules, structured errors, cancellation, and token events. Contain Rust panics at the boundary. Package libraries for supported Apple targets and set deployment requirements from qualified APIs.
- [ ] **M7.2 Add a minimal Swift harness.** Exercise load, prompt, token streaming, continuation, reset, and cancellation. Run inference away from the UI thread and define callback lifetime and serialization. Reuse the same Rust engine and local artifact format.
- [ ] **M7.3 Qualify the physical device.** Measure an independent iPhone 14 memory budget, thermal behavior, sustained throughput, and energy where instrumentation is available. Test inactive/background transitions, memory warnings, interruption during generation, resume/replay, and orderly resource release. Add iPad and newer Macs as separate device qualifications.

Exit: publish a reproducible device report and supported model/context limits. If physical hardware is unavailable, leave M7 pending; simulator results establish integration behavior only.

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

## First implementation queue

At 10–14 hours per week, aim to finish M0 and establish the first trustworthy CPU operations during the first two weeks. The full tiny decoder may require additional sessions.

| Order | Ticket | Reviewable result |
| --- | --- | --- |
| 1 | M0.1 | Single package, CLI help, clean build and checks |
| 2 | M0.2 | Frozen tiny arrays, manifest, independent oracle, manual linear example |
| 3 | M0.3 | Pinned model identity, benchmark token arrays, quality policy |
| 4 | M1.1 | Correct linear, embedding, bias, and residual operations |
| 5 | M0.4 | First reproducible external baseline with raw timings |
| 6 | M1.2 | Stable normalization, softmax, and MLP operations |
| 7 | M1.3 | Rotary positions and causal GQA with intermediate parity |
| 8 | M1.4 | Complete tiny decoder and a fixed greedy trace |

Implement one mechanism per reviewable change, including its memory layout, a small numerical trace, and targeted checks. Do not mark a milestone complete until its exit evidence exists. The next implementation action is M2.1.

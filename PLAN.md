# Toklet implementation plan

Prepared 9 October 2026. Status: M0, M1, and M2.1–M2.4 are complete; M2 is in progress, and M3–M7 remain pending.

Build a correct Rust decoder, run Qwen2.5-0.5B-Instruct with persistent session state, and use that reference to develop a measured Metal implementation. The first useful release is a CPU CLI with reliable continuation. The performance release adds FP16 Metal execution, Q4 weights, and separate prefill and decode paths. Physical iPhone qualification follows the Mac engine.

This plan translates the [Apple LLM performance proposal](apple_llm_performance_plan.md) into repository work. The proposal supplies design context; its references to an existing linear layer and earlier plan do not establish completed work in this checkout.

## Starting point

Before implementation, the repository contained a short README, license, ignore file, and the supplied performance proposal. There was no Cargo package, engine code, fixture suite, or benchmark harness. M0 establishes the package, frozen independent fixtures, workloads, quality gates, and the first native MLX baseline. M1 implements the complete scalar tiny decoder and greedy trace. M2.1 loads the pinned Qwen checkpoint into validated FP32 weights; M2.2 qualifies its full-prefix forward pass on short independent numerical probes. M2.3 adds bounded KV sessions and exact continuation; M2.4 adds bounded CLI generation; engine profiling remains next.

The environment reports Rust and Cargo 1.90.0, macOS 26.6.2 (25G83), and `/Library/Developer/CommandLineTools` as the active developer directory. Read-only inventory confirms Apple M1 Pro, MacBookPro18,3, and 16 GiB physical memory. AC power is attached with AC low-power mode disabled. `xcrun --find metal` fails; qualify an Xcode and Metal toolchain before engine GPU work. A hardware inventory does not establish numerical or performance qualification.

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

- [x] **M2.1 Import and validate Qwen weights.** Reuse safetensors/config parsers and a half-precision conversion library. Inspect actual dtypes; explicitly convert BF16/FP16 to FP32. Validate expected names, shapes, Q/K/V bias presence, normalization epsilon, rotary parameters, and shared embedding/output storage. Support an index manifest if the selected artifact is sharded. Reject unsupported configurations with a useful error. Convert tensor by tensor to bound peak memory.
- [x] **M2.2 Compare the real forward pass.** Export selected intermediates and logits from pinned Transformers in FP32 with dropout disabled. Compare short fixed token sequences, including more than one layer and nonzero positions. Retain small selected probes in Git and regenerate larger outputs locally. Do this before judging decoded prose.
- [x] **M2.3 Add the KV session.** Implement bounded per-layer K/V and reusable scratch, evaluated versus sampled positions, and exact append. Compare one-shot, irregularly chunked, and token-by-token execution with full-prefix recomputation. Prove position offsets and capacity checks at the 2,048-token boundary.
- [x] **M2.4 Add bounded CLI generation.** Reuse the tokenizer parser; reproduce the pinned template and special-token behavior with golden token IDs. Accept local model paths, prompt or token input, maximum output, context capacity, and backend selection. Add greedy streaming, EOS/stop handling, reset, continuation, cancellation, and statistics. Keep template rendering in the CLI layer.
- [ ] **M2.5 Add engine profiling and benchmark records.** Record model load/conversion, tokenization, prefill, first token, decode, sampling, and peak memory separately. Run the workload matrix and compare with the external baselines under the documented comparison track. CPU slowness is acceptable here; correctness, state behavior, and timing boundaries are the gate.

M2.1 follows M1.4; M2.2 follows M2.1. M2.3 can start on the tiny model after M1.4 and must pass real-model checks before M2 closes. M2.4 depends on M2.3; M2.5 completes after the generation path is stable.

Exit: a local Qwen session generates, appends a user/tool-result suffix, cancels, resumes or replays explicitly, and resets without cache corruption. The 1,920-input/128-output workload succeeds; a request exceeding capacity fails before evaluation. Loading and execution stay within the admitted memory budget.

M2.1 evidence, 9 October 2026: [local loader](src/weights/mod.rs) uses pinned safetensors, Serde, and half-precision parsers; validates config, tensor metadata and indexed shards before allocating FP32 weights; streams conversion with one 64 KiB buffer; and checks optional tied output tensors without retaining a duplicate table. [Seven ordinary loader tests](tests/weights.rs) cover tiny single/sharded oracle parity, independent FP16/BF16 bit values, conversion chunk boundaries/tails, tied heads, unsupported config/budget, and malformed artifacts. All forty-two ordinary Rust tests, formatting, Clippy with warnings denied, release compilation, and whitespace checks passed. The explicit SHA-verified pinned Qwen release check loaded all 290 BF16 tensors into 1,976,131,072 logical FP32 bytes, with physical process peak 1,981,335,520 bytes below 6 GiB; the [load report](benchmarks/results/2026-10-09-qwen-load.md) records identity, command and limitations. Astra accepted without findings, including the final chunk-boundary test. The metadata allowance is an estimate, and the single warm-cache load time is functional evidence; no real-model numerical or performance qualification is claimed. M2.2 is next.

M2.2 evidence, 9 October 2026: the [pinned Transformers exporter](tools/reference/qwen.py) captures 228 arrays over four short cases, including changed future tokens, absolute offset five, and one-token execution. [Versioned probes](tests/fixtures/qwen-v2/manifest.json) anchor selected values and complete-array hashes, shapes, and byte counts; the [explicit suite](tests/qwen_forward.rs) verifies source identity and compares all full intermediate arrays at layers 0, 1, and 23, final normalization, and all-position logits. Scalar model projections now accumulate in FP64 with FP32 storage, preserving the original FP32 kernels. All forty-four ordinary tests, formatting, Clippy with warnings denied, both fixture integrity checks, and the explicit real-model release suite passed. Maximum tolerance fraction is 0.634609, maximum logit absolute error is 0.000785828, all greedy IDs agree, and short teacher-forced NLL differences are at most 0.000217482 nats. The [qualification report](benchmarks/results/2026-10-09-qwen-forward.md) records an explicit v1-to-v2 policy change derived from independent FP32/FP64 envelopes and equivalent-prefix reference variation; tiny-v1 and historical real-model v1 remain unchanged. Astra independently verified the policy and accepted after complete-array identity anchors were added and requalified. This establishes short numerical parity, not held-out quality, longer-context, or performance qualification. M2.3 is next.

M2.3 evidence, 9 October 2026: [bounded CPU sessions](src/session.rs) borrow immutable weights and own separate head-major per-layer K/V, fixed chunk scratch, token history and staged/committed logits. Commits occur only after every layer and the final projection succeed; failed chunks leave a valid earlier prefix. Exact history extensions reuse KV, including pending sampled tokens; changed or shortened histories reset and replay. [Seven session tests](tests/session.rs) and [two cache-kernel tests](tests/cached_attention.rs) cover all frozen tiny cases, chunking, position offsets, failures after staged KV writes, reset/replay, input and memory admission, the actual 2,048-token small-model boundary, rejection beyond capacity, and zero measured steady-state allocations. All fifty-three ordinary tests, formatting, Clippy with warnings denied and whitespace checks passed. The [explicit Qwen session suite](tests/qwen_session.rs) passed 47 exact complete-logit comparisons on four short cases; the existing full-array M2.2 suite also passed with an identical numerical report. A default Qwen session allocates 48 MiB KV, admits 2,105,239,936 bytes including metadata allowance, and measured own-process physical peak was 2,108,770,344 bytes below 6 GiB. The [session report](benchmarks/results/2026-10-09-qwen-session.md) records scope and commands. Astra independently reviewed transaction semantics, numerical order, memory accounting and final evidence and accepted without findings. Real Qwen 2,048-token execution, held-out quality and performance remain pending for M2.5; cancellation can currently stop between synchronous chunk calls. M2.4 is next.

M2.4 evidence, 10 October 2026: [bounded greedy generation](src/generation.rs) validates complete prompt/output admission before mutation, keeps sampled stop/final tokens pending, and exposes synchronous cancellation and streaming events. The [CLI](src/bin/cli/mod.rs) accepts local raw/text/chat inputs, persistent generate/append/continue/reset, pinned Qwen formatting and tokenization, EOS/stop policies, statistics and Ctrl-C. [Chat-v1](tests/fixtures/chat-v1/manifest.json) freezes eight raw and seven chat cases from pinned Transformers, plus 1,010 Python float probes and a small offline BPE subset; the full pinned tokenizer passes the explicit local check. Sixty-nine ordinary Rust tests, formatting, Clippy with warnings denied, release build, fixture identity checks and whitespace checks pass. Ten [explicit real CLI cases](benchmarks/results/2026-10-10-qwen-cli.json) pass, including pending/cached history, chat output, prefill cancellation/recovery and over-capacity rejection; physical child-process peak is 2,115,782,768 bytes and admitted memory including tokenizer/application allowances is 2,343,243,600 bytes below 6 GiB. The [report](benchmarks/results/2026-10-10-qwen-cli.md) records reproduction and scope. Astra reviewed completed code, identified split-Unicode continuation, missing-config EOS fallback and a terminal cancellation race; all three were fixed and regression-tested, and Astra accepted the final code and qualification evidence. This establishes short functional generation, not throughput, held-out quality or the real-model capacity workload. M2.5 is next.

M2.5 profiling harness, 10 October 2026 (milestone still open): [teacher-forced timing](src/benchmark.rs), the [release worker](examples/cpu_bench.rs), and the [eight-workload adapter](tools/bench/cpu.py) record synchronous prefill/decode token events with fixed output IDs, reusable session buffers, three warmups and ten measured repetitions, raw checkpoints, schema validation, start-time identities and own-child physical footprint. Loader/CLI statistics now separate nested payload conversion from total load. Seventy ordinary Rust tests, twelve offline Python checks, formatting, Clippy and release compilation pass. The [short worker probe](benchmarks/results/2026-10-10-cpu-profiler-probe.json) checks cache scenarios and actual capacity rejection on Qwen; its physical peak is 2,033,928,208 bytes. Astra accepted the final profiling-harness code and short functional evidence after cleanup corrections. These are harness checks; the full workload matrix, external comparison, application/sustained records and real-model capacity execution remain pending, so M2.5 is not marked complete.

M2.5 matched external representation, 10 October 2026: the [MLX adapter](tools/bench/mlx_baseline.py) now preserves native BF16 by default and optionally converts floating parameters to FP32, waits for conversion completion and checks actual FP32 KV. A [128-input/128-output FP32 record](benchmarks/results/2026-10-10-mlx-fp32-128-128.json) retains three warmups and ten samples: median TTFT 60.260 ms, decode 68.967 steps/s, total 1.902777 s, physical process peak 3,342,387,216 bytes below 6 GiB. The [report](benchmarks/results/2026-10-10-mlx-fp32-128-128.md) records matching storage and timing scope, conversion and limitations. This is a controlled baseline measurement with no held-out-quality ranking; CPU measurements are still in progress and M2.5 remains open. Astra accepted the completed precision-adapter code and independently verified the record, hashes, schema, summaries and memory accounting.

M2.5 scalar/external probes, 10 October 2026 (milestone still open): the [scalar short report](benchmarks/results/2026-10-10-cpu-short.md) retains three warmups and ten samples, median TTFT 40.051132 s, decode 1.446565 steps/s and 2,043,742,224-byte physical child peak below 6 GiB. Ten requests total over 21 minutes with resets and recorded power/thermal state. Native profiling attributes 99.24% of samples to FP64 projections, selecting the M3 bottleneck. The separate-process [pinned llama.cpp comparator](tools/bench/llama/worker.cpp) builds with runtime Metal source compilation, independently verifies all 290 converted FP32 tensors, passes four short frozen numerical probes and executes all 2,048 positions with cache/rejection checks. Its [short record](benchmarks/results/2026-10-10-llama-fp32-short.md) gives median TTFT 35.759 ms and decode 75.407526 steps/s with matching token IDs/FP32 storage. The M1 matrix-matrix path uses half operands; small-batch parity does not qualify that path or held-out quality. Its footprint counter excludes clean mapped weights, so total mapped/shared physical peak remains unavailable; conservative admission is 5,088,231,424 bytes. Rust formatting/Clippy and seventy ordinary tests, fifteen offline Python checks, explicit conversion/build/GPU checks and evidence validation pass. Astra accepted the completed comparator code and independently verified final identities, raw records, timing/state series, summaries, tests and documented limits. Full workload matrix, application trace and real CPU capacity execution remain pending.

M2.5 application timing harness and external matrix, 10 October 2026 (milestone still open): [CLI statistics](src/bin/cli/mod.rs) now distinguish formatting/tokenization and detokenization/publication outside numerical kernels, and interactive readiness exposes logical ownership/accounting. The [application driver](tools/bench/application.py) records submission/visible-text/finalization latency for the frozen three-turn transcript, with separately retained startup and native components. Its [one-scenario functional probe](benchmarks/results/2026-10-10-application-probe.md) emits the expected ordered two-token events for all turns and has 2,126,055,536-byte physical child peak, below 6 GiB. Numerical code is unchanged; seventy ordinary Rust tests, formatting/Clippy, release compilation, sixteen offline Python checks and explicit real application probe pass. The [external matrix](benchmarks/results/2026-10-10-external-matrix.md) retains all eight llama.cpp cache/context workloads and four FP32 MLX prompt lengths, each with three warmups/ten samples, correct identities and raw events. Astra accepted the completed application timing code, verified final hashes, raw records, state/timing/memory and external summaries, and corrected the MLX dtype-check wording. Scalar matrix, real CPU final capacity execution and full application repetitions remain pending; the functional probe does not satisfy the performance gate.

## M3 Bound CPU optimization and ANE research

Preparatory candidate, 10 October 2026 (M3 remains pending): the profile-selected
[ordered FP64 projection tile](docs/experiments/2026-10-10-cpu-tile.md) preserves
scalar accumulation order and default decoder execution. Two complete synthetic
Qwen-shape runs use three warmup pairs and ten alternating measured pairs each,
with exact scalar/candidate output bits. Representative 128-token projections
improve 4.5–5.1×; single-token shapes improve 1.6–2.0×. These are microbenchmarks,
not full workload acceptance. Seventy-four ordinary Rust tests, formatting,
Clippy, release candidate build and eighteen offline Python checks pass. The
candidate awaits full-workload engine performance qualification after the scalar
M2.5 evidence completes; no milestone is marked complete by this preparation.
Astra accepted the completed candidate code and independently verified both
final records, raw checkpoints, identities, summaries and power settings.

Opt-in engine qualification, 10 October 2026 (M3 remains pending): the
[tiled model path](docs/experiments/2026-10-10-cpu-tiled-engine.md) selects a
concrete projection kernel once per operation, consumes model ownership without
copying weights, and preserves scalar defaults. Four pinned Qwen cases match
scalar bits across 1,740 complete trace arrays spanning all 24 layers and logits;
all 228 unchanged independent gates and twelve greedy IDs pass. Tiled cached
sessions pass 47 exact full-logit comparisons, with 2,067,597,328-byte physical
process peak and unchanged 2,105,239,936-byte session admission below 6 GiB.
All thirteen ordinary session tests run both variants, including actual
small-model capacity execution, failure recovery and allocation-free paths.
Formatting, Clippy, 75 ordinary Rust tests, 45 offline Python checks and explicit
release real-model checks pass. Builds and checks ran while the frozen scalar
numerical worker was idle between requests; its release binaries are unchanged.
This establishes an opt-in numerical candidate only. Scalar M2.5, full candidate
workloads and a second confirming performance run remain required before adoption.
Astra accepted the completed code and independently verified source/binary
identities, exact evidence copies, numerical counts, memory and scope.

Candidate measurement preparation, 10 October 2026 (M3 remains pending): the
[CPU workload adapter](tools/bench/cpu.py) can bind scalar or tiled execution
to the completed isolated-build qualification. Recorded source, binary and
forward/session report identities are checked before and after measurement;
readiness must acknowledge the requested kernel, chunk and capacity. Frozen
tokens, cache semantics, three warmups, ten samples, timing scope and capacity
checks are unchanged. Offline driver checks cover silent-default rejection,
changed identities, failed qualification and a complete synthetic protocol run.
No candidate workload or performance acceptance is claimed by this harness task.
Formatting, Clippy for all targets with warnings denied, 75 ordinary Rust tests
and 49 offline Python checks pass. The actual committed qualification receipt
verifies against the isolated worker and reports. Astra accepted the completed
adapter code without findings and independently checked identities and protocol
tests; required project checks ran while the frozen scalar worker was idle.

Public-runtime probe preparation, 10 October 2026: [coreml-slice-v1](tests/fixtures/coreml-slice-v1/manifest.json)
freezes 42 small arrays and independent FP32/FP16 projection and persistent-KV
attention gates from tiny-v1 before target evaluation. Scalar NumPy reconstruction,
frozen gate/payload corruption checks, formatting/Clippy and all 74 ordinary
Rust tests pass. This establishes the experiment's oracle only; no conversion,
ANE execution, complete decoder or milestone completion is claimed.
Astra accepted the completed oracle task after independently checking exact
regeneration, a separate vectorized attention reconstruction, KV states and gates.

Public-runtime slice results, 10 October 2026 (M3 remains pending): the
[public Core ML probe](docs/experiments/2026-10-10-coreml-slices.md) converts four
tiny projection/attention graphs with explicit types and persistent FP16 K/V.
FP32 projection and FP16/FP32-arithmetic attention pass three warmups and ten
measured repetitions across five frozen cases under both compute policies;
interleaved states, exact cache readback, verified poison installation and causal
masking pass. FP16 projection fails the unchanged first-warmup gate under both
policies and remains rejected. Every nonconstant compute-plan operation prefers
CPU; actual ANE placement is unobserved, so acceleration is inconclusive and no
engine path is adopted. Own-process peaks stay below 243 MB, with total external
Core ML service/device memory unavailable. Five offline probe tests, three oracle
tests, eighteen benchmark Python tests, formatting/Clippy and all 74 ordinary
Rust tests pass. The report preserves conversion/load costs, package/source
identities, raw samples, power policy and limitations. Astra reviewed the
completed code and required exact poison readback before prediction; it was fixed
and regression-tested before final evidence was regenerated. Astra accepted the
final code and independently verified the records and documented limits. Full scalar M2.5
and separately reviewed CPU engine qualification still precede M3 completion.

Future FP16 reference preparation, 10 October 2026: the separate
[fp16-v1 oracle](tools/reference/README.md) freezes 251 full tiny-decoder arrays
and 199 gates before any Metal execution, with FP16 weights/activations/KV and
FP32 accumulation/scores/probabilities/logits. Independent ordered FP32 and FP64
reconstructions use explicit storage-rounding boundaries and preserve the four
source greedy IDs. Five offline tests cover manual math, exact regeneration,
cache layout/causality, identities and gate corruption. This prepares numerical
evidence for M4 without changing current execution, existing FP32/Core ML gates,
real-model support or milestone status.
Formatting/Clippy, all 74 ordinary Rust tests and 31 offline Python checks pass.
Astra accepted the completed task and independently verified regeneration,
counts, rounding, layouts and all ten attention cases with a separate calculation.

- [ ] **M3.1 Improve one measured CPU bottleneck.** Spend approximately 4–6 hours on a profile-selected change such as contiguous traversal, vectorization, or bounded threading. Preserve the scalar oracle. Retain it only if full workloads improve without correctness or memory regressions.
- [ ] **M3.2 Evaluate public CPU and ANE execution.** Spend the remaining 8–14 hours converting the tiny decoder or a supported slice, comparing CPU-only and CPU-and-Neural-Engine policies, and attempting persistent state. Capture actual placement where tooling permits, fallback, conversion cost, completed-work latency, and memory.
- [ ] **M3.3 Close the experiment.** Record useful, uncompetitive, unsupported, or inconclusive, with commands and evidence. A slice result applies only to that slice. Adopt an ANE engine path only after complete stateful decode is measured; otherwise continue to M4 when the time budget ends.

Use the public [CPU-and-Neural-Engine compute policy](https://developer.apple.com/documentation/coreml/mlcomputeunits/cpuandneuralengine). Core ML supports stateful buffers from iOS 18/macOS 15; conversion success alone does not demonstrate ANE placement. [Core ML stateful models](https://apple.github.io/coremltools/docs-guides/source/stateful-models.html).

## M4 Implement complete FP16 Metal execution

Toolchain preparation, 10 October 2026 (M4 pending): the standalone
[Metal check](docs/experiments/2026-10-10-metal-toolchain.md) compiles through
public runtime APIs and passes exact FP32 add/buffer-tail checks on M1 Pro.
It records queried device/pipeline limits, MPS availability/row strides and
bounded build/run provenance. Standalone Metal/Instruments tools and GPU trace
capture remain unavailable. This changes no Rust execution or dependency and
does not qualify FP16, matrix multiplication, a decoder or a milestone.
Astra accepted the completed toolchain task after checking resource safety,
identities, functional records and both offline tests. Formatting/Clippy, all 74
ordinary Rust tests and 33 offline Python checks pass.

Real FP16 reference preparation, 10 October 2026 (M4 pending): the independent
[Qwen oracle](docs/experiments/2026-10-10-qwen-fp16-reference.md) executes all 24
layers over the four unchanged short source cases with FP16 storage and separate
FP32/FP64 arithmetic. It freezes 228 selected arrays and precision-specific
gates against the existing FP32 references before target Metal evaluation.
All twelve greedy IDs agree; recorded numerical drift does not qualify held-out
quality or persistent/long-context/GPU execution. Direct FP16 loading and bounded
weight conversion avoid a full FP32/FP64 copy. Exact full/probe regeneration,
six offline oracle tests, formatting/Clippy, all 74 ordinary Rust tests and 39
offline Python checks pass. No execution path or milestone changes.
Astra accepted the completed task without findings and independently verified
full/selected anchors, source/generator identities, pooled/prefix envelopes,
greedy IDs and the exact raw/result copy using light checks only.

Matrix primitive preparation, 10 October 2026 (M4 pending): the public
[MPS evaluation](docs/experiments/2026-10-10-mps-primitives.md) records two runs
at nine Qwen projection shapes with three warmups and ten samples. FP32-output
modes pass exact dyadic/manual cases and five unchanged tiny FP16 Q-projection
gates. FP16 output rejects a single-final-cast bias edge on both runs; its
failure and partial samples remain preserved. FP16 inputs with FP32 output
remain a candidate for further qualification, requiring resident bias setup
and activation casting. Accumulation precision, complete forward/quality and
end-to-end performance are unqualified. Checked allocation, completion,
immutability and physical guards pass; own-process peaks remain below 6 GiB.
Formatting/Clippy, all 74 ordinary Rust tests and 44 offline Python checks pass.
The Rust decoder and frozen scalar benchmark binaries remain unchanged.
Astra accepted the complete task without findings after checking safety,
protocol cleanup, five offline tests, final evidence and documented limits.

Resident projection preparation, 10 October 2026 (M4 pending): the
[GPU bias/projection/storage experiment](docs/experiments/2026-10-10-mps-resident-projection.md)
passes exact FP32 intermediate and final FP16 bits in two runs, including the
unchanged bias edge and five frozen tiny Q-projection gates. All three stages
share one command with a single completion wait, tracked resources, resident
bias and no CPU restaging. Immutable hashes, output row padding and physical
guards pass; own-process peaks are below 6 GiB. The original direct-FP16 MPS
rejection persists. Timings include GPU bias and final cast, with reported
variation and no performance acceptance. Formatting/Clippy, 74 ordinary Rust
tests and 45 offline Python checks pass. Rust execution and milestones remain
unchanged.
Astra accepted after a documentation precision correction, independently
verifying sequencing, bounds, six tests, raw identities, samples, edge payload
hashes, medians, tiny gates and the preserved rejection using light checks.

Rust binding preparation, 10 October 2026 (M4 pending): the standalone
[pinned Rust Metal probe](docs/experiments/2026-10-10-rust-metal-bindings.md)
passes 39 exact completed add/tail dispatches, structured shader-compilation
failure and budget rejection before device discovery. Private shared owners,
checked lengths/alignment, tracked hazards and retained command resources keep
host access outside execution; queried dispatch limits and explicit compile
policy readback pass. All native process peaks are below 6 GiB. The fresh utility
package pins its Cargo output directory and dependencies without changing the
engine package. Formatting/Clippy, 75 ordinary engine Rust tests, two standalone
Rust bounds tests and 54 offline Python checks pass. The first Clippy failure is
preserved; the corrected fresh attempt passes. Frozen CPU binaries and qualified
candidate source hashes remain unchanged. Astra accepted the complete task after
independently verifying safety, identities, exact evidence, dispatch values/guards,
failure propagation, memory and power with lightweight checks.
No FP16 decoder, trace, mobile qualification or milestone completion is claimed.

RMS primitive preparation, 10 October 2026 (M4 pending): the standalone
[FP16 RMS normalization](docs/experiments/2026-10-10-metal-rms.md) uses resident
Half input/weight/output with serial FP32 arithmetic, exact epsilon bits and
one final Half store. Its 25 frozen tiny normalization stages and three exact
manual range/subnormal cases pass 364 completed dispatches under unchanged
pre-GPU source gates; every output also matches the independent Half bits.
All payloads/guards remain intact and conservative admission and own-process
peak remain below 6 GiB. The generic shared-owner refactor preserves all 39
FP32 add/tail checks and structured failure behavior. Isolated/engine formatting
and Clippy, five standalone Rust tests, 75 ordinary engine tests and 60 offline
Python checks pass. Native/compiler/test work occurs only after the clean scalar
request series exits, with coordinators held and frozen CPU identities verified.
Width 32 only is qualified; Qwen, decoder composition, GPU traces and performance
acceptance remain pending. Astra accepted the completed task after a report
timestamp correction, independently verifying all numerical checks, compaction,
identities, ownership, memory and preserved regression failures with light reads.

Qwen RMS primitive qualification, 10 October 2026 (M4 pending): the
[width-896 extension](docs/experiments/2026-10-10-qwen-metal-rms.md) reads only
six norm tensors directly from pinned BF16 storage with bounded streaming
identity checks and one Half RNE cast. Four frozen short cases supply 24 selected
normalizations in nine bounded inputs. All 312 completed dispatches, 838,656
output values, immutable inputs/weights and guards pass unchanged pre-GPU gates.
An independent serial FP32 diagnostic matches every native value, explaining
46 one-Half-ULP differences from NumPy reduction order without a policy change.
Native peaks/admission remain below 6 GiB; preparation-process/system physical
memory is not inferred from child peaks. The isolated and engine checks pass:
five standalone Rust tests, 75 ordinary engine tests and 66 offline Python
checks, formatting and Clippy with warnings denied. Native/checkpoint/test work
uses a verified idle CPU boundary; frozen numerical identities remain unchanged.
Layer 23 input norm, all-layer composition, cache, quality, performance and trace
qualification remain pending. Astra accepted the complete primitive task without
findings, independently verifying identities, all numerical/guard proofs,
unchanged gates, the reduction-order explanation, memory and idle windows.

Embedding primitive preparation, 10 October 2026 (M4 pending): the standalone
[Half bit-copy lookup](docs/experiments/2026-10-10-metal-embedding.md) keeps one
immutable table resident across cases and copies UInt32-indexed rows exactly.
Five frozen tiny cases and a signed-zero/subnormal/max-finite/repeated/final-row
manual case pass all 78 completed dispatches, 25,376 active outputs and 546
output sentinels. Unchanged FP32-source gates also pass. Shared ownership,
checked admission and host access after completion preserve immutable inputs;
child peaks remain below 6 GiB. All 39 add/tail and 364 RMS regressions pass,
as do isolated/engine formatting and Clippy, eight standalone Rust tests,
75 ordinary engine tests and 71 offline Python checks. Heavy checks use a
verified idle CPU boundary, preserving frozen binaries and candidate identities.
Astra accepted the complete task without findings after independently checking
code, numerical proofs, identities, memory, final copies and isolation. Only
tiny width-32 lookup is qualified;
Qwen table, composition, cache, quality, performance and traces remain pending.

Residual primitive preparation, 10 October 2026 (M4 pending): the standalone
[in-place Half add](docs/experiments/2026-10-10-metal-residual.md) uses two resident
buffers, FP32 addition and one Half RNE store. Twenty frozen tiny stages and
independent rounding/zero/subnormal/range cases pass all 273 completed dispatches,
80,288 exact outputs, 1,911 state guards and unchanged FP32-source gates.
Completed overflow is explicitly invalid/discarded; no rollback/session claim.
Admission/physical peaks remain below 6 GiB. All add/RMS/embedding regressions,
isolated/engine formatting and Clippy, 11 standalone Rust tests, 75 ordinary
engine tests and 77 offline Python checks pass in a verified numerical-idle
window. Frozen CPU identities remain unchanged. Astra accepted the complete task
without findings after independently reproducing exact outputs, source gates,
overflow rejection, regressions and artifact identities.
Width 32 only, with Qwen, composition, cache, quality, performance and traces
still pending.

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

Implement one mechanism per reviewable change, including its memory layout, a small numerical trace, and targeted checks. Do not mark a milestone complete until its exit evidence exists. The next implementation action is M2.5.

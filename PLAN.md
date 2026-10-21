# Toklet implementation plan

Prepared 9 October 2026; updated 20 October 2026. Status: M0–M6 are complete in their qualified scopes; M7 remains pending. M6 qualifies explicit Online attention with FP16 KV for the selected small compensated Half and frozen larger group8 Q4 v2 models at4K/8K/16K/32K total tokens on Apple M1 Pro. Full quality, context boundary/state/memory and maximum application evidence pass within6GiB. Defaults and baseline limits remain. See the [M6 release decision](docs/experiments/2026-10-19-m6-release.md) and [M5 release decision](docs/experiments/2026-10-15-m5-release.md).

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
- [x] **M2.5 Add engine profiling and benchmark records.** Record model load/conversion, tokenization, prefill, first token, decode, sampling, and peak memory separately. Run the workload matrix and compare with the external baselines under the documented comparison track. CPU slowness is acceptable here; correctness, state behavior, and timing boundaries are the gate.

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

M2.5 real capacity evidence, 10 October 2026 (milestone still open): the
[complete scalar boundary run](benchmarks/results/2026-10-10-cpu-boundary.md)
retains three warmups and ten 1,920-input/128-output requests with all measured
states. The original driver then commits all 2,048 positions with no pending
token and rejects one more before mutation. Median warm TTFT is 653.578164 s,
request time 749.021123 s and decode rate 1.331183 steps/s. Maximum own-child
physical charge is 2,043,774,992 bytes below 6 GiB and conservative admission.
Exact source/result/raw identities, schema/summary checks and the final idle
window/resumption record are retained. Numerical work was never stopped;
between-request transport holds preclude a continuous sustained-run claim.
The real capacity criterion passes; remaining matrix, application and quality
qualification stay pending. Astra accepted the saved-evidence task without
findings after independently verifying all states, summaries, historical
assertions, identities and idle-window records.

M2.5 medium evidence, 10 October 2026 (milestone still open): the
[scalar medium report](benchmarks/results/2026-10-10-cpu-medium.md) retains
three warmups and ten 512-input/128-output requests, each with 639 committed
positions and one pending token. Median TTFT is 163.381240 s, completed
request time 253.195493 s and decode rate 1.413940 steps/s. Maximum measured
own-child physical charge is 2,043,840,528 bytes below 6 GiB and conservative
admission. Saved-state/schema/summary checks and historical driver identity
are retained. The medium worker completed uninterrupted; only orchestration
was held ahead of the next workload. Qwen SiLU and loss-runner checks ran in
the verified idle window before frozen identities were revalidated and the
parent resumed. Astra accepted the saved-evidence task without findings after
independent identity, state/event, summary, memory and window-order verification.
Remaining matrix/application/quality gates remain pending.

M2.5 long evidence, 10 October 2026 (milestone still open): the
[scalar long report](docs/experiments/2026-10-10-cpu-long.md) retains three
warmups and ten uninterrupted 1,024-input/128-output requests, each with
1,151 committed positions and one pending token. Median TTFT is 336.469968 s,
completed request time 428.750185 s and decode rate 1.377309 steps/s.
Maximum own-process physical charge is 2,043,774,992 bytes below 6 GiB and
conservative admission. Independent saved-data checks reproduce the schema,
raw states, summaries, memory, token and historical driver identities, and
launch/hold/idle ordering. Only orchestration was held ahead of the next
workload; numerical execution was never stopped. Astra accepted the saved
evidence and producer without findings after independent replay and identity,
state, timing, memory and window-order verification. Remaining cache/application
matrix and quality gates stay open.

M2.5 append evidence, 10 October 2026 (milestone still open): the
[scalar exact-prefix append report](docs/experiments/2026-10-10-cpu-append.md)
retains three warmups and ten uninterrupted measured requests. Each prepares
512 cached tokens outside timing, then appends 64 fixed IDs and emits 128
teacher-forced steps, finishing with 703 committed positions and one pending
token. Median warm TTFT is 21.348085 s, completed request time 111.628890 s
and decode rate 1.406308 steps/s; prefill throughput uses only the 64 appended
tokens. Maximum physical charge is 2,043,774,992 bytes below 6 GiB and
conservative admission. Independent saved-data checks reproduce raw states,
summaries, token and historical driver identities, memory and launch/hold/idle
ordering. Only orchestration was held; numerical work finished naturally
before root checks and native Qwen gating ran. Astra accepted the saved-data
task without findings after independent replay and raw-copy, state, timing,
memory, identity and window-order verification. Remaining cache/application
matrix and full quality gates stay open.

M2.5 cached evidence, 10 October 2026 (milestone still open): the
[fully cached scalar report](docs/experiments/2026-10-10-cpu-cached.md)
retains three warmups and ten uninterrupted measured requests. Each prepares
512 evaluated tokens outside timing and emits 128 fixed IDs, finishing with
639 committed positions and one pending token. Median numerical TTFT is
0.792 microseconds, accessing the prepared distribution; this excludes cache
preparation, sampling, transport and visible text. Median request time is
90.000207 s and decode rate 1.411108 steps/s. Prefill throughput is null for
zero prefill tokens; empty evaluation still incurs timer overhead. Maximum
physical charge is 2,043,856,912 bytes below conservative admission and 6 GiB.
Independent saved analysis reproduces all raw states, timings, summaries,
identities, memory and launch/hold/idle ordering. Only orchestration was held;
the numerical worker finished naturally before any checks or model work.
Astra accepted the saved-data task without findings after exact analysis replay
and independent raw-copy, state, stdout, timing, memory, identity and window-order
verification. The later pending-token record follows below; changed-prefix,
application and full quality gates remain open.

M2.5 pending-token evidence, 10 October 2026 (milestone still open): the
[scalar pending report](docs/experiments/2026-10-10-cpu-pending.md)
retains three warmups and ten uninterrupted measured requests. Each prepares
512 evaluated tokens outside timing, stages fixed ID 2077 as a pending-sample
stand-in, then emits 128 teacher-forced steps. All measured requests reuse 512
positions, evaluate one pending input and finish with 640 committed positions
and one pending token. Median numerical TTFT is 0.702565 s, completed request
time is 90.500725 s and decode rate is 1.414395 steps/s. Prefill throughput
uses the one pending input, not the prepared base. Maximum physical charge is
2,043,791,376 bytes below conservative admission and 6 GiB. Independent saved
analysis reproduces raw states, events, summaries, fixed identities, memory and
launch/hold/idle ordering. Only orchestration was held; the numerical worker
finished naturally before checks or native isolation work. Astra accepted the
complete saved-data task without findings after exact replay and independent
raw-copy, state/event, stdout, summary, memory, identity and ordering checks.
Changed-prefix, application and full quality gates
remain open.

Held-out scorer preparation, 10 October 2026 (M2.5/quality still open): the
[bounded loss runner](docs/experiments/2026-10-10-held-out-loss-runner.md)
implements the existing ten independent 1,024-target blocks with one-token
overlap and the 784-target tail. FP64 max-subtracted compensated loss scores
every label from its committed preceding prefix and retains full-logit hashes.
Four short real workers score twelve outputs across six independent labels;
scalar/tiled loss records and complete logits agree exactly, with maximum
source NLL discrepancy 0.000217481131459 under the unchanged 0.001 probe gate.
All workers commit four tokens with none pending. Maximum physical charge is
2,009,417,720 bytes and admission is 2,136,856,460 bytes below 6 GiB.
Root formatting, Clippy, 76 ordinary Rust tests, three example tests, isolated
release compilation and all 35 offline benchmark Python tests pass in the
verified medium-worker idle window. Interrupted checks kill/reap their owned
group and retain failure/output metadata. Astra accepted the completed code and
evidence without findings after independent NLL, alignment, identity, memory,
cleanup and test verification. Full 10,000-target scoring, the separate 100-task pass-rate evaluation,
lossy-model acceptance, performance and milestone closure remain pending.

Task response preparation, 10 October 2026 (M2.5/quality still open): the
[collector and deterministic scorers](docs/experiments/2026-10-10-task-response-collector.md)
validate all 100 frozen cases, submit prepared IDs directly and retain finalized
text through explicit reset. Pending EOS is distinct from visible output.
TruthfulQA and LAMBADA use the frozen exact text rules; HumanEval composition
preserves literal source and requires a separately qualified isolated executor.
Infrastructure failures prevent binary aggregation. Four authored native
requests across scalar/tiled modes emit 96 identical visible IDs in total and
replay all 106 events; maximum conservative CLI-plus-driver physical charge is
2,150,566,600 bytes below 6 GiB. Formatting, Clippy, 76 ordinary Rust tests and
54 offline benchmark tests pass after the cached numerical worker exits
naturally. Astra accepted the complete task without findings after independently
running the 19 targeted tests and reproducing the saved analysis exactly.
No held-out model response, generated code execution or quality score is claimed;
full quality and milestone gates remain open.

Code isolation preparation, 10 October 2026 (M2.5/quality still open): the
[authored native candidate](docs/experiments/2026-10-10-code-isolation.md)
provides bounded pipes, owned-group cleanup, fixed resource controls and planned
named-profile denied-access probes. Formatting, Clippy, 76 ordinary Rust tests
and all 62 offline benchmark tests pass after the pending worker's natural exit.
The first native child rejects 256 MiB `RLIMIT_DATA` admission and exits 2 before
applying the sandbox or running an authored operation. Exact saved replay
verifies runtime/source identities, capture hashes, checks and cleanup. The
candidate remains unqualified and is not adopted; limits are not dropped or
lowered. Astra accepted the completed experiment without findings after all
eight offline tests and exact rejection replay, with ABI, cleanup, identity and
ordering checks. A compatible isolated executor, generated HumanEval programs,
all 100 held-out responses, full quality and milestone gates remain pending.

Linux isolation preparation, 11 October 2026 (M2.5/quality still open): the
[authored Linux candidate](docs/experiments/2026-10-10-linux-code-isolation.md)
adds a fixed AArch64 seccomp policy, literal container controls, bounded owned
transport cleanup and independent policy/configuration tests. The first targeted
execution found a reversed descriptor high-half branch; it was corrected and
all 19 targeted tests pass. Formatting, offline locked Clippy, 76 ordinary Rust
tests and all 81 offline benchmark tests pass after the explicitly paused
benchmark is retired. Its six completed samples remain diagnostic; the
interrupted batch cannot qualify timing. Podman reports successful VM start,
but pull and inventory fail with connection refused before image/container
admission. Follow-up inventory verifies the original stopped machine state.
The [saved attempt](benchmarks/results/2026-10-11-linux-code-isolation-attempt-v1.json)
binds source, actual commands/captures, checks and retirement evidence. No native
policy, generated program, executor adoption, quality score or milestone closure
is claimed. Astra accepted the complete task without findings after independently
rerunning the 19 targeted tests and verifying the saved evidence and stopped VM.

Linux authored retry, 11 October 2026 (M2.5/quality still open): the
[fresh native run](docs/experiments/2026-10-11-linux-code-isolation-retry.md)
passes all fourteen modes on the pinned arm64 Python image, including denied
access, output/CPU/wall termination and both memory limits. Persistent startup
helpers and pipe readers survive until verified stopped-machine restoration.
The client binds the explicit selected-machine endpoint. All five process limits
remain fixed; AS is admitted by the trusted child because Podman rejects its OCI
ulimit spelling. Actual kernel capability masks must be zero, runtime metadata
is removed before admission and unknown exit codes remain unqualified. Separate
post-kill reaping avoids a redundant deadline signal without suppressing errors.
The exact saved replay preserves six excluded retries and validates all outcomes,
source/configuration identities and cleanup. Formatting, offline locked Clippy,
76 ordinary Rust tests and 97 offline benchmark tests pass. Final inspection
confirms owned helpers absent, original VM stopped and frozen binaries unchanged.
Astra accepted the complete task without findings after byte-identical replay,
35 targeted tests and independent identity/cleanup/restoration checks.
A general generated-code executor, held-out responses and quality scores remain
pending; this authored boundary is not an engine or milestone adoption.

Changed-prefix scalar result, 11 October 2026 (M2.5 still open): the
[fresh batch](docs/experiments/2026-10-11-cpu-changed.md) retains three warmups
and ten uninterrupted requests. Editing token 128 forces reset and replay of
all 512 prompt tokens, with zero reuse and final 639 committed positions plus
one pending token after 128 fixed output steps. Median numerical TTFT is
163.551572 s, completed request 253.054154 s and decode 1.416174 steps/s.
Maximum physical charge is 2,043,791,376 bytes, below admission and 6 GiB.
Exact saved replay verifies states/events, summaries, identities and natural
exit before checks. The earlier user-paused six-sample batch is excluded.
Astra accepted the complete evidence task after a power-query timing correction
and independent saved-data replay; links and whitespace checks pass.
Application measurements, full quality and milestone acceptance remain open.

Full scalar application evidence, 11 October 2026 (M2.5 still open): the
[completed trace](docs/experiments/2026-10-11-application-scalar.md) retains
three warmups and ten uninterrupted scripted three-turn scenarios. Median
scenario latency is 263.230241 s, with all thirty measured responses and
committed/pending states deterministic. Turns emit 128, 18 and 128 tokens;
the second stops at pinned EOS. Median first-token/visible-text latency is
11.606069 / 24.114077 / 36.922609 s for the three prompt depths. Maximum
own-child physical footprint is 2,128,758,896 bytes, below conservative
admission and 6 GiB. Saved replay verifies raw pairs, state/timing/text,
complete memory accounting, identities, actual session exit zero and exact
owned process absence. The original interrupted orchestration stays held.
Source/helper/binary code is unchanged, and checks and VM/native work remain
outside measurement. Astra accepted the completed task with no findings after
independent byte-identical replay, identity/state/timing/memory checks and
link/whitespace verification. Held-out quality, candidate application evidence
and milestone acceptance remain open.

Literal task evaluator preparation, 11 October 2026 (M2.5/quality still open):
the [child, receipt validator and bounded input transport](docs/experiments/2026-10-11-linux-task-preparation.md)
preserve whole-source completion, fixed preloads, strict evidence parsing and
separate owned-process cleanup. All 39 new offline tests and the full 136-test
benchmark suite pass; formatting, Clippy with warnings denied and 76 ordinary
Rust tests pass. Initial sandbox EPERM failures in existing process fixtures
are preserved separately from the passing rerun. No generated program was
executed. Astra accepted the checked preparation task with no findings.
Fixed container integration, native authored qualification and VM
restoration remain required before adoption or quality scoring.

Full scalar held-out loss, 11 October 2026 (M2.5/quality still open): the
[completed baseline](docs/experiments/2026-10-11-held-out-scalar-loss.md)
scores all 10,000 frozen targets with mean NLL 2.883725149929484 nats/token.
Nine 1,024-target blocks and the final 784-target block follow the unchanged
overlap/reset policy and finish with zero pending tokens. Own-worker peak
physical charge is 2,009,548,792 bytes, below admitted 2,136,856,460 and 6 GiB.
Formatting, offline locked Clippy, 76 ordinary Rust tests and three example
tests pass before scoring in isolated targets. Exact saved replay verifies all
10,012 native events, labels, sums, memory, identities, ordered checks, actual
exit zero and owned process absence; the original orchestration stays held
and the VM stays stopped. Astra independently reproduces the analysis
byte-for-byte and accepts the complete task without findings after documentation,
link and whitespace review.
At this checkpoint, full candidate loss, 100-task pass rates and milestone
acceptance remained open; the subsequent tiled loss comparison is recorded below.

Literal Linux controller, 11 October 2026 (M2.5/quality still open): the
[completed authored qualification](docs/experiments/2026-10-11-linux-literal-controller.md)
passes all 28 precise native expectations with bounded literal stdin, exact
trusted bootstrap/configuration, raw status agreement and owned cleanup.
Two stdout/stderr overflow cases exercise the narrow host-SIGKILL/remote-zero
race and resolve only as resource failure. Unknown Podman wait status may
retry with full evidence; persistent unknown remains infrastructure failure.
The corrected limit-change fixture checks libc return and exact EPERM directly.
Two excluded attempts and their exact source snapshots are retained.
All 161 offline benchmark tests, formatting, Clippy and 76 ordinary Rust tests
pass before the final native window. Every attempt removes its containers and
restores the stopped VM; fresh exact helper absence and protected identities
are verified. Saved replay validates both exclusions and all accepted outcomes.
Astra accepts the complete task without findings after independent byte-identical
replay, targeted tests, documentation and link review. No held-out generated
program, task score, adversarial grader integrity or milestone closure is claimed.

Full tiled held-out loss, 11 October 2026 (M2.5/quality still open): the
[complete comparison](docs/experiments/2026-10-11-held-out-tiled-loss.md)
replays all 10,000 aligned targets against the full scalar baseline. All loss
events, complete-logit hashes and target FP32 bits agree; mean NLL remains
2.883725149929484 nats/token, mean increase and maximum target difference are
zero, and the unchanged 0.05 nats/token gate passes. Both runs use independent
blocks and prefill chunk one; task prompts with chunk 128 require the separate
100-task evaluation. Own-worker peak is 2,009,532,408 bytes, below admission
2,136,856,460 and 6 GiB. Formatting, offline locked Clippy, 76 Rust tests and
three example tests pass before scoring in isolated targets. Saved replay
verifies all 10,012 native events, bindings, six ordered checks, actual exit
zero and owned process absence; the original parent stays held and the VM
stays stopped. Astra independently reproduces the comparison byte-for-byte.
Astra accepts the complete evidence/report task with no actionable findings;
local links and whitespace checks pass. Task pass rates, remaining candidate
workloads and milestone acceptance remain pending; scalar remains default.

Full task collector preparation, 11 October 2026 (M2.5/quality still open): the
[checked persistent collector](docs/experiments/2026-10-11-held-out-task-collector.md)
preserves all 100 frozen requests, independent reset state, literal finalized
text and empty immediate EOS without an authored first-token constraint.
Per-phase watchdogs are canceled/joined; late shutdown expiry invalidates
success. Unexpected descendants are terminated and boundedly rechecked,
with initial/final inventory retained and no successful collection outcome.
Eleven targeted tests include both Astra cleanup regressions and a real tiny
100-request fake CLI. All 172 offline tests, formatting, locked offline Clippy
and 76 Rust tests pass in an isolated target, with protected endpoints unchanged.
Astra accepts the complete checked code task after independent cleanup regression
checks and source/evidence/documentation review, with no remaining findings.
Local links and whitespace checks pass. No held-out model response, code grading,
pass-rate or milestone acceptance is claimed by this preparation.

Full scalar task response collection, 11 October 2026 (M2.5/quality still open):
the [completed collection](docs/experiments/2026-10-11-held-out-task-scalar.md)
retains all 100 frozen responses: 40 code, 30 prose and 30 continuation.
All 6,594 native events and 6,393 visible tokens replay with exact finalized
text, stopping, pending/committed state and reset semantics. Sixty-eight
responses stop on EOS and 32 reach category output limits. The clean checked
revision uses the existing qualified scalar CLI at capacity 2,048 and chunk 128.
Logical admission is 2,353,012,560 bytes; the maximum conservative joint CLI
and driver physical peak is 2,154,760,904 bytes, below 6 GiB. Native session
1132 exits naturally zero; its three owned processes and both groups are absent,
the VM is stopped, and protected identities and original held orchestration
are unchanged. Exact launch, ownership, completion, idle and replay sources
are retained. Astra independently reproduces byte-identical analysis and accepts
the complete evidence and documentation task with no remaining findings. All
144 local links, content and whitespace checks pass. No generated code execution,
task score, tiled task comparison, timing acceptance or milestone closure
is claimed by this collection.

Held-out grading integration, 11 October 2026 (M2.5/quality still open): the
[checked runner and runtime window](docs/experiments/2026-10-11-held-out-task-grading.md)
validate all 100 finalized responses and their qualified provenance before VM
startup. Forty code programs preserve literal prefix/completion/test composition;
prose and continuation use unchanged protected scorers. Known resource outcomes
remain resolved failures; unresolved compatibility/protocol/runtime evidence
prevents an aggregate, and uncertain cleanup stops later container creation.
The owned window binds the qualified runtime and cached image, refuses a
pre-existing running VM, retains unexplained live containers and owns helper
restoration through publication failures. All 21 targeted checks, 193 offline
Python tests, formatting, locked offline Clippy and 76 Rust tests pass, with real
completed scalar collection validation and protected idle endpoints unchanged.
Astra independently passes the targeted suite and accepts the complete code,
check evidence and documentation without actionable findings. No actual held-out program execution,
task score, tiled comparison or milestone closure is claimed by this code task.

M2.5 held-out scalar task baseline, 11 October 2026: the
[actual literal grading](docs/experiments/2026-10-11-held-out-task-scalar-grading.md)
resolves all 100 collected responses at 2/100: code 0/40, prose 1/30 and
continuation 1/30. All code outcomes are SyntaxErrors under unchanged literal
prefix/completion/test composition; no response repair or scoring relaxation
is applied. No infrastructure outcome is unresolved. All forty containers were
removed, owned helpers/groups exited and the VM was restored stopped, with
protected endpoints unchanged. Independent classification/text replay and Astra
confirm the baseline; Astra accepted the complete evidence and report without
findings. Tiled task collection,
grading and the unchanged maximum two-percentage-point drop gate remain open.
This does not close M2.5 or qualify candidate performance/application behavior.

Full tiled task response collection, 11 October 2026: the
[completed collection](docs/experiments/2026-10-11-held-out-task-tiled.md)
retains all 100 independent frozen responses, 6,393 visible tokens and 6,594
native events, with 68 EOS and 32 output-limit stops and no empty response.
The full journal and provenance replay passes. Actual native session 94680
exits naturally; all owned PIDs/groups are absent, the VM stays stopped and
protected endpoints remain unchanged. Conservative joint physical peak is
2,150,484,680 bytes below 6 GiB. No model, VM, test or build ran concurrently.
Astra independently reproduces the analysis byte for byte and accepts the
complete evidence and report without findings. No tiled task score, task-quality comparison
or retention is claimed; literal grading and remaining candidate workload/
application gates follow separately. Scalar stays default.

Full scalar/tiled task-quality comparison, 11 October 2026: the
[actual literal comparison](docs/experiments/2026-10-11-held-out-task-quality.md)
resolves both modes at 2/100 (code 0/40, prose 1/30, continuation 1/30), with
zero percentage-point drop against the unchanged two-point gate. All 100
visible texts, emitted token sequences, stop reasons and binary outcomes match.
All forty code failures remain SyntaxErrors; no response repair, score policy
change, resource failure or unresolved infrastructure is involved. All forty
containers were removed; actual session 74369 exited zero, the VM stopped,
owned helper groups exited and protected endpoints stayed unchanged. Exact
classification/text/aggregate/provenance replay passes; Astra independently
reproduces the analysis byte for byte and accepts the complete evidence/report
without findings. Combined with the identical full 10,000-target NLL comparison, the
frozen quality gates pass. This does not accept M3.1 retention: the seven
non-short tiled workload measurements, full tiled application and final review
remain required. Scalar stays default.

M2.5 completion, 11 October 2026: the
[consolidated scalar/external comparison](docs/experiments/2026-10-11-m2.5-scalar-comparison.md)
aligns all eight accepted scalar workloads, actual capacity commit/rejection,
separate load/conversion/request components, full application trace and sustained
request sequences with the scoped external matrix. FP64 projection accumulation,
external Half-operand GEMM, unavailable mapped-memory/energy/pressure accounting,
warm-filesystem startup and historical run/hold conditions remain explicit.
The completed frozen loss/task comparisons preserve their unchanged gates.
Astra verifies the table/source reports, links and milestone coverage and accepts
M2.5 closure without findings. Content/link/whitespace checks pass; no new engine
or benchmark run is needed for this documentation task. Tiled defaults and
M3.1 retention remain open for the seven other workload measurements, full
tiled application and final retention review.

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

Matched primary CPU result, 10 October 2026 (M3 remains pending): the
[comparison report](docs/experiments/2026-10-10-cpu-primary-comparison.md)
retains two complete three-warmup/ten-sample scalar/tiled pairs in the same
qualified executable, reversing order for confirmation. A Python-test-overlapped
scalar run is explicitly excluded and replaced by a complete clean run with
hash-bound activity/guardian evidence. Warm 128-input/128-output TTFT improves
77.6400% and 77.6009%, exceeding the 5% threshold and 0.3531% maximum observed
between-run median variation. Request medians improve 52.0024%/52.0992%; decode
rates improve 1.67460×/1.67830×. Logical weights/KV/scratch are identical and
all native child peaks remain below 6 GiB. Exact records, raw checkpoint hashes,
executed calculation/orchestration snapshots and power/thermal limitations
are retained. Scalar stays default; full matrix, application, sustained,
held-out quality and M2.5/M3 acceptance remain pending. Astra accepted the report
without findings after independently checking sample arithmetic, variation,
memory, artifact identities and the explicit clean-run substitution.

CLI selection preparation, 10 October 2026 (M3 remains pending): the
[opt-in CLI selector](docs/experiments/2026-10-10-cpu-tiled-cli.md) consumes
the model before session creation and reports actual scalar/tiled execution
at readiness and each request. Scalar remains default; invalid/duplicate modes
fail before model reads. Default, explicit scalar and tiled preserve tiny
cache/continuation/rejection/reset semantics and identical ownership. Four real
Qwen process roles, including the frozen CLI, agree on 24 generation requests over three
independent first-token cases and pending-token continuations. Maximum physical
child peak is 2,125,400,200 bytes; all admissions are 2,353,012,560 bytes below
6 GiB. Formatting, Clippy, 76 ordinary Rust tests and isolated release build
pass after the boundary worker exits, with the next matrix workload held and
frozen numerical/application binaries unchanged. Complete application and
matrix acceptance remain pending. Astra accepted the completed CLI task without
findings after independently checking raw state/token/text, source/receipt/binary
identities, checks, memory and idle-window evidence.

Application candidate preparation, 10 October 2026 (M3 remains pending): the
[qualified application adapter](docs/experiments/2026-10-10-cpu-tiled-application-adapter.md)
uses one checked CLI binary for explicit scalar/tiled and preserves legacy
default arguments. Receipt, compiled source, compiler/flags, reference/state,
binary and helper identities are checked, and live readiness/statistics must
acknowledge the selected mode. All 28 offline benchmark tests pass. After
accepted CLI commit `c52af3a`, three sequential real-model probes match all nine
two-token turns in visible text, token IDs, reset and non-timing generation
state. All logical/admission accounts agree; maximum physical peak is
2,128,513,136 bytes and admission is 2,353,012,560 bytes below 6 GiB. Root engine
checks and frozen release binaries are unchanged; the matrix remains held
after normal boundary exit. These functional probes establish wiring only;
complete application repetitions remain pending. Astra accepted the adapter
task without findings after independently checking code, checks, raw turn/state/
text evidence, receipt/source identities and memory.

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

Excluded candidate attempt, 11 October 2026: the [medium failure record](docs/experiments/2026-10-11-cpu-tiled-medium-excluded.md)
retains all ten sample/state rows and actual exit 1 after system Python lacked
`jsonschema` at publication. No result is reconstructed or accepted. The owned
launcher/driver/worker are absent, their group is empty, original orchestration
remains held, Podman is off and protected endpoint hashes match. A separately
reviewed replacement uses repository `.venv` Python and checks pinned schema
dependencies before dispatch. All seven non-short candidate rows remain pending.

Candidate medium evidence, 11 October 2026: the [complete replacement](docs/experiments/2026-10-11-cpu-tiled-medium.md)
uses qualified isolated tiled execution and pinned benchmark dependencies, with
three warmups and ten samples. All states commit 639 positions with one pending
fixed output token; median TTFT is 40.961032 s, completed request 97.892741 s,
and maximum own-child physical charge 2,043,693,048 bytes below 6 GiB. Historical
scalar reductions are descriptive, while the separate reversed-order primary
pairs retain the confirming acceptance evidence. The excluded first attempt
contributes no accepted sample. Natural completion, owned PID/group absence,
protected endpoint identities and exact source/qualification bindings are
retained. No numerical helper or qualified artifact changes. Six other non-short
rows, full tiled application evidence and M3.1 retention remain pending.

M3.2/M3.3 completion, 11 October 2026: the [feasibility conclusion](docs/experiments/2026-10-11-ane-feasibility-closure.md)
closes the bounded public-runtime research using the already accepted supported
slice experiment. FP32 projection and persistent-KV attention pass CPU policies;
FP16 projection remains rejected under unchanged gates. All nonconstant public
compute-plan operations prefer CPU; actual ANE placement remains unobserved,
so acceleration is inconclusive and no engine path is adopted. Conversion/load,
completed prediction times, exact state checks and scoped memory are retained.
No full decoder, Qwen generation, total service/device memory or ANE quality
claim follows. The probe and fixtures are retained; no further native experiment
runs for this documentation closure. M3 overall remains open for CPU workload,
application and retention qualification.

Candidate long evidence, 11 October 2026: the [complete 1,024-input workload](docs/experiments/2026-10-11-cpu-tiled-long.md)
uses three warmups and ten measured requests, all with 1,151 committed positions
and one pending fixed output token. Median TTFT is 85.683956 s and completed
request 142.368874 s; the whole observed request range is retained. Exact
source/qualification, preceding accepted medium admission, natural exit and
owned idle are bound to the saved analysis. Historical scalar reductions are
descriptive; no extra primary pair or quality run is implied. No numerical
helper or qualified artifact changes. Five other non-short rows, full tiled
application evidence and M3.1 retention remain pending.

Candidate boundary evidence, launched 11 October and archived 12 October 2026: the [capacity workload](docs/experiments/2026-10-11-cpu-tiled-boundary.md)
retains three warmups, ten 1,920-input / 128-output requests, actual successful
completion, owned idle and independent saved-data validation. All requests end
at 2,047 committed positions with one pending ID; the driver then verifies all
2,048 positions committed and rejects a further token before mutation. Median
TTFT is 186.521378 s and request 246.854774 s. Historical scalar reductions are
descriptive. No numerical helper or qualified artifact changes. Four cache
rows, full tiled application evidence and M3.1 retention remain pending.

Candidate append evidence, 12 October 2026: the [exact-prefix workload](docs/experiments/2026-10-12-cpu-tiled-append.md)
completes three warmups and ten requests, reusing 512 tokens and evaluating
64 appended tokens. Each measured request ends with 703 committed positions
and one pending fixed ID. Median TTFT is 5.839271 s and request 63.166372 s.
Preparation is outside request timing; all raw/sample pairs, state/event
chronology, source/qualification, actual natural exit and protected idle are
validated independently. Historical scalar reductions are descriptive. No
numerical helper or qualified artifact changes. Three cache rows (cached,
pending and changed), full application evidence and retention remain.

Excluded cached attempt, 12 October 2026: the [pause/recovery record](docs/experiments/2026-10-12-cpu-tiled-cached-excluded.md)
retains five partial rows solely as excluded evidence. All recorded benchmark
and original orchestration PIDs/groups are now absent; native session exit
status and disappearance cause are unavailable. Reviewed retirement stopped
at its pre-signal identity check. Protected endpoints and stopped VM are
verified; no resumed/reconstructed sample counts. Fresh cached qualification,
pending/changed workloads, full application and retention remain required.

Fresh cached evidence, 12 October 2026: the [complete retry](docs/experiments/2026-10-12-cpu-tiled-cached.md)
retains three fresh warmups and ten measured requests; all reuse 512 tokens,
evaluate zero prompt tokens and end at 639 committed positions plus one
pending ID. Median request is 54.425495 s and decode 2.333468 steps/s. The
empty evaluation is timed; prefill throughput is null. Cached TTFT near the
timer floor is not ranked by percentages. Exclusion recovery, current process
absence, natural completion, protected endpoints and source bindings pass
saved-data checks. No paused-v2 rows are reused. Pending/changed rows, full
application and explicit retention remain; scalar stays default.

Pending-token evidence, 12 October 2026: the [complete continuation workload](docs/experiments/2026-10-12-cpu-tiled-pending.md)
uses fixed staged token 2077, not a greedy sample, with preparation/staging
outside timing and its evaluation inside. Three warmups and ten measured
requests complete; all ten reuse 512 committed tokens, evaluate one token and
finish at 640 committed positions plus one pending ID. Median TTFT is
0.425491 s and request 54.797608 s. Source/recovery/qualification, actual
natural completion, protected idle and saved-data checks pass. Historical
scalar reductions are descriptive. Changed-prefix, full application and
retention remain; no numerical helper or qualified artifact changes.

Changed-prefix evidence, 12 October 2026: the [complete reset/replay workload](docs/experiments/2026-10-12-cpu-tiled-changed.md)
verifies the sole prompt edit at zero-based index 128, three warmups and ten
measured requests. Base preparation is untimed; reset and all 512 prompt
evaluations are timed. Every measured request reports zero reuse and ends at
639 committed positions plus one pending ID. Median TTFT is 38.530052 s and
request 93.038050 s. Source/recovery/qualification, actual natural completion,
protected idle and saved-data checks pass. Historical scalar comparisons are
descriptive. All eight numerical matrix rows now have candidate measurements;
full application and explicit retention remain. No numerical helper or
qualified artifact changes.

Full candidate application evidence, 12 October 2026: the [three-turn trace](docs/experiments/2026-10-12-application-tiled.md)
completes three warmups and ten scenarios. All thirty responses/token IDs,
visible text, stops and untimed generation states match the accepted scalar
reference. Median scenario latency is 129.141379 s, range 128.901821–
129.971217 s; historical reduction is descriptive. Tokenization/streaming,
EOS pending state, reset finalization, logical memory/admission, physical
peak and source/recovery/CLI-qualification/completion bindings pass saved-data
checks. All candidate matrix/application measurements are complete. M3.1
remains open for explicit retention; no numerical helper, kernel, qualified
artifact or quality gate changes.

CPU retention decision, 12 October 2026: [the consolidated evidence](docs/experiments/2026-10-12-cpu-tiled-retention.md)
retains the ordered FP64 tile as the qualified opt-in `--cpu-projection tiled`
variant, preserving scalar as default/reference. Both primary pairs exceed
unchanged acceptance thresholds; all eight numerical rows, full application,
full loss/task gates, bounded memory and capacity checks pass their recorded
scope. Historical comparisons, timer-floor cached TTFT, excluded attempts and
unavailable telemetry are disclosed. M3.1 and M3 close without changing any
numerical source, qualified executable or gate. Astra accepted closure after correcting cache-state and power-observation wording; numerical results and evidence are unchanged.

- [x] **M3.1 Improve one measured CPU bottleneck.** Spend approximately 4–6 hours on a profile-selected change such as contiguous traversal, vectorization, or bounded threading. Preserve the scalar oracle. Retain it only if full workloads improve without correctness or memory regressions.
- [x] **M3.2 Evaluate public CPU and ANE execution.** Spend the remaining 8–14 hours converting the tiny decoder or a supported slice, comparing CPU-only and CPU-and-Neural-Engine policies, and attempting persistent state. Capture actual placement where tooling permits, fallback, conversion cost, completed-work latency, and memory.
- [x] **M3.3 Close the experiment.** Record useful, uncompetitive, unsupported, or inconclusive, with commands and evidence. A slice result applies only to that slice. Adopt an ANE engine path only after complete stateful decode is measured; otherwise continue to M4 when the time budget ends.

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

Selected Qwen residual qualification, 10 October 2026 (M4 pending): the
[saved-operand driver](docs/experiments/2026-10-10-qwen-metal-residual.md) reuses
the accepted native primitive unchanged. Twenty width-896 stages across four
short cases pass 260 completed dispatches, 698,880 exact Half outputs, 1,820
state guards and unchanged source FP32 gates. Missing layer-22 state excludes
layer-23 attention residual. No checkpoint payload is read; frozen reference
identities, config, complete operands and all generated inputs are pinned.
Isolated formatting, Clippy, 11 Rust tests, five new offline checks and add/tail
and structured-failure regressions pass after tiny-task acceptance/commit in
the same verified idle window. Maximum native child physical peak is 15,630,960
bytes, below conservative 67,151,956-byte admission and the 6 GiB ceiling.
Frozen CPU identities are unchanged and orchestration resumed after all heavy
checks. Astra accepted the complete task without findings after independently
checking exact outputs, source gates, identities, memory and records.
Full residual coverage, decoder,
cache/session recovery, quality, performance and traces remain unqualified.

Stable SiLU primitive preparation, 10 October 2026 (M4 pending): the standalone
[Half activation](docs/experiments/2026-10-10-metal-silu.md) uses a bounded stable
FP32 formula and one Half store. Ten frozen tiny stages and signed-zero,
subnormal and saturation probes pass all 143 completed commands, 87,360 exact
outputs, 1,001 guards and unchanged FP32-source gates. Two resident owned shared
buffers preserve immutable input and enforce host access after completion.
Maximum admission is 67,129,414 bytes and own-child physical peak is 13,664,880
bytes, below 6 GiB. All add/RMS/embedding/residual regressions and required
failures pass, as do isolated formatting, Clippy, 13 Rust tests and 52 offline
experiment Python tests. Numerical checks run in the verified CPU idle window;
engine sources and frozen binaries stay unchanged. Astra accepted the complete
task without findings after independently reproducing numerical proofs,
regressions, identities, memory and final records.
This record qualifies width 64; subsequent selected Qwen evidence follows.

Selected Qwen SiLU preparation, 10 October 2026 (M4 pending): the
[width-4,864 extension](docs/experiments/2026-10-10-qwen-metal-silu.md) retains
the unchanged shader and 8,192-value inventory bound. All twelve saved stages
at layers 0, 1 and 23 split into 36 complete rows, totaling 175,104 values.
All 468 completed Qwen commands pass: 2,276,352 exact independent Half outputs,
3,276 guards, and original FP32-source gates with maximum fraction
0.495263925437. Same-binary tiny SiLU/manual and add/tail regressions pass;
650 positive commands and both required early failures are retained.
Maximum own-child physical charge is 13,091,392 bytes, below conservative
admission of 67,157,574 bytes and 6 GiB. Isolated formatting, Clippy, fourteen
Rust tests, release compilation and all 58 offline experiment Python tests
pass. No checkpoint payload is read. Checks occur only after the medium CPU
worker exits naturally. Astra accepted the complete code and evidence without
findings after independently reproducing all raw-to-compact events, numerical
checks, identities, test results and memory accounting.
Gating/MLP/decoder composition, cache, quality, performance and traces remain
unqualified, and M4 is open.

Stored-Half gating preparation, 10 October 2026 (M4 pending): the standalone
[product primitive](docs/experiments/2026-10-10-metal-gating.md) keeps two
immutable contiguous inputs and a distinct owned output, preserving the Half
SiLU boundary before an exact FP32 product and one Half store. Ten frozen tiny
stages and preselected rounding/zero/subnormal probes pass all 143 completed
commands, 87,360 exact independent Half outputs, 1,001 output guards and
unchanged source gates, maximum fraction 0.491684814232. Completed overflow is
discarded with a structured failure. Same-binary add/RMS/embedding/residual/SiLU
regressions pass: 1,040 positive commands and four expected negative runs.
Gating physical peak is 13,402,712 bytes; the largest own-process peak across
all native jobs is 17,121,856 bytes. Conservative gating admission is
67,141,744 bytes. Isolated formatting, Clippy, eighteen Rust tests and release
compilation pass, as do 63 experiment Python tests, 35 benchmark Python tests,
76 ordinary engine tests and root formatting/Clippy. Checks occur only after
the long CPU worker exits naturally; root engine sources and protected scalar
workers remain unchanged. Astra accepted the completed code and evidence without
findings after independently replaying outputs, gates, records and identities,
and checking memory, transcripts and idle-window scope.
Qwen gating, MLP/decoder/cache composition, full quality, performance and traces
remain unqualified; this does not close M4.

Selected Qwen gating preparation, 10 October 2026 (M4 pending): the
[stored-Half product extension](docs/experiments/2026-10-10-qwen-metal-gating.md)
admits one complete width-4,864 row while preserving the original 8,192-value
inventory ceiling. Both saved operands and independent product references
cover twelve stages, 36 rows and 175,104 products without checkpoint reads.
All 468 Qwen commands pass 2,276,352 exact Half outputs, 3,276 output guards
and unchanged source gates, maximum fraction 0.490092755385. Same-binary tiny
and add regressions pass; invalid shader, insufficient admission and completed
product overflow fail as required. Maximum own-process physical charge across
all jobs is 13,369,944 bytes; Qwen admission is 67,186,800 bytes. Isolated
formatting/Clippy, nineteen Rust tests and release build pass, alongside root
formatting/Clippy, 76 ordinary Rust tests, 70 experiment Python tests and 35
benchmark Python tests. Independent saved-output replay passes. Checks and
native work follow the append worker's natural exit and retained idle proof.
Astra accepted the completed code and evidence without findings after exact
independent replay and identity, memory, ownership and transcript verification.
Gating inactive-lane tails, fusion, MLP/decoder/KV composition, full quality,
performance, traces and M4 completion remain open.

Gating inactive-lane preparation, 12 October 2026 (M4 pending): the
[bounded tail extension](docs/experiments/2026-10-12-metal-gating-tails.md)
checks nine explicit one-row lengths, 5,344 products per inventory, with
queried-width rounded storage and seven additional sentinels. Both final
runs check 117 tail commands, 69,472 exact outputs and 2,483 guards each;
original primitive and selected Qwen regressions pass unchanged source gates.
Maximum admission is 67,186,800 bytes; own-child peak 16,974,448 bytes.
Direct final-size host allocations correct a reviewed capacity-accounting
issue; failed/superseded attempts remain retained outside acceptance.
Isolated native formatting/Clippy, twenty Rust tests and release build,
root formatting/Clippy and 76 ordinary Rust tests, 71 experiment and 193
benchmark Python tests, and byte-identical saved replay pass. Astra accepted
the completed code and evidence without findings after independent replay. No fusion, MLP/decoder, full quality,
performance or M4 completion is claimed.

Rust MPS projection preparation, 12 October 2026 (M4 pending): the
[public matrix boundary](docs/experiments/2026-10-12-rust-mps-projection.md)
uses pinned objc2 MPS bindings, explicit device support, queried strides,
immutable Half inputs/weights/bias and one bias→FP32-result→Half-store command.
Ten frozen tiny query stages and three manual probes pass 169 commands,
36,686 values and unchanged source gates, maximum fraction 0.498143005408.
Two repeated one-ULP Half differences are investigated; bit identity and MPS
internal accumulation remain unqualified. Own-child peak is 13,075,008 bytes;
projection admission 67,133,842. Same-binary add/gating/tails and required
negative jobs pass. Isolated formatting/Clippy, 22 Rust tests and release,
root formatting/Clippy and 76 Rust tests, 75 experiment and 193 benchmark
Python tests, and byte-identical saved replay pass. Astra accepted the completed
code and evidence after independent replay and a precision wording correction. Wider projection/MLP/decoder composition remains open.

Isolated tiny MLP projection preparation, 12 October 2026 (M4 pending): the
[actual-shape qualification](docs/experiments/2026-10-12-rust-mps-mlp-projections.md)
checks all thirty gate/up/down stages at 32→64 and 64→32. Six bounded batches
use saved Half inputs and weight identities, with zero bias for the existing
interface. All 390 commands/183,040 outputs match frozen Half bits and pass
unchanged source gates, maximum fraction 0.498911408659. Same-binary query,
manual, add, gating/tails and negative regressions pass. Compiled source/binary
identities are unchanged; peak 13,271,592 bytes/admission 67,152,466. Native
formatting/Clippy, 22 Rust tests and release, three targeted/all 78 experiment
Python tests and byte-identical replay pass. Astra accepted the completed code
and evidence without findings after independent replay. GPU stage propagation/composed MLP and M4 completion remain open.

Resident tiny MLP preparation, 12 October 2026 (M4 pending): the
[composed GPU graph](docs/experiments/2026-10-12-rust-metal-mlp.md)
propagates actual GPU outputs through gate/up, SiLU, gating and down, preserving
all Half storage boundaries in one command/wait. Ten graphs pass 130 command
buffers, 25,344 first-frame Half values and 36,608 independently retained final
outputs. All five original source gates pass, maximum fraction 0.498911408659;
first frames match frozen bits and native checks require all thirteen frames
identical. Repeated nonfinal/padding bytes are not archived; comparison scope
is explicit. Readbacks compact in place with two-snapshot bounds; admission
67,269,664 bytes and peak 13,337,152. Same-binary regressions and five required
negative jobs pass. Isolated formatting/Clippy, 25 Rust tests and release,
four targeted/all 82 experiment tests, root formatting/Clippy/76 Rust tests,
193 benchmark tests and byte-identical replay pass. Astra accepted the completed
code and evidence after independent replay and a frame-count wording correction. No full layer/decoder, KV, Qwen, performance or M4 closure claim.

Tiny attention projection preparation, 12 October 2026 (M4 pending): the
[K/V/output qualification](docs/experiments/2026-10-12-rust-mps-attention-projections.md)
covers all thirty saved-input stages:32→16 K/V with frozen biases,32→32 output
with zero bias. All unchanged source gates pass across390 commands/73,216
Half values; one V position differs by one normal Half ULP in all13 repetitions.
Independent FP64/feature-order FP32 reconstruction investigates its rounding
boundary; no tolerance or MPS accumulation guarantee changes. Same-binary
composed MLP/projection/gating/tail regressions total1,378 positive commands
and five required failures. Eleven build checks/25 standalone Rust tests and
85 experiment Python tests pass. Astra accepted code and evidence without findings after independent replay. Rotary/cache
writes, causal GQA, full decoder/Qwen and M4 completion remain open.

Tiny rotary preparation, 12 October 2026 (M4 pending): the
[split-half GPU primitive](docs/experiments/2026-10-12-rust-metal-rotary.md)
covers all twenty saved-input Q/K stages and exact position-zero/manual.
GPU FP32 angles/trigonometry/uncontracted products/sums precede Half stores;
all55,120 active values match frozen/manual bits across273 commands, with
5,863 full inactive/tail guard values replayed. Original source gates pass.
Finite-input output overflow is discarded. Same-binary regressions total
1,651 positive commands/six failures. Twelve build checks/27 standaloneRust,
88 experimentPython/193 benchmarkPython and76 rootRust tests plusfmt/Clippy
pass. The first pre-GPU Clippy failure is retained; finalnative completes
naturally with priorchains absent/VMstopped/protected endpoints unchanged.
Astra accepted code and evidence without findings after independent replay. KV writes, GQA, full decoder/Qwen and M4 remain open.

Tiny cache-write preparation, 12 October 2026 (M4 pending): the
[lossless head-major write](docs/experiments/2026-10-12-rust-metal-cache.md)
covers ten frozen local cache cases and four stale-prefix/tail/manual cases,
including last-slot/full capacity and pre-native over-capacity rejection.
RawUInt16 copies preserve exact Half storage;182 commands replay186,368 cache
payload values (52,416 new/133,952 untouched) and2,548 outputguards. Existing
rotary/attention/MLP/projection/gating regressions total1,833 positive commands
and seven negative jobs. Thirteen build checks/29standaloneRust and full
91experimentPython/193benchmarkPython/76rootRust+fmt/Clippy checks pass.
Natural completion, stoppedVM/priorchainsabsent and protected endpoints are
recorded. Astra accepted code and evidence without findings after independent replay. Saved initial caches reset outside
qualification timing; no integrated session-valid-prefix advances. Causal GQA,
propagated decoder/Qwen/quality and M4 remain open.

Tiny full-prefix GQA preparation, 12 October 2026 (M4 pending): the
[causal GPU attention](docs/experiments/2026-10-12-rust-metal-attention.md)
covers all ten saved-input jobs with stable orderedFP32 scores/softmax/context
and Half context. Original three intermediate gates pass; scores are frozen
bit-exact, probability differences are at most3FP32ULPs, and one normalHalf
contextULP is investigated at its midpoint. Independent orderedFP32 scores
and all finalcontext stores match; masked probabilities are zero and changed
future tokens preserve earlier context.130 commands retain36,608 finalcontext
values/first completeframes+2,354guards; native12laterframes equal first.
Same-binary cache/rotary/MLP checks total754positive/fournegativejobs;14build
checks/31standaloneRust and94experimentPython/193benchmarkPython/76rootRust
plusfmt/Clippy pass. Astra accepted the completed code and evidence without findings after byte-identical independent replay. Nonzero cached prefixes,
GPU propagation, full decoder/Qwen/quality and M4 remain open.

Tiny cached-prefix GQA preparation, 12 October 2026 (M4 pending): the
[absolute cache masking qualification](docs/experiments/2026-10-12-rust-metal-cached-attention.md)
covers both capacity32 layers over irregular partitions and last-token decode,
each paired with stale-unused-capacity controls. Checkedoffset+rows=valid<=32
separates query rows/cache positions. Original sliced scores/probability/context
gates pass across312 commands/54,912 finalcontextvalues; rawchunk assembly and
lastdecode equal same-binary fullprefix, and all12paired completeframes equal.
14,136 firstframeguards replayed; one previouslyinvestigated HalfcontextULP
repeats in the corresponding stalepair. Same-binary regressions total1,066
positive/fivenegativejobs.14buildchecks/31standaloneRust plus96experimentPython/
193benchmarkPython/76rootRust+fmt/Clippy pass. Astra accepted completed code and evidence without findings after byte-identical independent replay.
Read-only saved prefixes do not establish propagated cache/session behavior;
GPU layer/decoder/Qwen/quality and M4 remain open.

Propagated tiny attention preparation, 12 October 2026 (M4 pending): the
[resident branch](docs/experiments/2026-10-12-rust-metal-attention-graph.md)
composes actualQ/K/V projections→rotary→cache→causalGQA→O in onecommand/wait
from savednormalized input. Nine originalsourcegates pass; exactactualGPU
K/V layout copies have no inventedcachegate.130commands/36,608finalOoutputs,
first9Half/6F32arrays replayed; native12laterframes+guards checked. Known
Q/V/context differences propagate, and one O Halfmidpoint crossing is
investigated without tolerancechanges. Samebinary regressions total1,196
positive/sixnegativejobs.15buildchecks/33standaloneRust plus99experimentPython/
193benchmarkPython/76rootRust+fmt/Clippy pass. PreGPU buildfailure retained;
naturalcompletion/idle/VMstopped/protected endpoints recorded. Astra accepted completed code and evidence without findings after byte-identical independent replay. GPU normalization/residual/MLP layercomposition, decoder/
session/Qwen/quality and M4 remain open.

Complete tiny layer preparation, 12 October 2026 (M4 pending): the
[resident GPU layer](docs/experiments/2026-10-12-rust-metal-layer.md) adds
normalizations, two residuals and gatedMLP to actual attention propagation.
One command/wait and two boundedGPUblits preserve the first residual snapshot.
All18originalsourcegates pass; all36,608 finaloutputs match frozenHalf bits.
Known attention rounding differences are absorbed at the first residual in
these fixtures. First18Half/9F32 arrays replayed, native12laterframes/guards
checked; actual cache/cast/gating/residual boundaries and causality pass.
1,196positivecommands/sixfailures include completed-nonfinite discard.
16buildchecks/35standaloneRust and102experimentPython/193benchmarkPython/
76rootRust+fmt/Clippy pass. PreGPUtype-inference buildfailure retained;
naturalcompletion, idle/VMstopped/protected endpoints recorded. Astra accepted completed code and evidence without findings after byte-identical independent replay. Each layer consumes saved input; two-layer propagation,
embedding/finalnorm/tiedlogits/decoder/session/Qwen/quality and M4 remain open.

Complete tiny decoder preparation, 12 October 2026 (M4 pending): the
[resident two-layer decoder](docs/experiments/2026-10-12-rust-metal-decoder.md)
composes GPU embedding, both layers, final normalization and one tied table's
FP32 logits in one command/wait. No saved incoming layer activations are used.
Five prefixes and four frozen teacher-forced histories pass117completedcommands;
four greedy IDs are47 and target histories equal source histories. All original
gates pass:1,408embedding/1,408finalnorm Half values exact and2,466/2,816logits
frozenFP32bits exact. Known eight Half attention differences are absorbed at
first residual. Independent10,560RMS/9,536score values and allactualGPUcache/
cast/context/residual/gating boundaries pass. Future-token causality is exact.
Ten layer regressions+add/tails give286positivecommands/fivenegativejobs.
Combined admission67,909,198bytes/own-child peak14,402,184bytes;17buildchecks/
38standaloneRust/106experimentPython/193benchmarkPython/76rootRust+fmt/Clippy
pass. Both preGPUbuildfailures, final native evidence, idle/protected identities
and replay producers are retained. Astra accepted code and complete evidence
with no remaining findings after independent byte-identical replay. Root backend, incremental GPU session,
Qwen, quality, performance, trace and M4 closure remain pending.

Selected Qwen Q/K/V preparation, 13 October 2026 (M4 pending): the
[bounded layer0 projection slice](docs/experiments/2026-10-13-rust-metal-qwen-projections.md)
qualifies896→896Q/896→128K,V on four savednormalizedinputs. Six pinnedBF16
tensors convert once to Half; strictboundedexternalweightfile profile leaves
tiny parser/shader limits unchanged. All12originalsourcegates pass156commands/
179,712Halfoutputs. Q10,751/10,752,K1,536/1,536,V1,533/1,536frozenHalfexact;
four normal oneHalfULP differences diagnosed without tolerance changes.
AllactualFP32→Half casts/repetitions/future-prefixchecks pass; nine complete
tiny decoder frames match priorqualification. Independent1,033,344BF16word
conversion and byte-identical eventreplay pass. Same-binary regressions yield
481positivecommands/threenegativejobs. Admission71,995,858bytes/own-child peak
17,875,520bytes;17buildchecks/40standaloneRust/109experimentPython/193benchmark
Python/76rootRust+fmt/Clippy pass. Native/archive/idle/protectedidentity/replay
records retained. Astra accepted the complete code and evidence with no
remaining findings after independent byte-identical replay. Propagated Qwen normalization/attention/layer/decoder/session,
quality/performance/trace and M4 closure remain pending.

Qwen input normalization→Q/K/V preparation, 13 October 2026 (M4 pending): the
[resident layer0 front](docs/experiments/2026-10-13-rust-metal-qwen-front.md)
normalizes savedembeddingHalf once and feeds actualGPUHalf to allthree
projections in onecommand/wait. Fourprefixes pass16originalsourcegates and
52commands/179,712retainedQKVoutputs. Independent10,752orderedRMSvalues exact;
21normalization oneHalfULP differences from frozen reduction change309QKV
values versus savednormpath, with alloriginalgates unchanged. Actualprojection
casts and firstthree-token causality pass. SevenBF16tensors/1,034,240words
convert independently exact. Same-binary projections/tinydecoder match prior;
533positivecommands/fournegativejobs include completed-normalization discard.
Admission73,591,540bytes/own-child peak18,383,496bytes;18buildchecks/41standalone
Rust/113experimentPython/193benchmarkPython/76rootRust+fmt/Clippy pass. Both
attempts retained: v1self-driver identity omission fixed/retested in accepted
v2provenance. Idle/protected identities and byte-identical replay retained.
Astra accepted the complete code and evidence with no remaining findings.
Qwenrotary/cache/GQA/full-layer/decoder/session/quality/performance/trace and
M4closure remain pending.

Qwen rotary64 preparation, 13 October 2026 (M4 pending): the
[bounded split-half primitive](docs/experiments/2026-10-13-rust-metal-qwen-rotary.md)
uses14Q/twoKVheads and64dimensions under unchanged gates. Eight saved stages
plus exactposition0manual pass117commands/161,408Halfvalues/819guards.
Q10,749/10,752,K1,536/1,536Halfexact; gatefractions≤0.453451. Tiny/front/fulltiny
regressions give598positivecommands/threenegatives, preceding graphframesexact.
18buildchecks42standaloneRust115experimentPython193benchmarkPython76rootRust+
fmt/Clippy pass. Admission67,144,774bytes/own-child peak17,334,896bytes;
identities, idle, raw replay and guards retained. Astra accepted code and
complete evidence after independent replay. ActualQwenpropagation and
rootMetal residentbackend/session/trace remain pending; M4 stays open.

Root resident integration, 13 October 2026: the
[opt-in Metal backend](docs/experiments/2026-10-13-metal-engine.md) streams
validated Half weights, keeps KV/scratch/pipelines resident, uses the coarse
static session interface and completes each chunk in one command/wait. Five
tiny cases, full/3/1 chunks, all intermediate source gates, four greedy steps,
pending-prefix/generation/error/budget checks and zero warmed Rust allocations
pass. Full Qwen passes 612 selected intermediate/output comparisons and all
source final greedy IDs; all 2048 positions pass a separate exact zero-model
capacity oracle. Real overflow recovery and injected completion-status replay
pass. Three warmups/ten resident requests repeat exact IDs; complete chat reaches
EOS151645. Native charge peaks stay near1.135GB; admission with application
allowance is1.330GB. Identity-bound raw records, trace bundle hashes and unchanged
CPU release binaries are retained. 76 default/78 Metal-feature ordinary tests,
125 Python checks, formatting and both strict Clippy configurations pass. Astra
accepted code and complete evidence after independent replay. M4.1–M4.3 are
functionally closed. Actual65,099,992-byte GPU trace capture succeeds; per-kernel/
idle attribution remains M4.4. FP16 held-out10k-loss/100-task quality remains
explicitly unqualified; the CPU default and frozen gates are preserved.

Metal scheduling attribution, 13 October 2026: the
[final counter/capture report](docs/experiments/2026-10-13-metal-trace.md)
closes M4.4 at the public API boundary. Custom encoder timestamps, dependent MPS
markers and paired CPU/GPU clock conversion explain kernel/enclosed-MPS,
transition, host encoding, command creation, calibration and readback costs.
Three warmups/ten uncaptured samples/one separate capture retain exact logits
against ordinary sessions: 2,296 tiny and 25,704 Qwen stage events. Instrumentation
overhead is explicit; transition intervals are not global GPU idle claims.
Actual captures and source/binary/model/raw identities are hash-bound. Persistent
allocations stay fixed; Qwen uncaptured native charge grows 96 KiB after warmup.
Conservative capture admission estimate is 6,222,702,660 bytes under 6 GiB;
observed peak charge is 2,136,459,400 bytes. Default/Metal CPU tests pass 76/80,
Python checks 137, strict Clippy/fmt and final release tiny/Qwen gates pass.
Astra found and accepted a tested stale-profile failure fix, then independently
replayed both final analyses and verified every capture/manifest hash. Protected
CPU release binaries are unchanged. M4's functional exit is complete; CPU stays
default. Held-out FP16 10k-loss/100-task quality and performance replacement
acceptance remain unqualified, separately from this functional milestone.

- [x] **M4.1 Establish the Apple backend.** Qualify the Xcode/Metal toolchain, deployment target, device capability checks, buffer ownership, and command completion. Use a small `objc2`/`objc2-metal` boundary, following the [metal-rs migration recommendation](https://github.com/gfx-rs/metal-rs). Add a minimal kernel dispatch and CPU-versus-GPU comparison before porting the model.
- [x] **M4.2 Port the forward pass in slices.** Implement embedding, projections, RMSNorm, rotary/cache writes, causal GQA, gated MLP, residuals, final normalization, and output projection. Begin with one layer and then the full tiny model, followed by Qwen. Qualify reduction accumulation and FP16 ranges using intermediate checks. CPU greedy sampling is acceptable initially.
- [x] **M4.3 Keep model and session buffers resident.** Preallocate scratch for admitted shapes; cache pipeline objects. Load FP16 buffers without retaining an unnecessary full FP32 copy. Run CPU/GPU comparisons in separate processes when needed to respect peak memory. Introduce the narrow backend interface here.
- [x] **M4.4 Validate scheduling and failure boundaries.** Remove debugging waits after individual operators once parity passes. Preserve dependencies, wait at required request/token boundaries, and maintain committed state on cancellation or GPU failure. Capture a GPU trace that separates kernel time, host encoding, allocations, and idle gaps.

Before writing a substantial custom GEMM, compare an available Apple matrix primitive or TensorOps implementation at the actual projection shapes. Retain compatible kernels when the API is absent or slower; record the chosen device/OS requirements.

Exit: the M2 session suite passes with FP16-specific numerical gates, complete Qwen generation runs on Metal, allocations remain bounded across repeated tokens, and the trace explains remaining costs. Speed parity with mature engines is measured separately from functional completion.


## M5 Optimize decode and prefill

**Complete, 15 October 2026.** The [final release decision](docs/experiments/2026-10-15-m5-release.md) qualifies the small compensated Half configuration and frozen larger group8 Q4 v2 configuration. Larger v2 passes unchanged loss/task gates and confirms a22.20–23.69% named prefill gain; full matrix/application/confirmation/sustained/state/memory evidence is accepted. All rejections, defaults and baseline limits remain recorded. M6/M7 remain pending.

- [x] **M5.1 Freeze and test Q4 format version 1.** Specify row/group order, nibble order, rounding/clipping, constant groups, padding, alignment, endianness, FP16 metadata rounding, and excluded tensors. Start with groups of 64 along each linear row; reconstruct as `weight = scale * code + bias`, with codes 0–15. Keep norms, biases, and initially the tied embedding/output matrix in FP16. Implement a Python packer and simple Rust dequantization oracle. Test constants, extreme values, row tails, and exact packed-byte round trips. This is an engine format, not an implied claim of MLX compatibility.
- [x] **M5.2 Implement fused Q4 decode.** Dequantize tiles or values during matrix-vector execution; do not expand the whole model per token. Compare reconstructed-weight math with the CPU oracle and run the frozen quality suite against the unquantized checkpoint. Preserve the FP16 path if Q4 misses its quality gate.
- [x] **M5.3 Implement separate prefill.** Add tiled matrix-matrix execution and reuse dequantized weights across prompt rows. Sweep chunks of 128, 256, 512, and 1,024 within the memory ceiling. Compute vocabulary logits only for required positions. Verify irregular final chunks and nonzero cached offsets; retain simple attention until M6.
- [x] **M5.4 Reduce measured dispatch and traffic costs.** Evaluate GPU greedy selection returning a token ID, residual/normalization fusion, rotary/cache-write fusion, and MLP gating fusion individually. Add stochastic sampling only when required, with explicit RNG state and all filters/penalties in the measured path. Retain a small shape/device dispatch table rather than a general autotuner.
- [x] **M5.5 Optimize conversation continuation.** Find the longest exact shared token prefix, truncate only representable state, and replay from the required point. Handle an entirely cached prompt, a pending sampled token, and edits near the start or end. Compare continuation latency with reset-and-replay.
- [x] **M5.6 Qualify the performance release.** Repeat controlled and application comparisons, sustained runs, and quality checks. Select one larger checkpoint in the implemented Qwen family using actual tensor payload and memory admission; record its exact artifact before claiming behavior at larger scale. Reject configurations that exceed 6 GiB, including high-precision exclusions and load-time scratch.

M5.1 format and CPU oracle qualification, 13 October 2026: the [Q4 v1 contract](docs/q4-v1.md) freezes decoded FP32 source values, row-local affine groups of 64, rounded FP16 metadata, ties-to-even codes, canonical padding and explicit underflow/overflow policies. Python and Rust independently verify the frozen hand-calculated artifact. The [task report](docs/experiments/2026-10-13-q4-format.md) records checks and Astra review. This qualifies the matrix format only; Q4 GPU/model loading, quality and performance remain M5.2–M5.6.

M5.2 model-artifact prerequisite, 13 October 2026: [source-bound artifact report](docs/experiments/2026-10-13-q4-artifact.md). The row-streaming Python model packer and CPU-only Rust inventory bind config/shard identities, exact matrix shapes, version/quantizer and packed bytes, and admit packed residency plus one-matrix staging before allocations. Tiny14 and Qwen168 matrices pass explicit source/geometry/structure validation; ordinary CPU tests exercise mutation, malformed inventory, overflow and failure cleanup. This is an accepted prerequisite only: fused GPU decode, held-out quality and M5.2 closure remain pending.

M5.2 resident execution code task, 13 October 2026: [fused Q4 execution report](docs/experiments/2026-10-13-q4-decode.md). The root Metal library loads seven packed matrices per layer with streamed Half exclusions, checked full load/session admission, source rebinding after artifact identity establishment, and one queried SIMD group per projected row. No full-model/per-token weight expansion is retained. GPU evidence covers2,310 tiny stage comparisons against a separately frozen reconstructed-weight oracle,21 Qwen projections against the scalar dequantization oracle,8-step Qwen generation, pending/state/budget contracts and zero Rust allocations in20 warmed steps. The FP16 suites remain passing. These code tasks do not close M5.2: unchanged held-out quality and later performance acceptance remain pending.

M5.5 completion, 13 October 2026: [exact-prefix continuation report](docs/experiments/2026-10-13-prefix-continuation.md). CPU and Metal retain the longest exact committed prefix without KV copying, recover shortened/invalidated distributions by replaying the final required token, and preserve recovery provenance across repeated cleanup. Tiny FP16/Q4 and pinned CPU model checks pass. Two paired Q4 512-token runs confirm 21.6–21.8% lower engine TTFT when reusing 128 tokens and 93.0% for a final-token edit, with zero measured absolute logit difference and fixed admitted buffers. Astra accepted code/check/evidence closure; Q4 held-out quality and final M5 release gates remain pending.

M5.3 required-position computation code task, 13 October 2026: [task report](docs/experiments/2026-10-13-required-position-logits.md). Ordinary prefill uses a precreated final-row Half view and one-row vocabulary output; diagnostics retain every row. Native parity covers 76 full distributions across tiny/Qwen, FP16/Q4, offsets and irregular chunks under unchanged gates. Astra accepted code/check evidence; Float scratch remains unchanged and no speed/memory benefit is claimed. Tiled execution, chunk 128/256/512/1024 sweeps and M5.3 closure remain pending.

M5.2 held-out loss result, 13 October 2026: [Metal scoring/rejection report](docs/experiments/2026-10-13-metal-held-out-loss.md). The bounded all-required-position scorer passes code checks and Astra evidence review. Qwen 0.5B Q4 completes all 10,000 targets with mean NLL 3.057166 versus frozen scalar FP32 2.883725, increase 0.173441 > 0.05: quality rejected. FP16 short batch parity differs 0.004681 > 0.001 and stops before full scoring; no full FP16 quality conclusion is made. No gate/quantizer/corpus changes follow the results. Q4 on 0.5B remains quality-unqualified, CPU remains default, and M5.2/M5 exit stay open. Larger-checkpoint qualification is separate and cannot rehabilitate this result.

M5.6 single larger preselection, 13 October 2026: [selection report](docs/experiments/2026-10-13-larger-preselection.md). Qwen2.5-1.5B-Instruct revision 989aa7980e4cf806f80c7fef2b1adb7bc71aa306 is selected from its exact pinned BF16 header/payload, before larger quality results, using unchanged Q4 v1. Predicted Q4 residency 1,223,992,516 bytes; FP32 reference lower bound 6,367,852,544 bytes leaves 74,598,400 bytes below 6 GiB for remaining buffers/native overhead. Astra accepted selection-only scope. Full source/hash, actual admission, larger geometry, matched quality and performance evidence remain pending; no search/tuning from held-out feedback.

M5.6 bounded reference prerequisite, 13 October 2026: [caller-owned CPU all-row logits](docs/experiments/2026-10-13-cpu-all-row-logits.md) reuse existing FP32 weights/KV/scratch and FP64 scalar/tiled arithmetic. Checked external output permits all required causal rows without a full trace or duplicate model; normal final-row execution is unchanged. 84 tiny complete all-row comparisons in total (42 per kernel), plus 36 pinned-Qwen comparisons per kernel pass, with transactional failures and zero Rust allocations. This is a code prerequisite only; larger actual admission and matched quality remain pending.

M5.6 reference scorer/admission prerequisite, 13 October 2026: [bounded CPU row-loss report](docs/experiments/2026-10-13-cpu-row-loss.md). New all-required-row scorer/collector preserves frozen blocks, stable loss and owned bounded handshakes without changing protected CPU helpers. Small frozen probes and larger chunk1/8 consistency pass. Downloaded larger source/config/header identities match preselection; tokenizer semantics are byte-identical. Larger chunk8 admission 6,370,269,388 bytes and short-run combined native peak 6,288,004,880 bytes fit 6 GiB. Independent larger numerical geometry qualification and full reference loss remain pending; no GPU quality or M5 closure.

M5.6 independent larger CPU qualification, 13 October 2026: [oracle report](docs/experiments/2026-10-13-larger-independent-oracle.md). Bounded streamed Transformers eager FP32 parameters reproduce all228 old small-model arrays/envelopes exactly, then freeze larger layers0/1/27 under the unchanged gate formula before Rust comparison. Larger228 complete independent arrays pass (maximum fraction0.520006), all2028 scalar/tiled trace arrays have identical bits,36 cache distributions agree, and maximum short NLL difference0.000055770<0.001. Rust own-process peak6,391,338,192 bytes fits6GiB. Full larger reference loss and matched GPU quality/performance remain pending; the0.5B Q4 rejection is unchanged.

M5.6 larger FP32 held-out reference, 13 October 2026: [collection report](docs/experiments/2026-10-13-larger-fp32-loss.md). All 10000 frozen targets complete at mean NLL2.485958031019153, with1260 chunks and10 final-token-committed blocks. Five workers exit cleanly; allocation admission6,370,269,388 and joint native lifetime peak6,290,069,216 bytes fit 6 GiB. Astra independently replays all11306events and accepts frozen source/binary snapshot bindings. This supplies the matched larger reference only; GPU loss,100-task quality, performance and sustained gates remain pending, and the0.5B Q4 rejection is unchanged.

M5.3 experimental eight-row Q4 tile, 13 October 2026: [task report](docs/experiments/2026-10-13-q4-prefill.md). Fixed-index weight reuse improves512-input/one-output engineTTFT by35.1%/35.2% in two quiet paired runs with identical final logits. Independent projection bounds/tails and4550 complete Original/tile stage arrays pass refinement checks. A newly frozen offset17 independent decoder gate fails identically on both paths and remains separately runnable/unresolved; no tolerance is relaxed. Retain only the opt-in experimental mode, Original default, chunk<=128 and existing geometry. Chunk sweeps, full-graph numerical acceptance and M5 release gates remain open.

M5.6 independent larger Half/Q4 oracle prerequisite, 13 October 2026: [streamed storage oracle report](docs/experiments/2026-10-13-streamed-storage-oracles.md). One cached Half tied table plus streamed current tensors reproduces all228 small Half payloads/gates exactly, then freezes228 larger arrays per precision beforeGPUcomparison. Source/config/full-reference pins and exact-byte provenance, completed-memory-before-publication, and bounded probes are checked. Larger Half/Q4 admission1,159,593,984/1,205,469,248 and completed native peaks724,665,376/724,878,416 bytes fit 6 GiB;34 referencePython tests pass. GPUgeometry/parity, matchedquality and M5release remain pending; unchangedtinyindependentfailure and0.5BQ4rejection persist.

M5.2 depends on M5.1. M5.3 and M5.5 can start from M4 independently of Q4; M5.4 follows the relevant profile. M5.6 requires all accepted changes. Grouped affine quantization is a practical candidate, but container names alone do not establish format compatibility. [MLX quantization API](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.quantize.html).

Exit: Q4 passes quality checks and improves at least one named workload beyond measured noise; other accepted optimizations satisfy their declared tradeoffs. Report baseline gaps if they remain. Claim an external speed lead only when the same quality, memory, and comparison conditions support it.

M5.2/M5.6 numerical component, 13 October 2026: [compensated Q4 storage and larger GPU geometry](docs/experiments/2026-10-13-q4-compensated-storage.md) admits the exact preselected 28-layer geometry and bounded Q4 profiling. Larger Half passes969 selected-stage comparisons; larger Q4 fails its unchanged gate. An opt-in compensated projection fixes a localized double-rounding error at Half storage and passes4550 expanded tiny stage comparisons plus72 signed midpoint/overflow cases. Original/Tiled failures remain preserved, no quality/performance/default acceptance is claimed, and larger normalization/rotary investigation remains pending.

M5.2/M5.6 larger numerical qualification, 13 October 2026: [compensated RMS report](docs/experiments/2026-10-13-compensated-rms.md). Identical-input replay isolates 66 first-layer Half differences from ordered FP32 square summation; compensated reduction matches both frozen NumPy and FP64 outputs. A separate opt-in normalization mode passes 43,008 exact primitive outputs, 4,550 expanded tiny stage comparisons and 969 larger selected-stage comparisons (maximum gate fraction 0.732124), with native peak 1,344,226,768 bytes. Defaults, quantizer, source and all gates remain unchanged. This qualifies short larger Q4 numerical execution only; matched held-out/task quality, speed, sustained runs and M5 exit remain pending.

M5.2/M5.6 matched larger Q4 loss, 13 October 2026: [collector/rejection report](docs/experiments/2026-10-13-larger-metal-loss.md). The compensated numerically qualified larger path scores all 10,000 targets at mean NLL 2.588645591385506 against matched FP32 2.485958031019153: increase 0.10268756036635285 exceeds 0.05, so quality is rejected. Both unchanged 1e-3 short consistency guards pass (maximum 0.000008573); five owned workers exit 0 with empty cleanup/endpoint errors, and combined native peak 1,606,355,152 bytes fits 6 GiB. No source/quantizer/corpus/gate/default change follows. Both selected Q4 configurations remain quality-unqualified; M5 exit and remaining task/performance gates stay open.

M5.3 accurate eight-row Q4 tile, 13 October 2026: [component experiment](docs/experiments/2026-10-13-accurate-q4-prefill.md). Explicit compensated weight reuse passes unchanged 4,550 tiny independent stage comparisons, a separate 4,550-array strict bit refinement and bounds/midpoint checks. Two quiet larger 512-input/one-output paired runs improve engine TTFT by 25.4128%/25.6841%, preserving all 151,936 final-logit bits per pair and fixed persistent buffers. Retain only as an explicit experimental mode; both fixed Q4 quality rejections, defaults, chunk<=128 and M5 exit remain unchanged. Full workloads and chunk sweeps remain pending.

M5.3 explicit chunk admission/sweep, 13 October 2026: [task report](docs/experiments/2026-10-13-metal-chunks.md). Checked Metal admission extends through 1024 with default 128 unchanged. Eight independent exact-zero capacity/budget cases pass; fixed larger compensated Q4 reset1024/append7/edit1000 at offset 17 preserves all 151936 final-logit bits across 128/256/512/1024. The completed 3-warmup/10-measured rotating sweep has only 4.0506%/2.6294% gains at 256/512 and an 8.4712% loss at 1024, so no larger default meets 5% or merits a retention confirmation. Combined admission 3,521,063,508/native joint peak 3,239,218,488 fit 6 GiB. This completes this named component sweep, not every mode's independent numerical/quality qualification or the full 128-output release workload. Frozen Q4 quality rejections and M5 exit stay open.

M5.3 scoped functional completion, 13 October 2026: [prefill synthesis](docs/experiments/2026-10-13-m5-prefill-closure.md) binds required-position logits, accurate compensated eight-row reuse, irregular/tail/offset checks and the completed four-chunk sweep. The compensated path's independent tiny gate and strict refinement pass; two larger component runs confirm25.4128%/25.6841% TTFT gain. No larger chunk reaches5%, so retain default128 and simple attention. This closes the named prefill mechanisms only. Original/Tiled expanded-oracle failures, limited larger-chunk final-distribution coverage, both frozen Q4 loss rejections and pending M5.6 release/application/full-workload gates remain explicit. Defaults are unchanged; overall M5 remains open.

M5.4 GPU greedy code task, completed14 October2026: [selection report](docs/experiments/2026-10-13-metal-greedy.md). Explicit token-only sessions select the exact lowest-ID finite FP32 maximum during completed GPU evaluation, preserving pending/prefix/failure semantics and full diagnostics.160 exact CPU-oracle primitive cases,560 complete tiny Half/Q4 refinement arrays, failure/profile/replay and zero warmed Rust-allocation checks pass. Larger512/128 generation retains all128IDs/state across26requests; median79.169037458s is0.2890% slower than FullLogits78.9409020835s. Reject performance adoption/confirmation; retain explicit functional ID-only capability and1,215,448-byte host-admission reduction, defaults unchanged. Joint admission1,690,112,692/native peak1,534,445,800 fit 6 GiB. Remaining three fusion experiments, quality/task/application/release qualification and overall M5 remain open.

M5.4 MLP gating experiment,14 October2026: [negative task report](docs/experiments/2026-10-14-mlp-gating.md). Fused stableSiLU preserves its intermediateHalf round/check before multiplication, with actualdiagnostic stage materialization.2,178,580primitive comparisons,9,100tiny stagebitrefinements, ordinary/failure/allocation checks and969independent larger arrays perprecision pass. Completed paired512/128median gain0.006032% is belowdispersion/5%; reject andremove the livekernel/API/tests/harness, retaining boundprototype source/binaries andcompleteevidence. All26requests preserve128IDs/final-logitbits/state, admission1,691,328,140/native jointpeak1,534,347,448 fit 6 GiB. No confirmation/default/quality change. Two remaining fusion experiments andM5release gates stayopen.

M5.4 rotary/cache-write experiment, 14 October 2026: [negative task report](docs/experiments/2026-10-14-rotary-cache.md). One fused pair owner preserves explicit Half rotary storage, independent absolute/cache positions and raw V copying. All 1,280 primitive cases, 9,100 complete tiny stage refinements, ordinary/failure/allocation checks and 969 independent larger arrays per precision pass. Completed paired 512/128 median gain is only 0.014593%, below the 5% gate and Split dispersion; reject and remove the live kernel/API/tests/harness, retaining bound prototype source/binaries and complete evidence. All 26 requests preserve IDs/final-logit bits/state; combined admission 1,691,328,140 and joint native peak 1,534,068,848 bytes fit 6 GiB. No confirmation/default/quality change. Residual/post-RMS evaluation and M5 release gates remain open.

M5.4 residual/post-RMS experiment, 14 October 2026: [negative task report](docs/experiments/2026-10-14-residual-rms.md). A row owner preserves the explicit rounded resident Half residual and unchanged Original/Compensated reduction before normalization. All 4,379,072 primitive comparisons, 66 frozen reduction-mode output differences, 9,100 complete tiny stage refinements, ordinary/failure/allocation checks and 969 independent larger arrays per precision pass. Completed paired 512/128 Fused median is 0.553034% slower; reject and remove the live kernel/API/tests/harness, retaining bound prototype source/binaries and complete evidence. All 26 requests preserve IDs/final-logit bits/state; combined admission 1,691,328,140 and joint native peak 1,535,330,440 bytes fit 6 GiB. No confirmation/default/quality change. This finishes the final named individual fusion evaluation; scoped M5.4 closure follows final acceptance. M5.2/M5.6 and overall M5 remain open.

M5.4 named mechanism evaluations complete, 14 October 2026: [scoped synthesis](docs/experiments/2026-10-14-m5-dispatch-closure.md) links GPU greedy and the three individually qualified/measured fusion decisions. Retain only explicit functional token-ID readback; its 0.2890% slowdown prevents performance adoption. MLP gain 0.006032% and rotary/cache gain 0.014593% miss the 5% gate/noise policy; residual/post-RMS is 0.553034% slower. All three live fusion prototypes are removed after checks and Astra review. Existing bounded operation-level modes/queried device capabilities remain; no unused dispatch table or unnecessary stochastic sampling is added. This closes named evaluations only, with no retained fusion request-latency gain. Defaults, fixed quantizer/checkpoints/corpus/gates and both Q4 quality rejections remain unchanged. M5.2/M5.6 and overall M5 remain open.

M5.6 larger Half fallback prerequisite, 14 October 2026: [guard rejection report](docs/experiments/2026-10-14-larger-half-loss.md). Existing Original Half/MPS and RMS configuration, with unchanged checkpoint/corpus/guards, stops at the first short probe: target batch/single NLL difference 0.001523869051518 exceeds 0.001. Two owned workers exit 0 and all 17 events replay cleanly; full 10,000-target scoring never starts, so no full Half mean/quality conclusion is claimed. Admission 3,512,638,586/sampled joint native charge 3,362,491,920 bytes fit 6 GiB. Preserve the result and source/binary bindings; localize all-layer and identical-input projection differences before any arithmetic change or retry. Both Q4 quality failures, defaults and M5.2/M5.6/overall M5 remain open.

M5.6 Half consistency localization, 14 October 2026: [all-layer investigation](docs/experiments/2026-10-14-half-consistency.md). All six scored rows reproduce the pinned failure; complete451-stage/56-cache capture preserves diagnostic baseline bits. Identical-input replay localizes the first15 stored Half differences to layer0 Q Original MPS shape-dependent arithmetic and reproduces both graph outputs. Exact dyadic witnesses match single-row at6 coordinates, batch at8, neither at1; neither shape is an oracle. Admission4,798,494,330 and sampled joint native charge3,885,354,872 bytes fit6GiB. This is localization only; an explicit compensated Half candidate requires separate primitive/graph qualification and the unchanged consistency/quality gates. No production arithmetic or default changes; M5.2/M5.6 and overall M5 remain open.

M5.6 explicit compensated Half mechanism, 14 October 2026: [numerical task](docs/experiments/2026-10-14-half-compensated.md). Resident Half-only mode compensates exactly represented FP32 products, adds bias last, prepares Half storage with round-to-odd and preserves nearest FP32 vocabulary output. Compatible queried widths32/32, checked view offsets/strides, cached resources and unchanged normalization/attention pass1,668 primitive outputs,2,520 tiny captured independent stage arrays and969 selected larger arrays under frozen gates. Ordinary/diagnostic publication, token-only refinement, budgets, failure/replay and warmed zero-Rust-allocation checks pass. Larger480-event profile and sampled joint native charge3,205,254,408 bytes fit6GiB. The failed initial inventory harness is retained/excluded; fresh v2 qualification passes. This mechanism remains opt-in, with no held-out consistency/full quality/speed acceptance. Scorer integration and unchanged guards are next; both Q4 quality failures and M5.2/M5.6/overall M5 remain open.

M5.2/M5.6 compensated Half short scorer, 14 October 2026: [integration/probe task](docs/experiments/2026-10-14-half-scorer.md). Explicit Half-only selection validates actual geometry/Original RMS/compatible widths and preserves default calls. Small contract bytes are pinned before parsing and reasserted before launch; offline checks enforce guard-before-full and distinct probe/full-quality completion. Both fixed checkpoints pass the unchanged1e-3 guard with zero NLL difference and exact scored-row logits/target bits on both probes, four workers/34events/12observations per model. Admissions1,392,321,514/3,512,638,586 and joint native charges1,230,210,944/3,361,557,984 bytes fit6GiB. Full corpus is not collected in this task; Original failures and both Q4 full-loss rejections remain. Next is unchanged full Half loss; M5.2/M5.6/overall M5 stay open.

M5.2/M5.6 compensated Half full loss, 14 October 2026: [completed data qualification](docs/experiments/2026-10-14-half-full-loss.md). Both preselected models score all10,000 frozen targets across10blocks/88chunks after both unchanged short guards pass. HalfCompensated+Original RMS means2.8836405658255675/2.485924560227056 versus matched FP32 2.883725149929484/2.485958031019153; increases−0.0000845841039165407/−0.00003347079209703452 meet the unchanged0.05 limit. Fivecleanworkers/10,134raw events per model replay; admissions1,392,321,514/3,512,638,586 and sampled joint native peaks1,319,700,328/3,449,327,120 bytes fit6GiB. No code, corpus, quantizer, model, gate or default changes during collection. This qualifies full Half loss only; CLI/workload/application/task qualification and final-configuration release evidence remain. Both Q4 v1 full-loss rejections still block the explicit Q4 exit; M5.2/M5.6/overall M5 stay open.

M5.6 explicit compensated Half CLI, 14 October 2026: [functional task](docs/experiments/2026-10-14-metal-cli-half.md). Opt-in flag preserves defaults and publishes actual ready/done mode/dtypes/chunk and complete logical admission. Six sequential native workers pass79event replay, cached/pending/invalid/reset/stop/cancellation/idle-exit cases and independently finalized raw Unicode/chat text; maximum admission1,309,679,844/native joint sampled charge1,187,547,056bytes fit6GiB. The sandbox no-device attempt is retained/excluded; fresh approved native v2 passes.89CPU/98Metal-feature/189Python checks and strict formatting/Clippy pass. This closes CLI functional scope only; controlled/application/task/sustained release evidence and the unchanged Q4 quality-blocked exit remain open.

M5.6 controlled Metal worker/collector,14 October2026: [functional task](docs/experiments/2026-10-14-metal-controlled.md). Separate concrete HalfCompensated/Original/FullLogits worker mirrors frozen CPU timing, reuses all eight contract workloads and explicitly times changed reset/replay. Exact bounded consumed token bytes, initial/endpoint pins, complete history SHA/state, completed commands and fixed inventories pass16sequential workers/62events on both selected checkpoints; full2048boundary commits and next-token rejections pass. Maximum admissions1,262,602,960/3,396,602,720 and sampled joint native charges1,158,891,392/3,295,317,520bytes fit6GiB.89CPU/98Metal/196Python/fmt/strictClippy/release checks pass. These are single check-only requests, with no benchmark rows or measurement completion. Three-warmup/ten-sample collection, application/task/sustained evidence and the unchanged Q4-blocked M5 exit remain open.

M5.6 larger FP32 task-reference prerequisite,14 October2026: [bounded raw-ID task](docs/experiments/2026-10-14-cpu-task-reference.md). Fixed capacity562/chunk8 with separate64MiB application allowance admits all100frozen prompt+output maxima without a simultaneous tokenizer: actual6,343,713,560bytes. Owned model exits/reaps before independent final detokenization; decoder phase180,663,760bytes. Synthetic immediateEOS/max562/cap/prefix/reset checks,96CPU/108Metal all-target/201Python/fmt/strictClippy/release pass. Three actual larger short probes and four prefix-preserving rejections replay28events, with sampled model-phase joint native charge6,237,181,496 and decoder-driver lifetime peak74,777,176bytes. This is reference/protocol/memory qualification only; probes are excluded from100-task quality. Full references, matched Half/literal task scores, application/performance/sustained gates and the unchanged Q4-blocked exit remain open.

M5.6 backend-aware Half task collector,14 October2026: [tool-only task](docs/experiments/2026-10-14-metal-tasks.md). Actual Metal/CPUprojectionnone/HalfCompensated/Original/FullLogits metadata, exact qualified-binary/source admission and unchanged100 IDs/caps/EOS/reset grammar replace no CPU labels. Bounded parsed receipt and initial recheck reject replacement before launch.208Python/whitespace checks and six short probes on both fixed checkpoints replay55events with independent post-exit finalized text. Admissions1,443,266,720/3,577,266,480 and sampled model-phase native joint charges1,270,007,752/3,406,483,080bytes fit6GiB; exit/cleanup/endpoints pass. Probes are excluded from100-task quality. Full matched responses, separate literal-grading admission, application/performance/sustained evidence and the unchanged Q4-blocked M5 exit remain open.

M5.6 full task grading admission,14 October2026: [tool-only task](docs/experiments/2026-10-14-task-grading-admission.md). Full100-case truthful Half Metal and bounded larger FP32 collections require original IDs/caps, exact native replay, successful owned exit, complete memory and independent post-exit text. Exact backend inventories and pinned qualification executable hashes close Astra's missing-entry/replacement findings; nine targeted/all410Python checks pass. All three published short probes reject as grading inputs. No VM/grading/model run or score occurs in this task; backend-aware literal window, full matched responses/scores, application/performance/sustained gates and unchanged Q4-blocked exit remain open.

M5.6 backend-aware literal grading window,14 October2026: [outer adapter](docs/experiments/2026-10-14-task-grading-window.md). Separate runner preserves literal source/text/100-case scoring, actual identity and unresolved-no-aggregate behavior. Exact admitted hashes reject replacement before execution. Window requires all three full naturally exited collections and fresh PID/group absence immediately before startup, pinned qualified receipts/image/API and initially stopped machine with owned restoration. Existing controller/classifier/isolation/lifecycle unchanged.22mocked/all432Python checks pass; native grading/window execution is deferred until full collections exist. Application/full response/quality/performance/sustained gates and unchanged Q4-blocked exit remain open.

M5.6 Half application adapter,14 October2026: [functional task](docs/experiments/2026-10-14-metal-application.md). Exact frozen scripted three-turn36/76/117IDs, actual HalfCLI mode/memory, empty-turn reset/EOS/finalized text, native attribution and driver firsttoken/nonemptytext/done/reset timings preserve application policy. Seven targeted/all439Python checks pass; two sequential short workers/six turns/26raw events exit0 and independent text/endpoints pass. Admissions1,443,266,720/3,577,266,480 and sampled joint native charges1,331,464,136/3,465,186,736bytes fit6GiB. Probes excluded from latency acceptance; full128-output three-warmup/ten-sample traces, full matched task scoring, controlled/sustained evidence and unchanged Q4-blocked exit remain open.

M5.6 full frozen task responses,14 October2026: [completed data task](docs/experiments/2026-10-14-full-task-responses.md). All100 originalcases/caps for each small/larger Half and largerFP32 complete;18,027raw events, full inventories/pinned executables, state/memory and independent post-exit text replay. Allworkers naturallyexit0/cleanup/endpoints and fresh absence pass. Admissions1,443,266,720/3,577,266,480/6,343,713,560 and sampled model-phase joint native charges1,273,792,696/3,469,020,880/6,306,092,840bytes fit6GiB. Larger visible-ID sequences match99/100; no score/repair/quality acceptance follows. Literal grading, repeated application/controlled/sustained evidence and unchanged Q4-blocked exit remain open.

M5.2/M5.6 full matched Half task quality,14 October2026: [native scoring task](docs/experiments/2026-10-14-half-task-quality.md). Small Half2/100 matches frozenFP322/100; larger Half11/100 matches newFP3211/100. Both0pp drops pass unchanged2pp gate. Three sequential qualified literal windows resolve all300outcomes/120code programs, with no repair/unresolved infrastructure, successful container cleanup/stopped restoration/pipe lifetime and independent classifier/text/aggregate replay. Full Half loss and task quality now pass bothselectedconfigurations; absolute code passes remain0/40. No source/default/gate or performance adoption change. Controlled/application/sustained evidence and unchanged Q4-blocked M5 exit remain open.

M5.6 full small Half workload matrix,14 October2026: [completed native data](docs/experiments/2026-10-14-small-half-controlled-full.md). Eight workloads/24warmups/80measured requests replay187raw events with complete states/timings/memory, unchanged endpoint identities and clean owned exits. Median shortTTFT0.484318/request12.192120seconds; boundaryrequest109.611879seconds. Ten boundary samples total1096.033071seconds qualify sustained-duration scope with recorded AC/thermal endpoints; continuous sensors/energy unavailable. Maximum admission1,262,602,960/samplednativejointcharge1,166,575,656bytes fit6GiB. Larger controlled/application/confirming comparisons and Q4-blocked exit remain open. Initial post-run account-limit approval-review failure is retained; user retry restores access and fresh observations find no controlled worker/collector commands and a stopped/not-starting machine. Historical PID/group IDs were not retained. Astra accepts completed small-model evidence with no findings before commit.

M5.6 full larger Half workload matrix,14 October2026: [completed native data](docs/experiments/2026-10-14-larger-half-controlled-full.md). Eight workloads/24warmups/80samples replay187events with clean exits/endpoints. ShortTTFT/request medians2.037336/28.398590seconds; boundaryrequest264.684586seconds. Seven rows exceed ten measured minutes with AC/thermal endpoints; boundary total2646.948165seconds. Admission3,396,602,720/nativejointcharge3,301,821,896bytes fit6GiB. Recorded9PIDs absent/eightworker groups empty/machine stopped. Both full controlled matrices complete; application/confirmation/release decision and Q4-blocked exit remain open. Astra accepts completed larger-model evidence with no findings before commit.

M5.6 full Half application traces,14 October2026: [completed data](docs/experiments/2026-10-14-half-application-full.md). Both models3warmups/10three-turn scenarios,78total turns/60measured,3589/3576raw events, independent text/state/timing/memory/endpoints replay. Medians20.445239/43.550813seconds; measured totals204.468590/435.410454seconds do not add10min qualification. Admissions1,443,266,720/3,577,266,480 and nativejointcharges1,331,447,848/3,467,005,408bytes fit6GiB. RecordedmodelPIDs/groups absent/machine stopped. Small output128/14/128 differs fromFP32128/18/128; no identical-output backendgain claimed. Corrected nullable/timestamp audit and15invalidvariants pass. Astra accepts completed application evidence with no findings before commit. Confirmations/release/Q4-blocked exit remain open.

M5.6 independent Half primary confirmation,14 October2026: [completed repeats](docs/experiments/2026-10-14-half-primary-confirm.md). Both models secondshort128/128 threewarmups/tensamples complete26requests/30events with exact source/state/memory/timing/schema/endpoints and clean exits. SmallTTFT/request changes+0.003433%/+0.087202%; larger−0.168474%/+0.005098%. Full ranges retained; repeatability only, no new pairedbaseline/optimization adoption. Admissions unchanged/nativejointcharges1,176,504,144/3,311,848,736bytes fit6GiB. Astra clears artifact caller/fivefailurechecks and accepts final data with no findings before commit. Appconfirmation/finalrelease/Q4-blocked exit remain open.

M5.6 independent Half application confirmation,14 October2026: [completed repeats](docs/experiments/2026-10-14-half-application-confirm.md). Both second3warmup/10three-turn collections replay78turns/60measured/3589+3576events with exact initial-run IDs/text/stops/state signatures. Scenario medians20.456533/43.572448seconds change+0.055240%/+0.049678%; repeatability only. Admissions unchanged/nativejointcharges1,332,742,160/3,464,810,024bytes fit6GiB; recordedmodelPIDs/groupsabsent/machine stopped. Measured totals<600s, differingFP32output/end-power/timing limits retained. Astra accepts final evidence with no findings before commit; finalcomparison/release/Q4-blocked exit remain open.

M5 final evidence synthesis,14 October2026: [decision report](docs/experiments/2026-10-14-m5-performance-decision.md) and [saved-data checks](benchmarks/results/2026-10-14-m5-performance-decision-v1.json) consolidate both full eight-row Half matrices, second primary/application runs, passing unchanged loss/task gates, complete admissions and recorded sustained scopes. Compensated Half remains explicit opt-in; CPU scalar/GPU Original/FullLogits/chunk128 defaults stay. Historical FP32 CPU/app and external data are separate representation tracks; small boundary Half decode is slower than historical CPU and application outputs differ. No alternating matched-representation replacement or qualified external speed lead is claimed. Final report records baseline gaps and retain/reject decisions; M5.2/M5.6 and overall M5 remain open because both Q4 loss gates fail and replacement evidence is incomplete. Content/link/whitespace/saved arithmetic and protected binaries checked; Astra accepted after two wording corrections, with no remaining findings. No new numerical/source/default/gate change or M6/M7 work.

M5 new Q4 candidate storage task,14 October2026: [frozen v2 group8 contract](docs/q4-v2.md) and [storage qualification](docs/experiments/2026-10-14-q4-v2-format.md) define one separate candidate before further held-out evaluation. Distinct identities preserve all v1 bytes/policy/rejections and defaults. Eight Q4 codes plus Half scale/bias use8effectivebits/weight before padding. Python/source-bound packing and allocation-free Rust parser/static FP64 oracle pass manual cross-language fixtures,8/64boundary/tail/error/versionmix checks. Metal rejects v2 before GPU allocation until independently qualified execution exists. Predicted residentweights630,251,436/1,777,247,428bytes are layout-only.20Python, fullCPU/Metal-feature and final11targetedRust checks,fmt/Clippy pass; Astra accepted design/code/check/evidence with no findings. V2 independent GPU/state/memory, frozen quality and repeated performance tasks remain; no acceptance/gate/default changes. M5.2/.6 stay open.

M5 Q4 v2 GPU code task,14 October2026: [vector/prefill qualification](docs/experiments/2026-10-14-q4-v2-gpu.md). Separate staticgroup8/record8 compensated kernels, immutableformat dispatch and format-awareweights/staging admission preserve v1 and reject unsupportedv2modes before session allocation. Independent sixcase/210array reference freezes unchanged gateformula beforeGPU.4032exactprimitiveoutputs/16midpoint-overflow witnesses,9100v2 fulltiny stage comparisons, bothstate/admission/cancel/sinkfailure/20zeroRustallocations suites and4550unchangedv1 regression arrays pass. Astra's duplicateinventory/separatehash-read oracle findings are fixed with boundedconsumedbytes/exactinventory/endpoints and mutationtests; draftfixtures/failingoldhash harness attempt are excluded.93CPU/103Metal-featureordinary/39referencechecks,fmt/Clippy pass. Realcheckpoint geometries/native memory/fullfrozenquality and repeatedperformance remain; no gate/default/Q4quality or M5exit acceptance. Astra accepted finalcode/check/nativeevidence with no findings.

M5 Q4 v2 real artifact/reference prerequisite,14 October2026: [qualification](docs/experiments/2026-10-14-q4-v2-real-oracles.md). Both pinnedcheckpoints pack168/196matrices totaling357,837,312/1,310,208,256bytes, exactpredictions, with unchangedsource/helperendpoints and completepackingnativecharge≤17,253,048bytes. Independentgroup8decoder freezes6cases/342fullarrays permodel beforeGPU/heldout, adding fixed17token probes atoffset0/5 to original4cases; newcases disclose unavailableFP32source arrays/argmax. Sameprecision/equivalent-prefix gateformula. Logicaladmissions906,264,640/1,211,203,648 and completednativecharges433,210,280/724,534,280bytes fit6GiB; completed-memory-beforepublication. Astra stripe-overlap finding fixed by explicittemporaryrelease plusweakrefregression; exactinventory/version/source mutations covered.51Python/1portableRustprobeintegrity/fmt/Clippy checks pass. Astra accepted code/check/nativeevidence with no findings. Both GPUgeometries/realmemory, fullfrozenquality and repeatedperformance remain; no default/gate/M5exit change.

M5 Q4 v2 real GPU geometry task,14 October2026: [qualification](docs/experiments/2026-10-14-q4-v2-real-geometry.md). Both resident models pass14,892full-element stage-slice comparisons across sixfrozencases, vector/tile modes, chunks1/3/8/17 andoffset5, populatedKVprefixes/tails/all-rowgreedy. Weights630,251,436/1,777,247,428bytes matchpredictions. Capacity27admissions874,062,494/2,029,218,550 andnative lifetimecharges758,334,616/1,910,425,496bytes fit6GiB. Frozen gatesunchanged; ordinaryMetal-feature/fmt/Clippy/releasecompile pass. Astra accepted finalcode/evidence/all1,812bindings with nofindings. Full-context/quality/repeatedperformance remain; no default/gate/M5exit change.

M5 Q4 v2 scorer code task,14 October2026: [integration/shortguards](docs/experiments/2026-10-14-q4-v2-scorer.md). Explicitfrozenv2tile+Compensatednormalization reportsactualformat/quantizer/widths/dtype/residency, preserveslegacyHalf/v1calls. Bothmodelstwooriginalbatch/singleNLLguards pass; fourcleanworkers/34events each, unchangedendpoints/admission/jointnativecharge≤6GiB. Astra's shortprobeprovenancefinding fixed with exactboundedmanifestbytes, both-model orderedtupleprelaunchchecks andzero-launchmutationregressions.22Python/104ordinaryMetal-feature/fmt/Clippy/releasebuildchecks pass. Astraacceptedfinalcode/nativeevidence/all1,872bindings withnofindings; fullfrozenloss/tasks/repeatedperformance andM5exitremainopen.

M5 Q4 v2 full-loss data task,14 October2026: [frozen decision](docs/experiments/2026-10-14-q4-v2-full-loss.md). Both10,000target/10block/88chunk collections complete after unchangedshortguards,20,268raw events/10cleanownedworkers/endpoints/memory≤6GiB. Smallmean2.9350208557206288vsFP322.883725149929484 increases0.05129570579114473>.05: REJECTED. Largermean2.51426472772287vsFP322.485958031019153 increases0.02830669670371666≤.05: lossgatepasses only. Driverexit2smallreflectsqualityrejection; numericalcollectioncompleted. No grouping/metadata/exclusion/corpus/checkpoint/gate tuning. Astra confirmedplanallowsqualifiedsmallHalf fallback plusseparatelyqualifiedlargerQ4; bothQ4geometriesneednotpassquality. Larger100-task/CLI/fullcontext/application/controlled/sustained/repeatednamedgain remain beforeM5exit. Astraacceptedcompleteddata/all1,864bindings/20,268eventreplay withnofindings.

M5 Q4 v2 CLI code task,14 October2026: [functional qualification](docs/experiments/2026-10-14-q4-v2-cli.md). Explicitpaired--q4-weights/--metal-projectionv2 rejectsCPU/missing/conflicting/duplicateoptions andvalidv1beforeevent; actualdtype/version/group/quantizer/modes preservelegacyNone-omittedmetadata/defaults andexistingtokenizer/applicationadmission. Eightsequentialnativeworkers/109events pass tinyHalfcompat/v2cachedpendingresetinvalidstopcancelidleexit andbothrealgreedy/replay/Unicodechat/finalizedtext.95CPU/107Metal/13portable/fmt/Clippy/releasechecks pass. FailedpreflightomittingJSONretained/excluded. Omittedtransitivehelperinventorycoveredbypriorcommittedacceptedreceipts/snapshots plussupplementalendpoints, disclosedexplicitly. Astraacceptedfinalcode/all1,224bindings/109eventreplay withnofindings. Larger100-task/fullrelease/performance gates andM5exitremainopen;smallv2qualityrejected.

M5 larger Q4 v2 task collector code task,14 October2026: [qualification](docs/experiments/2026-10-14-q4-v2-task-collector.md). Fixedlargerloss-qualifiedcandidate/CLI/source/artifact/helper/taskpins, unchanged100cases/caps/stop/reset/state andpost-exitdecoding. Optionaltransporthooks preserveHalfdefaults. Astra prelaunchracefinding fixed bybinary/mappedcompiledRust/Cargopinsreassertion+zero-launchmutations.14portable/fmt/Clippy checks pass; threecategorycap8probe/24events/cleanownedexit/post-exittext/admit2,267,080,568/jointnative2,096,597,584bytes/endpoints pass. Astraacceptedfinalcode/all337bindings/24eventreplay withnofindings; full100taskcollection/literalquality andcompleteperformance stillrequired beforeM5exit.

## M6 Qualify longer context and attention

Complete,19 October2026, in the [M6 release decision](docs/experiments/2026-10-19-m6-release.md). Separate Online prefill/decode attention passes independent primitive/tiny/session checks, unchanged full10,000-target FP32 loss and100-task gates. All32 context roles at4K/8K/16K/32K replay22,752 events/429bindings; paired long numerical quality, profiles, full-capacity recomputation/overflow preservation and both actual32K applications pass within6GiB. The two exact original larger4K winner changes have separate manual review; original automatic flags stay preserved. Short512/128 complete numerical request median reductions repeat at76.26/76.24% small and73.02/73.03% larger, with small prefill/TTFT overlap disclosed; long timings remain descriptive. The conditional I8 experiment has positive small component gains and larger gains below5%; final review retains FP16 KV after complete context/application admission. No I8 adoption gates are waived. Qualified configurations, cyclic-stress limits, unchanged defaults and all evidence are recorded in the release report. M7 remains pending.

- [x] **M6.1 Implement tiled online-softmax attention.** Build separate decode and prefill paths without materializing the full attention-score matrix. Preserve causal boundaries, absolute positions, and GQA sharing. Compare irregular chunk sizes and substantial cached depths with the reference.
- [x] **M6.2 Extend context admission and profiling.** Measure 4K, 8K, 16K, and 32K total-token workloads only where the pinned model permits them. Reserve output capacity: at a validated limit of 32,768, a 128-output boundary run has 32,640 prompt tokens. Account for allocated KV, workspace, and model limits before dispatch.
- [x] **M6.3 Evaluate optional KV quantization.** Only if measured KV traffic or capacity warrants it, try an eight-bit cache with explicit K/V grouping and dequantization inside attention tiles. Repeat loss, long-context behavior, and latency checks. Retain FP16 KV if the tradeoff is unfavorable.

For the first Qwen config, calculated KV payload is `2 × 24 × tokens × 2 × 64 × bytes_per_element`: 48 MiB at 2,048 tokens in FP32, or 24 MiB in FP16. This is logical KV only; use measured footprint for admission validation. Selected depths must fit the budget and pass long-context quality checks before becoming supported configurations.

## M7 Integrate and qualify a physical iPhone

In progress,19 October2026. The first [CPU raw-ID C boundary component](docs/experiments/2026-10-19-m7-native-abi.md) adds explicit ownership/thread/callback/cancellation/error contracts and checked static library integration. The [Apple packaging component](docs/experiments/2026-10-19-m7-apple-package.md) adds three checked arm64 slices and MacCPU/MacMetal/simulatorCPU integration. The [Swift harness component](docs/experiments/2026-10-19-m7-swift-harness.md) adds checked iOS builds and simulator integration. Physical-device qualification remains before M7 exit.

- [x] **M7.1 Add the native boundary.** Build a thin C ABI or other explicitly chosen Swift bridge with opaque model/session handles, ownership rules, structured errors, cancellation, and token events. Contain Rust panics at the boundary. Package libraries for supported Apple targets and set deployment requirements from qualified APIs.
- [x] **M7.2 Add a minimal Swift harness.** Exercise load, prompt, token streaming, continuation, reset, and cancellation. Run inference away from the UI thread and define callback lifetime and serialization. Reuse the same Rust engine and local artifact format.
- [ ] **M7.3 Qualify the physical device.** Measure an independent iPhone 14 memory budget, thermal behavior, sustained throughput, and energy where instrumentation is available. Test inactive/background transitions, memory warnings, interruption during generation, resume/replay, and orderly resource release. Add iPad and newer Macs as separate device qualifications.

Physical integration update,20 October2026: the signed Release harness passes
actual CPU and Metal tiny checks on the wired iPhone14Pro (`iPhone15,2`,
iOS27.0.1). Exact token/state/replay/cancel/unload/close/owner-release assertions,
seven admitted installed-file proofs and independent process absence pass.
The [report](docs/experiments/2026-10-20-m7-physical-tiny.md) retains failures,
source/evidence pins and52 passing isolated helper checks. Development signing
and profile trust are resolved. Real-model numerical/state qualification,
whole-process memory limits, actual OS lifecycle transitions, sustained
throughput/thermal observations and base iPhone14 evidence remain pending;
M7.3 and overall M7 stay open. The
[measurement runner](docs/experiments/2026-10-20-m7-measurement-runner.md) now
passes checked memory/timing/artifact/failure mechanics on Mac CPU/Metal and
both iOS SDK builds. The [physical measurement controller](docs/experiments/2026-10-21-m7-physical-measurement.md)
now passes26 raw tiny CPU/Metal phone samples after filesystem-alias and startup
activation fixes; these qualify measurement mechanics only. Full real-model phone
qualification, supported limits, sustained/thermal and actual lifecycle evidence
remain pending. The [storage admission](docs/experiments/2026-10-21-m7-storage-admission.md)
now gates actual small/larger payload preparation and preserves a failed stale
observation. Thirteen small-model short requests pass exact raw IDs/state on
iPhone14Pro. The [bounded hashing fix](docs/experiments/2026-10-21-m7-hash-memory.md)
removes the file-sized proof retention:84 portable checks,17 Mac cases, both SDK
builds and two matching small phone trials pass. Their lifetime peaks are about
1.017GB versus2.098GB previously. The [larger short oracle](docs/experiments/2026-10-21-m7-larger-oracle.md)
now passes two actual phone runs (26 requests) with frozen IDs/state and lifetime
peaks about1.825GB. Full quality/state/context/lifecycle/sustained qualification
and base iPhone14 evidence remain open; M7.3 is pending.

The [committed-distribution score API](docs/experiments/2026-10-21-m7-score-api.md)
now passes independent CPU/selected-model Metal numerical checks, strict Cargo
checks, all three Apple build/link targets and actual Mac C CPU/Metal clients.
It enables later phone teacher forcing without full-logit copies. Full physical
quality and the remaining M7.3 gates are unchanged and still pending.

The [Swift score adapter](docs/experiments/2026-10-21-m7-worker-score.md) passes
Mac CPU/Metal committed-prefix, failure and deterministic lifecycle-publication
checks plus both SDK builds. The full phone quality collector remains pending.

The [teacher-forcing coordinator](docs/experiments/2026-10-21-m7-teacher-forcing.md)
passes frozen block-policy/failure simulation, independent tiny CPU and actual
Metal sequencing, plus strict three-target compilation. Full physical corpus/
task collection and app integration remain pending.

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

Implement one mechanism per reviewable change, including its memory layout, a small numerical trace, and targeted checks. Do not mark a milestone complete until its exit evidence exists. M4 and M5 are complete. The [15 October M5 release decision](docs/experiments/2026-10-15-m5-release.md) accepts the small compensated Half path and larger frozen Q4 v2 path, with unchanged full loss/task gates, confirmed useful prefill gain, complete repeated workload/application and sustained evidence within6GiB. Q4 v1 and small v2 remain rejected; CPU/GPU/readback/chunk defaults and baseline limitations remain. M6 is complete in its [19 October qualified scope](docs/experiments/2026-10-19-m6-release.md): explicit Online attention, selected4K–32K cases and retained FP16 KV. M7 remains pending.

M5.6 frozen larger v2 full responses,14 October2026: [collection report](docs/experiments/2026-10-14-q4-v2-full-tasks.md). All100original cases/caps complete with clean natural exit, exact raw/state/post-exit-text replay and unchanged compiled/model/helper endpoints. Complete admission/native memory fits6GiB. Literal grading, task-quality/performance acceptance and overall M5 remain open.

M5.2/M5.6 larger v2 full task quality,14 October2026: [quality report](docs/experiments/2026-10-14-q4-v2-task-quality.md). Exact frozen100responses grade11/100 versus matchedFP3211/100,0pp loss≤2pp;0/40code,10/30prose,1/30continuation with zero unresolved infrastructure.43portable admission/window tests pass. Qualified literal window starts only after model exit/fresh absence, preserves raw source tests/text rules, restores stopped VM and finishes helpers. Larger v2 now passes both unchanged loss/task quality gates. Small v2 and both v1 loss rejections remain; performance/application/sustained/confirmation and useful named Q4 gain remain pending, so M5 stays open.

M5.6 explicit v2 benchmark integration,14 October2026: [qualification report](docs/experiments/2026-10-14-q4-v2-bench.md). Truthful artifact/quantizer/mode/residency and cold hooks preserve legacy worker/timed paths.23portable/95CPU/107Metal/fmt/strictClippy/release checks pass. Eight frozen check-only workloads/31events qualify full2048commit/beyond-capacity preservation, cached/pending/changed state and≤6GiB accounting.13alternating512/1 pairs/15events retain identical all151936logit bits and greedy/state. Source/executable/provenance frozen; full clean-tree matrix/application/sustained/confirmation/affected-shape evidence remain before namedgain/release/M5 acceptance.

M5.6 explicit v2 application adapter,14 October2026: [task report](docs/experiments/2026-10-14-q4-v2-application.md). Frozen larger loss/task-quality/CLI/artifact/transcript and actual consumed parser helpers are admitted; cold hooks preserve legacy transport/timing/state/finalization.14portable/fmt/MetalstrictClippy checks pass. Original36/76/117prompts complete3cap2turns/13events with exactpost-exit text, clean natural exit and≤6GiB memory/endpoints. This qualifies integration only; full3warmup/10application/confirmation and performance release remain pending.

M5.6 v2 affected-shape qualification,14 October2026: [task report](docs/experiments/2026-10-14-q4-v2-shapes.md). Fixed512/1primary plus128reset andactual64append after512preparedoutside timers preserve39pairs/45events/all151936vocabularybits/greedy/state.13portable/oneRustboundedreader/fmt/strictMetalClippy/release checks pass. Staticn<8dispatch shares identicalvector kernel/grid/resources; otheroperations unchanged. Full128chunks cover longerprompt projection geometry;64coversappend. Allnativeexit/endpoints/memorypass. Preparatory512timings excluded for briefportabletestoverlap; twofreshprimaryrunsmandatory. Inheritedscope-label correction retainsold/newhelper snapshots andoriginalrecords, no numerical/timer change. Fullmatrix/application/sustained/confirmedgain/release/M5 remain open.

M5.3/M5.6 larger v2 fresh primary confirmation,14 October2026: [report](docs/experiments/2026-10-14-q4-v2-primary.md). Two samef24binary512/1 runs pass26pairs/30events/exactfullbits/greedy/state/endpoints/≤6GiB cleanexit. Gains23.691277%/22.202400% exceed5%, dispersion and betweenrunmedianvariation; preparatoryoverlapped512timings excluded. Namedtileprefill gain accepted; fullmatrix/shortconfirmation/application/sustained/release evidence remains before overallM5exit.

M5.6 larger v2 fullmatrix/shortconfirmation,15 October2026: [report](docs/experiments/2026-10-15-q4-v2-controlled.md). All8rows plusseparateshort pass117requests/202events,3warm10samplesperrow, exactcache/state/boundary2048/overflow/endpoints/cleanownedexit and≤6GiB. Sustainedcompletedrequestsum10,742.769647s includingwarmups,8,256.699921s measuredonly; setup/hash/loadexcluded. ShortTTFT/requestmedianchange−3.349513%/−1.463285%,repeatabilityonly. Application/finalrelease remain beforeoverallM5exit.

M5.6 larger v2 fullapplication/confirmation,15 October2026: [report](docs/experiments/2026-10-15-q4-v2-application-full.md). Both3warm10three-turncap128collections complete78turns/6086events, exactID/text/stop/state signatures, naturalownedexit/postexitdecode/initial-endpointpins/≤6GiB. Scenariomedians41.267054271/41.301725271s change+0.084016%, repeatabilityonly. Fullreceiptclockvectors unavailable; retainedtokenclocks/separatetextbounds verified. Finalrelease/retain synthesis remains beforeM5exit.

M5 finalclosure,15 October2026: [release decision](docs/experiments/2026-10-15-m5-release.md) and [qualification](benchmarks/results/2026-10-15-m5-release-qualification-v1.json). M5.1–M5.6 complete after accepted larger v2 fullquality,22.202400%/23.691277% repeatedsameartifactprefill gains, all8controlledrows+shortconfirmation, fullapplication+confirmation, exactraw/state/compiledsource/endpoints, complete≤6GiB memory and10,742.769647s completed sustainedsequence. SmallHalf staysqualified; smallv2/v1lossrejections, removednegativeexperiments, CPU/GPUOriginal/FullLogits/chunk128defaults and external/baselinegaps preserved. Noexternallead/defaultreplacement orM6/M7completion claimed.


M6 final closure,19 October2026: [release decision](docs/experiments/2026-10-19-m6-release.md) and [qualification](benchmarks/results/2026-10-19-m6-release-qualification-v1.json). M6.1–M6.3 complete after independent component/state checks, full unchanged loss/task gates,32-role context replay, all4depth profiles/descriptive timing/confirmations, full-capacity boundaries, both maximum applications and separate final FP16 retention. Mixed I8 component evidence is preserved without production adoption or waived gates.105Python/97CPU/110Metal-feature tests, formatting and strictClippy pass; current runtime/payload identities and protected CPU binaries are rechecked. Defaults, baseline limits and failed attempts remain; M7 is pending.

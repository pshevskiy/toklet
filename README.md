# Toklet
Toklet — Fast, lightweight LLM inference on Apple Silicon. An open-source Rust inference engine focused on low latency, efficient memory usage, and simple, understandable code.

See the [implementation plan](PLAN.md) for milestones, dependencies, and acceptance criteria, and the [Apple LLM performance proposal](apple_llm_performance_plan.md) for the design rationale.

The library provides a complete scalar decoder with FP32 storage, immutable weights, reusable scratch, tied output projection, and deterministic greedy selection. A separate diagnostic path retains all intermediate arrays. Projections, normalization, and softmax use FP64 accumulation for stability; the original FP32 projection functions remain available. The tiny decoder matches its frozen independent reference, and Qwen matches pinned Transformers on short numerical probes. Local safetensors loading supports FP32, FP16, BF16, and indexed shards. Bounded [CPU sessions](src/session.rs) retain per-layer KV, track pending versus evaluated tokens, and support exact append, cached prompts, reset, and replay. The CLI streams greedy generation with local tokenizer assets, the qualified Qwen chat template, stop IDs, cancellation and statistics.

Use Rust 1.90 or newer. The library and CLI build without model downloads or Apple frameworks. Safetensors, Serde, and half-precision conversion dependencies are pinned. Fetch the locked dependencies once before running the offline checks:

```sh
cargo fetch --locked
cargo build --locked --offline
cargo run --locked --offline -- --help
cargo run --locked --offline -- --version
```

Run the checks locally:

```sh
cargo fmt --check
cargo clippy --locked --offline --all-targets -- -D warnings
cargo test --locked --offline
```

The GitHub Actions workflow runs the same checks and a release build on Linux with Rust 1.90.0. Store downloaded models and conversion output in `models/` or `artifacts/`, and large raw benchmark output in `benchmarks/raw/`; these directories are ignored. Small fixtures, manifests, scripts, and compact reports remain versioned.

The [tiny reference](tools/reference/README.md) contains frozen FP32 arrays and an independent NumPy oracle. Rust tests read the saved arrays without Python; oracle regeneration and verification are separate commands.

The pretrained loader suite is explicit, requires the pinned local model, and runs separately from ordinary tests:

```sh
cargo test --locked --offline --release --test weights pinned_qwen_loads_within_the_initial_budget -- --ignored --nocapture
```

The [Qwen forward report](benchmarks/results/2026-10-09-qwen-forward.md) records short numerical parity and reference regeneration. With its full local reference prepared, run:

```sh
cargo test --locked --offline --release --test qwen_forward pinned_qwen_fp32_matches_full_transformers_forward -- --ignored --nocapture
```

The [session report](benchmarks/results/2026-10-09-qwen-session.md) records exact cache/full-prefix agreement, memory admission, and the checked capacity boundary. Its real-model suite uses the pinned local checkpoint; on macOS it also inspects its own process footprint:

```sh
cargo test --locked --offline --release --test qwen_session pinned_qwen_cached_sessions_match_full_prefix -- --ignored --nocapture
```

The [benchmark contract](benchmarks/README.md) fixes workloads and quality gates. The [first MLX baseline](benchmarks/results/2026-10-09-mlx-bf16-512-128.md) retains raw completed-work timings and memory measurements on the M1 Pro; it does not claim Toklet performance or quality qualification.

Generate from a local checkpoint after a release build:

```sh
cargo build --locked --offline --release
target/release/toklet --model models/Qwen2.5-0.5B-Instruct --prompt 'Answer briefly: what is Rust?' --max-new-tokens 32 --stats
target/release/toklet --model models/Qwen2.5-0.5B-Instruct --tokens 9707,11,358,2776 --max-new-tokens 4 --json
```

`--raw` bypasses chat formatting; `--messages` and `--tools` accept local JSON arrays. `--token-file` accepts little-endian u32 IDs. Prompt plus requested output must fit `--context-capacity` (at most 2,048); admission occurs before changing session state. The initial backend is `cpu`. Model EOS IDs are hidden from output but remain pending in history; `--no-eos` disables them and repeated `--stop-token` adds IDs. Sampling has no penalties or RNG.

`--cpu-projection tiled` selects the opt-in ordered FP64 projection tile before
session creation; `scalar` remains default. JSON readiness and statistics name
the selected mode. Full workload and application qualification precede default
adoption; see the [CLI qualification](docs/experiments/2026-10-10-cpu-tiled-cli.md).

`--interactive` loads once and accepts JSON lines, publishing a `ready` event before requests:

```json
{"prompt":"Answer briefly: what is Rust?","max_new_tokens":8}
{"action":"continue","max_new_tokens":8}
{"action":"append","prompt":"\n<|im_end|>\n<|im_start|>user\nGive an example.<|im_end|>\n<|im_start|>assistant\n","max_new_tokens":8}
{"action":"reset"}
```

Use `generate` (the default) with exactly one `prompt`, `messages`, or `tokens` field. Exact history extensions reuse evaluated KV; other changes reset and replay. `append` accepts raw text or IDs, so the caller supplies any turn delimiters. `continue` evaluates the pending suffix and resumes greedy output; `reset` accepts no other fields. Token events carry IDs and decoded text; `done` carries timings, state counts and admitted memory. Unfinished Unicode remains buffered across output limits or cancellation and subsequent `continue`; EOS, a new response, reset or process exit flushes it, sometimes through a separate `text` event. Parsed validation or admission errors produce a recoverable `error` event and preserve the valid prefix. An input line exceeding the 1 MiB transport limit terminates the process with an error on stderr.

Ctrl-C is checked between synchronous chunks/steps. An active interactive request reports `cancelled` and can be continued; idle Ctrl-C exits with status 130. `--prefill-chunk` controls the cancellation boundary. Numerical decode allocates no new buffers, while tokenization, JSON and output decoding use bounded application storage. Timers separate loading, input preparation, prefill, decode and sampling; total request time also includes output. These are functional statistics, pending controlled performance qualification.

The [CLI qualification](benchmarks/results/2026-10-10-qwen-cli.md) records short real-model checks. Ordinary tokenizer tests use a frozen small subset without downloading models. Explicit checks use the complete pinned assets:

```sh
cargo test --locked --offline --release --bin toklet full_pinned_tokenizer_matches_frozen_ids -- --ignored --nocapture
python3 tools/reference/cli.py --output artifacts/qwen-cli-check
```

The [profiling adapter](tools/bench/cpu.py) runs the frozen teacher-forced CPU
workloads and records completed token timings without sampling or text output.
See [benchmark instructions](benchmarks/README.md) for cache scenarios, raw
checkpoints and the workload matrix. Load statistics include nested
`conversion_seconds`; it is part of `load_seconds`, not an additional phase to
sum. Profiling harness checks pass; complete performance qualification is still
pending.

Standalone Apple preparation includes a
[pinned Rust Metal binding check](docs/experiments/2026-10-10-rust-metal-bindings.md)
with exact small dispatches, owned shared buffers, completion/error checks and
bounded isolated builds. It adds no engine dependency or GPU execution path;
complete FP16 Metal qualification remains pending.
The [matched short CPU comparison](docs/experiments/2026-10-10-cpu-primary-comparison.md)
confirms roughly 77.6% lower median TTFT for opt-in tiled projections in both
comparison orders. Scalar remains default while full workload and application
qualification continue.

The [qualified application adapter](docs/experiments/2026-10-10-cpu-tiled-application-adapter.md)
can compare scalar and tiled using one isolated CLI whose source, binary and
functional qualification are verified. Three short transcript probes preserve
token/text/session semantics and memory; full application measurements remain
pending.

The standalone [FP16 RMS primitive](docs/experiments/2026-10-10-metal-rms.md)
passes frozen tiny-model source gates and exact manual range/subnormal cases;
full decoder composition and Qwen qualification remain pending.
The [scalar capacity workload](benchmarks/results/2026-10-10-cpu-boundary.md)
now completes all 2,048 Qwen positions and rejects another token within the
memory budget. Remaining CPU matrix and application measurements continue.
The [scalar medium workload](benchmarks/results/2026-10-10-cpu-medium.md)
retains all ten 512-input/128-output measurements and valid prefix states,
with median TTFT 163.381 seconds and decode rate 1.414 steps/s.
The [scalar long workload](docs/experiments/2026-10-10-cpu-long.md)
retains all ten 1,024-input/128-output measurements, with median TTFT
336.470 seconds and decode rate 1.377 steps/s.
The [scalar exact-prefix append workload](docs/experiments/2026-10-10-cpu-append.md)
retains all ten requests with 512 reused tokens, 64 appended tokens and 128
fixed output steps; median TTFT is 21.348 seconds and request time 111.629
seconds, with completed Astra review. Remaining cache/application measurements
and full quality qualification remain pending.
The [fully cached scalar workload](docs/experiments/2026-10-10-cpu-cached.md)
retains all ten requests with 512 reused tokens and zero timed prefill tokens.
Median request time is 90.000 seconds and decode rate is 1.411 steps/s. Its
0.792-microsecond numerical TTFT reuses a prepared distribution and excludes
cache preparation and visible-text latency.
The [pending-token scalar workload](docs/experiments/2026-10-10-cpu-pending.md)
retains all ten requests with 512 reused positions and one timed pending input.
Median numerical TTFT is 0.703 seconds, request time is 90.501 seconds and
decode rate is 1.414 steps/s. Changed-prefix, application and full quality
checks stay open.
The [selected Qwen RMS check](docs/experiments/2026-10-10-qwen-metal-rms.md)
also passes width-896 source gates with norm-only checkpoint reads. Full-model
and session qualification remain pending.

The [Half embedding lookup](docs/experiments/2026-10-10-metal-embedding.md)
preserves every frozen tiny output bit, including signed zero and subnormals.
Wider/full-table and decoder execution remain pending.

The [Half residual check](docs/experiments/2026-10-10-metal-residual.md) also
passes exact tiny-model and rounding-edge outputs, with explicit invalid-state
rejection on Half overflow. Composed session recovery remains pending.
The [selected Qwen residual check](docs/experiments/2026-10-10-qwen-metal-residual.md)
passes 20 width-896 stages with exact Half outputs and unchanged source gates,
using saved operands without checkpoint payload reads.

The [stable Half SiLU check](docs/experiments/2026-10-10-metal-silu.md)
passes all ten frozen tiny stages and exact signed-zero, subnormal and saturation
edges using FP32 arithmetic and one Half store. The
[selected Qwen SiLU check](docs/experiments/2026-10-10-qwen-metal-silu.md) also
passes all twelve saved width-4,864 stages with exact Half outputs and unchanged
source gates. Complete MLP and decoder execution remain pending.

The [stored-Half gating check](docs/experiments/2026-10-10-metal-gating.md)
passes ten frozen tiny stages and exact product-rounding edges. Distinct output
ownership preserves both inputs, and completed Half overflow is discarded.
The [selected Qwen gating check](docs/experiments/2026-10-10-qwen-metal-gating.md)
passes all twelve saved width-4,864 stages with exact Half products and
unchanged source gates, with completed Astra review. MLP/decoder composition
remains pending.

The [held-out loss runner](docs/experiments/2026-10-10-held-out-loss-runner.md)
preserves the frozen block/label policy and passes independent short probes in
scalar and tiled modes with exact complete-logit agreement. The full 10,000
targets and separate 100-task quality evaluation remain pending.

The [frozen task collector](docs/experiments/2026-10-10-task-response-collector.md)
preserves prepared IDs, finalized text and pending EOS semantics, and supplies
the exact text scorers and literal HumanEval program composition. Four authored
scalar/tiled requests pass with matching outputs. Isolated code execution and
all 100 held-out model responses remain pending.
The [authored native isolation candidate](docs/experiments/2026-10-10-code-isolation.md)
passes offline checks but is rejected before sandbox admission because Python
cannot accept the fixed data limit. It is not used for generated code; a
compatible executor still needs qualification.

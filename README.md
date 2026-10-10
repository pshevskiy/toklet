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
The standalone [FP16 RMS primitive](docs/experiments/2026-10-10-metal-rms.md)
passes frozen tiny-model source gates and exact manual range/subnormal cases;
full decoder composition and Qwen qualification remain pending.

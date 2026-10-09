# Toklet
Toklet — Fast, lightweight LLM inference on Apple Silicon. An open-source Rust inference engine focused on low latency, efficient memory usage, and simple, understandable code.

See the [implementation plan](PLAN.md) for milestones, dependencies, and acceptance criteria, and the [Apple LLM performance proposal](apple_llm_performance_plan.md) for the design rationale.

The library provides a complete scalar full-prefix FP32 decoder with immutable weights, reusable scratch, tied output projection, and deterministic greedy selection. A separate diagnostic path retains all intermediate arrays. Normalization and softmax use FP64 reductions for stability. The tiny decoder matches the frozen independent reference. Local Qwen2 safetensors loading supports FP32, FP16, BF16, and indexed shards; real-model forward parity and cached generation are next. The CLI shell currently provides help and version information.

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

The [benchmark contract](benchmarks/README.md) fixes workloads and quality gates. The [first MLX baseline](benchmarks/results/2026-10-09-mlx-bf16-512-128.md) retains raw completed-work timings and memory measurements on the M1 Pro; it does not claim Toklet performance or quality qualification.

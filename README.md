# Toklet
Toklet — Fast, lightweight LLM inference on Apple Silicon. An open-source Rust inference engine focused on low latency, efficient memory usage, and simple, understandable code.

See the [implementation plan](PLAN.md) for milestones, dependencies, and acceptance criteria, and the [Apple LLM performance proposal](apple_llm_performance_plan.md) for the design rationale.

The library currently provides scalar linear projections, embeddings, residual addition, RMSNorm, masked softmax, SiLU, and gated MLP over checked slices. Buffers are FP32; normalization and softmax use FP64 reductions for stability. Model loading and generation are planned; the CLI shell currently provides help and version information.

Use Rust 1.90 or newer. The library and CLI have no runtime dependencies and build without model downloads or Apple frameworks. Fetch the locked test dependencies once before running the offline checks:

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

The [benchmark contract](benchmarks/README.md) fixes workloads and quality gates. The [first MLX baseline](benchmarks/results/2026-10-09-mlx-bf16-512-128.md) retains raw completed-work timings and memory measurements on the M1 Pro; it does not claim Toklet performance or quality qualification.

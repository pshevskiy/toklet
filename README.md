# Toklet
Toklet — Fast, lightweight LLM inference on Apple Silicon. An open-source Rust inference engine focused on low latency, efficient memory usage, and simple, understandable code.

See the [implementation plan](PLAN.md) for milestones, dependencies, and acceptance criteria, and the [Apple LLM performance proposal](apple_llm_performance_plan.md) for the design rationale.

The library currently provides scalar FP32 linear projections, embedding lookup, and in-place residual addition over checked slices. Model loading and generation are planned; the CLI shell currently provides help and version information.

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

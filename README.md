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
decode rate is 1.414 steps/s.
The [fresh changed-prefix scalar workload](docs/experiments/2026-10-11-cpu-changed.md)
retains three warmups and ten uninterrupted requests, excluding the earlier
paused batch. All 512 prompt tokens replay; median numerical TTFT is 163.552
seconds and request time 253.054 seconds. Application and full quality checks
stay open.
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
target [scalar baseline](docs/experiments/2026-10-11-held-out-scalar-loss.md)
now completes with mean loss 2.883725 nats/token and verified natural exit.
The [full tiled comparison](docs/experiments/2026-10-11-held-out-tiled-loss.md)
now matches all 10,000 complete-logit hashes and losses exactly, with zero mean
NLL increase. The complete task-quality comparison below now passes as well;
remaining candidate performance/application workloads stay pending.

The [frozen task collector](docs/experiments/2026-10-10-task-response-collector.md)
preserves prepared IDs, finalized text and pending EOS semantics, and supplies
the exact text scorers and literal HumanEval program composition. Four authored
scalar/tiled requests pass with matching outputs. The later scalar grading
report records actual held-out execution; the complete quality comparison
below now includes tiled grading.
The [full-task collector preparation](docs/experiments/2026-10-11-held-out-task-collector.md)
now handles 100 independent requests, including empty immediate EOS, with
phase watchdogs and owned descendant cleanup. Eleven targeted checks and all
172 offline checks pass. The [full scalar collection](docs/experiments/2026-10-11-held-out-task-scalar.md)
now completes all 100 frozen responses with natural exit and full journal
replay. Its conservative joint physical peak is 2,154,760,904 bytes, below
6 GiB. The scalar grading report below records its score; the later quality
comparison now includes both modes. Candidate performance remains pending.
The [authored native isolation candidate](docs/experiments/2026-10-10-code-isolation.md)
passes offline checks but is rejected before sandbox admission because Python
cannot accept the fixed data limit. It is not used for generated code. The
Linux qualifications below establish the compatible executor used for scalar
grading.

The [authored Linux isolation retry](docs/experiments/2026-10-11-linux-code-isolation-retry.md)
passes all fourteen native modes, 26 targeted checks and the full offline suite.
It retains the fixed limits, verifies actual zero capabilities and reconciles
completed container exit statuses. The VM's original stopped state is verified.
The later literal controller and scalar grading reports below extend this
qualification; the full candidate quality comparison remains pending.

The [literal Linux controller](docs/experiments/2026-10-11-linux-literal-controller.md)
passes all 28 authored native expectations and 161 offline tests, with literal
input, exact configuration, bounded transport and verified VM restoration.
Unresolved statuses remain infrastructure failures. The scalar grading report
below now records actual held-out execution and scores.
The [held-out grading integration](docs/experiments/2026-10-11-held-out-task-grading.md)
now validates complete collections before starting the owned runtime, composes
literal code programs and suppresses aggregates for unresolved infrastructure.
Twenty-one targeted checks, 193 offline Python tests and required Cargo checks
pass. The scalar grading report below records the subsequent actual run.

The [full scalar held-out task grading](docs/experiments/2026-10-11-held-out-task-scalar-grading.md)
resolves all 100 literal outcomes: 2/100 overall (0/40 code, 1/30 prose, 1/30
continuation), with no infrastructure ambiguity. All forty code outcomes are
SyntaxErrors under unchanged literal composition. The owned Linux window removed
every container and restored the stopped VM; independent replay confirms the
score. The full quality report below now includes tiled grading and comparison.

The [full tiled task collection](docs/experiments/2026-10-11-held-out-task-tiled.md)
now completes all 100 frozen responses with natural exit and exact journal
replay. Its conservative joint physical peak is 2,150,484,680 bytes below 6 GiB.
The separate quality report below records its literal grades and comparison;
candidate performance/application acceptance remains open and scalar stays default.

The [full task-quality comparison](docs/experiments/2026-10-11-held-out-task-quality.md)
passes the unchanged two-percentage-point drop gate: both modes score 2/100,
with no unresolved outcomes. All 100 visible texts, token sequences and stop
reasons match exactly. Combined with the identical 10,000-target loss result,
the frozen quality gates pass; remaining candidate performance/application
measurements and retention review still keep scalar as the default.

The [consolidated scalar/external comparison](docs/experiments/2026-10-11-m2.5-scalar-comparison.md)
aligns the complete reference workload matrix, full application/startup and
sustained measurements, scoped external timings and frozen quality results.
Astra accepted M2.5 closure without findings. Tiled retention remains a
separate M3.1 gate.

The [tiled medium workload](docs/experiments/2026-10-11-cpu-tiled-medium.md)
adds a complete three-warmup/ten-sample candidate row: median TTFT 40.961032 s,
request 97.892741 s and maximum child physical charge 2,043,693,048 bytes.
The first attempt failed publication and is preserved separately as excluded
evidence; the replacement uses pinned benchmark dependencies. Six non-short
rows, full tiled application evidence and the retention decision remain pending.

The [public CPU/ANE feasibility conclusion](docs/experiments/2026-10-11-ane-feasibility-closure.md)
closes M3's bounded runtime research with supported CPU slices and inconclusive
ANE acceleration. The rejected FP16 projection and unobserved actual ANE
placement remain explicit; no Core ML/ANE engine backend is adopted. M3 still
requires the CPU candidate retention decision.

The [tiled long workload](docs/experiments/2026-10-11-cpu-tiled-long.md)
adds ten checked 1,024-input/128-output requests after three warmups: median
TTFT 85.683956 s and request 142.368874 s. All finish with 1,151 committed
positions and one pending fixed token. Five non-short rows, full tiled
application evidence and the M3.1 retention decision remain pending.

The [tiled boundary workload](docs/experiments/2026-10-11-cpu-tiled-boundary.md)
completes three warmups and ten requests at 1,920 input / 128 fixed output
tokens, then verifies full 2,048-position capacity and rejection before mutation.
Median TTFT is 186.521378 s and completed request 246.854774 s. Scalar remains
default; four cache rows, full tiled application evidence and retention remain.

The [tiled append workload](docs/experiments/2026-10-12-cpu-tiled-append.md)
reuses 512 committed prefix tokens and evaluates 64 appended tokens, with
three warmups and ten complete fixed-output requests. Median TTFT is
5.839271 s and request 63.166372 s. Cache preparation is outside request
timing. Three cache rows, full application evidence and retention remain.

The [fresh tiled cached workload](docs/experiments/2026-10-12-cpu-tiled-cached.md)
completes three warmups and ten requests with zero prompt tokens to evaluate.
Median request time is 54.425495 s; all requests finish at 639 committed
positions plus one pending ID. The user-paused attempt remains excluded.
Pending/changed rows, full application and retention remain.

The [tiled pending-token workload](docs/experiments/2026-10-12-cpu-tiled-pending.md)
commits fixed staged token 2077 before generating the continuation. Three
warmups and ten measured requests complete; each measured request ends at
640 committed positions plus one pending ID. Median TTFT is 0.425491 s and
request 54.797608 s. Changed-prefix, full application and retention remain.

The [tiled changed-prefix workload](docs/experiments/2026-10-12-cpu-tiled-changed.md)
resets and replays the full 512-token prompt after an edit at zero-based index
128. Three warmups and ten measured requests complete. Median TTFT is
38.530052 s and request 93.038050 s. The numerical matrix is now measured;
full application evidence and explicit retention remain before M3.1 closure.

The [full tiled application trace](docs/experiments/2026-10-12-application-tiled.md)
completes three warmups and ten three-turn scenarios. All thirty responses,
token IDs, stops and untimed generation states match scalar. Median scenario
latency is 129.141379 s; historical scalar comparison is descriptive. All
candidate workload and application measurements are complete; M3.1 still
requires its explicit retention decision.

The [CPU retention decision](docs/experiments/2026-10-12-cpu-tiled-retention.md)
closes M3.1 and M3: retain the qualified tiled projection as an opt-in variant,
with scalar as default/reference. Both primary comparisons improve warm TTFT
by about 77.6%; complete workload/application and frozen quality/memory gates
pass their recorded scope. Metal composition and M4–M7 remain pending.

The [Metal gating tail extension](docs/experiments/2026-10-12-metal-gating-tails.md)
checks nine bounded lengths with exact Half outputs and full inactive-lane
padding from queried launch limits. Original tiny and saved Qwen regressions
pass; memory accounting includes final-size padded host and shared buffers.
This qualifies a primitive boundary; MLP/decoder composition remains open.

The [Rust MPS projection boundary](docs/experiments/2026-10-12-rust-mps-projection.md)
passes frozen tiny query gates and exact bias/stride/cancellation probes.
Half operands stay resident; GPU bias initialization, FP32 result storage and
one Half conversion share a command. Two repeated one-ULP Half differences
are investigated within unchanged gates. Wider projections, MLP and decoder
composition remain open.

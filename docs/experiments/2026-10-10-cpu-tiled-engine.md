# Opt-in tiled CPU engine qualification, 10 October 2026

The ordered FP64 [projection candidate](2026-10-10-cpu-tile.md) now has an
explicit, opt-in full-model execution path. Scalar remains the model and CLI
default. This task qualifies numerical and session behavior; full-workload
performance and a second confirming run still precede adoption or M3 completion.
The scalar M2.5 workload matrix remains in progress.

## Mechanism and ownership

[ProjectionKernel](../../src/cpu/tiled.rs) selects Scalar or Tiled once per
projection, before either function's statically dispatched numerical loops.
[Model::with_projection_kernel](../../src/model/mod.rs) consumes the model and
changes a small enum without copying weights or session buffers. A live session's
borrow prevents changing its model. Every Q/K/V, attention-output, MLP and tied
vocabulary projection uses the selection, including final-position-only logits.

Both paths consume row-major FP32 weights `[outputs, inputs]` and activations
`[tokens, inputs]`, and write `[tokens, outputs]`. The existing tile uses four
token rows and four output rows, sixteen independent FP64 sums, increasing
feature order, bias after the dot product and one FP32 storage rounding. One-row
and scalar tails handle irregular shapes. The tile itself is unchanged from the
reviewed microbenchmark candidate. Its largest accumulator array is 128 stack
bytes; there is no persistent allocation, packing, unsafe access, extra alignment
requirement, threading, reassociation or indirect call in the numerical loops.

Normalization, rotary offsets, causal GQA, separate head-major K/V storage,
session admission and transactional commits are unchanged. Selecting the tile
does not change FP32 model storage, allocated KV capacity or scratch sizes.
The [benchmark worker](../../examples/cpu_bench.rs) accepts
`MODEL [CHUNK] [scalar|tiled]` and reports the selected projection at readiness.
Invalid selection fails before model loading.

## Checks and evidence

The [functional qualification record](../../benchmarks/results/2026-10-10-cpu-tiled-qualification-v1.json)
retains commands, stdout, source/binary hashes, compiler/build configuration,
model identity, power policy and report/stderr hashes. Its byte-identical
[forward report](../../benchmarks/results/2026-10-10-cpu-tiled-forward-v1.json)
and [session report](../../benchmarks/results/2026-10-10-cpu-tiled-session-v1.json)
retain the complete numerical metrics and memory account.

On Apple M1 Pro, MacBookPro18,3, 16 GiB, macOS 26.6.2 (25G83), Rust 1.90.0,
ordinary optimized release settings with no RUSTFLAGS, the pinned Qwen
`7ae557604adf67be50417f59c2c2f167def9a775` passes:

- All 435 complete trace arrays per case, across all 24 layers and logits,
  match scalar FP32 bits: 1,740 arrays across four frozen cases. The check
  includes signed zero and every element, rather than sampled coordinates.
- All 228 existing independent full-array FP32 gates pass, with maximum
  tolerance fraction 0.6346080577. The original twelve greedy IDs agree.
  Maximum absolute short teacher-forced NLL delta is 0.0002174812, inside the
  existing 0.001 gate. Fixtures, tolerances and expected IDs are unchanged.
- Cached sessions pass 47 exact complete-logit comparisons, covering chunking,
  offsets, append, pending samples, fully cached prompts, changed-prefix replay
  and greedy continuation. Own-process physical peak is 2,067,597,328 bytes,
  below the unchanged 6 GiB ceiling.
- Logical model storage is 1,976,131,072 bytes, KV 50,331,648 bytes and scratch
  10,436,608 bytes. Session admission including history, logits, metadata and
  the existing 64 MiB allowance is 2,105,239,936 bytes for capacity 2,048.

Ordinary [tiny trace checks](../../tests/cpu_decoder.rs) compare complete arrays
for five cases. All thirteen [session tests](../../tests/session.rs) run both
variants, including failure rollback, cancellation/output failure recovery,
stop/pending semantics, actual 2,048-position small-model execution, rejection
beyond capacity and allocation-free numerical/session/event paths. Formatting,
Clippy for all targets with warnings denied, 75 ordinary Rust tests and 45 offline
Python checks pass. Explicit release Qwen forward/session tests and separate
one-request scalar/tiled worker admission checks pass. Worker timing fields are
retained as functional diagnostics, without a performance conclusion.

AC power was attached, AC low-power mode was disabled and no thermal/performance
warning was reported before or after qualification. Real-model checks here use
short frozen cases; they do not qualify held-out quality, real-model execution
at every capacity position, or end-to-end latency.

## Reproduction and isolation

The qualification starts from `de1c24c` plus the exact changed-source hashes in
the record. Builds use a separate target directory; the active scalar matrix
worker and CLI binaries remain byte-identical to their frozen revisions.
For this task, the transport driver was held at a request boundary while the
numerical worker finished normally, then stayed idle through compilation and
explicit checks. Only the transport driver was resumed afterward; no measured
numerical request was suspended. Other performance jobs were inactive during
the functional qualification.

```sh
CARGO_TARGET_DIR=artifacts/cpu-tiled-repro cargo build --locked --offline --release --example cpu_bench
TOKLET_QWEN_FORWARD_REPORT=artifacts/tiled-forward-repro.json CARGO_TARGET_DIR=artifacts/cpu-tiled-repro cargo test --locked --offline --release --test qwen_forward pinned_qwen_tiled_fp32_matches_full_transformers_forward -- --ignored --exact --test-threads=1 --nocapture
TOKLET_QWEN_SESSION_REPORT=artifacts/tiled-session-repro.json CARGO_TARGET_DIR=artifacts/cpu-tiled-repro cargo test --locked --offline --release --test qwen_session pinned_qwen_tiled_cached_sessions_match_full_prefix -- --ignored --exact --test-threads=1 --nocapture
artifacts/cpu-tiled-repro/release/examples/cpu_bench models/Qwen2.5-0.5B-Instruct 128 tiled
```

Explicit tiled report paths refuse overwrite. Raw qualification stderr is copied
alongside the result record with the same suffixes and checked against its hashes.
Decision: retain the opt-in path for full-workload measurement, preserve scalar
as default, and leave M2.5 and M3.1 open until their acceptance evidence exists.
Astra accepted the completed code task without actionable findings and
independently verified source/binary identities, exact raw/report/stderr copies,
trace and gate counts, session comparisons, memory and the documented limits.

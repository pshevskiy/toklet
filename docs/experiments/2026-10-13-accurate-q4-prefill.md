# Accurate eight-row Q4 prefill experiment

The explicit `ProjectionMode::Q4CompensatedTiled` reuses a reconstructed weight
across eight prompt rows while retaining the compensated projection arithmetic.
It is an experimental performance component. Both fixed Q4 configurations
remain rejected by the frozen held-out loss gate; this task does not qualify
Q4 quality, change the default backend, or complete M5.

## Mechanism and ownership

One queried SIMD group owns an output feature and an eight-row prompt tile.
Adjacent lanes traverse adjacent packed columns. Each lane reconstructs the
same rounded FP32 weight once, then updates eight independent two-component
accumulators with the product and its explicit FMA residual. The extracted
reduction helper preserves the disjoint adjacent-subtree operation sequence of
the previous compensated kernel. Bias is added after the reduction and the
existing round-to-odd FP32 intermediate prevents double rounding at Half storage.

All lanes use the same active-row predicate; inactive rows never read activation
padding or write output. Checked output overflow is unchanged. Chunks below
eight use the compensated single-row kernel. Mode selection occurs before the
column loop. Weights, separate KV buffers, scratch, views, command scheduling,
state publication and profile-event count are unchanged. Only one cached pipeline
is added; no expanded weights or persistent tensor allocation is introduced.

## Correctness

The primitive checks cover 1/7/8/9/17 rows, a 65-column tail, padded strides,
poisoned padding, subnormal inputs, overflow status and inactive output guards.
Both compensated kernels pass all 72 signed Half midpoint cases against direct
FP64 rounding, including the 22 cases that distinguish it from FP32-then-Half.

The tile passes 4,550 complete expanded tiny decoder stage comparisons against
the unchanged independent fixture. A separate refinement check preserves every
bit in 4,550 complete stage arrays against compensated single-row execution,
including irregular chunks and nonzero rotary offsets. The old Original/Tiled
independent failure remains recorded separately.

The paired larger checkpoint run requires equal queried SIMD widths and exact
bit equality across every one of the 151,936 final logits for every pair. It
compares the same Q4 artifact and compensated RMS mode; this is a refinement
check at a representative shape, not an independent larger full-graph oracle.

## Paired measurement

The worker shares one immutable larger model between two capacity-1024,
chunk-128 sessions. The primary metric is warm reset/prefill to the first token
event for 512 input tokens and one output token, including completed GPU work,
full final-logit readback and CPU greedy selection. Three warmup pairs precede
ten measured pairs, with order alternating throughout. Native worker/driver
lifetime charges are checked jointly against 6 GiB at idle handshakes. No
builds, tests or large hashes run concurrently with GPU measurement.

Hardware is Apple M1 Pro, 16 GiB, macOS 26.6.2, AC power with low-power mode
disabled; binaries use Cargo release defaults and Rust 1.90.0. Exact source,
config, packed matrices, prompt, worker binary and collector identities are
retained in each result and checked again after completion. The frozen larger
checkpoint and artifact are the same ones used for the rejected quality run.

| Run | Compensated single-row median | Tile median | Improvement | Single-row / tile MAD |
| --- | ---: | ---: | ---: | ---: |
| v1 | 13.495412396 s | 10.065848708 s | 25.4128% | 1.157 / 4.668 ms |
| v2 | 13.640405104 s | 10.136985833 s | 25.6841% | 56.749 / 29.179 ms |

All 26 pairs, including warmups, have identical final-logit bits and greedy IDs.
Between-run median variation is 1.0744% for single-row and 0.7067% for tile,
below both improvements and the 5% retain threshold. The second run has wider
dispersion: measured ranges are 13.583181417–13.990294417 s and
10.066158125–10.547523875 s. The gain remains larger than that observed variation.
Combined admission is 1,691,328,140 bytes; maximum joint native lifetime charge
is 1,535,215,752 bytes. Persistent buffer inventories are fixed. Both workers
finish with 15 protocol events, exit 0, empty cleanup errors and unchanged
endpoint identities.

Retain this explicit experimental component. There is no default change or
quality acceptance. [Run v1](../../benchmarks/results/2026-10-13-accurate-q4-prefill-larger-v1.json)
and [run v2](../../benchmarks/results/2026-10-13-accurate-q4-prefill-larger-v2.json)
retain all raw timings, numerical flags, memory and identity evidence.

This component measurement does not cover 128 output steps, application-visible
text, sustained thermal behavior, external runtimes, or the full M5 release
workload. The original projection and normalization remain defaults.

## Reproduction

```sh
CARGO_TARGET_DIR=artifacts/m5-checks cargo build --offline --release --features metal --example metal_prefill
.venv/bin/python tools/experiments/metal_prefill.py --binary artifacts/m5-checks/release/examples/metal_prefill --output benchmarks/results/2026-10-13-accurate-q4-prefill-larger-v1.json --accurate --larger
```

Evidence destinations are exclusive; use a fresh path for reruns. The second
run uses the identical command with a fresh `v2` output. Without either optional
flag, the collector retains the old Original/Tiled small-model protocol.

Formatting and strict all-target Clippy pass in default and Metal configurations,
along with 87 CPU tests, 94 Metal-feature ordinary tests and 164 experiment
Python tests. Four targeted native checks and 15 final native regressions pass
sequentially after completed builds. The first regression launcher rejected a
duplicated `--nocapture` flag before running any test; the corrected launcher
requires exactly one executed passing test per filter. No engine or gate change
was made for that harness correction.

The [qualification receipt](../../benchmarks/results/2026-10-13-accurate-q4-prefill-qualification-v1.json)
binds sources, fixtures, successful logs, exact immutable native binaries and
saved runs. Protected CPU release binaries retain their historical hashes.
Astra's source and compact saved-data reviews found no actionable issues.

# MLP SiLU/gating fusion experiment

The hypothesis is that fusing the two pointwise MLP operations can reduce
dispatch and intermediate traffic without changing stored numerical values.
This is an individual experiment against Split execution, with full-logit
readback and all projection/normalization choices matched. Neither frozen Q4
quality rejection is reconsidered.

## Mechanism and layout

Gate/up/gated tensors are contiguous Half `[rows, intermediate_size]`. Fused
execution completes the independent up projection before the pointwise kernel.
Each element computes the existing stable FP32 SiLU, explicitly rounds it to
Half, checks that former storage boundary, multiplies the rounded value by Half
up and checks the final Half store. Split retains gate → SiLU → up → gating.
No projection arithmetic, FMA policy, cache operation or committed-state rule
changes.

Ordinary fusion reads gate/up and writes gated: six bytes per element, versus
ten for the two split kernels. It removes four bytes of SiLU write/read traffic
and one dispatch per layer. Diagnostics execute actual fused arithmetic and
optionally materialize the rounded SiLU, preserving the complete stage inventory
and order. Existing SiLU scratch remains allocated for late diagnostics; no
resident-memory saving is claimed. Both modes retain one completed command per
chunk and persistent cached pipelines. Profile admission removes one event per
layer; the larger Q4 trace has 454 events instead of 482. Larger Half profiling
still exceeds the existing 1,024-event ceiling and remains unsupported.

## Numerical and state qualification

The primitive compares 2,178,580 elements across 20 guarded cases and both
diagnostic-write flags, including all 63,488 finite Half gates crossed with 15
signed/subnormal/zero/extreme up values. Separate all-finite sweeps with up zero
and one require status exactly zero; a sticky overflow elsewhere cannot hide
false rejection. Five isolated exceptional cases require rejection. Input and
output guards, optional-store omission and readonly inputs are checked.

Tiny Half and compensated Q4 compare 4,550 complete stage arrays each against
Split, at frozen shapes, tails and rotary offsets. Every stage independently
requires the expected identity, rows, columns and complete value count. Ordinary
Half/Q4 × offset0/5 cases compare 12 generated outputs per case, exact full
logits/history, eight warmed allocation-free steps, prefix edit/shortening/full
reuse, exact-budget admission, one-byte-short rejection, cancellation, hidden
stops and output failure. A separate failed-chunk test preserves the last
committed selection, checks replay and enables diagnostics after profiling.

Larger Half/Q4 each pass 969 unchanged selected-stage gate comparisons, covering
layers0/1/27 and complete selected arrays. Maximum gate fractions remain
0.5792791429143427 and 0.7321240893134191. Scores/probabilities are outside that
selected-stage inventory. Each saved report explicitly records Fused gating and
FullLogits readback. Native lifetime peaks are 3,146,697,488 Half and
1,340,769,792 Q4 bytes, below 6 GiB. These are short numerical checks, not quality
or performance acceptance.

The initial build caught a missing local Half-crate import; the corrected code
uses exact up-value Half words. A subsequent ordinary test referenced a nonexistent
session method and was corrected to `evaluate`. The first larger launcher matched
zero tests and supplies no qualification evidence. The corrected exact native
names each execute one passing test; no source gate was changed.

## Controlled measurement

The fixed larger Qwen Q4 model uses compensated eight-row projection and
compensated RMS in both sessions. One immutable model is shared, with separate
session-owned KV/scratch; persistent memory maps are required to be identical.
The frozen 512-token prompt produces 128 outputs, ignoring stop tokens for the
fixed workload. All generated IDs, committed history and every final-logit bit
are compared after timers. Reports retain 640-history/639-evaluated/one-pending
state; the pending sample is then discarded.

Three alternating warmup pairs precede ten balanced measured pairs. Primary
timing is complete warm generation, including GPU completion, full readback,
CPU greedy and the preallocated ID callback. The bounded owned collector checks
all modes, geometry, counts, timings, memory, source/artifact/prompt pins and
endpoint identities; failures retain primary and cleanup causes. Native worker
and driver lifetime charges are sampled jointly below 6 GiB. Power/thermal
endpoints are recorded; continuous sensors and energy are unmeasured. Builds,
tests and large hashes do not overlap GPU measurement.

Formatting, strict all-target Clippy in both feature configurations, 87 CPU /
94 Metal-feature ordinary tests and 179 Python experiment tests pass before
measurement. The six native primitive/state/tiny/larger checks pass. The exact
prototype sources, helper scripts and native/worker binaries are retained before
measurement so a rejected candidate can be removed without losing reproducible
evidence.

## Result and discard decision

Apple M1 Pro, 16 GiB, macOS 26.6.2, Rust 1.90.0 and Cargo release defaults.
AC power, low-power mode zero and unchanged power/thermal endpoints are recorded;
no thermal/performance warning is recorded. The source parent is `2b2cec9`, with
exact prototype source identities retained rather than attributing these changes
to that committed parent.

| Variant | Median complete request | MAD | Measured range |
| --- | ---: | ---: | ---: |
| Split | 78.9913938335 s | 7.994 ms | 78.647325500–79.019668292 s |
| Fused | 78.9866290625 s | 39.132 ms | 78.880865541–79.030386375 s |

Median improvement is **0.006032%**, only 4.765 ms. It is smaller than observed
dispersion and far below the frozen 5% threshold. Reject the candidate; no
promising result warrants a confirmation run. Fewer dispatches and pointwise
bytes do not produce a useful complete-request improvement on this workload.
This result does not identify the cause or establish reliable tail percentiles.

All 26 requests retain all 128 generated IDs, exact committed histories and
every final-logit bit. Persistent inventories remain equal and fixed. Combined
admission is 1,691,328,140 bytes; joint native lifetime peak is 1,534,347,448 bytes.
Ten measured pairs total 1,579.252164 seconds of completed repeated requests.
The [saved paired run](../../benchmarks/results/2026-10-14-metal-mlp-paired-v1.json)
has 15 exact events, clean completion, unchanged identities and empty cleanup
errors. Larger [Half](../../benchmarks/results/2026-10-14-mlp-half-parity-v1.json)
and [Q4](../../benchmarks/results/2026-10-14-mlp-q4-parity-v1.json) numerical reports
remain separate from this performance result.

The experimental kernel, API, tests and harness are removed from the live
package. Only task-owned files whose bytes still matched the measured source
snapshot were restored to `2b2cec9`; no unrelated edits were overwritten. The
bound local prototype snapshot and immutable binaries preserve reproduction.
All ordinary formatting/Clippy/tests are checked again after discard. The
[qualification receipt](../../benchmarks/results/2026-10-14-mlp-gating-qualification-v1.json)
distinguishes measured prototype paths from restored sources and binds both.
Protected CPU binaries are unchanged. Astra reviewed source, coverage repairs,
native checks, saved-data replay, provenance and the discard decision.

## Local reproduction

The ignored source snapshot is `artifacts/m5-mlp-prototype-source-v1`; its model,
artifact and fixture symlinks use the existing pinned assets. Rebuild from that
snapshot, with the absolute isolated target directory below. It contains the
exact candidate helper/worker source, while the live package retains Split.

```sh
cd artifacts/m5-mlp-prototype-source-v1
CARGO_TARGET_DIR=/Users/coloner/work/rust/toklet/artifacts/m5-checks cargo build --offline --release --features metal --example metal_mlp
```

To replay the retained binary using the bound snapshot collector, from the
repository root:

```sh
.venv/bin/python artifacts/m5-mlp-prototype-source-v1/tools/experiments/metal_mlp.py --binary artifacts/m5-mlp-native-binaries-v1/metal_mlp --output benchmarks/results/2026-10-14-metal-mlp-rerun.json
```

Use a fresh exclusive output path. No kernel is adopted, no quality gate or
default changes, and M5 release acceptance remains open under both Q4 failures.

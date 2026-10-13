# Explicit GPU greedy selection and token readback

`ReadbackMode::GreedyToken` selects the next token in the completed Metal
command and publishes one cached ID. It is an opt-in session mode; full logits,
CPU selection and all existing defaults remain available. It changes selection
and output traffic, not model arithmetic, weights or any quality gate.

## Layout, ordering and publication

One queried SIMD group reads contiguous FP32 vocabulary words from the ordinary
final-row output, or the final diagnostic row with an explicit word offset. Each
lane keeps its best integer-ordered finite value and lowest ID. A disjoint
shuffle tree merges candidates. Positive values use sign-bit inversion;
negative values use bit inversion; both signed zeros share one key. This
preserves exact finite FP32 ordering, including subnormals, without arithmetic
flush-to-zero effects. NaN or either infinity sets the existing failure status.

The cached pipeline replaces final-logit validation in ordinary token mode.
Diagnostics still validate every computed row and add selection afterward.
The output is one ID plus seven sentinel guard words in a 32-byte shared GPU
buffer. CPU access occurs only after command completion. Current/staged IDs
replace two host vocabulary distributions, using eight accounted bytes. GPU
Float scratch and all resident model/KV/activation buffers remain unchanged.
For vocabulary 151,936 this reduces admission by 1,215,448 bytes per session.

Selection is staged and published only after the whole step succeeds. Pending
tokens, invalid distributions and replay-required failures expose no ID. Prefix
reuse, cancellation, stop tokens and output failure use the same committed-state
rules as full-logit sessions. `InferenceSession::greedy_token` preserves the CPU
default by sampling its available logits; generation calls this coarse boundary.
Token mode's public `logits()` is unavailable. Explicit admitted diagnostics
retain complete logits and stages for inspection.

## Checks

The native primitive compares 160 exact cases against unchanged CPU greedy:
vocabularies 1/7/31/32/33/65/257/151936, two offsets and ten input patterns. It
checks lowest-ID ties across lanes, signed zeros, positive/negative subnormals,
adjacent finite values, extrema, NaN/infinities, input preservation and guards.
The measured execution width is 32; launch uses the queried device capability.

Tiny Half and compensated Q4 sessions compare 560 complete diagnostic arrays
across reset, append, edit and shortening at two rotary offsets. Both snapshots
independently require the exact 35-stage inventory, shapes and complete value
counts before bit comparison. Fully cached prompts retain selection without
evaluation. Generation sequences, pending/committed state, repeated continuation,
stop/cancellation/output failure, exact-budget admission, one-byte-short rejection
and zero warmed Rust allocations are checked. A separate native failure test
checks nonfinite rejection, last committed selection recovery, completion replay
and profile-before-diagnostics capacity/event accounting.

## Measurement protocol

The fixed larger Qwen Q4 model uses `Q4CompensatedTiled` projection and compensated
RMS. Two sessions share immutable weights; all local buffers and framework/application
allowances are jointly admitted below 6 GiB. The frozen 512-token prompt produces
128 greedy outputs with no stop tokens. Every pair checks all generated IDs,
the exact committed 639-token history and final cached selection. Reports retain
the pre-discard 640-history/639-evaluated/one-pending state.

Three alternating warmup pairs precede ten balanced measured pairs. Primary
timing is complete warm engine request time through generation and its preallocated
ID callback, including GPU completion. Full-logit mode's host sampling counter
includes CPU greedy. Token mode's GPU selection is charged inside prefill/decode;
its sampling counter only fetches the cached ID. Component-counter reductions
alone are not speed claims. Tokenization, text decoding and UI are outside this
component measurement.

The owned collector freezes/checks source, artifact, prompt, runtime source,
helper and binary identities before/after execution. It enforces exact readiness,
13 samples plus completion, persistent memory, completed timings and token/state
coverage. It bounds events/stdout/deadlines, samples joint worker/driver lifetime
memory and retains primary/cleanup failures with exclusive evidence paths.
Five offline tests mutate the protocol, state, memory, times and frozen pins,
and verify owned child cleanup on failure.

## Completed result and decision

Collection started 13 October and completed 14 October 2026 in Moscow local
time. Hardware is Apple M1 Pro, 16 GiB, macOS 26.6.2, Rust 1.90.0 and Cargo
release defaults with the Metal feature. AC power is attached, AC low-power mode
is zero and endpoint observations show no recorded thermal/performance warning.
Continuous sensors and energy are unmeasured. No builds, tests or large hashes
overlap the GPU run. The source parent is `9b165e0`, with exact dirty-task hashes;
the concurrent `74f00bf` commit changes only reviewed prefill documentation.

| Readback | Median complete request | MAD | Measured range |
| --- | ---: | ---: | ---: |
| Full logits / CPU greedy | 78.9409020835 s | 28.734 ms | 78.609639542–79.233264209 s |
| Token ID / GPU greedy | 79.1690374580 s | 8.862 ms | 79.092989291–79.536114791 s |

The token mode is **0.2890% slower** by the declared median metric. Reject it as
a default performance optimization; there is no promising gain to confirm in a
second run. The small host sampling counter is not a speedup: medians fall from
76.008 ms to 0.031 ms because GPU selection is inside evaluation, while completed
prefill/decode remain included in total time. This experiment does not establish
a cause for the small latency difference or reliable tail percentiles.

Retain the explicit ID-only API and its verified host-storage/traffic behavior.
It supports ordinary generation without host vocabulary distributions, with the
same selection and transactional semantics; full logits remain the default.
This is functional capability retention, not acceptance of a sub-5% speed claim.
All 26 requests, including warmups, preserve every generated ID, committed token
and cached final selection. Ten measured pairs provide over 26 minutes of
completed repeated request activity, with fixed memory inventories. This supplies
sustained component evidence, not the complete release/application qualification.

Combined admission is 1,690,112,692 bytes. The maximum observed joint native
lifetime charge is 1,534,445,800 bytes, below 6 GiB. The
[paired result](../../benchmarks/results/2026-10-13-metal-greedy-paired-v1.json)
finishes with 15 exact protocol events, exit zero, unchanged endpoint identities
and empty cleanup errors. Raw samples remain beside the summary.

Final formatting and strict all-target Clippy pass in default and Metal builds;
ordinary suites pass 87 CPU / 94 Metal-feature tests and 174 experiment Python
tests. Three final selection/state native checks and four default Half/Q4 native
regressions pass sequentially from immutable binaries. The expanded compensated
Q4 independent oracle remains passing. Protected CPU release binaries are
unchanged. The
[qualification receipt](../../benchmarks/results/2026-10-13-metal-greedy-qualification-v1.json)
binds sources, fixtures, measured/test binaries, checks and complete saved data.
Astra reviewed code, repaired coverage findings and completed evidence; scope is
the explicit functional capability and negative performance decision above.

## Reproduction

```sh
CARGO_TARGET_DIR=artifacts/m5-checks cargo build --offline --release --features metal --example metal_greedy
.venv/bin/python tools/experiments/metal_greedy.py --binary artifacts/m5-checks/release/examples/metal_greedy --output benchmarks/results/2026-10-13-metal-greedy-paired-v1.json
```

Use fresh exclusive destinations for reruns. Both selected Q4 held-out quality
rejections remain in force; this task cannot establish quality acceptance or M5
completion.

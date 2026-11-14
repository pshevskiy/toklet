# Ordinary Gemma CLI and request metrics

[Source2 conformance](../artifacts/gemma-cli-metrics-2026-11-14/conformance-join-1/result.json)
passes required controls and policy/mode parity across its84-entry inventory.
Unavailable controls remain explicit in that receipt. The final resource-header
revision passes required checks and both fresh header/export smokes. The complete
fixed overhead campaign passes collection/integrity checks; three of eight
predeclared timing comparisons miss the1% gate. Summary remains disabled by
default and explicitly opt-in, with overhead unqualified. Independent raw replay agrees with every captured value and statistic. Astra
approves scoped functional retention with no actionable findings. Commit/push
and exact-head CI remain pending; broader performance/adoption stays unqualified.

The explicit `gemma4-bf16` profile uses the existing official
`google/gemma-4-E2B-it` BF16 owner at revision
`70af34e20bd4b7a91f0de6b22675850c43922a03`. It borrows that owner's model and
single codec, then constructs the existing G2 session. It does not load a second
tokenizer. Immutable weights remain separate from session KV and scratch.
Numerical kernels, source readers, native ABI and other CLI profiles are unchanged.

Apple Silicon macOS and the `metal` feature are required. Capacity defaults to
512 positions and prefill chunks to32; neither may exceed those bounds. Prompt
and requested output must fit together. Tools, thinking, multimodal input and
kernel overrides are unsupported. `--submission per-tile` remains the default;
`--submission batched-single-row` explicitly selects the previously qualified
[bounded submission route](gemma-single-row-submission.md).

```sh
cargo run --release --features metal -- \
  --model-profile gemma4-bf16 --backend metal \
  --model /path/to/official-gemma \
  --prompt "Describe the stages of a request." --max-new-tokens 128 \
  --stats --request-metrics summary \
  --metrics-output /tmp/toklet-gemma-metrics.jsonl
```

Raw token IDs, raw text and basic chat use the existing owner codec and stop IDs.
`--json` requests token and terminal events; text output can use `--stats` on
stderr. These requested outputs are separate from routine telemetry. Interactive
JSON lines support `generate`, `append`, `continue` and `reset`. Sampled history,
committed KV and pending tokens remain separate. Continue preserves unfinished
UTF8 output; reset/replay follows admission. An output failure can require a new
response or reset while retaining the authoritative numerical prefix.

## Bounded collection and export

Metrics default to `--request-metrics disabled`. Summary collection requires
both `--request-metrics summary` and `--metrics-output PATH`. It uses fixed typed
fields and a bounded circular ring: the fixed Collector value is at most16KiB,
its fixed value plus retained ring capacity is at most1MiB, and each JSONL header,
request summary or footer is at most4KiB. There are no routine
prompt/output/token arrays or per-token telemetry records. Circular replacement,
invalid records and exporter failures have explicit counters.

The manifest reports `collector_fixed_bytes`, `retained_ring_capacity_bytes`,
`collector_peak_bytes` and `telemetry_admitted_bytes`. The first three describe
logical Collector storage after construction: fixed value plus ring capacity
equals the known high-water storage, which stays fixed until drop. They exclude
allocator overhead, constructor transients, temporary export storage and native
or OS memory. This is not a sampled process peak. A failed ring reservation can
report zero capacity while preserving dropped-record counters.

Export runs after active inference ends, including interactive EOF, separately
from token callbacks. It creates a new file and never overwrites an existing
path. Telemetry failure cannot replace the inference result. A complete export
requires `request_metrics_manifest`, ordered `request_summary` records and a
terminal `request_metrics_export` footer. Process death or a failed partial export
does not manufacture a completed request or footer.

Each handled attempt receives an identity before input validation. One retained
summary describes its outcome and before/after state. If terminal metadata output
fails, the most recent matching summary is corrected to output failure rather
than appended again. Immutable owner generation remains1 across cache resets;
cache action and state snapshots describe logical changes.

## Clock and memory meanings

Summary durations are milliseconds from monotonic host clocks. Request content
time begins at the handled invocation before formatting/tokenization and ends
after content publication. Core generation time is nested within that interval.
Terminal metadata/collector completion and explicit export have separate scopes;
overhead trials measure external submit-to-terminal round trips so collector
work is included. New-process load and route-start-to-ready are separate from
warm request timing, and do not establish a cold filesystem cache.

Callback-token time and first nonempty stdout publication are distinct. Neither
is a UI/render acknowledgment. Decode rate is `(N-1)/(last-first)` using one
callback event source, only for `N>1` and a positive interval. Completed numerical
decode-evaluation throughput is separate. Errors retain observed duration and
authoritative state; unavailable completed core phases/counts are null.
Logical prompt count includes existing history for Append/Continue. Replayed rows
count only completed prefill overlapping formerly committed positions; new or
previously pending input is separate. Overflow prevents an invented attempt ID
and makes attempted-outcome rates unavailable.

Whole-owner admission reserves the application, stream, active telemetry and
32MiB helper before loading, within6GiB. The codec's allowance is counted once
in the owner. The manifest attaches the full numerical logical-memory report,
including positional/PLE, history, logits and metadata owners. Numerical and
additional owner/application admission partition the whole total; they are not
added to it again. KV capacity and valid positions remain distinct. Admission
estimates are separate from externally sampled process memory. The1,064,960-byte
telemetry admission already covers the reported logical storage; actual storage
is not added to whole admission again. The independent reader checks these
integer bounds and equations without importing a platform-specific Rust layout.

Ordinary backend GPU/host/wait/readback/pipeline/allocation counters, process
footprint, UI, energy and unsupported signal-latency observations remain null
with fixed availability reasons. In particular, `cancel_to_terminal_ms` remains
null even for a known cancellation outcome: the manifest states
`signal_timestamp_not_observed; cancellation_outcome_and_state_do_not_measure_signal_to_ack_latency`.
No signal timestamp or additional request clock is introduced by these header
facts. Diagnostic profiles or inventory predictions
cannot substitute for ordinary measurements. These remaining metrics are part
of the wider [observability roadmap](model-observability-plan.md#metric-contract).

The completed resource header is required by the new strict schema1 reader.
Source2 exports keep their original immutable reader and evidence pins; they are
not rewritten or silently admitted as final-header captures. Final checks pass
57 CPU/72 Metal affected tests (two ignored each),36 independent reader controls
per Python mode, formatting and strict CPU/Metal/probe Clippy. Both fresh policy
smokes match output/state/text and report zero telemetry loss or errors. Their
actual fixed/ring/peak storage is9,000/1,032,192/1,041,192 bytes, within the
1,064,960-byte admission. [Checks](../artifacts/gemma-cli-metrics-2026-11-14/required-checks-3.json),
[fresh smoke](../artifacts/gemma-cli-metrics-2026-11-14/resource-header-smoke-1/result.json).

[Interface and ownership contract](../benchmarks/gemma-cli-v1/contract.json),
[first interface checks](../artifacts/gemma-cli-metrics-2026-11-14/early-integration-1.json),
[first stable Metal integration](../artifacts/gemma-cli-metrics-2026-11-14/integrated-metal-2-source.json).

## Fixed overhead measurement

The final optimized Metal binary runs on Apple M1 Pro/MacBookPro18,3,16GiB,
macOS26.6.2 build25G83, with AC power, Low Power Mode off and thermal state0.
The authenticated official BF16 model, tokenizer, source3 and binary stay fixed.
Input384/output128, capacity512/chunk32, greedy/no-EOS, explicit JSON output
and reset before every generation are identical between modes. Each fresh owner
has three warmups and five measured generations; independent ABBA and BAAB
cohorts yield10 measured requests per mode/cohort, separately for both policies.
All16 owners,128 generations (48 warm/80 measured),128 resets and eight complete
16-record exports validate. Collection takes6,264.96s; maximum observed owned
charge is5,004,482,128 bytes including the32MiB helper, within6GiB. All children
are reaped. This sampled charge is separate from logical admission and is not
a continuous allocator cap. Source/model endpoints remain unchanged.

Each table entry is an overhead percentage. Marginal overhead is
`100*(median_summary/median_disabled-1)`; upper is the predeclared paired-median
95% bootstrap upper endpoint (10,000 resamples, seed1, indices250/9749). Both
marginal and upper must be at most1% for each metric/cohort/policy. Timing starts
at the parent send timestamp before bounded request journaling and nonblocking
stdin writes, and ends at the first-token or done newline receipt before JSON
parsing. Capture bookkeeping is included in both modes. Cold start and EOF
export are separate.

| Policy/cohort | TTFT marginal | TTFT upper | Complete marginal | Complete upper | Original gates |
| --- | ---: | ---: | ---: | ---: | --- |
| per-tile / 1 | +3.9002% | +7.2139% | +10.7172% | +19.4624% | fail TTFT, fail complete |
| per-tile / 2 | -0.1178% | +0.5932% | +0.0290% | +0.5160% | pass, pass |
| batched-single-row / 1 | -0.5387% | -0.1478% | -0.1825% | +0.0576% | pass, pass |
| batched-single-row / 2 | +1.1991% | +1.9127% | +0.0874% | +0.5733% | fail TTFT, pass |

Three of eight metric/cohort comparisons fail. The frozen policy permits
functional retention with summary opt-in and the overhead bound unqualified.
All observations remain in the dataset; there is no addedN, discarded failing
cohort, new noise threshold or default/adoption qualification. Independent raw
receipt/statistical replay validates128 generations,128 resets,eight exports and
all4,116 frozen inputs with zero discrepancies. It independently derives the
same eight gate results from receive timestamps. In the first per-tile cohort,
second-versus-first block complete-time medians change−12.03% for disabled and
−26.87% for summary. These are descriptive drift observations; they assign no
cause, invalidate no cohort and exclude no sample. The comparisons do not isolate
collector cost. Astra independently reproduces all eight gate results and approves
scoped opt-in functional retention with no actionable findings.

Two earlier attempts contain zero measured samples and stay excluded: the first
stops on a runner lookup of a wrapper counter inside the core Report; the
corrected adapter keeps the outer127-evaluation check and passes actual saved
wire/report regressions. The second stops on actual battery/LPM admission during
warmup, with clean cleanup. The fresh complete campaign uses the originalN/order.
[Protocol](../artifacts/gemma-cli-metrics-2026-11-14/overhead-protocol-1.json),
[complete capture](../artifacts/gemma-cli-metrics-2026-11-14/cli-overhead-3/result.json),
[guard](../artifacts/gemma-cli-metrics-2026-11-14/cli-overhead-guard-3/result.json),
[statistics](../artifacts/gemma-cli-metrics-2026-11-14/cli-overhead-3/offline-overhead-replay.json).

[Independent raw/statistical replay](../artifacts/gemma-cli-metrics-2026-11-14/independent-overhead-replay-1.json).

[Combined Astra review](../artifacts/gemma-cli-metrics-2026-11-14/astra-review-1/verdict.json).

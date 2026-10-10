# Matched CPU primary comparison, 10 October 2026

The opt-in [ordered FP64 tile](2026-10-10-cpu-tiled-engine.md) is measured
against scalar projections in the same qualified numerical executable. Both
complete comparison pairs pass the short-workload primary gate. Scalar remains
the default; full workload matrix,
application and sustained qualification still precede M2.5/M3 acceptance.

## Hypothesis and measurement

Profiling attributed 99.24% of scalar numerical samples to FP64 projections.
The four-token/four-output tile reuses activations and contiguous weight rows
while preserving feature order, bias placement and final FP32 rounding. Existing
real-model qualification proves exact scalar trace bits and unchanged source
gates, greedy IDs and session/cache comparisons. No weights, KV, precision,
allocation policy or numerical oracle changes for these measurements.

The declared primary metric is warm TTFT for the frozen 128-input/128-output
teacher-forced workload. Both paths use FP32 weights/KV, capacity 2,048 and
prefill chunk 128. Each measured request starts with an empty cache and uses
identical pinned prompt/continuation IDs. Sampling, tokenization, detokenization,
publication and JSON transport are outside request timing; completed synchronous
session evaluation through final vocabulary projection determines token times.

The confirming pair reverses order: clean scalar, tiled, tiled, scalar. Every
run has three warmups and ten measured repetitions in a separate process using
one executable SHA-256. Numerical build provenance is the qualification's
`de1c24c` base plus recorded source hashes, committed as `49df89c`; per-run
`engine_commit` identifies later dispatch/checkouts and does not identify a
newly rebuilt numerical executable. Ordinary optimized release settings, Rust
1.90.0 and no RUSTFLAGS are unchanged.

The [first originally scheduled scalar run](../../benchmarks/results/2026-10-10-cpu-primary-pair-1-scalar.json)
is retained but excluded because
offline Python checks overlapped a timed request at 02:00:46–02:00:56 UTC.
Its entire run is replaced by the completed clean scalar record. The
[activity](../../benchmarks/results/2026-10-10-cpu-primary-series-activity.json)
and [guardian](../../benchmarks/results/2026-10-10-cpu-primary-clean-guardian-v1.json)
records bind that substitution; the
[orchestration file](../../benchmarks/results/2026-10-10-cpu-primary-pairs-v1.json)
still contains the excluded first scalar entry. It must not be used as pair 1's
comparison baseline. No measurements or thresholds are edited to conceal it.

One original scalar matrix process remained idle with unchanged CPU time during
the primary series. Only orchestration/transport parents are held between work;
active numerical workers always finish normally. Native/check/test work uses
verified numerical-idle boundaries between runs, with frozen binaries and
qualified source/receipt identities checked before resumption. The between-run
holds are not part of request timing. They are disclosed rather than treated as
continuous-load evidence for this four-run comparison. The coordinator finished
and resumed the original scalar transport at
`2026-10-10T03:48:17.025759+00:00`; all native/test work had finished before then.
The [tiny RMS](2026-10-10-metal-rms.md),
[Qwen RMS](2026-10-10-qwen-metal-rms.md),
[embedding](2026-10-10-metal-embedding.md),
[tiny residual](2026-10-10-metal-residual.md) and
[Qwen residual](2026-10-10-qwen-metal-residual.md) reports retain the individual
idle-window records. The orchestration `finished_utc` fields include parent
holds after numerical work completed and are not native request durations.

## Results and decision

Each linked record contains all ten measured samples and the unchanged
three-warmup policy. TTFT and total request intervals are seconds.

| Run, in effective order | TTFT median [min, max] | Request median [min, max] | Decode median, steps/s | Prefill median, steps/s |
|---|---:|---:|---:|---:|
| [Pair 1 scalar, clean](../../benchmarks/results/2026-10-10-cpu-primary-pair-1-scalar-clean.json) | 40.22137 [40.03753, 40.67199] | 128.56179 [127.94370, 129.72492] | 1.43909 | 3.18239 |
| [Pair 1 tiled](../../benchmarks/results/2026-10-10-cpu-primary-pair-1-tiled.json) | 8.99348 [8.91733, 9.07835] | 61.70652 [61.57578, 62.14016] | 2.40989 | 14.23253 |
| [Pair 2 tiled](../../benchmarks/results/2026-10-10-cpu-primary-pair-2-tiled.json) | 9.02524 [8.90721, 9.24878] | 61.74968 [61.38452, 62.24287] | 2.40838 | 14.18247 |
| [Pair 2 scalar](../../benchmarks/results/2026-10-10-cpu-primary-pair-2-scalar.json) | 40.29282 [40.14719, 40.67347] | 128.91154 [128.46110, 130.78412] | 1.43501 | 3.17675 |

| Pair | TTFT reduction | TTFT speedup | Request reduction | Decode speedup |
|---|---:|---:|---:|---:|
| 1, scalar then tiled | 77.6400% | 4.47228× | 52.0024% | 1.67460× |
| 2, tiled then scalar | 77.6009% | 4.46446× | 52.0992% | 1.67830× |

Between-run median variation, `(higher-lower)/lower`, is 0.1776% scalar and
0.3531% tiled for TTFT, 0.2720%/0.0699% for request time, and 0.2844%/0.0630%
for decode rate. Both TTFT improvements exceed the declared 5% threshold and
observed between-run variation. Ten samples support median/min/max dispersion,
without a formal confidence interval or tail percentile claim. Decode rate
counts the 127 intervals after the first of 128 emitted fixed-token events.

Peak child physical charges in effective order are 2,043,643,920,
2,043,725,840, 2,043,660,304 and 2,043,709,456 bytes, all below 6 GiB.
Each run retains identical 1,976,131,072-byte weights, 50,331,648-byte allocated
KV and 10,436,608-byte scratch. The largest scalar/tiled peak difference is
81,920 bytes; the allocation layout did not change.

The [calculation record](../../benchmarks/results/2026-10-10-cpu-primary-analysis-v1.json)
binds every exact input/result and local complete checkpoint SHA-256, validates
the result schema and regenerates each native summary. Its
[executed source snapshot](../../benchmarks/results/2026-10-10-cpu-primary-analysis-v1.py.txt)
SHA-256 is `0c5c5a9b4e054bbb25dfbd54243964ffad627a208c99608360b849781aded78e`.
This is the historical calculation with its original local paths, rather than
a generic benchmark tool. The
[orchestrator snapshot](../../benchmarks/results/2026-10-10-cpu-primary-pairs-v1.py.txt)
is also retained exactly. Analysis pins the schema, current shared summary
helper, clean guardian, substitution record and orchestration identities.
It requires identical model/tokenizer/build/device/workload/precision/timing
identity, complete request state, startup AC source and low-power policy,
start/end warning reports, and memory limits before calculating the comparisons.
No model, native or test work is performed during report generation or review.

All four effective runs use qualified executable SHA-256
`3601c63ae83b0d9b20f8362bbda27781e4dd969d527918c072ccf270d2d4874d`
and qualification receipt SHA-256
`2ecfb27ef923973d327fa1be3c328d6acc85e10ab33ae55d6ca05a4c4d4bfbdc`.
The pinned Qwen model/tokenizer revision is
`7ae557604adf67be50417f59c2c2f167def9a775`; model SHA-256 is
`fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe`
and tokenizer SHA-256 is
`c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539`.
Record dispatch revisions, in effective order, are `0f122d7`, `d6baf81`,
`3de376d` and `e721bf6`; the numerical binary/source identities remain the same.

The host is Apple M1 Pro, MacBookPro18,3, 16 GiB, macOS 26.6.2 (25G83).
Each run records the actual AC source line and AC low-power mode disabled at
startup; `pmset` captures no recorded thermal/performance warning at start/end.
End-of-run power source, actual temperature, system memory pressure and energy
are unavailable. A warning-free report does not establish controlled thermal
conditions or energy efficiency.

The short workload alone does not establish absent regression in the other
seven cache/context workloads, application latency, full real-model capacity,
held-out quality or an external speed ranking. Retain the opt-in path for those
measurements, preserve scalar as default, and leave M2.5/M3 open.
Astra accepted the completed report without findings after independently
reproducing all 40 samples' summaries, pair arithmetic, variation and memory,
and checking identities, exact copies, source snapshots and clean substitution.

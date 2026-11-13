# Gemma console cost attribution

`toklet profile` exposes the existing selected-evaluation official Gemma 4 E2B
BF16/G2 profiler. It runs plain and profiled requests sequentially on one
authenticated owner, checks the frozen output/history/work counts and compares
all 262,144 selected logits bit for bit. Numerical kernels, the native ABI and
existing session defaults remain unchanged. The additive `new_g2_session` owner
method selects the already implemented last-row prefill configuration.

Build the explicit Mac Apple Silicon route with `cargo build --release
--features metal --bin toklet`, then run:

```sh
toklet profile --model /absolute/pinned/model \
  --workload /absolute/frozen-workload.json --evaluation 13 \
  --output /absolute/new-profile-directory
python3 tools/bench/gemma_profile_report.py /absolute/new-profile-directory \
  --json /absolute/new-report.json --markdown /absolute/new-report.md
```

Help is available on all builds; unsupported builds fail explicitly. The
command neither downloads a model nor overwrites an existing output directory.
Malformed options/descriptors return 2 before model access/output creation;
runtime, authentication, parity, cancellation or export failures return 1.
Partial files remain evidence, but only the final successful `manifest.json`
allows the offline consumer to accept a bundle.

The input uses the existing frozen O1 workload schema: at most 16 KiB of JSON,
8 KiB of ordinary typed basic-text conversation, exact conversation/rendered-token
hashes and nonempty expected output with explicit stop/state semantics. Capacity
512, prefill chunk 32, offset zero and request reset are fixed. The selected
ordinal must exist in the expected chunk/decode schedule. Other admitted
workloads are diagnostics; the primary qualification workload is input384,
output128, selecting first/last decode evaluations 13 and 139 separately.

The numerical session owns KV and scratch; the immutable owner retains the
weights and codec. Plain KV/scratch is dropped before profiled session creation.
The whole 6 GiB admission includes the owner reserve exactly once, the existing
64 MiB profiler reserve, 16 MiB caller allowance and 32 MiB helper allowance.
Two bounded selected-logit buffers remain in memory; full logit traces are not
exported. `profile.json` and `inventory.json` each have a 1 MiB limit; the final
manifest has a 64 KiB limit and binds their exact bytes/SHA-256 hashes.

The report validates current stable operation IDs and source geometry, all 349
projection tiles and two CPU gathers, complete/nonoverflow state and paired-clock
calibration. It groups the five exclusive host buckets by role/layer/source/shape
and reports completed dispatch/command/wait counts and logical copy bytes. GPU
encoder interval unions stay separate from overlapping host waits and nested
encoding intervals. Global and role unions are recomputed from intervals, never
added from per-operation unions. Copy bytes do not measure DRAM traffic; gaps
between attributed GPU samples do not establish physical idleness. Unattributed
host time remains explicit.

This captures one evaluation, not whole-request operator totals. Host generation
times include selected-logit witness work and are not acceptance timings.
Parity is checked by the producer; the offline consumer authenticates the bundle
and verifies its internal bindings without receiving the private full logits.
No optimization, ordinary summary-overhead qualification, default adoption,
quality, UI or iPhone claim follows from these diagnostics. The V2 summary TTFT
bound remains unqualified and summaries remain disabled/opt-in.

Both full384/128 plain/profile comparisons pass exact output/history/state,
all selected logit bits and complete349tile/2gather coverage. Fmt, strict CPU/Metal
Clippy,815 CPU/974 Metal tests, final28 CLI unit controls (2 ignored) and11 portable controls
per Python mode pass. The first private-selector integration failure and a root
optimized replay-wrapper failure remain recorded and are corrected.
Astra approves scoped functional/diagnostic retention with no findings.
[Review/commit/push/CI](../artifacts/gemma-cost-attribution-2026-11-13/publication-checks-1.json).

| Selected evaluation | Frame ms | Exclusive host ms | Host wait ms | GPU encoder union ms | Unattributed host ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| First decode,13 |226.079|161.259|134.909|52.048|64.813|
| Last decode,139 |235.896|151.546|127.566|51.906|84.337|

Each frame records349 dispatches,349 commands and349 host waits. The largest
single operation is the32tile output head (about17ms exclusive host); the
feed-forward gate/up roles each sum55tiles and about25–27ms exclusive host.
This suggests investigating bounded multi-tile submission next, with exact
numerical/state tests and complete-request timing before any retention decision.
Host/GPU columns overlap; their difference is not measured queue latency. Two
selected frames establish neither whole-request operator totals nor a speedup.
Declared combined admission is5,931,514,488B; observed guarded peaks are
4,801,925,944/4,802,024,176B, including the helper reserve, below6GiB.
[Final checks](../artifacts/gemma-cost-attribution-2026-11-13/checks-final-1.json),
[actual captures](../artifacts/gemma-cost-attribution-2026-11-13/actual-captures-1.json),
[offline cost replay](../artifacts/gemma-cost-attribution-2026-11-13/actual-report-replay-1.json).
[Frozen contract](../artifacts/gemma-cost-attribution-2026-11-13/implementation-contract-1.json),
[integration boundary fix](../artifacts/gemma-cost-attribution-2026-11-13/interface-adjustments-2.json).

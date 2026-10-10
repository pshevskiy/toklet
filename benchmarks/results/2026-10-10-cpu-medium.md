# Scalar Qwen medium workload, 10 October 2026

The frozen scalar engine completes the 512-input/128-output workload with
three warmups and ten measured requests. Every saved request finishes with
639 committed positions and one pending fixed output token. This adds the
medium row to the CPU matrix; M2.5 remains open for the remaining workloads
and application measurements.

The [result](2026-10-10-cpu-medium.json),
[complete measured states](2026-10-10-cpu-medium.json.samples.jsonl),
[analysis](2026-10-10-cpu-medium-analysis-v1.json) and its
[executed source](2026-10-10-cpu-medium-analysis-v1.py.txt) retain exact
identities, all ten raw/sample pairs, schema validation, input hashes,
prefix states, monotonic token events and independently reconstructed summaries.
The analysis reads only small saved records; it performs no new numerical
execution or model/binary hashing.

| Metric | Median | Minimum | Maximum |
|---|---:|---:|---:|
| Warm TTFT, seconds | 163.381240 | 162.868807 | 166.383823 |
| Completed request, seconds | 253.195493 | 252.250703 | 256.925113 |

Median decode rate is 1.413940 steps/s over the 127 intervals after the first
token event; median prefill rate is 3.133778 tokens/s. FP32 weights and KV,
empty cache, chunk 128 and capacity 2,048 are fixed. The controlled workload
uses the frozen teacher-forced IDs and ignores EOS. Timers cover synchronous
session evaluation through final projection and staging of the final pending
ID; tokenization, sampling, detokenization and JSON transport are excluded.
Visible text, formal confidence intervals, tail percentiles and held-out
quality are not measured. This record establishes no speed ranking.

Logical weights are 1,976,131,072 bytes, allocated KV is 50,331,648 bytes and
scratch is 10,436,608 bytes. Conservative admission is 2,172,348,800 bytes;
maximum measured own-child physical charge is 2,043,840,528 bytes, below
6 GiB. Shared physical allocations are counted once. System memory pressure
and energy remain unavailable.

Hardware is Apple M1 Pro, MacBookPro18,3, 16 GiB RAM, macOS 26.6.2 (25G83),
Rust 1.90.0 and ordinary optimized release settings. AC power is observed
before repetitions. The AC low-power configuration is queried at the end,
when it reports zero; this is not a start-time low-power measurement. End
battery and energy observations are unavailable. Start/end thermal reports
contain no recorded thermal or performance warning.

The dispatch revision is `8a5ebc8ba320d073cbb4d1f2db8cbded22e0760f`, with
a clean working tree at dispatch. The separately frozen scalar executable
SHA-256 remains
`e3debabbae7db9e5801bc9bfba177275444ebe1d9849eb3ad0f2dff487ac8dea`;
its numerical sources were built at `996f8e7`. Later opt-in tiled source
changes are not part of that executable. The
[executed Python driver](2026-10-10-cpu-medium-source-driver-v1.py.txt)
SHA-256 is `887436019c6f634eff60ba8a6ba73e22452d4550bed1115f4a75a46fa483a21a`;
the summary/footprint helper is
`911806d7254e905979217145b6e173ed1def09576ba3685b2c3c105176984b34`.
Pinned model/tokenizer revisions and hashes are retained in the result.

Only the orchestration parent was
[held](2026-10-10-cpu-qwen-silu-check-window-hold.json) ahead of the next
workload. The medium transport and numerical worker continued normally
through every request and exited naturally before the
[idle observation](2026-10-10-cpu-qwen-silu-check-window-idle.json) at
`2026-10-10T05:51:15.319572+00:00`. Qwen SiLU and loss-runner checks then ran
while orchestration remained held. The
[completion record](2026-10-10-cpu-qwen-silu-check-window-completed.json)
binds reviewed commits `706edb5` and `95733e4`, validates frozen binaries,
qualified sources, inputs, evidence and all local model assets, and records
resumption of only the parent at `2026-10-10T06:01:44.668055+00:00`.
No tests, builds, GPU work or model/binary hashing overlap the next timed
workload. The medium worker was never stopped. A continuous medium workload
does not establish the separate sustained-condition or application criteria.

Result and raw-state SHA-256 are
`a9f0e34d3d7f9a83486c27b6d17cd262975f922973466a82999a0894ca38aad5`
and `994e0147192d265059b26cda9268353d4f51f3800140e849300c9769f35ed050`.
The analysis binds the window records, source and producer as well.

Saved-data calculations, documentation links and whitespace checks apply to
this task; no engine code changes or new numerical tests are needed.
Astra accepted without findings after independently verifying exact copies,
historical identities, all states/events, summaries, memory and window ordering.

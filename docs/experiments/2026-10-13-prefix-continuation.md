# Exact-prefix conversation continuation

The session retains the longest exact token prefix already committed to KV.
The immutable borrowed model fixes weights, precision and position semantics;
each session owns its existing per-layer cache and scratch. Editing history
changes the valid length without copying or clearing KV. Future slots are
overwritten before being consumed and remain outside causal validity meanwhile.

Only the latest distribution is retained. A shortened entirely cached prompt
therefore retains all but its last token and replays that token to reconstruct
its distribution. A currently cached prompt with its valid distribution requires
no execution. Pending sampled tokens never count as reusable KV. Completion
failure still requires complete replay. Invalid input is rejected before changes.

The recovery token carries explicit provenance so cancellation and repeated
pending-token cleanup preserve it. Cleanup cannot repeatedly remove committed
positions, including after appending fresh input or replacing the same history.
This edge case and the collector's missing failure-path reap were Astra findings.

## Correctness and checks

Both scalar and tiled CPU paths compare full logits exactly with fresh model
execution at offsets 0 and 5 for first/middle/tail edits, shortening, full cache,
pending samples, empty history and recovery cleanup. The pinned Qwen CPU suites
each pass 47 exact comparisons; own-process lifetime maximum charge is
2,120,665,152 bytes scalar and 2,110,572,608 bytes tiled, including hash/setup.
The bounded libproc v4 diagnostic replaces an unbounded external sampler.

Native tiny FP16 and Q4 sessions cover the same edit/recovery sequences and
offsets against complete reset/replay, with unchanged numerical gates and exact
greedy IDs. Q4 tiny/Qwen/application, FP16 tiny/Qwen/capacity, and native
transaction/failure suites pass. Default and Metal ordinary tests, formatting,
strict all-target Clippy, and 143 experiment tests pass (84 default / 90 Metal
ordinary Rust tests). All numerical builds use
the isolated `artifacts/m5-checks` target; protected CPU release binaries remain
unchanged. Device tests and measurements run without overlapping builds or large
file hashing jobs.

## Measurement

The release worker shares one immutable Q4 Qwen 0.5B model between two separately
admitted sessions (capacity 1024, chunk 128, Half KV). Each pair first prepares
the frozen 512-token base outside the timer. Timed requests include explicit
reset or prefix selection, completed GPU work, full-logit readback, CPU greedy
selection, sampled-history append and first token event. It generates one token,
then compares complete distributions with the predeclared gate
`1e-4 + 1e-4 * abs(reset)`. Request order alternates. No tokenizer or visible-text
latency is claimed. Three warmups and ten measured samples cover each case.

The named changed-prefix fixture first differs at position 128 (384 tokens
replayed); the tail edit changes only position 511 (one token replayed). Base
prompt and changed prompt identities are frozen in the benchmark contract.
Combined numerical/metadata/application admission is 903,729,044 bytes, counting
shared weights once and both 64 MiB per-session framework allowances plus 64 MiB
application allowance. Native libproc charge counters can omit shared/clean
pages and complement allocation accounting.

Run v1 hit the collector's legacy 180-second lifetime after 15 pairs. Its partial
raw journal is retained and excluded from acceptance. The reviewed collector
uses a bounded 120-second event and 1200-second total policy; it reaps only its
owned worker, preserves original failures through cleanup, and saves exclusive
summary/raw/stderr files. Endpoint identities, compiler/build description,
OS/device and battery/power-policy/thermal observations are recorded. Thermal
endpoints do not measure continuous pressure.

| Case | Run v2 reset/reuse median | Run v3 reset/reuse median | Reduction v2/v3 |
| --- | --- | --- | --- |
| changed128 |3.075579 /2.411682 s |3.079011 /2.408257 s |21.586% /21.785% |
| tail edit |3.076046 /0.215859 s |3.078898 /0.215219 s |92.983% /93.010% |

All 52 paired distributions (including warmups) have zero measured absolute
difference; greedy IDs
match. Reset medians differ by at most 0.12% between runs and reuse medians by
at most 0.30%, well below both gains and the 5% retain threshold. Raw samples and
ranges remain in [run v2](../../benchmarks/results/2026-10-13-metal-continuation-q4-v2.json)
and [run v3](../../benchmarks/results/2026-10-13-metal-continuation-q4-v3.json).
Native lifetime maximum charge is 721,716,304 bytes in v2, unchanged after warmup;
v3 increases 65,536 bytes from 721,306,704 to 721,372,240 after its remaining 20
paired requests. Allocated session bytes remain exactly fixed. This bounds the
observed native drift without claiming allocation-free native frameworks.
Apple M1 Pro/macOS 26.6.2, AC 100%, AC low-power mode 0 and no recorded thermal or
performance warning at either endpoint are shared by both runs. The
[qualification record](../../benchmarks/results/2026-10-13-prefix-continuation-qualification-v1.json)
binds final sources, check logs, binaries and evidence. A preserved prior binary
allows later comparisons without rebuilding or changing the CPU baseline.

Retain the mechanism: both named workloads exceed the declared threshold in a
second run, with numerically identical distributions and fixed admitted buffers. Astra accepted the final code, checks and evidence; M5.5 is complete.

This qualifies the prefix mechanism only. Q4 held-out quality, separate prefill,
individual fusion/greedy experiments, sustained/external comparisons and larger
checkpoint admission remain separate M5 work.

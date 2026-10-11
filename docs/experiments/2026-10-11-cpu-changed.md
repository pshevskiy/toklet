# Scalar changed-prefix replay, 11 October 2026

The fresh scalar batch completed three warmups and ten measured requests without
interruption. Each prepares 512 tokens outside timing, then changes only token
128 of the prompt. The current reset-and-replay policy reuses zero tokens and
evaluates all 512 prompt tokens within timing. Each request emits 128 fixed
teacher-forced steps and ends with 639 committed positions and one pending token.
Capacity is 2,048 and chunk size 128.

| Measurement | Result |
| --- | ---: |
| Median numerical TTFT | 163.551571521 s |
| TTFT minimum / maximum | 162.97588425 / 166.233767625 s |
| Median completed request | 253.0541543125 s |
| Request minimum / maximum | 252.639919291 / 256.66743925 s |
| Median decode throughput | 1.4161741519307007 steps/s |
| Median prefill throughput | 3.1305110891159416 tokens/s |
| Maximum own-process physical charge | 2,043,791,376 bytes |
| Conservative admitted process estimate | 2,172,348,800 bytes |

The [result](../../benchmarks/results/2026-10-11-cpu-changed-uninterrupted-v1.json)
and [ten raw state/sample pairs](../../benchmarks/results/2026-10-11-cpu-changed-uninterrupted-v1.json.samples.jsonl)
retain every measured numerical event. Result SHA is
`ad4fa810bce6c54c9c07217c671a2e966d293a29c833e457ff807addae3c14fd`;
raw SHA is `e86014ddf5722e7703f8eee300c8ad0be27932db8ce56dfc249638266dc7ad5d`.
The independent [analysis](../../benchmarks/results/2026-10-11-cpu-changed-uninterrupted-analysis-v1.json)
and exact [producer](../../benchmarks/results/2026-10-11-cpu-changed-uninterrupted-analysis-v1.py.txt)
verify schema, all ten exact state/event pairs, fixed arrays, summaries, memory,
launch identity, natural exit and exclusion of the interrupted earlier batch.
The [observed stdout](../../benchmarks/results/2026-10-11-cpu-changed-uninterrupted-observed-stdout-v1.txt)
retains three warmup lines, ten measurements and the final summary. Raw warmup
states and timestamps are unavailable.

The user paused the earlier run after six completed measurements. Those samples
remain diagnostic and do not enter this result. Explicit continuation authorized
[retirement](../../benchmarks/results/2026-10-11-user-resume-retirement-v1.json.txt)
of its stopped worker, transport and obsolete guard; original orchestration
17694 remains held. The fresh [launch](../../benchmarks/results/2026-10-11-cpu-changed-uninterrupted-launch-v1.json)
at 06:18:28 UTC binds clean revision `746456f`, eight assets, the previous 108
source/artifact bindings and qualified candidate receipts. Its exact
[producer](../../benchmarks/results/2026-10-11-cpu-changed-uninterrupted-launch-v1.py.txt)
is retained. Fresh launcher 53099, transport 53106 and numerical worker 53114
all exit naturally before [idle verification](../../benchmarks/results/2026-10-11-cpu-changed-uninterrupted-idle-v1.json)
at 07:49:45 UTC, using the saved
[verifier](../../benchmarks/results/2026-10-11-cpu-changed-uninterrupted-idle-v1.py.txt).
No tests, builds, native workloads or model/binary/large-artifact hashes overlap
the numerical batch. Cold Python VM-control source preparation changes no
numerical source or frozen executable.

The frozen scalar executable has numerical source revision `996f8e7` and SHA
`e3debabbae7db9e5801bc9bfba177275444ebe1d9849eb3ad0f2dff487ac8dea`.
The actual [driver](../../benchmarks/results/2026-10-10-cpu-medium-source-driver-v1.py.txt)
SHA is `887436019c6f634eff60ba8a6ba73e22452d4550bed1115f4a75a46fa483a21a`.
Qwen2.5-0.5B-Instruct and tokenizer revision is
`7ae557604adf67be50417f59c2c2f167def9a775`; checkpoint SHA is
`fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe` and
tokenizer SHA is
`c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539`.
BF16 source weights convert once to resident FP32, with FP32 activations and KV.
Logical weights occupy 1,976,131,072 bytes, KV 50,331,648 bytes and scratch
10,436,608 bytes. Physical charge is below admission and the 6 GiB limit.
Loading takes 1.835885041 s, including nested conversion of 1.2158366309999964 s;
these times must not be added.

Hardware is Apple M1 Pro, MacBookPro18,3, 16 GiB, macOS 26.6.2 (25G83), Rust
1.90.0 and driver Python 3.12.8. The frozen release command is
`cargo build --locked --offline --release --example cpu_bench`. Launch reports
AC attached at 80%, not charging; AC low-power mode off was queried at the end.
Endpoint thermal reports contain no recorded thermal or performance warning. Continuous thermal
sampling, system memory pressure and energy are unavailable.

Timing excludes base-cache preparation, admission, bounded event allocation,
JSON transport, sampling, tokenization and text output. TTFT ends at the first
numerical event; visible-text latency is unset. Decode throughput uses 127
intervals between the first and last events. Wall time also includes thirteen
untimed base preparations. EOS is ignored for this fixed-length workload.
These measurements do not establish quality or a runtime speed ranking.

Exact analysis replay, local links and whitespace checks pass. Astra accepted
the complete task after correcting the power-query timing description and
independently checking saved copies, states/events, stdout, identities and pause
exclusion. Review executed no engine, build, VM or signals.

Application measurements, full held-out quality and M2.5/M3 acceptance remain
open. This task archives evidence and changes no engine code.

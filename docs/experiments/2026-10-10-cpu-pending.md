# Scalar Qwen pending-token continuation, 10 October 2026

The original scalar workload completed three warmups and ten measured requests
without intervention. Each prepares a fully evaluated 512-token prefix outside
timing, stages fixed ID 2077 as a pending-sample stand-in, then continues with
128 teacher-forced IDs. Before the request, history has 513 tokens and 512
committed positions. Every measured request reuses 512 positions, evaluates one
pending input within timing and ends with 640 committed positions, one pending
token and 128 emitted steps. Capacity is 2,048 and chunk size 128.

| Measurement | Result |
| --- | ---: |
| Median numerical TTFT | 0.702565104 s |
| TTFT minimum / maximum | 0.700897458 / 0.721566417 s |
| Median completed request | 90.500724792 s |
| Request minimum / maximum | 90.444751084 / 90.872934458 s |
| Median decode throughput | 1.4143951955885101 steps/s |
| Median pending-input prefill throughput | 1.4233560586623355 tokens/s |
| Maximum own-process physical charge | 2,043,791,376 bytes |
| Conservative admitted process estimate | 2,172,348,800 bytes |

The [result](../../benchmarks/results/2026-10-10-cpu-pending.json) and
[ten raw sample/state pairs](../../benchmarks/results/2026-10-10-cpu-pending.json.samples.jsonl)
retain all 128 token event timestamps per measured request. The
[analysis](../../benchmarks/results/2026-10-10-cpu-pending-analysis-v1.json)
reproduces the schema, exact raw states, independent summaries, fixed input
identities, memory and launch/hold/idle ordering with its exact
[producer](../../benchmarks/results/2026-10-10-cpu-pending-analysis-v1.py.txt).
Result SHA is `d200f0658d199708fc8c83b80331397c469e5645189707745604f3d3bc11d666`;
raw SHA is `6788d119eea5f260b911624f81100518a0d11ed0e408fd86937c707d98cbd66a`.
The result's `prompt_tokens = 512` and prompt SHA identify the frozen prepared
base. The contract separately pins ID 2077 and the initial 513-token history;
prefill throughput uses only the one pending token, rather than the base length.

The historical scalar executable has numerical source revision `996f8e7` and SHA
`e3debabbae7db9e5801bc9bfba177275444ebe1d9849eb3ad0f2dff487ac8dea`.
The record's clean launch revision is `3393962`; subsequent opt-in tiled code
does not describe that frozen executable. The actual driver SHA is
`887436019c6f634eff60ba8a6ba73e22452d4550bed1115f4a75a46fa483a21a`, with
its [saved source](../../benchmarks/results/2026-10-10-cpu-medium-source-driver-v1.py.txt).
This saved-data task changes no numerical source or protected release binary.

Qwen2.5-0.5B-Instruct and its tokenizer are pinned at revision
`7ae557604adf67be50417f59c2c2f167def9a775`. Checkpoint SHA is
`fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe`;
tokenizer SHA is
`c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539`.
BF16 source weights convert once to resident FP32; activations and KV are FP32.
Logical weights occupy 1,976,131,072 bytes, allocated KV 50,331,648 bytes and
scratch 10,436,608 bytes. Physical charge is below admission and the 6 GiB limit.
Model load is 1.88008525 s, including nested conversion of 1.1374397479999971 s;
these durations must not be added. Compilation and tokenization are outside this
saved numerical workload.

The [completed prior window](../../benchmarks/results/2026-10-10-cpu-task-cases-check-window-completed.json)
verified reviewed commits `e54d6f8` and `3393962`, frozen executables, eight model
assets and 102 source/artifact bindings before resuming only orchestration 17694
at 10:12:43 UTC. Only that parent was
[held ahead of changed-prefix timing](../../benchmarks/results/2026-10-10-cpu-pending-check-window-hold.json)
at 10:13:24 UTC. Transport 80030 and numerical worker 80039 completed all three
warmups and ten measured requests naturally. The
[idle proof](../../benchmarks/results/2026-10-10-cpu-code-isolation-check-window-idle.json)
at 11:08:49 UTC confirms the numerical worker absent and transport defunct under
the held parent; its [producer](../../benchmarks/results/2026-10-10-cpu-code-isolation-check-window-idle-v1.py.txt)
is retained. Tests, compilation and native isolation work began only after this
proof. Source preparation during the run performed no model/binary/large hashes.

The [observed stdout](../../benchmarks/results/2026-10-10-cpu-pending-observed-stdout-v1.txt)
retains three warmup lines, ten measured lines and the final summary, read from
the original session before resumption. Measured durations match saved samples
rounded to three decimals. Raw warmup states and timestamps are unavailable.

Hardware is Apple M1 Pro, MacBookPro18,3, 16 GiB, macOS 26.6.2 (25G83), with
Rust 1.90.0 and driver Python 3.12.8. The frozen release build uses
`cargo build --locked --offline --release --example cpu_bench`. Launch shows AC
attached at 80%, not charging; AC low-power mode off was queried at the end.
Endpoint thermal reports contain no recorded thermal or performance warning.
Continuous power/thermal sampling, energy and system memory pressure are unset.

Timing excludes cache preparation, admission, bounded event allocation, JSON
transport, tokenization, sampling and detokenization. TTFT ends at the first
numerical event; visible-text latency is unset. Decode rate is
`127 / (last token event - first token event)`. The full orchestration duration
also includes thirteen untimed 512-token preparations. EOS is ignored in this
fixed-length teacher-forced workload; ID 2077 is staged without greedy sampling.
These results do not measure generated-text quality or establish a speed ranking.

Local links and whitespace checks pass. Astra accepted the complete saved-data
task without findings after exact analysis replay and independent verification
of raw copies, all ten states/events, stdout, summaries, memory, source identities
and window ordering. Review performed no engine execution, tests, builds, large
hashes or signals.
Changed-prefix and application measurements, full quality and M2.5/M3 acceptance
remain open; no unrelated test suite or numerical benchmark is needed for this
evidence-only report.

The later [completed idle window](../../benchmarks/results/2026-10-10-cpu-code-isolation-check-window-completed.json)
verifies reviewed commits `2bec633` and `22dc0f4` at clean `22dc0f4`, the frozen
executables, eight model assets and 108 source/artifact bindings. All verification
finishes before resuming only orchestration at 11:21:29 UTC, using its retained
[source](../../benchmarks/results/2026-10-10-cpu-code-isolation-check-window-resume-v1.py.txt).
Changed-prefix transport 85019 and numerical worker 85027 then run uninterrupted;
only their parent is [held ahead of application timing](../../benchmarks/results/2026-10-10-cpu-changed-check-window-hold.json)
at 11:21:49 UTC. Its exact [hold source](../../benchmarks/results/2026-10-10-cpu-changed-check-window-hold.py.txt)
is retained. This follow-up adds launch evidence, not a changed-prefix result.
Astra accepted the archival follow-up without findings, verifying source copies,
the recorded producer hash, reviewed clean commits, asset/binding counts,
ordering, ancestry and parent-only signaling from small saved evidence.
No tests, builds, native/model work or binary/model/large-artifact hashes run
while that numerical worker is active.

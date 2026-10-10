# Scalar Qwen fully cached prompt, 10 October 2026

The original scalar workload completed three warmups and ten measured requests.
Each request starts after a fully evaluated 512-token prefix is prepared outside
timing, reuses its final distribution and emits 128 fixed teacher-forced IDs.
All ten measured states contain 512 reused tokens, zero prefill tokens, 639
committed positions and one pending token. Capacity is 2,048; prefill chunk size
is 128. The state reuse avoids evaluating the cached prompt within the request.

| Measurement | Result |
| --- | ---: |
| Median numerical TTFT | 0.792 microseconds |
| TTFT minimum / maximum | 0.583 / 1.000 microseconds |
| Median completed request | 90.0002074375 s |
| Request minimum / maximum | 89.680068084 / 91.457112 s |
| Median decode throughput | 1.4111083615746347 steps/s |
| Maximum own-process physical charge | 2,043,856,912 bytes |
| Conservative admitted process estimate | 2,172,348,800 bytes |

The sub-microsecond TTFT is access to an already computed distribution followed
by the numerical event timestamp. It excludes preparing the 512-token cache,
sampling, transport and visible text, and does not measure user-visible response
latency. Zero pending input still calls the timed empty `evaluate()` operation;
measured prefill durations range from zero to 125 nanoseconds. Prefill throughput
is null because its token numerator is zero. These timer samples do not qualify
the precision of sub-microsecond timing on this host.

The [result](../../benchmarks/results/2026-10-10-cpu-cached.json) and
[ten raw sample/state pairs](../../benchmarks/results/2026-10-10-cpu-cached.json.samples.jsonl)
retain all 128 event timestamps per measured request. The
[analysis](../../benchmarks/results/2026-10-10-cpu-cached-analysis-v1.json)
independently reproduces the schema, states, summaries, input identities, memory
and launch/hold/idle ordering using its exact
[producer](../../benchmarks/results/2026-10-10-cpu-cached-analysis-v2.py.txt).
Result SHA is `a78bee940fec0110a477551337ee1b1c32154afd1d06b6505dc8b7b0ca7e083e`;
raw SHA is `1a0193cad5e391890718b0c4e12deec7887202ebd10ec0c69ecbf6a97d003d55`.

The historical scalar executable has numerical source revision `996f8e7` and
SHA `e3debabbae7db9e5801bc9bfba177275444ebe1d9849eb3ad0f2dff487ac8dea`.
The record's clean launch revision is `f36ad90`; subsequent opt-in tiled code
does not describe that frozen executable. The original driver SHA is
`887436019c6f634eff60ba8a6ba73e22452d4550bed1115f4a75a46fa483a21a`, with
its [saved source](../../benchmarks/results/2026-10-10-cpu-medium-source-driver-v1.py.txt).
No numerical source or protected release binary changes in this evidence task.

Qwen2.5-0.5B-Instruct and its tokenizer are pinned at revision
`7ae557604adf67be50417f59c2c2f167def9a775`. Checkpoint SHA is
`fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe`;
tokenizer SHA is
`c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539`.
BF16 source weights convert once to resident FP32; activations and KV are FP32.
Logical weights occupy 1,976,131,072 bytes, allocated KV 50,331,648 bytes and
scratch 10,436,608 bytes. Physical charge is below admission and the 6 GiB limit.
Load time is 2.098833875 s, including nested conversion of 1.2125740930000024 s;
these durations must not be added. Compilation and tokenization are outside this
saved numerical workload.

The [completed prior window](../../benchmarks/results/2026-10-10-cpu-qwen-gating-check-window-completed.json)
verified both reviewed commits, frozen binaries and eight model assets before
resuming orchestration PID 17694 at 09:01:08 UTC. Only that parent was
[held ahead of pending](../../benchmarks/results/2026-10-10-cpu-cached-check-window-hold.json)
at 09:01:57 UTC. Transport 74583 and numerical worker 74591 completed all three
warmups and ten measurements naturally without intervention. The
[idle proof](../../benchmarks/results/2026-10-10-cpu-task-cases-check-window-idle.json)
at 09:58:13 UTC confirms the worker is absent and transport defunct beneath the
held parent; its [producer](../../benchmarks/results/2026-10-10-cpu-task-cases-check-window-idle-v2.py.txt)
is retained. Tests, compilation, native/model work and binary/model hashing began
only after this proof.

The [observed stdout](../../benchmarks/results/2026-10-10-cpu-cached-observed-stdout-v1.txt)
retains three warmup lines, ten measured lines and the final summary read from
the original session before resumption. The measured durations match saved
samples rounded to three decimals. This output does not contain raw warmup
states or token timestamps.

Hardware is Apple M1 Pro, MacBookPro18,3, 16 GiB, macOS 26.6.2 (25G83), with
Rust 1.90.0 and driver Python 3.12.8. The frozen release build uses
`cargo build --locked --offline --release --example cpu_bench`. Launch power
shows AC attached at 80%, not charging; AC low-power mode off was queried at
the end. Endpoint thermal reports contain no recorded thermal or performance
warning. Continuous power/thermal sampling, energy and system memory pressure
are unavailable.

Request timing excludes cache preparation, admission, event allocation, JSON
transport, tokenization, sampling and detokenization. Decode throughput is
`127 / (last token event - first token event)`. The full orchestration duration
also includes thirteen untimed cache preparations. EOS is ignored for the
fixed-length teacher-forced continuation, so this record supplies no generated
text quality result. It is one controlled cache workload without a speed ranking.
Pending-token, changed-prefix and application measurements, full quality and
M2.5/M3 acceptance remain open.

Local links and whitespace checks pass. Astra accepted the complete saved-data
task without findings after exact analysis replay and independent verification
of raw copies, all ten states, observed stdout, summaries, memory, historical
identities and window ordering. The review also confirmed the timed empty
evaluation in the historical source. No unrelated tests, builds, model work or
benchmarks are required for this evidence-only task.

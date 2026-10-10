# Scalar Qwen exact-prefix append, 10 October 2026

The original scalar workload completed three warmups and ten measured
requests. Astra accepted the completed saved-data task without findings after
independent replay. The frozen case prepares a fully evaluated
512-token prefix outside request timing, then appends 64 fixed IDs and emits
128 teacher-forced continuation steps. Every measured request finishes
with 512 reused tokens, 64 timed prefill tokens, 703 committed positions and
one pending token. Capacity is 2,048 and prefill chunk size is 128.

| Measurement | Result |
| --- | ---: |
| Median warm TTFT | 21.3480848955 s |
| TTFT minimum / maximum | 21.200767709 / 21.545358209 s |
| Median completed request | 111.6288904375 s |
| Request minimum / maximum | 111.347230375 / 113.082251125 s |
| Median decode throughput | 1.40630756730723 steps/s |
| Median appended prefill throughput | 2.99793248301749 tokens/s |
| Maximum own-process physical charge | 2,043,774,992 bytes |
| Conservative admitted process estimate | 2,172,348,800 bytes |

The [result](../../benchmarks/results/2026-10-10-cpu-append.json) and
[ten raw sample/state pairs](../../benchmarks/results/2026-10-10-cpu-append.json.samples.jsonl)
retain all 128 token timestamps per measured request. The
[analysis](../../benchmarks/results/2026-10-10-cpu-append-analysis-v1.json)
reproduces the schema, states, independent summaries, fixed input identities,
memory and launch/hold/idle ordering with its exact
[producer](../../benchmarks/results/2026-10-10-cpu-append-analysis-v1.py.txt).
Result SHA is `a20328d513870c82f681593603e42966e2f8306e1500fb82443c93292bd03753`;
raw SHA is `450ff84f7764b1f83489af298f3c83635f6acf35eb7875109efb7a777077094c`.

The historical scalar executable remains separate from subsequent opt-in
tiled code. It has numerical source revision `996f8e7` and SHA
`e3debabbae7db9e5801bc9bfba177275444ebe1d9849eb3ad0f2dff487ac8dea`.
This workload launched at clean revision `2bc36db` after the
[completed gating window](../../benchmarks/results/2026-10-10-cpu-gating-check-window-completed.json)
verified frozen binaries and all eight model assets. The original driver SHA
is `887436019c6f634eff60ba8a6ba73e22452d4550bed1115f4a75a46fa483a21a`,
with its exact [saved source](../../benchmarks/results/2026-10-10-cpu-medium-source-driver-v1.py.txt).
The record's engine revision describes the clean checkout at launch, while
the frozen executable's numerical source is the earlier revision above.
The bounded numerical implementation has not changed during this run.

The model and tokenizer are pinned Qwen2.5-0.5B-Instruct revision
`7ae557604adf67be50417f59c2c2f167def9a775`. Source checkpoint SHA is
`fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe`;
tokenizer SHA is
`c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539`.
The source BF16 weights are converted once to resident FP32; activations and
KV are FP32. Logical weights occupy 1,976,131,072 bytes, allocated KV
50,331,648 bytes and reusable scratch 10,436,608 bytes. Maximum physical
charge is below conservative admission and the 6 GiB working-set limit.
Model load is 2.020343167 s, including the nested 1.2237589559999953 s
conversion; those durations must not be added together. Compilation and
tokenization are outside this saved numerical run.

Only orchestration PID 17694 was
[held ahead of the next workload](../../benchmarks/results/2026-10-10-cpu-qwen-gating-check-window-hold.json).
Append transport 68696 and numerical worker 68705 completed the three warmups
and ten measured requests without intervention. The
[idle proof](../../benchmarks/results/2026-10-10-cpu-qwen-gating-check-window-idle.json)
at 08:49:00 UTC confirms the numerical worker has exited and transport is
defunct under the held parent. Tests, builds, GPU/model work and binary/model
hashing began only after this proof. No numerical worker was stopped.
The original driver's warmup policy and complete saved measured records are
retained; separate warmup stdout is not claimed.

The named host is Apple M1 Pro, MacBookPro18,3, 16 GiB, macOS 26.6.2
(25G83), Rust 1.90.0 and driver Python 3.12.8. The frozen release build uses
`cargo build --locked --offline --release --example cpu_bench`.
Launch records show AC attached at 80%, not charging. AC low-power mode off
was queried at the end, rather than attested at launch. Start/end thermal
reports contain no recorded thermal or performance warning; continuous
power/thermal sampling, energy and system memory pressure are unavailable.

Request timing excludes the 512-token cache preparation, admission, bounded
event allocation, JSON transport, tokenization, sampling and detokenization.
TTFT ends at the first numerical token event; visible-text latency is unset.
Decode throughput is `127 / (last token event - first token event)`.
Prefill throughput uses the 64 appended tokens, while the full sequence's
elapsed time also includes repeated untimed cache preparation. EOS is ignored
for this fixed-length teacher-forced workload; these outputs do not measure
greedy sampling or generated-text quality.

This task changes saved evidence and documentation only. Local link and
whitespace checks pass. No unrelated engine tests or benchmarks are required
for the report. Astra independently reproduced the retained analysis exactly
and verified raw copies, all ten cache states, 64-token prefill throughput,
timing summaries, memory, historical identities and window ordering. No
findings remain.

This is one controlled cache workload. Remaining fully cached, pending-token,
changed-prefix and application measurements, full quality and M2.5/M3 closure
remain pending. No numerical source or protected release binary is changed
by this saved-evidence task.

# Full small-model compensated Half workload matrix

All eight frozen controlled workloads complete with three warmups and ten measured
requests each on the fixed Qwen2.5-0.5B checkpoint. The unchanged qualified worker
uses HalfCompensated projections, Original normalization, FullLogits readback,
capacity 2,048 and chunk 128. Each request emits 128 teacher-forced IDs with EOS
ignored. There is no new numerical implementation, default or quality policy.

| Workload | Median TTFT, s | Median request, s | Request MAD, s | Median decode steps/s |
| --- | ---: | ---: | ---: | ---: |
| Short, 128 input | 0.484318 | 12.192120 | 0.003813 | 10.847458 |
| Medium, 512 | 2.359421 | 31.840999 | 0.022576 | 4.308656 |
| Long, 1,024 | 5.939004 | 59.266852 | 0.288322 | 2.380557 |
| Boundary, 1,920 | 15.210350 | 109.611879 | 0.022951 | 1.345305 |
| Exact append, 512 + 64 | 0.528379 | 32.838057 | 0.006477 | 3.930478 |
| Fully cached, 512 | 0.0000006875 | 29.379901 | 0.002780 | 4.322683 |
| Pending token, 512 + 1 | 0.208863 | 29.636882 | 0.006743 | 4.315615 |
| Changed prefix, reset/replay 512 | 2.350358 | 31.715965 | 0.006145 | 4.324617 |

Timers cover completed GPU evaluation, full final-logits readback and request
state handling. Token timestamps precede staging the next fixed ID. Decode rates
count 127 intervals. Sampling, tokenization and text are outside this controlled
track; there is no first-visible-text result. Cached prefill throughput is unset.
Prefix preparation is outside timing. Changed-prefix reset/replay is inside timing.
The cached TTFT is near the timer floor and does not support a percentage ranking.

Every request has the complete expected token-history SHA and committed/pending
counts. The boundary worker additionally commits all 2,048 positions and rejects
an extra token while preserving that history. All eight workers exit 0 with empty
cleanup errors; their owned transport finishes without incomplete events. The
collector exits 0 and both endpoint identity inventories match. There are 187
raw events, 104 completed requests, 80 measured samples and 24 warmups.

The ten measured boundary requests total 1,096.033071 seconds, exceeding ten
minutes in one repeated reset/request sequence. No manual holds, builds, tests,
other owned model/VM runs or large hashes overlap its numerical requests. This
qualifies that sustained workload duration. The ten measured long requests total
597.909383 seconds, just below ten minutes; they do not independently meet that
measured-duration threshold. Recorded start/end conditions are AC attached,
AC low-power mode zero and no reported thermal/performance warning. Endpoint
queries do not establish continuous temperature, clocks, energy or system pressure.

Maximum complete admission is 1,262,602,960 bytes, including 988,069,596-byte
weights, allocated KV/scratch/history/logits/metadata, framework allowance and
separate 64 MiB application allowance. Maximum sampled worker-plus-driver native
lifetime charge is 1,166,575,656 bytes. Both fit 6 GiB. These native charge counters
are not a complete shared physical allocation measurement. Persistent inventories
match at every phase; no memory advantage over an external runtime is inferred.

The [full collection](../../benchmarks/results/2026-10-14-small-half-controlled-full-v1.json),
[raw events](../../benchmarks/results/2026-10-14-small-half-controlled-full-v1.raw.jsonl),
[stderr](../../benchmarks/results/2026-10-14-small-half-controlled-full-v1.stderr),
[independent analysis](../../benchmarks/results/2026-10-14-small-half-controlled-full-analysis-v2.json)
and [exact analyzer](../../benchmarks/results/2026-10-14-half-controlled-full-analysis-v1.py.txt)
are byte-identical copies of retained originals. Replay verifies every raw response,
request/phase, warmup/sample label, state, timing, summary and complete memory
observation, using the pinned [worker qualification](../../benchmarks/results/2026-10-14-metal-controlled-qualification-v1.json).
The analyzer hashes the exact bounded receipt/summary/raw bytes it parses and
rechecks their paths. Six meaningful provenance, command, event, memory and
repetition mutations reject. An initial analysis output mixed a contract diagnostic
with JSON; that ignored v1 output is retained and excluded. The corrected v2 sends
diagnostics to stderr and parses successfully. Native data is unchanged.

Dispatch parent is `91d5758` with a clean tree. Compiled worker SHA-256 remains
`31434813900abe5f2fd5cd9ed80d593bde95185cea06c2a3fddabe67f1278396`;
its qualified snapshot identifies compilation separately from dispatch. Hardware
is Apple M1 Pro / MacBookPro18,3 / 16 GiB, macOS 26.6.2 (25G83), Rust 1.90.0.
Exact fixed checkpoint/tokenizer/contract identities and build flags are retained.
The [qualification receipt](../../benchmarks/results/2026-10-14-small-half-controlled-full-qualification-v1.json)
binds the completed data, source, checks and unchanged protected CPU binaries.

This is full small-model measurement and sustained-boundary evidence. Larger full
controlled/application runs, confirming comparisons and the release decision remain
open. Both selected Half configurations separately pass the unchanged frozen loss
and task gates; both Q4 v1 full-loss failures still block the explicit M5 exit.
No external speed ranking, default change or completed M5 claim follows here.

After successful collector exit, automatic approval review initially rejected the
read-only process-inventory request because the account usage limit was reached.
It was a review failure, not a determination that the action was unsafe; that first
action was not executed. The user's retry restores approval review. Fresh v2
observations find no controlled worker/collector commands and show the machine
stopped/not-starting. The collector did not retain historical worker PID/group IDs;
this is current command absence, not reconstructed individual PID/group proof.
The failed attempt remains recorded separately. Saved-data, content/link and
whitespace checks pass. Astra accepts the full small-model evidence after exact
published replay, all 335 bindings, row arithmetic, mutation checks and saved
post-retry observations, with no actionable findings. Acceptance is limited to
the full small-model measurements and sustained boundary duration.

For a sequential native rerun, use a fresh output and the qualified worker:

```sh
.venv/bin/python tools/experiments/metal_controlled.py --binary artifacts/m5-controlled-native-binary-v1/metal_bench --model small --output artifacts/half-small-full-rerun.json
```

Replay while native workers are idle; the archived analyzer also works from its
published location. The original audit source is retained in `artifacts/`.

```sh
.venv/bin/python benchmarks/results/2026-10-14-half-controlled-full-analysis-v1.py.txt artifacts/half-small-full-rerun.json > artifacts/half-small-full-rerun-analysis.json
```

# Qualified tiled CPU long workload, 11 October 2026

The opt-in ordered FP64 tile completes the frozen 1,024-input/128-output
workload with three warmups and ten measured requests. All saved requests
start from an empty session, commit 1,151 positions and leave one fixed output
token pending. This adds the long candidate row; scalar stays default and
M3.1 remains open for the remaining five non-short rows, full application
trace and explicit retention decision.

The [result](../../benchmarks/results/2026-10-11-cpu-tiled-long-v2.json),
[raw measured states](../../benchmarks/results/2026-10-11-cpu-tiled-long-v2.json.samples.jsonl),
[analysis](../../benchmarks/results/2026-10-11-cpu-tiled-long-v2-analysis.json)
and [exact analysis source](../../benchmarks/results/2026-10-11-cpu-tiled-long-v2-analysis.py.txt)
retain schema validation, all ten raw/sample pairs, token-event chronology,
committed/pending state, independently reconstructed summaries and source
bindings. The analysis launches no numerical workload and reads no model or
binary. The [launch](../../benchmarks/results/2026-10-11-cpu-tiled-long-v2-launch.json),
[owned-process observation](../../benchmarks/results/2026-10-11-cpu-tiled-long-v2-owned.json),
[actual tool completion](../../benchmarks/results/2026-10-11-cpu-tiled-long-v2-session-completion.json)
and [post-exit idle proof](../../benchmarks/results/2026-10-11-cpu-tiled-long-v2-idle.json)
bind this fresh run to its exact executed sources.

| Metric | Median | Minimum | Maximum |
| --- | ---: | ---: | ---: |
| Warm TTFT, seconds | 85.683956 | 85.406178 | 86.432885 |
| Completed request, seconds | 142.368874 | 142.084439 | 143.227613 |

Median decode is 2.239058 steps/s over the 127 intervals following the
first fixed-token event; median prefill is 11.950897 tokens/s. The
[historical scalar long](2026-10-10-cpu-long.md)
has 336.469968-second TTFT and 428.750185-second request medians. This run's
descriptive reductions are 74.5344% and 66.7944%. These are
separate historical runs, not an alternating same-executable comparison;
the [paired primary experiment](2026-10-10-cpu-primary-comparison.md) supplies
the confirming primary-metric acceptance evidence.

Pinned prompt/continuation IDs, FP32 weight/KV storage, capacity 2,048 and
prefill chunk 128 are unchanged. The ordered tile preserves FP64 projection
accumulation and final FP32 rounding. It does not change attention precision,
weights, KV ownership, sampling or output policy. Timers cover synchronous
session evaluation through final projection and staging of the final pending
ID, excluding admission, event allocation, JSON transport, tokenization,
sampling and detokenization. Visible text, energy, system memory pressure,
formal confidence intervals and reliable tail percentiles remain unavailable
for this numerical row. The separate frozen loss/task comparisons already
pass unchanged gates; this workload is not a quality evaluation.

Logical weights are 1,976,131,072 bytes, allocated KV 50,331,648 bytes and
scratch 10,436,608 bytes. Conservative admission is 2,172,348,800 bytes;
maximum measured own-child physical charge is 2,043,840,528 bytes, below 6 GiB.
Physical peaks describe the numerical child, not complete system pressure.
Sources are verified before execution, warming file cache; startup is not
cold-storage qualification. Loading and conversion are separately retained
in the result, and conversion overlaps model loading.

Hardware is Apple M1 Pro / MacBookPro18,3 / 16 GiB / macOS 26.6.2 (25G83),
Rust 1.90.0 with recorded optimized release build commands. Dispatch revision
is `fabc6ceba2f2bd7bbfbc5f6a7adc39756b6cb29f`, with a clean tree.
Qualified isolated executable SHA-256 is
`3601c63ae83b0d9b20f8362bbda27781e4dd969d527918c072ccf270d2d4874d`;
the [qualification](../../benchmarks/results/2026-10-10-cpu-tiled-qualification-v1.json)
retains its build base and committed numerical source hashes. Later dispatch
commits do not identify a new executable. The exact CPU driver remains
`887436019c6f634eff60ba8a6ba73e22452d4550bed1115f4a75a46fa483a21a`.
The result retains pinned model/tokenizer hashes and power/thermal observations.
AC attachment is observed before repetitions. AC low-power mode reports zero
in the configuration queried at the end; no start-time low-power measurement
is claimed. Start/end thermal reports contain no recorded warning. Continuous
temperature, system pressure, end battery and energy observations are unavailable.

The [preceding accepted medium run](2026-10-11-cpu-tiled-medium.md) finished
and reached owned idle before this launcher could hash model/binary data or
start numerical work. Its exact completion, owned PIDs/group and accepted
analysis are admitted alongside prior task collection/grading proofs. The
launcher requires repository `.venv` Python 3.12.8 and pinned `jsonschema`
4.25.1 before dispatch, and
created an exclusive one-off claim before model/binary hashing. Only the
owned launcher/driver/worker ran; no other model workload, VM grading,
tests, builds or large hashes overlapped numerical requests. Bounded process
and progress observations and inactive source preparation/review continued.
Original orchestration PID 17694 remained held. Owned PIDs 42842/42858/42866
and their shared group 42842 exited naturally before endpoint protection
checks and archival hashing; Podman remained off. Launch is observed at `2026-10-11T18:10:41.720357+00:00`; actual execution-tool
completion is observed at `2026-10-11T18:42:37+00:00` and owned idle is verified
at `2026-10-11T18:42:53.863700+00:00`.

The ten measured requests total 1424.459786 seconds, forming one
uninterrupted repeated-request sequence with resets. Any sustained claim is
limited to this run's recorded power and endpoint thermal conditions.
Saved-data calculation, documentation link/content and whitespace checks
apply; no engine/kernel code, protected artifact or numerical helper changes
and no new test suite are needed for this evidence task. Astra independently replayed byte-identical analysis and accepted the actual
evidence task without findings. All twelve report links, combined local
document links and whitespace checks pass.

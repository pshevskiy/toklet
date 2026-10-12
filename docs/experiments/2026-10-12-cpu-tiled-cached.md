# Qualified tiled CPU cached workload, 12 October 2026

The opt-in ordered FP64 tile reuses a fully prepared 512-token prompt with
zero prompt tokens to evaluate. It times the empty evaluation call and uses
the retained next-token distribution for the first fixed event. After three
warmups, all ten measured requests end at 639 committed positions with one pending
ID after 128 frozen continuation events. Preparation is outside request timing.
Scalar remains default; two cache rows, full application and retention remain.

The [result](../../benchmarks/results/2026-10-12-cpu-tiled-cached-v3.json),
[raw measured states](../../benchmarks/results/2026-10-12-cpu-tiled-cached-v3.json.samples.jsonl),
[analysis](../../benchmarks/results/2026-10-12-cpu-tiled-cached-v3-analysis.json)
and [exact analysis source](../../benchmarks/results/2026-10-12-cpu-tiled-cached-v3-analysis.py.txt)
retain schema validation, ten exact raw/sample pairs, 128-event chronology,
committed/pending state and independently reconstructed summaries. The
saved-data analyzer launches no numerical workload and reads no model/binary.
The [launch](../../benchmarks/results/2026-10-12-cpu-tiled-cached-v3-launch.json),
[owned-process observation](../../benchmarks/results/2026-10-12-cpu-tiled-cached-v3-owned.json),
[actual tool completion](../../benchmarks/results/2026-10-12-cpu-tiled-cached-v3-session-completion.json)
and [post-exit idle proof](../../benchmarks/results/2026-10-12-cpu-tiled-cached-v3-idle.json)
bind this fresh run to exact executed sources.

| Metric | Median | Minimum | Maximum |
| --- | ---: | ---: | ---: |
| Warm TTFT, seconds | 0.0000008545 | 0.0000006660 | 0.0000010420 |
| Completed request, seconds | 54.425495 | 54.339616 | 55.054898 |

Median decode is 2.333468 steps/s over 127 intervals after the first event.
Prefill throughput is null because zero prompt tokens are evaluated; the empty
evaluation duration may be zero at clock resolution. The
[historical scalar cached workload](2026-10-10-cpu-cached.md)
has 0.0000007920-second TTFT and 90.000207-second request medians.
The descriptive request reduction is 39.5274%. Cached first-event
latencies are near the timer floor; their tiny absolute differences do not
support a percentage speed ranking. These are separate historical runs,
not an alternating comparison; the
[paired primary experiment](2026-10-10-cpu-primary-comparison.md) supplies
confirming primary-metric acceptance evidence.

Pinned prompt/continuation IDs, FP32 weight/KV storage, capacity 2,048 and chunk
128 remain unchanged. The ordered tile preserves FP64 projection accumulation
and final FP32 rounding. Timers cover synchronous session evaluation through
final projection and staging the final pending fixed ID, excluding admission,
cache preparation, event allocation, JSON transport, tokenization, sampling
and detokenization. Visible text, energy, system pressure, formal confidence
intervals and reliable tail percentiles remain unavailable. Frozen loss/task
gates already pass; this numerical workload is not a quality evaluation.

Logical weights are 1,976,131,072 bytes, KV 50,331,648 bytes and scratch
10,436,608 bytes. Conservative admission is 2,172,348,800 bytes; peak
own-child physical charge is 2,043,774,992 bytes, below 6 GiB. This does not
measure complete system memory pressure. Source hashing warms file cache;
startup is not cold-storage qualification. Loading/conversion are separately
retained, with conversion overlapping model loading.

Hardware is Apple M1 Pro / MacBookPro18,3 / 16 GiB / macOS 26.6.2 (25G83), Rust
1.90.0 with recorded optimized release build commands. Dispatch revision is
`79490da0b7b9a3d49496fb4d32b090196cd81810`, with a clean tree. Qualified isolated
executable SHA-256 is
`3601c63ae83b0d9b20f8362bbda27781e4dd969d527918c072ccf270d2d4874d`;
the [qualification](../../benchmarks/results/2026-10-10-cpu-tiled-qualification-v1.json)
retains its build base and compiled source hashes. Dispatch does not identify
a newly compiled executable. Exact driver SHA-256 is
`887436019c6f634eff60ba8a6ba73e22452d4550bed1115f4a75a46fa483a21a`.
Pinned model/tokenizer identities are retained. AC was attached before execution. The AC low-power policy was queried at the end and was zero; this does not establish its start setting. Endpoint reports recorded no thermal or performance warning. Continuous temperature, energy, system pressure and end battery observations are unavailable.

The [preceding accepted append run](2026-10-12-cpu-tiled-append.md) reached
verified owned idle before this launcher could hash model/binary data or start
numerical work. Exact completion, owned identities and accepted analysis are
admitted alongside prior collection/grading proofs. The launcher requires
repository .venv Python 3.12.8 and jsonschema 4.25.1 before dispatch and makes a
fresh exclusive reservation before heavy hashes. The
[paused v2 attempt](2026-10-12-cpu-tiled-cached-excluded.md) remains wholly
excluded; none of its five rows is reused. Its native exit status/time and
disappearance cause are unknown. The fresh launcher explicitly requires its
[current recovery proof](../../benchmarks/results/2026-10-12-cpu-tiled-cached-v2-excluded-idle.json),
all eight recorded old PIDs absent, both old groups empty, committed archive
bindings and Podman stopped before model/binary hashing. No other model/VM/test/build
or large-hash work overlaps numerical requests. Bounded observations and
inactive source preparation/review continue. The original orchestration graph
is absent, as admitted by the reviewed exclusion recovery proof.
Owned PIDs 29542/29577/29597 and shared PG 29542 exited naturally before
endpoint protection and archival hashes. Actual successful tool completion was observed at 2026-10-12T08:13:03.597935+00:00; owned idle was verified at 2026-10-12T08:13:07.670103+00:00. These are observation times, not a retained exact exit timestamp. Podman remained stopped.

The ten measured requests total 545.308331 seconds, one repeated-request
sequence with preparation/reset outside each request timer. The recorded
power and endpoint thermal observations apply only to this sequence. Saved-data,
link/content and whitespace checks apply; no engine/kernel code, protected
artifact or numerical helper changes and no new test suite are needed for
this evidence task. Astra accepted the complete evidence task without findings after byte-identical independent replay, all ten request states, source/recovery bindings, timing/memory/chronology checks, four exact raw copies and all fourteen report links.

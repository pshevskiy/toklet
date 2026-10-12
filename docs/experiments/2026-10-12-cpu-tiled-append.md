# Qualified tiled CPU append workload, 12 October 2026

After three warmups, the opt-in ordered FP64 tile reuses a prepared 512-token prefix, evaluates
64 appended tokens and emits 128 frozen continuation IDs. All ten measured
requests end with 703 committed positions and one pending ID. Preparation
is outside request timing. This adds the append candidate row; scalar remains
default and M3.1 stays open for three remaining cache rows, full application
trace and an explicit retention decision.

The [result](../../benchmarks/results/2026-10-11-cpu-tiled-append-v2.json),
[raw measured states](../../benchmarks/results/2026-10-11-cpu-tiled-append-v2.json.samples.jsonl),
[analysis](../../benchmarks/results/2026-10-11-cpu-tiled-append-v2-analysis.json)
and [exact analysis source](../../benchmarks/results/2026-10-11-cpu-tiled-append-v2-analysis.py.txt)
retain schema validation, ten exact raw/sample pairs, 128-event chronology,
committed/pending state and independently reconstructed summaries. The
saved-data analyzer launches no numerical workload and reads no model/binary.
The [launch](../../benchmarks/results/2026-10-11-cpu-tiled-append-v2-launch.json),
[owned-process observation](../../benchmarks/results/2026-10-11-cpu-tiled-append-v2-owned.json),
[actual tool completion](../../benchmarks/results/2026-10-11-cpu-tiled-append-v2-session-completion.json)
and [post-exit idle proof](../../benchmarks/results/2026-10-11-cpu-tiled-append-v2-idle.json)
bind this fresh run to exact executed sources.

| Metric | Median | Minimum | Maximum |
| --- | ---: | ---: | ---: |
| Warm TTFT, seconds | 5.839271 | 5.621927 | 5.950998 |
| Completed request, seconds | 63.166372 | 60.247734 | 64.368049 |

Median decode is 2.214753 steps/s over 127 intervals after the first event;
median prefill is 10.960313 tokens/s over the 64 appended tokens. The
[historical scalar append](../../docs/experiments/2026-10-10-cpu-append.md)
has 21.348085-second TTFT and 111.628890-second request medians.
Descriptive reductions are 72.6473% and 43.4140%.
These are separate historical runs, not an alternating comparison; the
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
own-child physical charge is 2,043,840,528 bytes, below 6 GiB. This does not
measure complete system memory pressure. Source hashing warms file cache;
startup is not cold-storage qualification. Loading/conversion are separately
retained, with conversion overlapping model loading.

Hardware is Apple M1 Pro / MacBookPro18,3 / 16 GiB / macOS 26.6.2 (25G83), Rust
1.90.0 with recorded optimized release build commands. Dispatch revision is
`32d1ab327109e6a51040b490e56bd4fa417ea2d6`, with a clean tree. Qualified isolated
executable SHA-256 is
`3601c63ae83b0d9b20f8362bbda27781e4dd969d527918c072ccf270d2d4874d`;
the [qualification](../../benchmarks/results/2026-10-10-cpu-tiled-qualification-v1.json)
retains its build base and compiled source hashes. Dispatch does not identify
a newly compiled executable. Exact driver SHA-256 is
`887436019c6f634eff60ba8a6ba73e22452d4550bed1115f4a75a46fa483a21a`.
Pinned model/tokenizer identities are retained. AC was attached before execution. The AC low-power policy was queried at the end and was zero; its start setting is not established. Endpoint reports recorded no thermal or performance warning. Continuous temperature, energy, system pressure and end battery observations are unavailable.

The [preceding accepted boundary run](2026-10-11-cpu-tiled-boundary.md) reached
verified owned idle before this launcher could hash model/binary data or start
numerical work. Exact completion, owned identities and accepted analysis are
admitted alongside prior collection/grading proofs. The launcher requires
repository .venv Python 3.12.8 and jsonschema 4.25.1 before dispatch and makes a
fresh exclusive reservation before heavy hashes. No other model/VM/test/build
or large-hash work overlaps numerical requests. Bounded observations and
inactive source preparation/review continue; original PID 17694 remains held.
Owned PIDs 93877/93893/93902 and shared PG 93877 exited naturally before
endpoint protection and archival hashes. Actual successful tool completion was observed at 2026-10-12T05:35:10.713908+00:00; owned idle was verified at 2026-10-12T05:35:14.940343+00:00. These are observation times, not a retained exact process-exit timestamp. Podman remained stopped.

The ten measured requests total 629.436638 seconds, one repeated-request
sequence with preparation/reset outside each request timer. Sustained claims
are limited to recorded power and endpoint thermal conditions. Saved-data,
link/content and whitespace checks apply; no engine/kernel code, protected
artifact or numerical helper changes and no new test suite are needed for
this evidence task. Astra accepted the complete evidence task without findings after byte-identical independent replay, all ten request states, source/archive bindings, timing/memory/chronology checks and all twelve report links.

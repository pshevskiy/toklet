# O2.B1: Gemma numerical workload coverage

7 November 2026. Measurements, checks and scoped Astra review pass. This extends the
[O2.A warm Chat report](o2-gemma-cost-report.md) with a separate raw-token track;
it does not change production inference or complete O2/UI observability.

`tools/experiments/o2_gemma_workloads.py` compiles the strict Swift 6 release
diagnostic against the pinned native46 Mac archive, without downloads. It refuses
an existing output directory, pins inputs, bounds raw capture to 64 MiB per
process and 128 emitted IDs per request, and rejects incomplete or inconsistent
measurements. These explicit benchmark captures may contain token arrays;
routine request summaries retain their separate bounds and privacy contract.

The official audited Gemma 4 E2B BF16/G2 route keeps capacity 512, chunk 32,
FP32 KV, FullLogits and the 6 GiB memory ceiling. The audited model loader
authenticates the pinned six-file source/codec identity. This is the reference
route; uncensored MLX conversion and adoption remain separate quality work.

The frozen primary prompt is the 15-token encoding of “Name the capital of
France.” Fixed four-step generation uses no implicit stop and must begin with
`[818,5279,529,7001]`. It is deliberately distinct from Chat's eight visible
tokens plus hidden stop. Comparisons never mix the two workloads.

| Workload | Requested work and accounting |
| --- | --- |
| Reset | Full 15-row prompt, four emitted IDs; 0 reused rows |
| Exact append | Commit first 14 input IDs, append final ID; 14 reused/1 prefill row |
| Fully cached | Commit all 15 input IDs, request identical prompt; 15 reused/0 prefill rows; no prefill rate |
| Pending | Sample first ID, then continue for three IDs; 15 reused/1 prefill row, same full four-ID result |
| Edited prefix | Change the last prompt ID, compare with reset/replay of that edited input; Gemma's current owner replays all 15 rows (0 reused), rather than rewinding ring/shared state |
| Capacity and interruption controls | Reject 385+128 without mutation; pre-cancel without mutation; callback Stop preserves one pending ID; hidden stop samples one and emits none |
| Boundary | 384 input IDs plus 128 fixed output steps; exact legacy/observed output/state parity; no performance qualification from this single pair |
| New process | Three warmups plus ten measured processes per batch, two batches; audited load, session creation, codec setup and launch-to-readiness receipt |
| Sustained | Three warmups, then at least ten minutes of repeated four-step resets; duration, active duty, memory and thermal observations; no p95/p99 claim |

Continuation cases use three warmups and ten matched legacy/observed pairs in
each of two batches, alternating order. Setup requests lie outside each measured
request. Both modes collect the same bounded callback IDs/timestamps; only the
observed call copies the existing native phase report. This is a diagnostic
observed-ABI comparison, not a new summary-mode overhead qualification.

All host stamps are process monotonic nanoseconds. Native Report phase durations
are nested inside the timed C call; they are not added to caller durations.
Callback decode rate is `(N-1)/(last-first)`. No emitted event means no TTFT;
zero prefill rows mean no prefill rate. Raw records retain authoritative before/
after committed/pending state. Error metrics remain unavailable.

New-process readiness has two instruments: child entry-to-prepared on the Swift
clock, and parent process-launch-to-first-ready-record receipt on the parent
clock. Neither subtracts timestamps from the other clock. The receipt includes
startup, pipe delivery and scheduling. Model loading includes authentication,
conversion and pipeline preparation where the implementation does them; those
components are not separately measured. Filesystem/driver cache temperature is
uncontrolled, so these are not cold-storage results.

The next O2.B2 task must correlate UI publication and qualified visible-frame observations with
request IDs. The present Friendly Chat store publishes completed response text
only after the terminal worker result. Native callbacks do not establish first
visible text or first final-answer visibility; those fields remain unavailable.

The [result receipt](../benchmarks/results/2026-11-07-o2b-gemma-workloads-v1.json)
links all raw process captures, source pins and checks. Both independent batches
pass exact output/state parity and authoritative reuse/prefill accounting.

| Batch | New-process load median | Session creation median | Reset prefill median | Append prefill median |
| --- | ---: | ---: | ---: | ---: |
| 1 | 50.819 s | 24.995 ms | 397.292 ms | 227.242 ms |
| 2 | 50.821 s | 25.512 ms | 396.715 ms | 225.494 ms |

Fully cached requests evaluate no prefill rows; edited committed prefixes replay
all 15. These are different work/cache states, not optimization gains. The
384+128 boundary pair reaches history 512/committed 511/pending 1 with exact parity.
The sustained run completes 601.060 seconds and 554 measured requests, with 99.99%
active caller duty, 1078.661 ms median request time, nominal thermal readings and
4,922,413,304-byte lifetime peak. The maximum across all captures is 4,923,953,400
bytes. These are child-process lifetime peaks, not per-request or whole-app peaks.

Strict Swift 6 compilation and six new normal/optimized Python tests pass. The
exact named CI commands pass 14 offline + 9 measurement-contract tests locally;
hosted CI execution is not claimed. Required Rust/native/production-Swift checks
are reused after verifying all 352 native and 42 Swift source pins unchanged.
[Combined evidence](../artifacts/o2b-gemma-workloads-2026-11-07/checks.json).

The first interrupted cohort exposed an incorrect reporter assumption about
Gemma rewind; its output/state parity passed, and the existing owner source
confirmed full replay. A wholly fresh run supersedes that cohort and potentially
CPU-contended old load trials. After the fresh run, offline checks were tightened
for setup-state continuity and sampled-capacity consistency. The exact collecting
runner is retained with its original hash; all 17 measured inputs match, and
stricter recomputation produces identical reports. No numerical, state, quality
or performance threshold was loosened. See the checks receipt for all failures,
source-equivalence evidence and the report-only diff.


[Astra approves this separate numerical milestone](../artifacts/o2b-gemma-workloads-2026-11-07/astra-review.md)
with no actionable findings. UI timing, O3 attribution, model adoption and
performance qualification remain pending under the full active goal.

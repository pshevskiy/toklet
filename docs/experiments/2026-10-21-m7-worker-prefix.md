# Completed-prefix lifecycle diagnostic

`NativeWorker(captureSavedPrefix:true)` now publishes a bounded exact saved-ID
snapshot after completed commands. Only its owning thread reads engine history;
the condition lock protects publication. Ordinary callers keep the default
`false` and receive no prefix array/copy. Submit, lifecycle interruption and close
clear old diagnostics, and queued cleanup/close suppress publication. Numerical
kernels, native ABI, model formats and quality gates are unchanged.

The diagnostic is a prerequisite for proving exact-prefix resume/replay during
later real OS lifecycle tests. It does not itself establish physical transition
or memory-pressure qualification. Existing app/default worker callers remain
unchanged. Close now clears history/evaluated/pending/reason under the publication
lock after owning-thread release, so a terminal worker reports no active prefix.

The first native attempt exposed that last bug: the CPU executable completed
cleanly, but its strict independent parser rejected `race-close` counts
`(4,4,0,1)` after owners/saved prefix were released. That failed receipt, all five
clean commands and exact old sources remain preserved. The expected all-zero
closed state was retained; no tolerance or oracle was loosened.

The corrected run passes26 exact states/backend on actual Mac CPU and Metal,
including pending cancellation, hidden EOS, recoverable failed scoring,
suspension, cancelled/successful replay, default `nil`, reset/unload and weak
owner release. Four actual post-publication witnesses inspect prefix absence
before queued unload/close cleanup can erase a transient publication. Astra's
initial test findings led to those witnesses and a complete per-row native
state matrix; the previous tests/sources remain retained.

All34 owned native/build commands pass, including existing task/quality/worker/
measurement regressions and both unsigned iOS app builds. Six focused parser
methods and all159 portable methods across17 isolated modules pass. No combined
suite or new phone run is attributed to this task. Full phone loss/task gates,
actual lifecycle/pressure observations, sustained behavior, supported limits and
base iPhone14 evidence remain separate; M7.3 stays open.

Astra accepted source, close-state fix and completed evidence with no findings.
Retain this opt-in diagnostic and corrected terminal state report.

The [qualification record](../../benchmarks/results/2026-10-21-m7-worker-prefix-qualification-v1.json)
binds source/parent/failed correspondence, passing/failing receipts and exact raw
public copies.

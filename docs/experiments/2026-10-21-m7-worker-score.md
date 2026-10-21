# M7 Swift committed-distribution scoring

The dedicated-thread Swift worker now accepts a cold `.score(target:)` command
and returns a bounded scalar snapshot from the additive native score API. This
prepares the later phone quality collector while preserving generation, native
numerical kernels, model representations and all quality gates.

`NativeDistributionScore` contains target/greedy IDs, vocabulary/evaluated counts,
exactly widened FP32 logits and FP64 loss. The engine checks ABI/record size,
identity, finite outputs and exact fully committed history before publishing it.
Only the engine thread calls the native API. The worker's normal accepted-command
boundary still clears cancellation while idle; the underlying C score query
itself does not change the flag. A caller must associate a score with its exact
committed prefix, and score before appending the next teacher-forced token.

Accepted commands clear the earlier score immediately. Failures, reset,
suspension, unload, lifecycle interruption and close expose no stale score.
Completion publishes only when it succeeded and neither close nor queued
lifecycle cleanup supersedes it. This last check occurs under the worker's
condition lock. Without it, a score completing after interruption could briefly
reappear before cleanup; Astra and local review identified that race.

Optional internal diagnostic observers pause before score publication and report
the actual publication decision afterward, outside the condition lock and
without exposing native owners. Production callers leave both absent. The
regressions hold a completed score, request interrupt/unload or close, release
the barrier, then require that exact publication decision to be false before
cleanup can hide a transient error. They also check terminal cleanup and weak
owner release. An isolated negative control changes only the former completion
predicate: compilation succeeds, the CPU check exits1 at `Race final
publication/release`, and no success result is emitted. Production source stays
unchanged. This is deterministic synchronization coverage, not a sleep race.

The [qualification receipt](../../benchmarks/results/2026-10-21-m7-worker-score-qualification-v1.json)
binds source/package/fixture/tool inputs, native binaries, both app builds,
raw outputs and the failed attempt's exact historical source. Seven portable
parser checks reject missing/reordered/Boolean state fields, stale/missing score
records, wrong identities, nonfinite/invalid values, changed read-only/recovery
results, loss identity violations, missing race evidence and unsuccessful
lifetime/backend results.

Ten owned native validation steps pass with no cleanup or endpoint errors.
Strict optimized Swift compilation uses complete concurrency checking and
warnings as errors. Actual Mac CPU and Metal each pass20 score-state rows and
two publication races. CPU loss matches the independent frozen tiny oracle
within1e-5; Metal checks the frozen greedy ID and score identities without
inventing a new full-quality tolerance. Both backends also pass the original
10-row worker continuation/replay/cancel/reset/release checks and unchanged
`[47,47,47,47]` output. Both iOS device and simulator Release SDK builds pass
platform/deployment/layout/link checks. Existing Xcode orientation/AppIntents
warnings remain in raw logs; Swift warnings are errors. No simulator boot or
phone launch occurs in this component.

The first native attempt stopped during strict compilation because two weak
references were unnecessarily mutable. Changing them to `weak let`, as in the
existing checks, fixes the compiler diagnostic. The failed receipt, exact source
and raw logs remain; it is not counted as runtime success. The fresh second
attempt passes. No tolerance or numerical behavior changed.

Astra cleared the worker, controller and deterministic regressions before the
successful run. Astra accepted completed evidence with no remaining findings after verifying156
bound paths,66 exact staged copies, four code pins, the historical source,
all score/race/legacy/negative and SDK replays, privacy,376 links and whitespace. Retain this adapter and
its targeted checks. The full10,000-target/100-task physical quality collector,
real OS lifecycle qualification, supported phone limits, sustained behavior and
base iPhone14 evidence remain before M7.3 exit. The old signed phone context
matrix continues separately on its frozen sources and native package.

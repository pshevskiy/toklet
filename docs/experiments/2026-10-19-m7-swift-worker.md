# M7 dedicated Swift inference worker

The Foundation worker owns model/session calls on a dedicated OS Thread.
A condition lock protects the single queued command, bounded snapshot and atomic
cancellation handle lifetime. A serial queue or actor does not supply native
OS-thread affinity. Idle flag clearing and request enqueue share the same lock
as cancellation, so cancellation after enqueue cannot be silently reset by the
worker. Only the independent atomic flag is used from the caller thread.

Token callbacks append at most128 raw IDs to a polled snapshot. They do not queue
per-token UI closures. `withExtendedLifetime` retains the synchronous callback
sink across the complete C call; no borrowed callback pointer escapes. Each
command and final native release runs inside an autorelease pool. Numerical
weights/KV/scratch remain in Rust; Swift owns bounded prompt/output/history
storage and command metadata, which are separate application memory costs.
The last whole model/session admission remains a conservative upper bound while
immutable weights are retained after session suspension; it is not physical RAM
or a complete application budget.

Suspend releases KV/scratch after active work returns and preserves exact
sampled IDs plus the same immutable model owner. Replay is transactional:
precancellation, partial reconstruction or failure releases the partial session
and keeps the original prefix. Successful replay verifies exact history and
fully committed KV before offering continuation. Full unload clears model,
session and replay state. A queued full unload cannot be weakened by a later
suspend event. Close cancels, discards queued work, drains active work and
releases all native owners/flag on their proper threads. The caller must close;
UI code must not wait synchronously for close. No inference continues by design
in a background task.

Optimized Swift5 compilation with complete concurrency checking and warnings as
errors passes using a repository-local module cache. Both actual Mac CPU and
Metal native tests pass the frozen tiny oracle: prompt6/5/1, continuation8/7/1,
suspend retaining8IDs, precancelled replay retaining8IDs without a session,
complete replay8/8/0, continuation9/8/1, reset, deterministic callback
cancellation5/4/1 and resume6/5/1. Full lifecycle unload, close acknowledgement
and weak-owner deallocation pass. The initial prompt-plus-continuation list is[47,47,47,47]; subsequent replay,
cancellation and resume emissions have separate asserted oracle IDs.

Astra's initial review found replay-prefix loss on precancellation and missing
explicit callback lifetime. Both are fixed and covered before runtime checks.
An initial compile was blocked by the default external module cache; the next
caught a weak-binding warning, fixed with the supported immutable weak binding.
No Rust numerical/default change occurs. This is Mac Swift/native integration;
the iOS app, physical-device lifecycle, quality, memory and sustained behavior
remain separate M7 tasks.

A controlled rebuild records the exact compiler command, full allowlisted
environment, successful exit, source/header/module/archive inputs and new binary
identity. Fresh CPU/Metal runs pass the same ten states with clean exits, no
cleanup errors and unchanged endpoints. The [component receipt](../../benchmarks/results/2026-10-19-m7-swift-worker-qualification-v1.json)
retains the initial run and binds three sources,27 evidence files and18 exact
published copies.

Astra accepted the worker component after verifying30source/evidencebindings,
18copies, the exact compiler command and four clean executions.

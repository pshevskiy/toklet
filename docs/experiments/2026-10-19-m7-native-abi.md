# M7 CPU native boundary component

The first M7 task adds a versioned synchronous C ABI over the existing CPU
model loader, session and raw-ID generator. It retains scalar arithmetic and
all production defaults. Opaque model/session ownership, thread affinity,
structured errors, cancellation, token events and panic containment are
specified in the [native ABI contract](../native-abi.md).

A private owning wrapper retains the immutable stable model allocation while a
session borrows it, with the session dropped first. Model-handle release cannot
invalidate a live session. No blanket Send/Sync is introduced. Model/session
calls remain on one dedicated thread; only the separate atomic cancellation
handle may be used concurrently. Busy checks reject callback reentry, including
session destruction. Callback failure leaves the pending sampled ID and reports
all prior successful emissions. C inputs/outputs remain caller-owned under
explicit alignment, extent, non-overlap and lifetime obligations.

Eight portable ABI tests reuse the frozen independent tiny weights/oracle and
cover model retention/drop, exact greedy continuation, pending KV history,
hidden stops, capacity/ID/version/pointer errors, callback reentry, later sink
failure, cross-thread cancellation, wrong-thread calls and panic poisoning.
The C smoke checks actual C/Rust layouts, local loader, retained model, four
frozen greedy IDs, reentry, pending continuation, cancellation, reset and release.
Its fixture exporter admits the exact manifest SHA and every consumed array
before writing a new local artifact. No download or held-out tuning occurs.

Astra found the initial later-error emission count omission and test-admission
weaknesses. Successful emission tracking, a third-callback-failure regression,
exact C greedy assertions and pre-write pinned fixture admission address them.
An initial edit failed compilation before any native test run; its log remains
preserved. Astra accepted final code, check and native evidence after verifying all39source
pins,8evidencepins,4exactcopies and26exported tensor payloads. Formatting, strict all-target
Metal Clippy,105 CPU tests (8 ignored),118 Metal-feature tests (58 ignored) and
the static library build pass with Rust1.90 in an isolated Cargo target. The
compiled C client emits exactly the frozen[47,47,47,47] IDs.

The [component receipt](../../benchmarks/results/2026-10-19-m7-native-abi-qualification-v1.json)
binds current Rust/C/header/fixture sources, library/client binaries, both local
fixture files and exact published check/smoke logs. Ordinary ignored device/model
tests remain distinct from this explicitly run tiny C integration check.

This is a CPU ABI component, not complete M7.1 or mobile qualification. Explicit
Metal modes, iOS target/platform checks and library packaging, the Swift harness,
and actual physical-device memory/lifecycle/sustained evidence remain. M7 stays
open. CPU admission remains capped at 2,048, with an explicit budget and 4 KiB
bridge reservation rather than any inferred iPhone budget.

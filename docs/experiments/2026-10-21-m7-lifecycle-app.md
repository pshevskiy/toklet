# Lifecycle app integration

The SwiftUI harness now forwards actual scene-phase callbacks and memory-warning
notifications to a bounded lifecycle recorder while a separate background
coordinator exercises one native worker/model. The normal UI shares the same
lifecycle action mapping. Diagnostic capture is opt-in; ordinary decode still
avoids saved-prefix array copies. Cold handlers request cancellation, session
release or full unload; foreground return never silently replays history.

The explicit `--lifecycle` mode admits a canonical consumed-byte job and all
model files before loading. It requires a visible token while generation is
still active and the app is foreground, emits a READY marker, and waits for
actual scene-phase inactive/background/active observations. Both direct return
and inactive/background/inactive/active are accepted without replacing the
original interruption. Callback observation clocks are captured before lock
acquisition. READY must precede the original inactive observation, whose worker
state must still show active generation and a matching visible prefix.

After the round trip the worker must retain the exact sampled history with no
live session. Explicit replay continues three IDs; an independent reset repeats
the original prompt and the same number of single-token decode steps, then
continues three IDs to compare. This preserves the original prefill/decode pattern
instead of comparing two batched replays. A separately labelled synthetic memory
warning uses the shared handler, requires full unload and rejected replay, then
fresh load reproduces the original prompt continuation. Final close acknowledges
zero native state, detaches weak references and proves owner release. Artifact
reproof still runs if a memory-cap check fails, and unadmitted artifacts are not
read again. Failed memory observations remain in evidence.

Astra's fixes cover the optional return-inactive phase, original callback/READY
chronology, independent endpoint checks after budget failure, strict checkpoint
replay and consistent READY/trace mutations. All five memory checkpoint phases,
positive typed footprints, lifetime-peak ordering and trial limits are replayed.
Thirty-four focused cases per tiny CPU/Metal execution exercise config rejection,
phase/provenance/READY admission and combined actual-file mutation plus synthetic
budget failure. Tiny runs explicitly fail closed: they do not claim a useful
active OS interruption window. The actual small-model Mac Metal run records 19
synthetic/coordinator events and passes original-prefix, replay/reference, warning,
fresh-load and owner-release checks.

All 51 owned native/build commands and 173 portable methods in 19 isolated
modules pass, including both unsigned iOS app builds with the new sources.
Eight focused parser methods pass. Initial native attempt v1 compiled but its
tiny CPU child exited −5 with empty logs: the test called a background-only
coordinator from the main thread. The async test entry now uses a detached task,
matching the app contract. Its original receipt, logs and eight source copies
remain; three changed failed-input bindings resolve to exact historical copies.
No quality gate or numerical tolerance was changed.

This is app integration and Mac synthetic evidence. No physical OS transition,
real system memory-pressure notification, device supported limit, energy or M7
completion is claimed. The diagnostic requires a separate physical collection
with actual scene-phase provenance and the same app PID across the round trip.
It builds on the [bounded lifecycle core](2026-10-21-m7-lifecycle-core.md).

The [qualification](../../benchmarks/results/2026-10-21-m7-lifecycle-app-qualification-v1.json)
retains current/parent/failed sources and exact evidence. Astra accepted the source fixes and completed staged app evidence with no
findings. Retain this integration
for physical testing; M7.3 remains open.

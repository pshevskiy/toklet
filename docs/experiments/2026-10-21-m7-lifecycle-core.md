# Bounded lifecycle core

The cold lifecycle trace records command identity, exact bounded worker states
before and immediately after lifecycle requests, observation/serialized record
clocks, process footprint and explicit system/synthetic/coordinator provenance.
It retains at most64 events and1152 saved IDs per state, weakly references the
worker, and requires completed close before explicit detach. Partial failure
evidence remains readable while successful admission stays failed closed.

Inactive requests cancellation, background requests session release, active
performs no implicit replay, and memory warning requests full unload. Native
cleanup stays on the worker thread. Trace errors or capacity exhaustion never
prevent those requests. Both retained state snapshots are bounded, including
when cleanup finishes between snapshots. A nil-by-default non-UI test hook
makes that interleaving deterministic without waiting in production callbacks.

Astra found that failed tracing could block resource cleanup and that only the
post-action snapshot was bounded. Both are fixed and exact previous source is
retained. Native CPU/Metal tests explicitly prove background/warning cleanup
after exhaustion/prior failure and rejection of an oversized pre-action prefix
after full unload has already shrunk the post-action snapshot.

Fourteen trace events/backend and13 scenarios/backend cover pending cancellation,
suspend/replay and uninterrupted-prefix equivalence, explicit foreground resume,
unload precedence, required fresh load, failed trace evidence, bounds, command
identity, independent observation clocks and weak owner release. Every event
in these checks is labelled synthetic/coordinator; no real OS transition is
claimed. The next task must connect the recorder to actual app notifications.

All43 owned native/build commands pass: actual Mac CPU/Metal, explicit iPhoneOS
and simulator type checks of this new core, and existing prefix/task/quality/
measurement/unsigned-app regressions. Existing app builds do not yet integrate
the new recorder. Six focused parser checks and165 methods across18 isolated
portable modules pass. The initial parser probe failed one of six checks because
its fixture aliased the expected case list; copying that list fixed the test,
with the failed output and old two Python sources retained. The initial shell
included a trailing log command, so its overall exit0 is not a passing unittest
status. Final owned receipts and fail-fast checks establish success.

This adds no kernel, native ABI, model format, gate or default inference change.
It builds on the [completed-prefix diagnostic](2026-10-21-m7-worker-prefix.md).
Astra accepted source and completed component evidence with no findings. Retain the
bounded core for explicit app integration. Actual phone lifecycle/pressure,
remaining quality/sustained/supported-limit and base iPhone14 gates stay open;
M7.3 is pending.

The [qualification record](../../benchmarks/results/2026-10-21-m7-lifecycle-core-qualification-v1.json)
binds current/previous sources, exact public copies, commands and accepted worker
prefix evidence.

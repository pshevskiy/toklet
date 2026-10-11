# Literal Linux controller qualification, 11 October 2026

The [fixed controller](../../tools/bench/linux_task_controller.py) now passes
all 28 authored native expectations for bounded literal source-test execution.
The [accepted attempt](../../benchmarks/results/2026-10-11-literal-native-v3-probe.json)
retains ten intentional passes, eight model failures, five resource failures
and five unresolved infrastructure outcomes. These are qualification fixtures,
not a model task pass rate. No held-out response or generated HumanEval program
was executed; full task quality and M2.5 acceptance remain open.

The controller builds an exact trusted bootstrap from the existing policy and
literal child. It rejects an altered, NUL-containing or oversized bootstrap
before any container operation. Candidate program text travels only through
bounded stdin, with its UTF-8 source hash and case binding. Actual inspected
configuration must retain the qualified resource/security declarations, with
interactive stdin enabled and TTY disabled. Only the two verified configuration
differences are adapted in a private copy for the shared validator.

Each fresh container is checked against its full ID, image, name and ownership
label. The controller captures input/output, retains the post-attach State,
kills a still-running owned container when required, waits and reinspects it,
and removes it with both ID and name absence checks. Raw transport and container
exit statuses remain literal. The [precise authored checker](../../tools/bench/linux_task_native_check.py)
also requires the intended exception, normal-completion state, resource flags,
framed text/channels, capture bound and absence of unexpected controller stderr.
The [runner](../../tools/bench/linux_task_native_probe.py) binds all implementation
and cleanup dependencies and preserves the first failed execution.

The six-second execution and combined 65,536-byte output limits remain fixed.
Both accepted overflow fixtures actually finish remotely with code zero after
the owned attach client is killed with SIGKILL. The
[receipt validator](../../tools/bench/linux_task_evidence.py) accepts that mismatch
only with complete input/admission, hash-verified exact output saturation,
owned group kill, clean stopped remote exit, no wall timeout/OOM/forced remote
kill, and successful cleanup. It returns false resource failure, never pass.
Mutation tests reject missing evidence, unsaturated or altered captures,
other transport statuses and contradictory flags. The existing exit-137
ambiguity rule is unchanged.

The [fixed authored inventory](../../tools/bench/linux_task_native_cases.py)
covers whole-source completion, definition-only and assertion failures, syntax
errors, caught and raw exits, ordinary/status-like text, closed and unsupported
streams, imports, memory admission, CPU ambiguity, wall/output termination,
and exact EPERM for filesystem, socket, fork, exec and limit-change denial.
Both memory limits remain unchanged. The denied limit setter uses a checked
16-byte, eight-aligned LP64 structure, explicit ctypes argument/result types
and a live byref owner; it requires return -1 and errno EPERM directly.

Two prior attempts are excluded and retained with their exact check records and
22 source snapshots each. [V1](../../benchmarks/results/2026-10-11-literal-native-v1-archive.json)
passes 27 expectations, then observes Python ValueError in the original
limit-change fixture instead of its expected OSError. The fixture is corrected
to inspect the libc return and errno without accepting an exception substitute.
[V2](../../benchmarks/results/2026-10-11-literal-native-v2-archive.json)
passes 13 expectations, then receives healthy Podman wait output `-1` at the
wall fixture. The controller rejects that unknown status; it is not normalized
or counted as a resource outcome.

The final controller may retry only literal unknown `-1`, at most three times
with two 0.1-second pauses and the existing bounded control calls. It retains
every wait and ownership-validated State observation. Acceptance still requires
a later integer wait code 0–255 exactly matching stopped/no-error inspected
state. Persistent unknown, malformed output, stderr, control/ownership failure
or nonnegative disagreement remains infrastructure failure. This occurs after
execution termination and does not extend the six-second candidate budget.
The retry branches pass offline mutations; every accepted V3 wait completed
on its first attempt, so V2 is not claimed as an observed transient resolution.

The [enclosing V3 window](../../benchmarks/results/2026-10-11-literal-native-v3-window.json)
starts at 11:51:11.029627 UTC and finishes at 11:52:07.351450 UTC. It requires
the completed scalar-loss idle proof, the previously qualified runtime/image
and explicit machine connection, and an initially stopped VM. It retains the
reviewed bounded fixed-machine SSH inventory fallback, never stops the machine
with unexplained live containers, and keeps startup pipe readers until verified
shutdown. The pinned interpreter remains Python 3.12.15 on Linux AArch64,
UID/GID 65534 with zero capability masks and the unchanged seccomp filter.

All containers are removed in every attempt and all three windows restore the
stopped VM. The [fresh final idle proof](../../benchmarks/results/2026-10-11-literal-native-v3-idle.json)
verifies startup PID 82722 and helpers 82723/82724 absent, the original held
orchestration unchanged, loss workers absent and protected binary/model/source
identities unchanged. [Actual session completion](../../benchmarks/results/2026-10-11-literal-native-v3-session-completion.json)
retains exit zero. The
[final archive](../../benchmarks/results/2026-10-11-literal-native-v3-archive.json)
binds the full raw records and exact source inventory.

The [final checks](../../benchmarks/results/2026-10-11-linux-literal-controller-checks-v3.json)
pass all 161 offline benchmark tests, formatting, offline locked Clippy with
warnings denied, and 76 ordinary Rust tests. Rust uses the existing isolated
target; root release binaries are unchanged. All native operations are mocked
in offline controller/transport tests. Four additional window-fallback mocks
passed before native execution. Checks precede each native attempt.
The [saved replay](../../benchmarks/results/2026-10-11-literal-native-analysis-v1.json)
and [exact producer](../../benchmarks/results/2026-10-11-literal-native-analysis-v1.py.txt)
verify accepted and excluded outcomes, captures, literal input/configuration,
commands, source identities, waits, checks and restoration evidence.

The external isolation boundary does not establish adversarial grading integrity:
candidate code and source tests share the Python interpreter, as documented in
the [child preparation](2026-10-11-linux-task-preparation.md). Unresolved outcomes
must stop aggregate quality scoring. Astra accepted the complete code/evidence/
documentation task with no actionable findings after 25 targeted tests and
independent byte-identical replay of all three attempts. All 144 local links
across this report, README and PLAN resolve, and whitespace checks pass.

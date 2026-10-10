# Authored code isolation qualification, 10 October 2026

The native candidate is rejected: this Python process cannot admit the fixed
256 MiB data limit. All required offline checks pass, but the child exits during
resource admission before applying the named sandbox or running any authored
operation. Astra accepted the completed experiment without findings. No dataset program, generated code or
held-out model prompt executes in this task, and no quality result is claimed.

The [static child](../../tools/bench/code_isolation_child.py) accepts only fourteen
authored mode names, with no arbitrary source or expression input. It imports
trusted standard-library modules before applying fixed resource controls and
the SDK's named `kSBXProfilePureComputation` profile through `sandbox_init` with
`SANDBOX_NAMED = 1`. The installed SDK marks this API deprecated; its availability
and enforcement on this host still require actual qualification. No reserved
flag or private custom-profile interface is used.

The SDK export stub assigns the public legacy function and named-profile
constant to `/usr/lib/system/libsystem_sandbox.dylib`; the child loads that
specific exporter. The distinct `libsandbox.1` stub lists profile-compilation
interfaces, which this task does not use. Header and export-stub identities will
be retained with runtime evidence.

The exported C profile string is borrowed from the retained library owner.
The error pointer comes only from the API and is released exactly once using
`sandbox_free_error`. Setup failure emits infrastructure evidence and prevents
running the authored operation. All candidate restrictions begin after trusted
initialization; a future generated-code executor is a separate implementation
that must preserve and qualify this boundary before dataset use.

| Fixed control | Prepared value |
| --- | --- |
| Data allocation (`RLIMIT_DATA`) | 256 MiB, soft and hard |
| Address space (`RLIMIT_AS`) | 1 GiB, soft and hard |
| CPU time (`RLIMIT_CPU`) | 4 seconds, soft and hard |
| Parent wall deadline | 6 seconds, including child startup |
| Captured stdout plus stderr | 64 KiB total |
| Core file | zero bytes |
| Open files | 64, soft and hard |
| Network / filesystem / fork / exec after initialization | must be denied |

These values are prepared before any held-out response generation. They are
requirements to qualify, rather than claims that the Mac/runtime enforces them.
An ineffective or incompatible control fails this task; the suite does not
silently drop it. A policy replacement requires separately identified limits
and fresh qualification before generated-code evaluation.

The [controller](../../tools/bench/code_isolation_probe.py) starts one fresh
session group per authored operation. It supplies an empty stdin, isolated
Python with user/site imports and bytecode writes disabled, an empty case
working directory, closed inherited descriptors and a minimal environment.
It requires optimization level zero so source assertions remain executable.
It reads both pipes with nonblocking selectors, retaining at most 64 KiB across
them. Excess output or the wall deadline kills only the owned group. Cleanup
also handles an exited leader, reaps it and requires no live group members;
failed cleanup or unavailable process inspection prevents qualification.
Captured-byte SHA values bind the retained bytes, including intentionally
truncated flood evidence; they do not claim a complete flood transcript.

Native qualification must attest active sandbox admission and all exact resource
limits before checking any probe result. It includes a benign function with an
explicit source-test call and post-return record, a failed assertion and zero
exit before that record. Other authored cases attempt reading/writing parent
owned sentinel files outside the case directory, localhost networking, fork,
exec, flooding each output pipe, endless CPU work, excess sleep, a 512 MiB data
allocation and a 2 GiB anonymous mapping. No private file or remote service is
used. Network refusal/timeout is not evidence of denied access; permission
errors are required. Memory probes describe enforcement under the combined
controls, rather than independently attributing a failure to one limit.

A passing early-exit probe means the qualification recognized absent test
completion; it never means a program passed source tests. A future HumanEval
executor must append the literal completion and source tests, invoke the frozen
entry-point check and require completion evidence. This capability task does
not expose that executor or assign binary model outcomes.

The [offline tests](../../tools/bench/test_code_isolation_probe.py) are prepared
with synthetic captures and fake process/pipe objects. They exercise denied
versus refused access, exact memory requests, missing completion, incorrect
termination, record corruption, failed cleanup, combined output and wall bounds,
pipe setup failure and idle admission. They require neither model downloads nor
the Apple sandbox API. All eight offline tests pass, including regressions that
distinguish successful fork followed by denied wait from denied fork, and denied
socket timeout setup from denied connection. These errors cannot establish the
requested isolation.

The native driver ran only after the
[pending idle proof](../../benchmarks/results/2026-10-10-cpu-code-isolation-check-window-idle.json)
at 11:08:49 UTC confirmed numerical worker 80039 absent and transport 80030
defunct beneath held orchestration 17694. The original numerical worker finished
all three warmups and ten measured requests uninterrupted. The
[exact idle producer](../../benchmarks/results/2026-10-10-cpu-code-isolation-check-window-idle-v1.py.txt)
is retained. Root formatting, Clippy with warnings denied, 76 ordinary Rust tests
and all 62 offline benchmark tests pass. No root release binary is rebuilt.

The [native record](../../benchmarks/results/2026-10-10-code-isolation-probe-v1.json)
retains all check output, runtime/SDK/helper identities, the one attempted worker,
its bounded raw stdout and empty stderr, exit code 2 and successful cleanup.
The only native message is `Resource admission failed for RLIMIT_DATA: current
limit exceeds maximum limit`. The data setter fails first; later limits,
`sandbox_init`, denied-access probes and source-test execution are unqualified.
The record's `completed` field remains false.

The [rejection analysis](../../benchmarks/results/2026-10-10-code-isolation-rejection-analysis-v1.json)
and its exact [producer](../../benchmarks/results/2026-10-10-code-isolation-rejection-analysis-v1.py.txt)
reproduce the stored failure, all raw capture hashes, runtime/source bindings,
check outcomes and zero active sandbox/operation counts. Analysis completion
does not imply native qualification. Hardware is Apple M1 Pro, MacBookPro18,3,
16 GiB and macOS 26.6.2 (25G83), with Python 3.12.8. Launch checkout is `8404f81`
with this task uncommitted; explicit hashes identify the executed source.

Retain the checked prototype and rejection evidence; do not adopt this native
candidate or lower/drop the fixed limits. A compatible executor needs fresh
qualification before any held-out response or generated-code execution. The
[prepared resumption verifier](../../benchmarks/results/2026-10-10-cpu-code-isolation-check-window-resume-v1.py.txt)
binds the reviewed rejection and completed pending records while explicitly
leaving executor adoption false. Original matrix progression is independent of
this rejected candidate. HumanEval, full quality and milestone gates remain open.

Astra independently ran all eight offline tests and reproduced the rejection
analysis exactly, checking C ABI/ownership, cleanup, identities, idle ordering,
report links and the prepared resumption sequence. This accepts the experiment
and its rejection evidence; the native candidate remains unqualified. Review
ran no new native probe or orchestration signal.

# Proposed process CPU witness for isolated Gemma task grading

**Design awaiting qualification.** HumanEval/39 produced a valid readiness
record followed by exit 137, without a terminal result, OOM or watchdog event.
The first run remains unresolved. Its 4.191 seconds of wall time cannot establish
four seconds of evaluator CPU use. Keep the [qualified classifier](../tools/bench/linux_task_evidence.py)
and the existing grader unchanged until an independent witness passes review.

## Decision rule

For the otherwise ambiguous 137 path only, require all existing admission,
readiness, program hash, transport, restoration and cleanup checks. Require an
incomplete literal evaluation, no conflicting infrastructure failure, and a
trusted guest observation proving that the exact evaluator process consumed
**at least 4,000,000 microseconds of user plus system CPU** during its lifetime
under the unchanged process limit. This permits a resource-failure outcome
without attributing SIGKILL to a particular sender. Use integer units, round
down and require a conservative lower bound. The limit covers the process
including trusted bootstrap CPU; readiness still gates applicability.

Bind the guest boot ID, container ID and qualified image, evaluator's guest PID
and TGID, process start identity, PID namespace and cgroup to the program's
authenticated readiness record. Establish the binding before releasing its
payload when necessary. Detect PID reuse and missing, ambiguous or dropped
events. Record the measurement source, clock units, original raw result and
timing relative to exit. The observer must be outside the candidate container.
Wall time, container-wide CPU, conmon CPU and counters from a different process
are insufficient. Missing identity or a below-threshold measurement leaves
the original ambiguous exit unresolved. A trusted complete terminal event
continues through its existing classifier.

The first authored prototype used `clock_getcpuclockid` (SCHED). Its spinner
recorded3,930,666,775ns and **failed** the frozen four-second rule. Keep that
failure. Linux6.17 enforces RLIMIT_CPU against process `CPUCLOCK_PROF`, the
user+system counter; SCHED is a different scheduler-runtime counter. Before
fresh controls, select process PROF **only**, signed ID `((~tgid)<<3)|0`, with
the per-thread bit clear. Record both PROF and SCHED at the same kernel exit
stop, never select their maximum or add a rounding allowance. Require integer
**PROF ≥4,000,000,000ns**, with the unchanged four-second limit.

Use `PTRACE_SEIZE` with `PTRACE_O_TRACEEXIT`, query those clocks with
`clock_gettime`, resume the exit stop and require matching final SIGKILL status.
The process clocks exclude child-process accounting. Final `wait4(RUSAGE_BOTH)`
alone would need proof that no descendants were ever reaped; an attachment-time
snapshot cannot establish that prehistory. No `EXITKILL` or candidate register,
memory, syscall or isolation changes are permitted. Forward real signal-delivery
stops and preserve seized group stops. Observer loss or unknown stops remain
inconclusive. Qualify teardown-pause overhead against the original watchdogs.
[Linux process clocks](https://man7.org/linux/man-pages/man3/clock_getcpuclockid.3.html) ·
[ptrace stop semantics](https://man7.org/linux/man-pages/man2/ptrace.2.html) ·
[kernel enforcement and clock sampling](https://github.com/torvalds/linux/blob/v6.17/kernel/time/posix-cpu-timers.c).

## Bounded inventory in an owned VM window

Record the stopped initial machine, existing rootless connection, tool hashes,
start/exit status and independently verified stopped restoration. Bound each
guest command to 10 seconds and 64 KiB of output. Run read-only queries under
the same guest identity that would operate the observer. No image pull,
package install, sysctl change or generated program is needed.

- Record `uname -srmo`, `cat /etc/os-release`, `id`,
  `cat /proc/sys/kernel/random/boot_id` and `getconf CLK_TCK`.
- Use `command -v` for python3, strace, conmon, crun, getdelays and bpftool.
  Record versions, resolved paths, SHA-256 and installed RPM build IDs for
  available tools. Check `podman info` for the actual conmon and OCI runtime.
- Read `/proc/sys/kernel/yama/ptrace_scope` if present, capability lines in
  `/proc/self/status`, and CPU online mask in `/sys/devices/system/cpu/online`.
  Check readable kernel config for TASKSTATS, BSD_PROCESS_ACCT, TASK_XACCT,
  IKCONFIG_PROC and tracepoint support. Check installed Python for
  `socket.AF_NETLINK` and `socket.NETLINK_GENERIC` (a missing Python constant
  alone does not establish kernel support). Record absent/inaccessible items.

Prefer an existing trusted process-exit accounting record. Upstream conmon's
exit path calls `waitpid` and retains status; tracing a wait with an unused
rusage pointer does not recover CPU counters. Verify the installed build before
pursuing this route. If TASKSTATS is available, check whether an existing
version-aware receiver can capture the target TGID exit record across all
online CPUs, including kernel sender identity, CPU fields and loss detection.
Use targeted output only; discard unrelated guest records. A procfs read at a
validated ptrace exit stop is an alternative if permitted, provided the
observer can establish a conservative accounting bound and preserve the
grader's deadlines and lifecycle. Instrumentation requiring an extra process
inside the container or weaker isolation is outside scope.
[conmon exit handling](https://github.com/containers/conmon/blob/main/src/ctr_exit.c) ·
[Linux taskstats](https://docs.kernel.org/accounting/taskstats.html)

## Authored controls before any generated retry

Use the same image, trusted bootstrap, seccomp profile, one-PID limit, CPU
limits [4,4], six-second wall watchdog and cleanup rules for all candidate
controls. Pin the observer and exact control programs in advance.

| Authored case | Required result |
| --- | --- |
| CPU spinner past readiness | Incomplete 137 plus authenticated process CPU at or above the limit. Repeated under-budget measurements fail qualification. |
| Early self-termination | An explicit exit status 137 far below budget must stay unresolved. If testing self-SIGKILL, first check whether seccomp permits it; retain a denial as a denial. |
| External SIGKILL below budget | A controlled guest-side kill after readiness must not produce a CPU-exhaustion verdict. Keep independent kill provenance. |
| Normal completion | The existing terminal event determines the result. The observer must neither delay cleanup nor alter the decision. |

Also reject replayed/wrong identity, missing counter, duplicates, event loss
and one microsecond below threshold. Compare authored control runs with and
without observation to detect perturbation. On capture failure, retain raw
evidence and verify VM restoration. Review the controls, observer lifetime,
resource overhead and classifier changes before any new generated-code run.
Apply an accepted rule uniformly to both profiles in a fresh isolated grading
window. The previous run never acquires a retrospective classification.

## Authored qualification

The owned read-only guest inventory finds Python3, passwordless guest-root
access and ptrace support. No existing strace/getdelays/bpftool is present;
kernel-config files are unavailable. The inventory's first launcher shadows
stdlib `inspect` before VM startup; its failure is retained and the renamed
launcher passes with independently stopped restoration.

The final observer binds the container and process before the original
transport releases stdin, samples PROF/SCHED at the kernel exit stop and waits
for final status. Seven pure witness/identity tests plus existing evaluator checks
total46 passing controls. Ten paired authored executions pass: baseline and
observed normal completion, two spinners, raw early-exit137 and a controlled
external SIGKILL. PROF reaches4,000,864,950/4,000,141,624ns for the observed
spinners; normal/early/external measurements remain below budget. Both positive
and negative final statuses match the original controller, without wall/OOM
watchdogs. Every container is removed and the VM returns stopped with its
lifetime owner finished. Astra approves this authored-witness scope.

Grading integration remains separate: classify every original execution
normally first. Only an otherwise unresolved137 triggers one observed rerun of
the **identical** authenticated payload under this same rule for either profile.
Require both readiness/payload/image/configuration/cleanup bindings to agree.
A rerun with an observation failure, watchdog or insufficient PROF leaves the
original unresolved. Preserve both raw attempts and the original verdict;
qualifying a rerun does not identify the cause of the original signal.
[Qualification evidence](../benchmarks/results/2026-11-04-linux-process-cpu-witness-v1.json).

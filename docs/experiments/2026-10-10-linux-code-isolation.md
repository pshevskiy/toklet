# Authored Linux isolation candidate, 10 October 2026

The first attempt passed its offline checks on 11 October 2026, but native
qualification was unavailable after its actual setup attempt. A later
[retry](2026-10-11-linux-code-isolation-retry.md) records fourteen passing
authored modes and the enforcement adaptations; this document retains the first
attempt's design and historical evidence. Podman reported
successful machine start, then both image pull and container inventory failed
with exit 125 and connection refused at its configured local socket. No image
was pulled and no container, authored operation, dataset program, model response
or generated code ran. A follow-up machine inventory confirmed the original
stopped state. This is an infrastructure failure, not evidence for or against
the seccomp policy. The [native Mac candidate](2026-10-10-code-isolation.md)
remains rejected without weakening its limits.

The user paused the changed-prefix benchmark after six completed measurements.
That interrupted batch is excluded from timing qualification. On explicit resume,
the owned worker, transport and obsolete waiting guard were retired while the
orchestration remained held. Checks ran only after live process inspection
confirmed the worker and guard absent and transport defunct. A fresh uninterrupted
three-warmup/ten-measurement batch subsequently
[completed](2026-10-11-cpu-changed.md).

The [Linux child](../../tools/bench/linux_code_isolation_child.py) admits only
Linux AArch64, little-endian LP64. A fixed classic-BPF program first checks
`AUDIT_ARCH_AARCH64`, kills a different syscall architecture, then defaults to
`EPERM`. It permits bounded interpreter memory operations, signals, clocks and
exit. Descriptor operations are restricted to inherited standard streams;
mapping requires anonymous, non-executable memory and protections cannot add
execute permission. File opens, sockets, fork/clone/exec, resource changes,
ptrace, namespace changes and subsequent prctl/seccomp calls are denied.
No path or pointer content is inspected by BPF. This distinction follows the
[kernel's seccomp interface](https://docs.kernel.org/userspace-api/seccomp_filter.html).

The child imports trusted modules, fingerprints its interpreter, closes every
inherited descriptor above two and admits the fixed resource pairs before
installing the filter. Localized ctypes calls validate the public structure
sizes, offsets, alignment and unsigned-long calling slots. Owned BPF storage
remains live through the kernel's synchronous copy. Successful admission emits
actual interpreter/kernel/UID/GID/filter identities before a fixed authored
probe. The existing [authored operations](../../tools/bench/code_isolation_child.py)
are reused unchanged; no general Python source runner is introduced.

| Control | Fixed value |
| --- | --- |
| Data soft/hard | 256 MiB |
| Address-space soft/hard | 1 GiB |
| CPU soft/hard | 4 s |
| Wall watchdog | 6 s, including start/attach and interpreter initialization |
| Combined stdout/stderr | 65,536 captured bytes |
| Core / open descriptors | 0 bytes / 64 |
| Container memory / memory plus swap | 256 MiB / 256 MiB |
| Container process count | 1 |

The [host driver](../../tools/bench/linux_code_isolation_probe.py) requires a
complete changed-workload proof or an explicit interrupted-batch retirement proof
that marks timing ineligible, with worker 85027 absent, transport 85019
defunct and orchestration 17694 held under its original parent. It does not
start the VM, pull an image or select a floating tag. Setup must supply a full
immutable Linux/arm64 image ID and one of two fixed Python executable paths.
The image's identity, layers and repository digests, client/server versions and
client binary are bound separately from the authored policy.

For future idle setup, the [official Python image inventory](https://github.com/docker-library/official-images/blob/master/library/python)
lists `3.12.15-slim-bookworm` for arm64. A tag is only a setup selector;
qualification must use the observed full immutable image ID. No image has yet
been inspected or pulled, and the interpreter must pass the actual probes.

Each invocation creates a fresh UUID-named container, verifies its exact
ID/image/ownership label and configuration, then starts one trusted bundled
Python argument with `-I -S -B`. No shell, arbitrary expression/source path or
host mount is supplied. Image volumes are ignored, proxy forwarding and default
environment variables are disabled, and only static PATH/locale/timezone values
are admitted. The container uses a read-only filesystem, no network, UID/GID
65534, no effective/bounding capabilities and no-new-privileges. The explicit
limits are also declared before OCI exec. These controls use the installed
Podman 5.7.1 client's documented [run/create options](https://docs.podman.io/en/v5.7.1/markdown/podman-run.1.html).

Creation, inspection and cleanup are trusted control operations with separate
20-second and 256-KiB bounds. The 6-second execution watchdog starts before
spawning start/attach; a separate Podman timeout also bounds the remote process.
Bounded selectors capture both pipes and kill/reap only the newly owned host
transport group. A failed transport must additionally kill/remove the verified
owned container and prove both its full ID and fresh name absent. A failed
create may already have created a server object; cleanup requires matching its
ownership label and image first. A transport exit alone does not prove remote
cleanup. The driver retains raw host/container exit codes and explicitly
normalizes authored SIGKILL status 137, rejecting OOM and infrastructure failures.

The fourteen fixed modes cover explicit successful check invocation, failed
assertion, early zero exit without check return, denied read/write/network/
fork/exec, two output floods, CPU/wall termination and oversized data/address
allocations. Qualification requires every mode, active fixed resource admission,
raw capture hashes and completed owned cleanup. Future generated programs and
the frozen HumanEval suite need a separate reviewed integration after this
boundary is qualified; early exit is an executor probe, not a passing model task.

All nineteen [targeted offline tests](../../tools/bench/test_linux_code_isolation.py)
passed. Their first execution caught a reversed branch in the descriptor
high-half guard; it was corrected without weakening the admission policy.
They independently interpret classic BPF against adversarial
architecture, syscall, descriptor and memory arguments; validate ABI packing
and owned ctypes pointer lifetime with mocked calls; and reject changed
container identity/configuration, capture/runtime corruption, missing cleanup,
false success and live numerical workers. VM/kernel qualification remains
necessary even when these checks pass.

The syscall and structure inventory is derived from pinned Linux v6.17 UAPI
[syscalls](https://raw.githubusercontent.com/torvalds/linux/v6.17/include/uapi/asm-generic/unistd.h),
[audit architecture](https://raw.githubusercontent.com/torvalds/linux/v6.17/include/uapi/linux/audit.h),
[seccomp layout/actions](https://raw.githubusercontent.com/torvalds/linux/v6.17/include/uapi/linux/seccomp.h),
[BPF layout](https://raw.githubusercontent.com/torvalds/linux/v6.17/include/uapi/linux/filter.h)
and [mapping flags](https://raw.githubusercontent.com/torvalds/linux/v6.17/include/uapi/asm-generic/mman-common.h).
Container metadata uses the public
[inspect fields](https://docs.podman.io/en/v5.7.1/markdown/podman-container-inspect.1.html).
Source documentation and CLI availability are not native qualification.

The setup attempt used the installed Podman 5.7.1 and its existing AppleHV
default VM (five CPUs, 2 GiB), with no configuration changes. Its start command
returned zero, but pull and inventory both returned 125 before admission. The
subsequent machine inventory showed `Running=false` and `Starting=false`; no
additional stop was needed. The attempted official image selector was
`docker.io/library/python:3.12.15-slim-bookworm`; no immutable image identity or
interpreter runtime was established.

Formatting, offline locked Clippy with warnings denied, all 76 ordinary Rust
tests and all 81 offline benchmark tests passed in the retired-workload idle
window. The first sandboxed capture could not verify process-group cleanup
because `ps` was denied; that attempt is retained as incomplete. The complete
rerun had no cleanup errors or surviving owned process groups. The
[check and runtime report](../../benchmarks/results/2026-10-11-linux-code-isolation-attempt-v1.json)
binds exact source, raw commands/captures, the pause/retirement evidence and the
final stopped machine inventory. The candidate remains unadopted and establishes
no engine latency, quality score or milestone closure.

# Authored Linux isolation retry, 11 October 2026

The fresh v8 run passes all fourteen fixed authored modes on Linux AArch64.
It verifies admission, successful check return, assertion failure, early zero
exit, denied file read/write/network/fork/exec, both output floods, CPU and wall
termination, oversized data allocation and oversized address mapping. Every
owned container is removed, with no cleanup errors or surviving host transport
groups. The originally stopped VM is stopped again and its owned helpers exit.
This qualifies the authored boundary only. No model response, dataset program
or generated program runs; a general executor remains unadopted.

The [complete native record](../../benchmarks/results/2026-10-11-linux-code-isolation-probe-v8.json)
retains actual OCI configuration, runtime identity, raw pipes/hashes, host and
container statuses, bounded exit-wait captures and all fourteen outcomes. The
[VM window](../../benchmarks/results/2026-10-11-linux-runtime-window-v8.json)
and exact [producer](../../benchmarks/results/linux-runtime-window-2026-10-11-v8.py.txt)
bind selected machine/connection, image, startup helper ownership and restoration.
The independent [saved replay](../../benchmarks/results/2026-10-11-linux-native-retry-analysis-v1.json)
has an exact [producer](../../benchmarks/results/linux-native-retry-analysis-2026-10-11-v1.py.txt).
Separate [final inspection](../../benchmarks/results/2026-10-11-linux-native-final-restoration-v1.json)
and its [producer](../../benchmarks/results/linux-native-final-restoration-2026-10-11-v1.py.txt)
verify actual stopped inventory, absent owned helpers and unchanged frozen CPU
and CLI executables after the window.

The installed Podman client is 5.7.1. The existing rootless AppleHV machine has
five CPUs and 2 GiB; its configuration is unchanged. The client explicitly uses
the selected machine's verified loopback SSH URI and identity, independent of
the ambient default connection. Official selector
`docker.io/library/python:3.12.15-slim-bookworm` resolves to immutable Linux/arm64
image ID `eb7d16fec874f15b584063275012ac719885a2b87574d124c0da9c156d504629`.
Admission records Python 3.12.15, executable SHA
`7863a4d5e03fde7791c7f8c2c304cf3522f435e19745364ad18a6b7a0458af57`,
kernel `6.17.7-300.fc43.aarch64`, UID/GID 65534 and zero effective/bounding
capability masks. Filter SHA is
`7584e24ca69a1c2da42105b5956ef2e5bdccf215eca35cd3bea7da717c6c9e12`.

The original [policy](2026-10-10-linux-code-isolation.md) retains data 256 MiB,
address space 1 GiB, CPU 4 s, wall watchdog 6 s, combined output 65,536 bytes,
core zero and 64 open descriptors. Podman rejects the `as` ulimit option, so
OCI declares exactly data/CPU/core/nofile. The trusted child still sets and
verifies all five unchanged soft/hard process pairs before seccomp admission,
readiness or any authored operation. Container memory/process/no-network/read-only
controls apply during trusted interpreter startup; the complete address-space
admission occurs afterward. Failure to admit it rejects qualification.

Podman's [pinned inspect implementation](https://raw.githubusercontent.com/containers/podman/v5.7.1/libpod/container_inspect.go)
copies OCI capability slices directly, allowing JSON `null` for empty slices.
Inspection requires both fields explicitly present, accepts only `null` or `[]`
and rejects any named capability. Independently, the trusted child bounded-reads
kernel status and requires unique zero `CapEff`/`CapBnd` masks. Missing, malformed,
nonzero or oversized status rejects admission. This resolves representation
differences without relying on null alone as enforcement evidence.

The runtime injects `HOME` and `HOSTNAME` after creation inspection. Trusted
initialization removes only those two keys and records their names, not values.
Readiness still requires exactly PATH/LC_ALL/TZ; another unexpected key fails.
This sanitization happens after trusted interpreter startup and makes no claim
that the two metadata variables were absent during bootstrap.

Successful VM startup now preserves its helpers and drains inherited pipes
until verified machine shutdown and an empty owned startup group. Evidence
publication failures cannot drop that ownership. Ordinary finite commands retain
their bounded owned-group cleanup. At the execution deadline or output cap,
SIGKILL is requested and reaping uses a separate 20-second cleanup bound;
execution admission is not extended. Ordinary completion retains its original
deadline. Every EPERM diagnostic remains a qualification failure.

After termination, a bounded `podman wait --condition=exited` on the verified
container ID must return exactly one code in 0..255. Reinspection must agree,
with no running state, OOM or runtime error. The raw wait is retained before
validation; unknown `-1` is never converted into success. CPU termination in v8
has host/container status 137 without the wall flag. Wall termination has
host -9/container 137 with its watchdog flag, 6.045757 s including bounded
post-kill reaping. This elapsed capture duration is not a relaxed wall budget.

The excluded retry records remain exact and independently replayable:

| Version | Reason qualification stopped |
| --- | --- |
| v2 | Container creation failed; bundled command text obscured the truncated diagnostic. |
| v3 | Preserved raw diagnostic identified unsupported OCI `as` ulimit. |
| v4 | Strict configuration rejected null capability serialization. |
| v5 | Benign operation returned, but runtime-injected environment keys failed admission. |
| v6 | Eleven modes passed; wall termination had unknown exit status and host EPERM cleanup. |
| v7 | Completed exit wait resolved status 137; host EPERM still rejected qualification. |

Each excluded window removed its owned containers and restored the stopped VM.
Their raw records, exact wrapper producers and matching child/host/test source
snapshots are bound by the saved replay. The v8 post-kill reap change removes the
redundant one-millisecond second-kill race; v8 has no EPERM cleanup. This supports
the race hypothesis, without retroactively qualifying v6/v7 or suppressing errors.
The earlier v1 finite-command startup cleanup likely killed persistent helpers;
its recorded infrastructure failure remains unchanged.

Formatting, offline locked Clippy with warnings denied, 76 ordinary Rust tests
and 97 offline benchmark Python tests pass in v8. Targeted policy/control tests
cover literal resource values, environment/capability admission, ABI/BPF guards,
explicit endpoint ownership, delayed/malformed exit publication, separate reaping
and fail-closed cleanup. Nine lifecycle tests cover helper pipes and restoration
through publication failure. All native work occurs after the fresh changed
numerical worker's natural exit. This task changes no engine code or numerical
binary, and establishes no engine performance, quality score or milestone closure.

Astra accepted the complete code and evidence task without findings after
byte-identical saved analysis replay, all 35 targeted tests, raw-copy and owned
container checks, and final restoration verification. Local links and whitespace
checks pass.

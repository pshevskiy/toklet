# Excluded tiled medium publication failure, 11 October 2026

The fresh tiled medium attempt saved all ten measured numerical requests after
three warmups, then failed at schema validation because the launcher used system
Python without `jsonschema`. No validated result was published. This whole
attempt is excluded from performance acceptance; its sample journal is retained
without reconstruction, substitution or a new summary.

The [actual execution-tool completion](../../benchmarks/results/2026-10-11-cpu-tiled-medium-v1-session-completion.json)
records exit 1 and both tracebacks. The unchanged CPU driver reaches its
`finally` cleanup, kills its own still-live child and reaps it. This is failed
execution followed by cleanup, not a successful natural benchmark completion.
The [ten saved sample/state rows](../../benchmarks/results/2026-10-11-cpu-tiled-medium-v1.json.samples.jsonl),
[launch](../../benchmarks/results/2026-10-11-cpu-tiled-medium-v1-launch.json),
[owned-process snapshot](../../benchmarks/results/2026-10-11-cpu-tiled-medium-v1-owned.json),
[exact launcher](../../benchmarks/results/2026-10-11-cpu-tiled-medium-v1-launch.py.txt),
[exact driver](../../benchmarks/results/2026-10-11-cpu-tiled-medium-v1-driver.py.txt)
and [executed archive source](../../benchmarks/results/2026-10-11-cpu-tiled-medium-v1-failure-idle.py.txt)
remain available.

The [failure idle proof](../../benchmarks/results/2026-10-11-cpu-tiled-medium-v1-failure-idle.json)
at `2026-10-11T17:43:16.530124+00:00` binds eleven exact archived files,
verifies PIDs 29311/29334/29347 absent and shared group 29311 empty, keeps
original orchestration 17694 held and verifies Podman off. Protected root
executables and model/contract assets match the prior accepted endpoint hashes.
The first, weaker ownership observer and receipt are also retained; the corrected
observer records exact commands, active states and ancestry before failure.
No numerical workload remains active during archive and review.

The replacement launcher has a distinct one-off prefix and exclusive claim,
requires the preceding failure's actual exit and current owned-process/group
absence before model/binary hashes, and imports/checks pinned `jsonschema`
4.25.1 under repository `.venv` Python 3.12.8 before dispatch. Its source is
reviewed before execution. The dependency import was checked while idle;
no engine, qualified binary, numerical helper, workload, quality gate or test
policy changed. M3.1's seven non-short rows remain pending until accepted
replacement records exist. The extra run adds approximately twenty minutes of
numerical work plus evidence checks/review.

Astra independently verified all eleven archived hashes, ten excluded state rows,
actual dependency failure, cleanup and protected endpoint identities, and
accepted this exclusion evidence task without findings. All eight report links
and whitespace checks pass. This accepts the archive, not the benchmark.

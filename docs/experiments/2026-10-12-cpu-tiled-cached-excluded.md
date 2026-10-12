# Excluded paused cached attempt, 12 October 2026

The tiled cached-v2 attempt is wholly excluded. The user requested pause at
05:50:30 UTC; all three owned processes were observed stopped with SIGSTOP.
Five measured journal rows had been retained, but no successful result JSON
was published. No timings from this partial attempt count toward acceptance.
M3.1 remains open for a fresh cached run, pending-token and changed-prefix
workloads, the full application trace and retention.

On resume, the recorded benchmark PIDs 96097/96112/96121 and original
orchestration graph 17694/85019/85027/88184, plus old execution parent 35939,
were absent. Native session60442 returned “Unknown process id”; its exit status,
exact exit time and disappearance cause are unavailable. The reviewed retirement
script failed its initial process-identity assertion before opening its dispatch
receipt or sending signals. The numerical worker was not resumed by recovery.
These observations do not demonstrate natural completion or a successful run.

The [pause receipt](../../benchmarks/results/2026-10-12-cpu-tiled-cached-v2-excluded-user-pause.json)
preserves full before/after process snapshots. Its producer was an inline tool
command; a separate exact source file was not retained. The
[recovery observation](../../benchmarks/results/2026-10-12-cpu-tiled-cached-v2-excluded-recovery-observation.json)
preserves the unknown-session query, actual failed retirement tool result and
bounded PID-absence observations. The
[failed retirement source](../../benchmarks/results/2026-10-12-cpu-tiled-cached-v2-excluded-failed-retirement.py.txt)
is retained exactly; its current-state admission prevented signals against
unavailable identities.

The [excluded journal](../../benchmarks/results/2026-10-12-cpu-tiled-cached-v2-excluded.samples.jsonl)
and [current idle proof](../../benchmarks/results/2026-10-12-cpu-tiled-cached-v2-excluded-idle.json)
retain eleven exact archived files and their hashes, including the original
claim, launch, owned observation and executed launcher/observer/driver sources.
The [archive producer](../../benchmarks/results/2026-10-12-cpu-tiled-cached-v2-excluded-idle.py.txt)
checks all eight PIDs absent and both process groups empty before endpoint
hashing. At 07:49:31 UTC it verifies Podman stopped, clean dispatch revision
`06d120ac1cc1e28a743d6d2e2f6181fcc3410305`, both protected root binaries, all
eight model/tokenizer assets, the isolated worker and compiled sources.
Current absence replaces the former live held-orchestration assumption; it
must be admitted explicitly by the fresh retry rather than calling the old
helper that requires PID17694 still held.

The fresh retry must reserve a separate prefix, use pinned repository Python
3.12.8/jsonschema 4.25.1, admit this recovery proof and current process absence
before heavy hashing, and retain the accepted append predecessor and frozen
quality/qualification bindings. No resumed or reconstructed v2 samples may
substitute for three fresh warmups and ten measured requests. Obsolete natural
completion/EOF-based v2 archivers and the v2 success analyzer remain unexecuted.
Only saved-evidence, content/link and whitespace checks apply; no engine,
qualified artifact, numerical helper or quality gate changes are made.
Astra accepted archival and exclusion integrity only, without findings, after all eleven hashes/exact original copies, five excluded rows, saved observations and six report links passed independent checks. No performance acceptance is implied.

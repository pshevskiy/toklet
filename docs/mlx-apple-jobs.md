# Authenticated native MLX diagnostic jobs

Measurement, teacher-forced quality, task-response and lifecycle/state jobs now
have an explicit version-2 selection: `"version":2,"mlx":true,"metal":true`
with an empty separate Q4 folder. Every other field and its existing bounds,
SHA/inventory policy, foreground checks and numerical behavior stay unchanged.
Version 1 retains its exact original field set and has no `mlx` field. Missing,
null, false or wrongly typed MLX selection, CPU/Q4 conflicts, unknown versions
and extra fields reject before loading; there is no implicit fallback.

All five runners propagate the authenticated selection into the existing worker
and report backend `metal-mlx`. The shared lifecycle/state canonical job location
remains `M7Jobs/<SHA256>.json`; the directory name does not change the versioned
contract. Results continue to keep full-quality, performance/device and milestone
qualification separate from a completed diagnostic request.

The [explicit Mac checker](../apple/tests/swift/MLXJobsMain.swift) admits actual
SHA-authenticated decoded version-2 objects before execution. Four config schemas
reject ten malformed variants each and also admit their legacy version-1 forms.
On the frozen packed tiny fixture it completes 13 measurement requests (three
warmups/ten measured requests with fixed four-ID outputs), three teacher-forced
targets, two task responses and the bounded prefix/capacity/state suite. Each
positive component closes/releases its native owners and rechecks artifact inputs.
These small diagnostic samples establish routing and instrumentation, not held-out
quality or model performance.

The negative lifecycle check actually reached READY, then deliberately threw
`Forced READY sink failure`. Native owners closed and released, and artifact
reproof passed. The existing trace nevertheless reported `Lifecycle detach
requires completed close and no active command` in cleanup errors. This is retained
as an open trace-cleanup gap for a separate code task; it is not accepted as clean
lifecycle completion or actual OS behavior. The check permits either the injected
READY error or an absent tiny active window; this recorded run reached READY.

Strict Mac Swift compilation/execution also passes the existing CPU/Half
measurement, quality and task checkers (17, 19 and 31 case labels respectively).
The complete iOS18+ app sources typecheck, and fresh release linked simulator/
unsigned device apps build against the previously reviewed native package. All
12 owned outer commands plus nine app-build child commands pass with unchanged
source/archive/protected-binary endpoints. Rust/shader bytes are unchanged;
ordinary Rust suites are not reported as fresh runs for this Swift-only task.
The [task receipt](../benchmarks/results/2026-10-22-native-mlx-apple-jobs-v1.json)
retains the positive results and the negative lifecycle trace without editing them.

The real public test checkpoint is now acquired at immutable revision
`a5339a4131f135d0fdc6a5c8b5bbed2753bbe0f3` of
[mlx-community/Qwen2.5-0.5B-Instruct-4bit](https://huggingface.co/mlx-community/Qwen2.5-0.5B-Instruct-4bit).
Ten files total 289,599,545 bytes; the 278,064,920-byte weight SHA256 is
`ddffab9cbc7bf6dde941c6724841eeca8981fcfa81ca20ff8efff1396326d153`.
The data-only acquisition uses HTTPS, exact revision/literal names, a 300 MiB cap,
per-file sizes and LFS/Git-object identities, a captured inventory SHA and atomic
exclusive publication. Three portable tests and all eleven download commands pass.
Astra's pre-execution size, inventory-binding and publication findings were fixed
before downloading. This is experimental data acquisition, not the app's future
search/download feature.

Header inspection finds Qwen2, affine four-bit/group64 tensors, Half metadata,
a packed tied embedding and no separate lm_head. Vocabulary, merge rules and
added ID/token pairs match the frozen source tokenizer. Full tokenizer files and
the chat-template hash differ, so native CLI chat-template compatibility is not
assumed. No real target-model numerical inference has run yet. Independent gates,
real-model trials and physical iPhone testing remain next in the
[M8 plan](mlx-implementation-plan.md).

# Shared app-worker MLX metadata preflight

This H1 step adds a small `NativeCompatibility.swift` adapter for the immutable
[native report](mlx-compatibility-consumers.md). Before an MLX load, the existing
inference thread creates, copies, decodes and releases the report synchronously.
The app passes the exact caller budget. There is no extra numerical worker,
architecture parser, network request or per-token work.

The decoder checks schema, backend, context, budget, pending checks and exact
reason/admission/backend-availability combinations. Compatible metadata permits
an attempt to load; it never grants a text profile or skips actual loading.
The original loader still checks current files, payloads, GPU limits and session
admission. The three pinned text/download profiles remain separate.

A captured summary identifies the last attempted artifact. A new attempt clears
it before argument validation. Suspend, unload and close clear both worker and
published state; superseded work cannot republish an old report. In particular,
a queued load replaced by Suspend must not resurrect the preceding artifact's
metadata. No live-model replacement or cancellation contract changes.

The helper states an **additional application allowance requirement** of
655,616 logical bytes: the native 131,328-byte serialization/owner allowance plus
512 KiB for Swift output/decode storage. It is separate from numerical admission,
not an enforced reservation, allocator-overhead estimate or process-peak claim.
The current pinned-profile process allowances have room for that requirement;
the real runtime checks remain authoritative.

The C allowance macro uses the equivalent literal so Swift can import it. A C
static assertion verifies its formula. All native layouts, entrypoints and Rust
numerical sources remain unchanged. The Xcode module registration and explicit
checker source lists include the new module; indexed source subsets also include
it for their platform checks.

## Checks and boundaries

All 89 portable checker methods, the strict default/Metal Swift checks and
existing CPU/Half-Metal/MLX worker regressions pass. The wrappers exercise independent malformed/wire
mutations, exact budgets, unsupported geometry, unknown graphs and actual real
Qwen3 metadata. Deterministic dequeue gates assert the idle Suspend/Unload
snapshot is clear **before** Close, so Close cannot mask a stale-report bug.
The existing CPU, Half-Metal and MLX worker checks still pass ownership,
pending-token, cancellation and replay behavior. C compile/link and strict iOS
and simulator module checks cover the additive boundary. Portable checker tests
remain independent of model downloads and GPUs.

The first captured portable run selects the old system Python, which lacks
`hashlib.file_digest`. The retry uses the orchestrator's explicit Python
interpreter and passes without a production change. Both captures are retained.
The fresh unsigned native package passes Mac, iOS and simulator build/link checks.
These checks alone do not qualify physical iPhone execution.

The three existing real Mac Chat profiles (SmolLM2, Qwen and Mistral) also pass
through the new preflight using immutable source snapshots: nonempty short
replies, exact fresh replay, context-rejection prefix preservation, pre-cancelled
turns, explicit Send after suspend and owner release. These are short functional
regressions, not a new quality or performance qualification. No Qwen3 Chat
profile is introduced.

A separately authorized mobile UI task owns the Friendly/Hub/Workspace/Expert
presentation changes. Its source snapshots and engine checks must be identified
separately; this task does not commit or qualify another task's moving UI files.
The [task receipt](../benchmarks/results/2026-10-24-mlx-compatibility-swift-v1.json)
records exact sources, commands and inherited unchanged Rust checks.

Retain shared-worker metadata preflight and truthful invalidation. H1 remains
open for actual runtime resource planning and exact qualification receipts.
General Hub acquisition, downloaded text profiles and full app/device flows
remain open. No new graph, text profile, numerical policy, performance or physical
phone qualification follows from this adapter.

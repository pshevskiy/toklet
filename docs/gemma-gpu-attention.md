# Reusable FP32 GPU attention workspace

15 November 2026. G2 building block for the [competitive GPU goal](mlx-gpu-performance-plan.md).
The ordinary Gemma route is unchanged. This operator and its borrowed encoder
prepare resident block composition; complete graph, quality/state and request
performance qualification remain open. Astra approves the scoped owner/encoder with no findings.

## Ownership and execution

[Context and Workspace](../src/metal/gemma_attention.rs) cache four precise
pipelines and fixed buffers. Options admit up to 32 queries, 1,025 keys, eight
query heads, one KV head and D512; scale is exactly one. Q/context are
`[queries,heads,D]`; diagnostics are `[heads,queries,keys]`. Three immutable
logical KV spans preserve ordering and absolute/window masking.

Every capacity, shape, length, span origin/stride, position endpoint, visible
interval and finite input is checked before staging. Insufficient budget rejects
before numerical buffer allocation. Context policy reserves 64 MiB separately;
workspace bytes include buffer payloads and its Rust value. Caller inputs,
outputs, diagnostic arrays, compiler/allocator/driver overhead and process
footprint retain separate scopes.

The largest tested capacity admits 15,743,664 workspace bytes. Persistent
buffers are reused when query/key counts or span splits change. Four dependent
encoders share one command and completion wait. Successful `apply` publishes
only context after completion, full inactive/guard checks, status and finite
outputs pass. `copy_diagnostics` explicitly copies the last successful scores
and probabilities. A preflight error preserves previous results; an execution
failure invalidates diagnostics and permanently poisons the workspace. Pending
command/resources remain owned; recreate a poisoned owner.

The crate-private `Pipelines` encoder uses an existing Gpu and a caller-owned
command/status. Checked buffer ranges support element offsets into shared
allocations: read-only wrapped KV spans may alias; ranges overlapping writable
scratch/status are rejected before encoder creation. Same-device, retained
resources and NotEnqueued command state are required. The caller owns
synchronization, completion validation and publication. This lets future block
operators share a command without a second Context or queue. It does not yet
change the synchronous projection/CPU graph boundary.

The [shader](../src/metal/gemma_attention.metal) is byte-identical to the approved
feasibility experiment: three-term compensated dot/context accumulation with
product-error recovery, Safe/Precise FP32 softmax and explicit FP32 boundaries.
The independent scalar CPU implementation remains unchanged.

## Verification and measured scope

Fmt, strict all-target CPU/Metal Clippy, 847 CPU tests and 1,022 Metal tests pass
(53/369 ignored), plus the release build. Explicit GPU tests cover borrowed ring
offsets/disjoint shared scratch, numerical reuse, rejected preflight and isolated
masked-overflow poisoning. A separate optimized test observes zero Rust
allocations in 32 warmed changing-shape operations and rejected admission;
its owner admits 311,152 bytes. This does not count native driver allocations.

Independent normal and optimized-Python replay agree across 2,844,296 values:

| Input family | Comparisons | Scope |
| --- | ---: | --- |
| Eight frozen mathematical cases, two layouts | 932,968 | Full/local/translated/cancellation, D3–D512 and up to 1,025 keys |
| Saved original-BF16 graph inputs, two layouts | 376,320 | 24 four-position prefill/decode cases; early/late/shared-KV layers |
| Independent capacity controls, two layouts | 1,535,008 | 32-row prefill, 1,025-key decode and translated W512 spans; synthetic uniform/dyadic oracle |

All use the original `1e-5 + 1e-4 * abs(expected)` bound. Maximum bound fraction
is 0.011921; 136,568 masked probabilities are exact positive zero. Segment
layouts agree bitwise. Eight active-output reader corruption controls reject,
including a masked 1e-6 value and negative zero. The isolated finite-input fault
has exactly one nonfinite score at the masked pair; caller output remains
unchanged, status rejects and subsequent reuse/diagnostics reject.

Actual GPU guards close cleanly below 512 MiB: largest sampled charge is
370,509,264 bytes and separate terminal direct-pair charge 435,404,800 bytes,
including helper reserve. These are observed owned-process scopes, not a
continuous allocator cap or whole-model memory result.

The first compile stops at a 3 GiB screening limit; another attempt finds a
closure annotation error. Both are retained. Metal test compilation then uses
its existing 6 GiB suite ceiling; the inference ceiling stays 6 GiB. Frozen
fixture data now uses bounded authenticated runtime reads. A controller folder
collision occurs before spawning the own-input probe; completed stages are
reused and the remaining probes pass under separate guard paths.

[Required checks](../artifacts/mlx-gpu-gap-2026-11-15/g2-workspace-required-progress-4.json),
[explicit GPU/allocation checks](../artifacts/mlx-gpu-gap-2026-11-15/g2-workspace-hardware-progress-2.json),
[independent replay/raw identities](../artifacts/mlx-gpu-gap-2026-11-15/g2-workspace-replay-1.json),
[source freeze](../artifacts/mlx-gpu-gap-2026-11-15/g2-workspace-source-4.json).

## Shared normalization and rotary encoding

The crate-private RMS and rotary pipelines now compile on the same existing
Gpu as attention and accept checked borrowed ranges and a caller-owned retained
command/status. Standalone owners use these same encoders. The three shaders
remain byte-identical; coefficient preparation retains the original FP64 phase
and trigonometry followed by FP32 storage. Public standalone rotary keeps its
eight-token limit; only the internal composable path admits 32 tokens.

One guarded shared allocation runs RMS -> rotary -> attention with one command
and completion wait, without intermediate CPU staging. The frozen small case
and synthetic 32-query D256 local / D512 global cases pass. Independent normal
and optimized replay reconstructs 622,632 reference values from captured inputs,
with 9,025 exact masked positive zeros and 98,308 inactive rotary copies; all
33 corruption controls reject. Command/wait counts follow the pinned source and
capture metadata, rather than an independent timing trace. Invalid shapes,
lengths, overflow, writable overlaps, nonretaining and completed commands reject
before encoding; valid read-only aliases work. Foreign-device execution cannot
be exercised on this single unified GPU, though pointer admission is explicit.

Fmt, strict CPU/Metal Clippy, 847 CPU/1,024 Metal tests (53/371 ignored), release,
seven explicit GPU tests and the reused 32-operation allocation witness pass.
Raw diagnostic captures use bounded authenticated binary sidecars. Earlier
JSON captures exceeded memory/log limits; those failures remain preserved.
[Failure and repair](../artifacts/mlx-gpu-gap-2026-11-15/g2-shared-encoders-memory-failure-1.json).
Astra approves the scoped encoder/composition change with no findings. This qualifies composition primitives; the
ordinary model graph, whole-block/state/quality and request performance remain
open. No CPU/GPU buffer copy reduction is claimed for ordinary generation.

[Required checks](../artifacts/mlx-gpu-gap-2026-11-15/g2-shared-encoders-required-progress-3.json),
[GPU/allocation/readers](../artifacts/mlx-gpu-gap-2026-11-15/g2-shared-encoders-hardware-progress-4.json),
[independent raw replay](../artifacts/mlx-gpu-gap-2026-11-15/g2-shared-encoders-replay-normal-2.json),
[source freeze](../artifacts/mlx-gpu-gap-2026-11-15/g2-shared-encoders-source-3.json).

## Next mechanism

Use this borrowed seam for one complete accurate producer block on the existing
BF16 context. Preserve projection admission using cold thresholds and GPU
finite/max/L1 checks, qualify FP32 GELU separately, and keep proposed KV private
until the final checked publication boundary. Retain file-backed PLE and commit
only new shared KV rows. [Apple hardware focus](apple-silicon-gpu-goal.md).
No engine speedup, summary-overhead pass or external win follows from these
primitive checks.

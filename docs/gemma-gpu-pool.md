# Gemma interlayer GPU scratch pool

16 November 2026. Private continuation of the [three-layer chain](gemma-gpu-chain.md).
The ordinary model, session and CLI selection is unchanged. Source3 passes all
eight original cells: 2,375,680 values, 480 exact masked zeros, 64 causal-prefix
states and eight atomic dual-cache commits, with no replay. Original unpooled numerical/state/RMS/warm and producer regressions pass.
Fmt/strict CPU+Metal Clippy, 1,047 Metal tests (397 ignored), release CLI and the
certified release unit pass; 847 CPU tests reuse 4,009 byte-identical inputs.
Astra approves scoped code retention with no required fixes; no request speed is qualified.

One session-owned shared tracked pool holds 28 numerical state spans and four
GELU spans, disjoint within each node. Entry Hidden is borrowed; later Hidden
views borrow persistent boundary outputs. Only PLE inputs need inter-node copies.
All three outputs, both producer K/V proposal pairs, committed caches, status/RMS
buffers and immutable entry checkpoint remain outside the pool. `Last` narrows
only consumer15 queries; both producer proposals retain every new row. Qualified
active work remains at most four rows/positions with physical cache capacity 512.

All three recipes and the complete immutable audit worklist are validated before
encoding. After nodes 13/14, a GPU integer audit checks every active/inactive/guard
word before sentinel fill; node 15 audits without fill. Dedicated chunk receipts
start invalid, so missing work cannot imply success. Queried execution width,
checked ranges and tracked resource dependencies retain one command/wait.
The nominal chain has 102 numerical plus 3 audit dispatches.

Guards, descriptors, missing/unknown receipts and status/proof grammar are global
fatal contracts. Ordered numerical uncertainty may replay the whole chain from
its entry checkpoint and committed prefix before candidate-nonfinite receipts
are considered. It cannot hide integrity failures. Successful replay validates
all CPU logical states and restores persistent boundaries/proposals plus the
final pool; speculative retired nonfinites are discarded. Final-pool/live-state
checks and receipt snapshots repeat before prevalidated cache copies; both cache
metadata updates remain last. Failure/cancellation preserves the committed
prefix, and queued Drop drains before unsafe reuse is poisoned.

| T4/K512 owner scope | Bytes |
| --- | ---: |
| Shared GPU pool payload | 3,298,176 |
| GPU audit descriptors/active ends/receipts | 24,780 |
| CPU receipt snapshots | 10,044 |
| Session owner without retired-state diagnostics | 13,813,420 |
| Session owner with retired-state diagnostics | 15,867,084 |

Host byte counters cover instrumented logical spans. Active-count rewrites, snapshot destination writes and publish receipt-comparison reads are not all charged separately; these counters do not establish exhaustive CPU traffic.

Session figures include actual Rust owner/capacity accounting and retain the
7,163,992-byte CPU reference payload. Shared model weights, graph, certified
GELU table and existing pipeline policy are separate, counted once. The audit
adds one 24-byte PSO wrapper in the shared residual cache under its existing
policy; native driver/PSO bytes are unknown. These owned bytes are not measured
physical working-set usage. T1/T32 capacity ledgers are planning, not new target
qualification.

Nominal diagnostics add 56 explicit retirement-state copy calls; normal pooled
work has three PLE copy calls, while the resident-entry diagnostic adds four
checkpoint copies. Test fault fills are counted separately. All 23 lifecycle/
priority controls and both readers pass, including 19 atomic failures and whole
replay after a speculative descendant NaN. Thirty-two changing-row evaluations
have zero measured Rust allocations, one command/wait each and no replay;
native allocations are unmeasured. The integer leaf separately passes 58 GPU
cases and 1,083,904 raw words at queried widths including 1,3,7 and 32.

The initial controls reader rejected the intentionally unobserved queued-Drop
timing record. Reader-v6 recognizes only that exact cleanup phase, requires its
complete null/reason schema, and rejects coherent fabricated timestamps. Both
modes retain 14 prior timing corruptions plus 3 cleanup controls. No GPU execution
or source was changed for this reader repair. Coarse begin-to-publish timing
includes any caller diagnostic gap; it cannot establish utilization or a speedup.

[Astra code review](../artifacts/mlx-gpu-gap-2026-11-15/astra-scratch-pool-code-review-1/verdict.json),
[publication/source receipt](../artifacts/mlx-gpu-gap-2026-11-15/g2-scratch-pool-publication-1.json),
[Implementation](../src/metal/gemma_affine/bf16/resident/producer/chain/pool.rs),
[source/check preparation](../artifacts/mlx-gpu-gap-2026-11-15/g2-scratch-pool-final-prep-1/handoff-v1.json),
[initial actual hardware and retained reader failure](../artifacts/mlx-gpu-gap-2026-11-15/g2-scratch-pool-hardware-progress-1.json),
[remaining hardware/readers](../artifacts/mlx-gpu-gap-2026-11-15/g2-scratch-pool-hardware-progress-2.json),
[reader-only repair](../artifacts/mlx-gpu-gap-2026-11-15/g2-scratch-pool-draft-1/reader-handoff-v2.json),
[final actual regressions/checks](../artifacts/mlx-gpu-gap-2026-11-15/g2-scratch-pool-final-prep-1/runs-continuation-1/result.json).

The first old-nominal reader was interrupted at the unchanged 2 GiB disk floor
after native success. Sources/captures are unchanged; the continuation
authenticates passed native/listing receipts, reruns both readers and executes
all remaining work. Only task-owned regenerable Cargo debug cache was reclaimed;
actual recovered free space was 8,158,031,872 bytes.
[Retained failure/continuation](../artifacts/mlx-gpu-gap-2026-11-15/g2-scratch-pool-final-prep-1/continuation-handoff-v1.json),
[cleanup receipt](../artifacts/mlx-gpu-gap-2026-11-15/owned-debug-target-cache-reclaim-1.json).

The isolated observed nominal peak is 4,696,183,416 bytes including the 32 MiB
helper under the unchanged 6 GiB sampled process guard. This is not a continuous
allocator cap or full 35-layer/codec fit. CPU staging/final validation/new-row
commit copies remain. Full-graph CLI integration, original whole-quality/state,
representative request latency, external-engine wins and the original overhead
gates remain open; summary defaults stay disabled.

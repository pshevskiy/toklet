# Gemma GPU RMS scheduling

16 November 2026. All 32 final regression/check stages pass. Astra approves
retention of this private bounded graph with no required fixes.

This private continuation of the [resident chain](gemma-gpu-chain.md) and
[scratch pool](gemma-gpu-pool.md) changes scheduling of the qualified higher
precision RMS operation. The ordinary model, session, producer pilot and CLI
selection remain unchanged. `ResidentGraph::new_two_phase` selects the paired
family once at cold construction; the existing constructor retains its original
family. Both use unchanged CPU reference/replay and original numerical gates.

Phase A computes each row's scaled compensated statistics once. Phase B applies
the same qualified denominator and output arithmetic to independent columns.
The original guard2 helpers, operation order, epsilon bits, precise compiler
policy and finite/subnormal uncertainty rules remain. This is a scheduling
experiment over the existing arithmetic, with two dependent dispatches per RMS
operation and no host wait between them. The three-node chain has 22 RMS
operations: 124 total dispatches without pooling, or 127 including the three
pool audits, within one retained command/completion.

A session owns one disjoint bank for all 22 operations, including 32-byte row
contexts, operation headers and guards, plus a CPU snapshot. Cold descriptors
bind active rows/widths, queried execution width, operation identity and bounded
epoch. Pending status and exact completed-group counts distinguish missing work
from success. Reset validates every shape and epoch before mutation; exhaustion
rejects rather than truncates. At T4 the GPU-bank and CPU-shadow payload totals
14,232 bytes; actual owner/plan capacities and pipeline wrappers are accounted
separately. The existing precision pipeline policy remains; native PSO/driver
allocation is unknown. T32's 91,288-byte payload is planning only: active
qualification remains at most four rows with physical cache capacity 512.

All node contracts, row/header grammar, guards, pending/unknown status and group
counts are checked before ordered numerical replay. Whole-chain uncertainty
replays from the immutable entry checkpoint and committed prefix. Candidate
finite checks remain after an otherwise valid replay decision. Descriptor and
receipt snapshots are checked again before publication; both producer cache
commits retain validate-all, copy-all, metadata-last semantics. Partial encoding
makes dispatch count unavailable and drains/discards without publication;
cancellation, stale evidence, failure and queued Drop preserve the valid prefix
or poison unsafe reuse.

Both pooled/unpooled eight-cell frozen state sets pass independently in normal
and optimized readers: each checks 2,375,680 values, 480 masked zeros, 64 causal
prefix states and eight atomic dual-cache commits, with no replay. All logical
state bits also match the retained default guard2 captures. Source4 retains the
source2 three pure/one Rust leaf and 16 nominal successes under an exact inverse
source proof. The 28 fresh lifecycle controls and 64 warmed changing-shape
iterations pass; measured Rust allocations are zero, native allocations unmeasured.
All twelve guarded paired readers pass, including rehashed receipt/state,
core-table and timing corruption controls. The observed model-process peak is
4,711,633,504 bytes including the 32 MiB helper under the 6 GiB sampled guard;
this is not a continuous allocator cap.

Completed GPU-command medians on the attested M1 Pro, in milliseconds:

| Run | Work | Default guard2 | Paired | Reduction |
| --- | --- | ---: | ---: | ---: |
| 1 | Prefill, four rows | 167.874375 | 50.896708 | 69.68% |
| 1 | Decode, one row | 137.289708 | 43.983979 | 67.96% |
| 2 | Prefill, four rows | 167.905333 | 51.337688 | 69.42% |
| 2 | Decode, one row | 137.272333 | 43.766187 | 68.12% |

Each run retains first use, three warmups and ten samples per family/workload,
with reversed comparison order in run2: 56 commands/waits and 6,496 dispatches.
These are completed three-layer pooled-command intervals including caller work
inside the command, excluding reference replay. Timing records contain no
independent numerical output arrays; unchanged Rust per-sample checks and the
separate 16 raw-state cells provide numerical evidence. No additive extrapolation
or full-model, request or external-engine speed follows from these medians.

The first build failed strict visibility/argument-count lints. Narrow scopes and
argument grouping fixed them without shader changes. Two controls attempts
then exposed storage-specific test helpers: reference injection required an
unpooled owner, and a successful pooled replay lacked retired-state diagnostics.
Only the new test helper/admission was corrected; failed attempts remain retained.
The reader preparation also retained and fixed a no-op mutation by selecting an
actual paired timing record. No tolerance, fixture or numerical arithmetic changed.

Source4 fmt, strict Metal Clippy and the certified release unit pass. The actual
unit listing is 1,038 tests/311 ignored. All 32 final stages pass: original
chain/pool/producer regressions, unchanged normal/optimized readers, fresh strict
CPU Clippy, 1,050 full Metal passes/403 ignored and release CLI. Reuse of 847 CPU
tests is supported by all 4,009 byte-identical non-Metal inputs and original logs.
The combined Astra code review approves this scope with no required fixes.
Byte counters represent instrumented logical spans, rather than exhaustive CPU
or memory-controller traffic.
CPU staging, snapshot/finite validation and new-row cache copies remain. Fit of
all 35 layers, public route selection, request latency, external-engine wins,
overhead and full-quality gates remain open.

[Implementation](../src/metal/gemma_affine/bf16/resident/producer/chain/rms_phase.rs),
[paired encoder](../src/metal/gemma_rms/precision/two_phase.rs),
[source4 reuse proof](../artifacts/mlx-gpu-gap-2026-11-15/g2-rms-two-phase-graph-source4-reuse-proof-1.json),
[paired raw readers](../artifacts/mlx-gpu-gap-2026-11-15/g2-rms-two-phase-graph-readers-run-1/result.json),
[timing run1](../artifacts/mlx-gpu-gap-2026-11-15/g2-rms-two-phase-graph-readers-run-1/timing-1-normal.json),
[timing run2](../artifacts/mlx-gpu-gap-2026-11-15/g2-rms-two-phase-graph-readers-run-1/timing-2-normal.json),
[final-check preparation](../artifacts/mlx-gpu-gap-2026-11-15/g2-rms-two-phase-final-prep-1/handoff-v1.json).

[Final required checks](../artifacts/mlx-gpu-gap-2026-11-15/g2-rms-two-phase-code-review-prep-1/final-check-completion-1.json),
[combined Astra code review](../artifacts/mlx-gpu-gap-2026-11-15/astra-rms-two-phase-code-review-1/verdict.json).

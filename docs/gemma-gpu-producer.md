# Gemma resident producer-block pilot

15 November 2026. Explicit private experiment for official Gemma 4 E2B BF16;
the ordinary CLI graph remains unchanged. The pilot covers layers 0 and 4,
local/global attention, one to four input rows and positions through four. It
does not qualify the complete 35-layer graph, full model quality, request speed,
summary overhead or an external-engine win.

## GPU ownership and work

One existing Metal device/queue owns the shared tracked buffers. The block
borrows nine authenticated BF16 matrices, the seven learned normalization vectors
and learned final scalar. It shares cached RMS, rotary, attention, residual,
certified GELU and admission pipelines; weights are neither cloned nor expanded.
The immutable GELU table is shared and admitted against its native compiler,
CPU leaf, host/OS and reviewed certificate. An unmarked/debug build fails closed.
[Table certificate and limits](gemma-gpu-gelu.md).

Linear weights retain row-major `[out,in]` packed BF16 storage. Activations use
`[rows,width]` regions of a checked guarded FP32 arena: hidden width 1,536,
per-layer input 256 and FFN width 6,144. Attention has eight query heads, one KV
head and head dimension 256 or 512. Separate K/V buffers use
`[kv_heads,capacity,head_dim]`; local capacity is 512, while this bounded global
pilot has capacity four. Proposed K/V remain private until publication.

The complete block encodes 35 compute dispatches into one retained caller-owned
command. Host-input cases have no device blits; resident-input cases checkpoint
the two source views with two GPU blits in that same command. There is no CPU
readback or wait between projections/operators. Completion, validation and
publication still touch shared memory on the CPU; this pilot does not remove
those costs or establish full-graph transaction semantics.

## Conservative projection admission

A full integer finite/max scan runs after each input producer. Finite max misses
then use a source-order positive FP32 L1 upper sum on the GPU. Nonzero subnormals
are inflated to the smallest normal value; each subsequent addition advances
two positive FP32 encodings. Overflow or uncertainty requests the unchanged
reference path. The cold cutoff is found through the original upward-rounded
`l1_bounds` helper, with passing endpoint and failing successor checks. Q/K/V and
gate/up groups select the largest actual scanned weight maximum and verify both
cutoffs cover every member. Original admission/error limits and projection
arithmetic are unchanged. [Proof and saved-input analysis](../artifacts/mlx-gpu-gap-2026-11-15/astra-gpu-l1-admission-research-1.md).

The max-only pilot exposed a real bottleneck: all four prefills requested
whole-block reference replay at FF-product admission. Retained raw runs show:

| Frozen workload | Max-only commands/waits | With GPU L1 commands/waits | Reference replays with L1 |
|---|---:|---:|---:|
| Four nominal prefills | 10 each | 1 each | 0 |
| Four nominal decode cases | 1 each | 1 each | 0 |

These are completed command/wait counts from the guarded explicit harness,
checked against actual publication records; they are not latency measurements
or independent GPU profiler counts. Both versions encode 35 speculative compute
dispatches. The older prefill additionally executes nine unchanged reference
projections. No speedup percentage follows from these counts.

## Publication and failure behavior

The encoded guard holds exclusive block/cache/source borrows through terminal
completion. Validate statuses, guards, exact source/frame identity, copy ranges
and cancellation before any committed write. Copy only new K/V rows and optional
caller output; update history, offset and generation last. Cancellation or
failure preserves the committed prefix. Dropping queued speculative work drains
the command before releasing resources and poisons an unvalidated block.
Unknown/reserved status and GELU certificate faults globally reject; an earlier
reference request may discard later ordinary numerical faults. Reference replay
uses bounded scratch and the original CPU operation order, without a second live
FP32 model. Graph consumers must discard/replay dependent speculation when an
earlier layer replays; per-block publication is not whole-graph atomicity.

Metrics separate encoded/speculative/completed dispatches, submission/wait
ownership, reference deltas, input/checkpoint/readback/new-KV bytes and shared
resource identities. Pipeline allowances are logical policy, not measured
physical allocations; shared storage is counted once by identity. Explicit
inspection bytes stay distinct from ordinary copies. No per-element logging or
operator clocks are added.

## Validation and remaining scope

The fresh original-checkpoint authentication and guarded model run pass. All
eight nominal cases have zero replay and one submitted/completed command/wait.
Independent normal/optimized readers compare 847,360 frozen values at the
unchanged `1e-5 + 1e-4*abs(reference)` bound, maximum fraction 0.381, 192 exact
masked positive zeros and 46 exact causal-prefix pairs. Every validated nominal
arena equals its raw GPU candidate bit-for-bit. Twenty-two fault controls,
resident imports, reset/apply reuse, queued-drop progress and actual status
injections pass. Twenty-one block-reader corruptions reject.

Fifteen small GPU L1 controls and independent replay verify actual helper
branches/upper bits, 4,304 partial enclosures and seven corruptions. Required fmt,
strict CPU/Metal Clippy, 1,034 Metal tests (382 ignored), attested release
unit and release CLI builds pass. CPU tests reuse 847 passing cases because all
non-Metal source/dependency inputs are unchanged. Astra approves the combined
scoped pilot with no remaining findings. Steady-state block allocation and
request speed remain unqualified. The observed model-run
peak charge is 4,672,000,608 bytes including the helper, within 6 GiB; the guard
is sampled and does not prove an allocator cap. Earlier compile/reader failures
and the max-only replay remain retained evidence.

- [Source/build/compiler](../artifacts/mlx-gpu-gap-2026-11-15/g2-producer-build-progress-5.json),
  [actual GPU/model/readers](../artifacts/mlx-gpu-gap-2026-11-15/g2-producer-hardware-progress-4.json),
  [independent block result](../artifacts/mlx-gpu-gap-2026-11-15/g2-producer-replay-normal-4.json).
- [Combined Astra review](../artifacts/mlx-gpu-gap-2026-11-15/astra-producer-block-review-1/verdict.json),
  [publication](../artifacts/mlx-gpu-gap-2026-11-15/g2-producer-publication-1.json).
- [Current status and failed attempts](../artifacts/mlx-gpu-gap-2026-11-15/g2-producer-prerequisites-applied-1.json).
- [Next graph integration and remaining CPU touches](../artifacts/mlx-gpu-gap-2026-11-15/g2-resident-graph-integration-prep-1.md),
  [strided projection prerequisite](../artifacts/mlx-gpu-gap-2026-11-15/astra-bf16-multitile-research-1.md).

Next qualify checked multi-tile output, deeper own-input producer/consumer chains
and one evaluation-wide transaction. Full graph numerical/quality/session/memory
and request cohorts precede any default/adoption decision. The original eight
summary-overhead comparisons still have three failures; summary stays disabled
by default. [Competitive acceptance plan](mlx-gpu-performance-plan.md).

# Modular Qwen3 dense admission and scalar execution

24 October 2026. This is the native scalar part of E1 in the
[engine expansion plan](engine-expansion-plan.md), following the independently
frozen [Qwen3 oracle](mlx-qwen3-oracle.md). It adds local diagnostic Rust execution;
it does not enable Qwen3 in the installed app or Hugging Face download catalog.

## Module responsibilities

| Module | Responsibility |
| --- | --- |
| [mlx/qwen3.rs](../src/mlx/qwen3.rs) | Family-specific config validation: explicit head width, class, SiLU, epsilon/positions, full attention and ordinary RoPE; reject unsupported graph options |
| [mlx/dense.rs](../src/mlx/dense.rs) | Coarse architecture selection; existing family admission remains unchanged |
| [mlx/qwen.rs](../src/mlx/qwen.rs) | Existing shared bounded artifact reader, exact tensor inventory, packed/float validation and load admission; require both per-head norm vectors for Qwen3 |
| [mlx/reference.rs](../src/mlx/reference.rs) | Explicit diagnostic expansion and immutable weight ownership; select the shared Q/K-normalized dense constructor |
| [model](../src/model/mod.rs) | Checked dense execution, independently sized query/KV geometry, per-head normalization mechanism, session scratch reuse and numerical weight accounting |
| [model/trace.rs](../src/model/trace.rs) | Diagnostic-only Q/K norm stages and independently sized projection/context traces |

There is no new trait hierarchy, per-element architecture dispatch or runtime
model code loading. The existing `Source::open` and `reference::load_qwen`
entrypoints remain Qwen2-only. `Source::open_dense` and
`reference::load_dense` explicitly opt into the implemented dense graphs. The
reader's historical `qwen` module path remains API-compatible.

## Shapes, order and ownership

Let `H` be hidden width, `D` head width, `Q` query heads and `K` KV heads. Query
width `Q*D` need not equal `H`. Row-major weights are Q `[Q*D,H]`, K/V `[K*D,H]`
and O `[H,Q*D]`. Hidden activations remain `[tokens,H]`; Q and attention context
are `[tokens,Q*D]`; new K/V chunks are `[tokens,K*D]`. Session-owned K/V remain
separate per-layer head-major `[K,capacity,D]` buffers.

Each projected Q/K head receives RMSNorm using its own learned `[D]` vector,
shared across that kind of head, and the configured epsilon. Normalization
precedes split-half RoPE. The scalar implementation preserves ordered FP64
products/reductions and FP32 storage as the existing reference path does. V is
not normalized. Causal GQA, residuals, SiLU MLP, final norm and tied/untied output
ownership retain their existing order.

Normalization writes to existing rotary scratch, then RoPE writes back to the
Q/K projection buffers. Explicit traces capture raw projections before reuse;
normal decode adds no temporary allocation. Immutable norm vectors are loaded
once. Weight and session admission count their allocated capacities and the
allocated norm-table capacity, including deliberately over-reserved vectors.
Overflow and shape checks precede numerical execution.

The generic config now admits independent query width. The existing Toklet Q4
inventory explicitly retains its equal-width constraint, preventing an accidental
widening of that format's support. Qwen3 Metal loading rejects before GPU creation
until resident execution has its own frozen-gate result.

## Verification and remaining scope

The targeted suite compares all 430 FP32 stage arrays for tied/untied graphs,
five prompt/position/capacity cases and both actual upstream CPU/GPU final-logit
witnesses under the unchanged `1e-5 + 1e-4 * abs(reference)` gate. Thirty cached
chunk/offset combinations cover full reuse, reset and overflow recovery.
Further tests cover malformed/missing/nonfinite norm tensors, unsupported
configurations, exact load/session budgets, smaller/larger query widths, norm
ownership/capacity accounting, allocation-free decode, cancellation before start,
pending sampled-token continuation and changed-prefix replay/append.

Rust1.90 formatting and strict default/Metal-feature Clippy pass. The ordinary
CPU suite passes 161 tests (13 ignored); the Metal-feature suite passes 179
(91 ignored). All three existing tiny Qwen and two tiny Llama explicit GPU
regressions also pass on the Mac. The
[task receipt](../benchmarks/results/2026-10-24-qwen3-scalar-v1.json) binds all seven
owned check commands, exact sources/fixtures and raw results.
No shader or Swift source changes in this task. Protected release binaries and
the installed iPhone/Mac apps remain unchanged.

This qualifies the tiny scalar graph only. Resident Qwen3 Metal execution,
immutable real checkpoints, tokenizer/thinking/chat behavior, quality,
performance and physical iPhone trials remain open. The app continues to expose
its existing qualified profiles; general Hugging Face admission follows native
graph support in H1–H4.

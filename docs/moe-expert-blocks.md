# Selected-expert scalar blocks

5 November 2026. Implements the projection/reduction part of the
[Granite sparse-MoE plan](granite-moe-next.md). These are standalone FP32 CPU
reference leaves, with no model importer or Metal MoE execution yet.

`selected_expert_glu` borrows input `[T,H]`, distinct selected IDs `[T,K]`,
fused gate/up weights `[E,2I,H]` and down weights `[E,H,I]`. Gate rows precede up
rows **within each expert**. It writes `[T,K,H]` in the supplied token/slot order;
two caller-owned `[T,K,I]` scratch buffers retain gated SiLU and up values.
No table splitting, weight copy, runtime trait dispatch or allocation is added.
The readable reference reuses the existing checked scalar GLU, including its
ascending FP32 projection sums and stable SiLU; it is not a performance path.

`reduce_selected_experts` multiplies `[T,K,H]` expert values by nonnegative
finite FP32 gates `[T,K]`, then sums slots in their supplied order into `[T,H]`.
Each product rounds FP32 before accumulation. It does not renormalize weights,
reorder experts, add a residual or perform a Half conversion. All-zero gates
are valid; gates need not sum exactly to one.

Both leaves check positive dimensions, checked extent products, exact lengths
and finite operands before writes. Projection additionally requires at most
32 experts and valid, per-token distinct IDs; reduction allows at most 32 slots.
Invalid data anywhere, including unselected immutable weights, preserves all
outputs/scratch. Arithmetic overflow is a separate error: partial buffers must
be discarded. Existing scalar kernels and dense graphs are unchanged.

The independent nested-axis Python oracle was frozen before target execution.
Seven cases contain 29 token rows and 969 expected stage values. They cover odd
hidden/intermediate widths, 32 experts/top 8, expert permutation, perturbed
unselected weights and 7/8/9-token batches. Tests compare activated/up/expert and
reduced stages and repeat with token rows reversed. The prospective comparison
gate is `2e-6 + 2e-5*abs(reference)`; no threshold was changed after execution.

Four exact reduction controls distinguish supplied slot order, FP32 versus FP64
summation, a separately rounded product versus FMA, and zero gates. Validation
controls cover every slice length, shape overflow, duplicate/out-of-range IDs,
late nonfinite values and arithmetic errors. The existing allocation harness
checks 48 repeated projection/reduction pairs after warmup with preallocated
buffers, including 7/8/9-token shapes; no unsafe code was added.

[Fixture](../tests/fixtures/moe-experts-v1/fixture.json),
[checks](../artifacts/moe-experts-checks-2026-11-05/).
Half rounding, source import/bounded conversion, exact model planning, resident
Metal kernels and real sparse-MoE model qualification remain separate tasks.

Astra approves the final code/evidence with no actionable findings. All eight
check jobs pass: formatting, strict Clippy/default+Metal, 567 CPU/676 Metal-feature
tests, release library build, four optimized expert tests and optimized allocation
control. Source replay is byte-identical; 48 measured projection/reduction pairs
allocate zero Rust buffers.
[Result](../benchmarks/results/2026-11-05-moe-expert-leaves-v1.json).

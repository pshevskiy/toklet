# Granite checkpoint-defined dense scaling

5 November 2026. Implementation and checks pass Astra review. Real checkpoint
execution and managed Chat admission remain separate tasks.

Granite reuses dense projections, RMSNorm, SiLU, ordinary split-half rotary,
causal grouped-query attention and tied/untied head ownership. One immutable
`DenseScales` value supplies four source parameters: embedding multiply,
explicit attention-score multiply, branch multiply at both residual additions,
and final-logit division. Other adapters retain identity values and the old
attention division by sqrt(head dimension). Scalars resolve outside hot loops.
Weights remain immutable; session KV and scratch ownership do not change.

The adapter requires positive normal FP32 values for all four explicit fields.
Metal additionally rejects embedding/residual coefficients that cannot remain
normal Half. Those two scalar literals adopt Half dtype, matching the stock MLX
source experiment. The branch product rounds to Half before its residual sum
rounds to Half. Logit division remains FP32 division, before diagnostic readback
and greedy selection. No reciprocal substitution or fused residual store is used.

Metal pointwise launches index logical rows/columns through physical strides;
weights, KV and scratch stay resident. The scaled Reference attention entry
point preserves existing score/cache strides. Its small separate kernel leaves
legacy arithmetic unchanged; no new tensor layout or per-element variant
selection is introduced. Granite admits Reference only, at the initial512
position ceiling. Online, compensated Reference and experimental Qwen3 staged
schedules are not qualified for this adapter.

The independent source fixture has8 cases,312 stages, tied/untied heads and
absolute offsets0/5/509. Half FP32-ordered and FP64-wide profiles use the unchanged
frozen gate `max(1e-4,2*source-envelope)+1e-3*abs(reference)`; the scalar FP32
storage profile uses the existing `1e-5+1e-4*abs(reference)` gate. All five scale
omissions are detectable. Fixture regeneration must be byte-identical before
native execution. Authored Metal controls distinguish Half coefficient
promotion from FP32 multiplication and distinguish two residual stores from a
single store; padding and guard values are checked independently.

The source-only MLX experiment checks19,456 finite Half operands:
3,838 distinguish coefficient promotion and959 distinguish residual-store
boundaries. Neither source profiles nor gates were fitted to target results.

Evidence: [block checks](../artifacts/granite33-block-checks-2026-11-05/),
[source scalar experiment](../artifacts/granite33-inventory-2026-11-05/scalar-probe.json),
[frozen fixture](../tests/fixtures/granite-half-graph-v1/fixture.json).
No full-model fidelity, task quality, performance, app or iPhone claim follows
from these tiny graph checks.

Checks:558 CPU/667Metal tests, strict Clippy/default+Metal, formatting and
release build pass. Five explicit GPU graph/regression tests plus the authored
rounding/padding test pass. Astra approves the bounded block scope.
[Result](../benchmarks/results/2026-11-05-granite-scaling-blocks-v1.json).

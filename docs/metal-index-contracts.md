# Composite Metal index contracts

M8 B1c2a, 28 October 2026. Retained after ordinary checks, targeted GPU regression
and independent Astra review. [Evidence](../benchmarks/results/2026-10-28-metal-index-contracts-v1.json)
binds the exact two edited production files, frozen test binary and raw results.

The MLX shader uses UInt32 index arithmetic. Previously, packed product checked
each shape scalar and physical input/output length, but not the composed strided
address against UInt32. Gather had the same output-address gap. One small
`active_extent` helper now checks the last-row exclusive end with overflow-safe
host arithmetic, both the physical element bound and UInt32 bound before shape
conversion/encoding. Its actual consumers are product input, product output and
gather output. Existing alignment, packing and rounded-dispatch checks remain.

The session planner already calculated separate guarded KV/score counts with
checked host arithmetic. Those counts now also reject values above UInt32 before
allocation or casts. Positive dimensions make this bound cover the Reference
head/chunk/capacity intermediates. Existing queried activation `Layout` bounds
remain the source of truth for strides; no formulas are duplicated. Online keeps
its existing small score allocation and does not receive a Reference-only limit.

Cold tests cover exact limits, physical ends, stride/offset overflow and invalid
shapes. Synthetic planner configurations cross KV and Reference-score UInt32
thresholds without allocating numerical buffers. They are not executable models
or memory-fit claims. All existing eligibility and numerical policies remain.

Formatting, strict default/Metal Clippy, 383 default / 263 Metal-library tests,
targeted 9/10 Llama tests and release test compilation pass; counts overlap.
The unchanged strided MLX projection/gather GPU test passes with its existing
offset/padding/guard checks. Astra authenticates all sources, raw outputs and
nine closed command groups. The shader and model-admission dependencies are
unchanged. No real checkpoint, phone, larger context or speed claim follows.

Next compose the [isolated scaled rotary](llama3-metal-rotary.md) with model-owned
residency, structural Llama admission and independent tiny Half graph/session
validation. See [TODO.md](../TODO.md).

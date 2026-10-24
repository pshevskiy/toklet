# Native per-layer input composition

Two standalone CPU functions implement the frozen
[per-layer input contract](gemma-per-layer-reference.md) with caller-owned
buffers. `prepare_per_layer_projection_input` copies raw `[tokens,H]` embeddings
while multiplying by FP32 `sqrt(H)`, computed as FP64 sqrt then cast. Projection
remains an external backend operation; it consumes those scaled embeddings.
`finish_per_layer_inputs` accepts raw projected `[tokens,L*P]` values and optional
unscaled selected lookup rows. It owns no model/table weights and does not
require expanding quantized projections or embeddings to a second FP32 model.

Finish writes FP32 `projection * H^-0.5` to scratch, normalizes `[tokens*L,P]`
rows with existing FP64 RMS/direct learned weights `[P]` to FP32 output, then
optionally combines lookup. Lookup multiplication by FP32 `sqrt(P)`, FP32
addition and FP32 final `sqrt(0.5)` multiplication retain distinct rounding
boundaries. Constants are computed in FP64 then cast, without algebraic
cancellation. Projected-only output is the normalized projection; dispatch is
outside element loops. Token rows can flatten independent batch/sequence axes.

Geometry/size/length/epsilon checks and all finite main/projection/lookup/norm
inputs precede scratch/output writes. Zero dimensions reject; the reference
fixture's empty case denotes a no-call policy, not acceptance by these functions
or existing RMSNorm. Arithmetic overflow can leave partial scratch/output, which
the caller discards together before replay from immutable inputs. Intermediate
lookup overflow rejects even when a later algebraic simplification might hide
it. No buffers are allocated, retained or grown by these operations.

Six methods authenticate the prior fixture and compose existing scalar embedding
lookup and FP64-accumulation projection with the new functions. All six nonempty
cases compare scaled-main/projection/selected-row/scaled-projection/output
stages against both original FP64 and staged FP32 contracts under unchanged
policy. Scaled-main bits and selected rows match exactly. Hand cases distinguish
normalization axis/direct scale and projected-only behavior. Other tests cover
late invalid inputs before writes, main/norm/lookup/addition overflow, immutable
inputs and replay, exact independent-row chunks, buffer reuse, signed zeros,
empty policy and size overflow.

Formatting, strict default/Metal Clippy,251 CPU/273 Metal-feature tests and both
iOS cross-checks pass. Hardware suites remain ignored in the general checks.
The [receipt](../benchmarks/results/2026-10-24-per-layer-scalar-v1.json) binds
source, fixture, checks and unchanged prior engine/shader/protected binary
hashes. Astra reviews exact final evidence before commit.

This is the FP32 composition mechanism. Quantized projection/selected-row fetch,
BF16 checkpoint execution, per-layer gate/norm/residual/scalar order, full
Gemma/session/Metal, rotary parity, quality/performance and device memory fit
remain separate gates in the [Gemma4 plan](gemma4-implementation-plan.md).
No native ABI, capabilities, app limits or installed phone package changes.

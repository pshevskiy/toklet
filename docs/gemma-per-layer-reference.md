# Independent Gemma per-layer input reference

This G2 witness freezes token lookup and model-projection inputs before native
composition. Raw main embeddings are `[B,N,H]`; the model first multiplies them
by `sqrt(H)`. The projection weight is row-major `[L*P,H]`. Its output is scaled
by `H^-0.5`, reshaped to `[B,N,L,P]`, and RMS-normalized over P with direct learned
weights `[P]` and the admitted epsilon. Lookup selects token rows from
`[vocabulary,L*P]`, multiplies by `sqrt(P)` and reshapes to `[B,N,L,P]`. The normal
model path adds lookup and normalized projection, then scales by `sqrt(0.5)`.
The projected-only helper returns normalized projection directly.

The [fixture](../tests/fixtures/gemma-per-layer-v1/math.json) contains173 arrays
and253823 values in seven cases. Ideal FP64 stages and a separately labeled
staged FP32 contract both preserve operation order. The staged contract casts
scale constants, scaled main embeddings, projection, projection scaling,
normalization, addition and final output to FP32 at their explicit boundaries.
Projection products and normalization reductions use FP64. Neither contract
algebraically cancels `sqrt(H)` against `H^-0.5`; a test distinguishes that
shortcut by actual FP32 bits. Staged FP32 is not BF16 checkpoint qualification.

Cases cover hand literals, batches/repeated IDs, projected-only helper, P1,
empty mathematical no-call policy, actual H1536 with small L/P and actual
L35/P256 with small H. The actual hidden scale and layer-input shape are separate
bounded tests, not a combined full E2B-shape layer. Unequal nonunit/negative RMS
weights distinguish direct scale from `1+weight`; the hand case distinguishes
normalizing P from L*P and lookup `sqrt(P)` from `sqrt(H)`.

The adapter authenticates MLX-LM `gemma4_text.py` at revision
`0d4d3d04f8ab7ce4a51e7209e8615755a48ba94d`, SHA256
`73ffa75e9069ce7b2f864ccd8f0267bb80c5b5d4e2b73102b8a2a93fe03f936d`.
Its unchanged `_get_per_layer_inputs` and `_project_per_layer_inputs` helper
bodies execute with real FP32 MLX Embedding, Linear and RMSNorm weights. Only
annotations are deferred. Token IDs are admitted by the harness; the nearest-
token fallback for missing IDs/input embeddings is excluded. The admitted raw
main embeddings and observed scaled embeddings make the caller boundary
explicit. Projected-only helper coverage is separate from the ordinary model
path that supplies lookup inputs.

Each real operator stage is separately evaluated and retained before later
probe operations can fail. Those probes are independent executions, not
internal captures or fusion qualification of the unchanged helper calls. Both
helper outputs are also captured. Seven portable, three owned-process and
three runtime-admission methods pass. All six owned commands pass. Admitted
NumPy2.3.2/MLX0.32.3 CPU and GPU each retain71 raw arrays and pass142 comparisons
against both contracts under unchanged `1e-5 + 1e-4*abs(expected)`. Maximum
error/policy ratios are about0.02092/0.02354 CPU and0.04990/0.05188 GPU for
ideal/staged contracts. These are numerical errors, not timing measurements.

Independent stdlib scalar replay checks every seeded operand and all stages.
FP32 stages match exactly; maximum FP64 absolute replay difference is about
`7.05e-12` within `1e-13 + 1e-13*abs(expected)` for the independent replay. The
[receipt](../benchmarks/results/2026-10-24-gemma-per-layer-reference-v1.json)
binds exact source/runtime, fixtures, commands, observations and replay. Astra
reviews final evidence before commit; engine/shader sources and protected
binaries remain unchanged.

The [native FP32 composition](per-layer-scalar.md) now passes the frozen cases.
Quantized backend integration, per-layer gate/residual/scalar order, whole Gemma/session,
BF16/quantized checkpoint execution, nearest-token fallback, rotary parity,
quality/performance and real-model/device fit remain open in the
[Gemma4 plan](gemma4-implementation-plan.md). No ABI, app capability, limit or
installed phone package changes.

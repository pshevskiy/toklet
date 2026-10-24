# Gemma4 tanh-GELU and GeGLU reference

This is the first G2 numerical witness after the
[pinned Gemma4 E2B inspection](gemma4-e2b-inspection.md). It freezes the actual
activation variant before a native target. No model, Metal graph or Chat profile
is implemented by this task.

For FP32 gate x and up u, the independent NumPy producer widens to FP64 and
computes `0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x³)))`, then multiplies by u.
This fixes tanh-GELU rather than erf-GELU or SiLU, and fixes the gate/up order.
Five vector cases cover empty/no-call, scalar,129-point grid,17 tail/tiny values
and33 odd-width values. Separate hand, symmetry and chunk tests retain input
immutability and distinguish the other activation variants. Mathematical FP32
extreme checks are additional portable tests, not upstream extreme qualification.

The bounded inline JSON fixture requires an external hash, exact fields/cases,
finite numeric values, non-Boolean shapes and exactly representable FP32 inputs.
The [frozen fixture](../tests/fixtures/gemma-gelu-v1/math.json) contains independent
FP64 activation and gated-output expectations. Producer/source/runtime identities
and qualification exclusions are explicit.

The explicit adapter authenticates `gemma4_text.py` at MLX-LM revision
`0d4d3d04f8ab7ce4a51e7209e8615755a48ba94d`, executes only its captured `geglu`
definition and invokes the actual admitted MLX0.32.3 `nn.gelu_approx`. Independent
CPU/GPU processes each retain eight raw arrays and compare360 scalar values at
the unchanged `1e-5 + 1e-4*abs(expected)` policy. Maximum error/policy ratios are
approximately0.02754 CPU and0.09168 GPU. These are numerical errors, not latency
or throughput measurements. Every completed result is retained before later
construction/evaluation or dtype/finite checks can fail.

Three owner, seven portable mathematical/format/failure and three cold runtime
admission tests pass. The owner binds actual NumPy2.3.2/MLX0.32.3 package/native/
Metal/metadata/interpreter files before numerical imports, owns fresh child
outputs/logs/cache/process groups and rechecks input/output identities. A source
review corrected four owner filename references before execution; the unexecuted
proposal remains distinct from the passing capture. The
[receipt](../benchmarks/results/2026-10-24-gemma-gelu-reference-v1.json) binds exact
source, raw observations, checks and independent stdlib replay.

The scalar GELU/GeGLU target is next. Proportional RoPE, local/shared KV, value
normalization, per-layer paths, decoder order, logit capping, artifact/session/
Metal execution and real-model/tokenizer/quality/device gates remain open. Current
native eligibility and output/context capabilities are unchanged. This witness
does not qualify131072 output tokens or a new phone package.

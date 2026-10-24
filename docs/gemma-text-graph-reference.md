# Complete tiny Gemma text graph witness

The independent fixture now covers the complete token-to-logit dense text path.
The original upstream CPU graph passes; the GPU intermediate-stage comparison
fails under the unchanged tolerance. This is a synthetic short CPU reference,
not a runnable Gemma checkpoint, completed G2 milestone or device qualification.
The [acceptance and retained rejection receipt](../benchmarks/results/2026-10-25-gemma-text-graph-reference-v1.json)
binds every mathematical operand, source capture and raw comparison.

## Concrete graph and ownership

The explicit tiny configuration uses hidden8, vocabulary17, four alternating
sliding/full layers, two query heads and one KV head. Local heads have dimension8,
window4 and full rotary; global heads have dimension16, four rotary coordinates
and the proportional full-dimension frequency convention. The first two layers
own KV; the last two reuse the preceding producer of their kind. Their MLP
widths are12/12/24/24. Per-layer input width is2; all direct learned norms,
unlearned V normalization, tanh-GELU gates, residuals and learned scalars execute.

The graph starts with token lookup and sqrt(hidden) scaling, combines the
per-layer table and normalized projection, and ends with final direct RMS,
the same tied embedding table as output projection and cap30 softcapping.
Weights are row-major `[out,in]`; hidden arrays are `[1,tokens,hidden]`, and
attention arrays are `[1,heads,tokens,head_dim]`. Learned weights remain immutable
and separate from the two source cache owners. Consumer layers have no K/V
projection weights or cache. This fixture does not choose native storage or
authorize expanded full-checkpoint FP32 weights.

All absolute positions0..31 execute in three schedules: a32-token prefill,
32 single-token steps and mixed chunks3/1/4/2/7/1/8/6. The mixed and decode paths
exercise repeated window wraps, source cache offsets and shared-array identity.
The source caches' normal updates do not establish native atomic commit/rollback.

## Independent numerical contract

[The fixture](../tests/fixtures/gemma-text-graph-v1/math.json) is SHA256
`aef5ddcb46b89606c7532cca7441b7144ee6ed9f8b0c4768a201addee5267c9c`.
Deterministic rational FP32 weights and a non-monotonic repeated token sequence
are frozen before native composition. Independent ideal arithmetic uses FP64
`fsum` reductions and Decimal80 rotary frequencies/absolute angles. The second
contract rounds at explicit FP32 operator boundaries, including individual
rotary products before addition/subtraction. Both are mathematical references;
they do not promise exact bits from upstream fast kernels or a complete native
graph. A separate standard-library reconstruction checks all4,358 weight values,
32 IDs and180 stage arrays/68,224 values. Staged values match exactly; ideal
maximum difference is5.551115123125783e-15 under1e-13+1e-13 relative replay policy.

The source comparison authenticates the same four captured MLX-LM files used by
the component witnesses at revision
`0d4d3d04f8ab7ce4a51e7209e8615755a48ba94d`. It admits exact NumPy2.3.2/MLX0.32.3
runtime files before numerical imports. Original class bodies, selected
constructors, mask/cache helpers and compiled decorators execute unchanged;
base/cache namespaces stay separate. Complete learned parameter names, shapes,
dtypes and assigned values are checked, and retained weight handles remain
immutable after every call. Derived private rotary frequencies are separate.

Each device runs41 pristine whole-graph calls and41 calls with real module-boundary
instrumentation, retaining3,731 raw arrays and7,462 comparisons against both
contracts. Instrumentation materializes its captured boundaries; its execution
is not a fusion/performance claim. All164 final-logit comparisons per device
pass, but they cannot replace intermediate checks.

## Retained GPU gap and decision

CPU passes all7,462 comparisons under `1e-5 + 1e-4 * abs(expected)`. Maximum
error/policy ratio is0.8011257545865864 ideal and0.9269659441144206 staged.
GPU completes every call but fails15 comparisons. Its maximum ratio is
1.3557632102577382 at token13, layer1 output; layer2 input duplicates that tensor.
The raw source output there is-0.13776704668998718, versus staged
-0.1377348154783249. The overall owned collection remains failed with GPU exit1;
its logs, raw arrays, complete comparison list and guarded endpoint records are
retained. Five commands succeed, and the sixth records the numerical rejection.

Retrospective reconstruction from the captured incoming residual, normalization
and scalar stages reproduces that output exactly. Recomputing its post-FF norm
from the captured projection differs locally by only1.1920928955078125e-7.
Accumulated differences and normalization/residual sensitivity explain how small
local errors can exceed a later graph gate. They do not identify a specific GPU
kernel root cause. Neither weights nor tolerances were changed to obtain a pass.

Nine portable methods, three owner/failure methods and three cold-runtime
admission methods pass. The independent collector verifies all raw results and
their original pass/fail outcomes, process-group cleanup and protected binaries.
Astra reviews exact code, evidence and CPU-only scope before commit. No Rust or
shader change is included, so unrelated Rust suites are not repeated for this
reference-only task.

ModelArgs parsing, sanitizer/quantization paths, BF16/quantized storage,
multimodal/embedding-only inputs, alternate graph options, native session
recovery/sampling, real-model quality, performance and memory fit remain open.
The [long-position rotary discrepancies](gemma-rotary-precision.md) remain
unwaived. Native graph/session, Metal execution, real checkpoint and app/device
work follow in the [ordered plan](gemma4-implementation-plan.md); this short
reference grants neither Gemma app eligibility nor a131072-token reply.

# Gemma V normalization witness

This freezes unlearned mean-square RMS normalization before its native mapping.
Gemma projects V as `[batch,tokens,kv_heads,head_dim]` and normalizes each final
D-wide row separately before transposing into the attention layout. Its pinned
`RMSNormNoScale.__call__` invokes `mx.fast.rms_norm(x,None,eps)`; there is no
learned scale, added-one scale, sequence/head reduction or sum-only L2 norm.
Source revision is `0d4d3d04f8ab7ce4a51e7209e8615755a48ba94d`, and the
`gemma4_text.py` SHA is
`73ffa75e9069ce7b2f864ccd8f0267bb80c5b5d4e2b73102b8a2a93fe03f936d`.

Inputs are FP32. The independent ideal contract computes
`x/sqrt(mean_D(x*x)+f32(1e-6))` in FP64. The staged contract rounds that result
once to FP32. Eight bounded cases include an empty no-call, two anisotropic
heads with hand literals, batched/multiple heads, D1, zeros, epsilon-dominated
nonzero values, D256 and D512. They distinguish axis, mean-vs-sum, identity and
epsilon errors. These shapes do not qualify a full E2B attention layer.

The [producer](../tools/reference/gemma_v_norm.py) and
[portable tests](../tools/reference/test_gemma_v_norm.py) are separate from
Toklet code. The unchanged selected source body executes with actual FP32 MLX
on explicit CPU/GPU devices after source and runtime admission; only annotations
are deferred. Constructor/config selection is outside the witness. Empty input
uses the mathematical no-call policy. Each evaluated raw output is retained
before dtype/finite validation; earlier completed rows survive later failures.
The [controller](../tools/experiments/gemma_v_norm_reference.py) owns exclusive
outputs and process groups, retains failure logs, checks endpoint guards and
leaves protected binaries unchanged. Its [regressions](../tools/experiments/test_gemma_v_norm_reference.py)
exercise source substitution, failed/raised children and capture preservation.

An independent stdlib `fsum`/`sqrt` replay reproduces all24 arrays/7,350 values
exactly, including staged FP32. The upstream gate remains
1e−5+1e−4×abs(expected). All six owned commands pass, including seven
portable, three owner and three runtime-admission methods. Each CPU/GPU capture
retains seven outputs and passes14 comparisons against the two contracts.
Maximum ideal/staged error-policy ratios are0.00106586/0.00104450 CPU and
0.00085679/0.00104450 GPU. The
[acceptance receipt](../benchmarks/results/2026-10-24-gemma-v-norm-reference-v1.json)
binds all source/runtime/fixture/raw evidence and final checks.

The [scalar recipe](gemma-v-norm-scalar.md) now passes the accepted fixture.
It reuses `rmsnorm` with a persistent immutable exact-one
`[D]` vector, normalized over flattened B×T×Hkv rows. Its adapter must establish
that invariant and include the small shared vector in allocation accounting;
the test qualifies that numerical recipe, while adapter ownership/integration
remain open. Extreme finite-FP32
robustness belongs to the native/math path unless the upstream fast reduction
also passes the unchanged numerical gate.

V projection, normalized-V storage precision, shared KV, attention integration,
BF16/quantized checkpoint, full model/session, Metal graph, device fit, quality,
performance and long replies remain unqualified. The eight
[upstream rotary discrepancies](gemma-rotary-precision.md) remain unwaived.

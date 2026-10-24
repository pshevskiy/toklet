# Scalar recurrent gate construction

`cpu::gated_delta_gates` completes raw scalar inputs for the checked
[normalizer](gated-delta-normalization.md) and
[recurrent update](gated-delta-scalar.md). This remains an E4/L4 operator
foundation; no model/session uses it yet.

Inputs A, B, A_log and dt_bias are equal-length FP32 vectors `[value_heads]`.
Decay is `exp(-exp(A_log)*softplus(A+dt_bias))`; beta is sigmoid(B). Softplus uses
`max(x,0)+log1p(exp(-abs(x)))`; sigmoid uses `exp(-abs(B))` and a sign branch.
Intermediate arithmetic is FP64, followed by one FP32 output rounding. This
avoids positive-exponential overflow in softplus/sigmoid and preserves tiny
nonzero update gates until actual FP32 underflow. Outputs lie in `[0,1]`.

Finite-input, exact-length and nonempty-vector admission precedes all writes.
A nonfinite FP64 rate or exponent rejects as arithmetic failure; both gate
outputs must then be discarded, even if earlier heads completed. No state,
weights or output buffers are allocated/owned by the function. A later graph
must include its two caller-owned `[value_heads]` FP32 gate vectors and any
conversion buffers in admission. This task does not precompute checkpoint rates
or establish a performance-selected execution path.

The independent frozen mathematical producer remains unchanged. Three new tests
authenticate its accepted receipt and compare gate values plus the complete raw
normalization→gates→recurrence state/output path against all five frozen cases.
The numerical policy stays `1e-5 + 1e-4*abs(expected)`. Independent hand cases
check 0.5/0.25 values, bias placement, saturation, and a tiny nonzero beta with a
stronger dedicated relative assertion so zero cannot pass the absolute floor.
Malformed lengths, every nonfinite input kind and FP64 intermediate overflow
exercise rejection. The empty mathematical case makes no step call.

Exact source/check coverage is bound by the
[receipt](../benchmarks/results/2026-10-24-gated-delta-gates-v1.json).
Formatting, strict default/Metal Clippy, 203 CPU tests, 225 Metal-feature tests
and both iOS cross-checks pass (15/107 ignored respectively).
Earlier normalization and recurrent-update bodies remain unchanged. Distinct
copy-on-write compiler-cache targets keep prior accepted target files separate;
this reuse is not a build-time or runtime performance result.

Masks, vector gates, negative-eigenvalue updates, convolution, gated output
normalization, hybrid config/tensor mapping, state/session commit, resident Metal
execution and real-model/long-context/device qualification remain open. Existing
app/native model eligibility and 256/128/512 execution limits are unchanged.

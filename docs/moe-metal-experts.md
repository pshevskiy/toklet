# Metal Half expert projection and reduction

5 November 2026. The qualification harness exercises four borrowed Metal
stages in one command buffer. The stages implement fused gate/up projection,
GLU activation, down projection and ordered weighted reduction.

Weights retain the source `[E,2I,H]` and `[E,H,I]` layouts. Each token keeps its
supplied route order. Gate and up dots accumulate in FP32 and store Half. GLU
stores Half after SiLU and after multiplication. Down projection accumulates
in FP32 and stores Half. Reduction rounds each weighted product in FP32,
accumulates in slot order and stores the result once in Half.

The owner consumes a checked plan for padded input, route, intermediate and
output strides. The plan counts tight weight payloads, guarded buffers and
fixed Rust owner bytes. Device limits and budget are checked before allocation.
Encoders borrow aligned, disjoint, retained buffers on the same device. The
pipeline reservation accounts separately for compilation and driver resources.

Projection and down stages check expert IDs before accessing weights.
Reduction checks IDs, duplicates and finite nonnegative gates. Arithmetic
failure and invalid routes set bits in shared status. The caller discards
the whole candidate when status is nonzero. Two invocations in the same
command buffer demonstrate that a valid invocation preserves prior errors.

The independent fixture checks exact Half words at five stage boundaries,
with odd dimensions, expert permutation, unselected weights and 7/8/9-token
batches. A sparse case exercises Granite's `H=1024`, `I=512`, 32 experts and
top 8. Three authored reductions distinguish reordered or widened accumulation
and fused products. The initial eight cases passed, then a source-side check
found missing gate/up rounding discriminators. The added case independently
exercises both stores with the same arithmetic policy.

The GPU controls also check immutable input and weight bytes, padding, guards,
budget limits, late invalid routes, overflow and Rust allocations during
unprofiled encoding. Full graph wiring and the combined session allocation
plan follow this test-gated leaf.

[Fixture](../tests/fixtures/moe-experts-half-v1/fixture.json) ·
[Check logs](../artifacts/moe-experts-half-checks-2026-11-05/final/)

On the Apple M1 Pro, both release GPU tests passed. Nine cases checked 45,612
exact stage words across 18 launches, with zero measured Rust allocations during
encoding. The independent source fixture replayed byte for byte. All six
required jobs passed, including 574 CPU and 685 Metal-feature tests. Astra
approved the code with no findings.
[Result](../benchmarks/results/2026-11-05-moe-metal-experts-v1.json).

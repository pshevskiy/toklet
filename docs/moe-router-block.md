# Sparse-expert router leaf

5 November 2026. This is the first numerical component of the
[Granite sparse-MoE plan](granite-moe-next.md), not model execution support.

`cpu::route_topk` consumes row-major FP32 scores `[tokens,experts]` and writes
caller-owned IDs/gates `[tokens,selected]`. It supports up to 32 experts, checks
all shapes, lengths, overflow and finite inputs before writing, and allocates
nothing. Bounded repeated scans keep the scalar correctness reference readable.

Scores rank descending; ascending expert ID breaks ties, including signed zero.
Exactly the selected scores enter a stable softmax. Subtraction, exponentials,
ranked-order mass accumulation and division use FP32 boundaries. Extreme finite
score subtraction may produce negative infinity, contributing zero mass; the
largest score always contributes one. No existing softmax or dense path changes.

The independent Python oracle sorts complete rows and explicitly stores FP32
boundaries. Its fixture was frozen before target execution: 11 cases, 32 rows,
234 selected IDs/gates. Coverage includes ties at selection boundaries, all-equal
and negative scores, extreme finite values, expert permutations, single/all
selection and batches of 7/8/9 tokens. IDs are exact; the prospective exponential
comparison gate is `2e-7 + 2e-6*abs(reference)`. This is a routing-leaf gate, not
a full-model fidelity threshold. Byte-identical replay authenticates the source.

An additional authored `exp(-17)` control requires the highest gate to be exactly
one: ranked FP32 additions each lose a term below half an ULP. FP64 summation or
summing small terms first changes that result. Invalid input controls preserve
every output, including a nonfinite score in the last row. The existing session
allocation harness checks repeated 7/8/9-token calls without new unsafe code.

[Fixture](../tests/fixtures/moe-router-v1/fixture.json),
[checks](../artifacts/moe-router-checks-2026-11-05/).
Expert projections, weighted expert reduction, source import, exact model owner
planning, Metal routing and real checkpoint execution remain unimplemented.

Astra approves final code/tests with no outstanding findings. Full 562 CPU and
671 Metal-feature tests, strict Clippy/default+Metal, formatting and release
build pass. Fresh controls in both feature configurations cover all-selected
opposite extremes and mixed ties at the cutoff. Forty-eight measured calls
allocate zero Rust buffers. No new Metal numerical capability is claimed.
[Result](../benchmarks/results/2026-11-05-moe-router-leaf-v1.json).

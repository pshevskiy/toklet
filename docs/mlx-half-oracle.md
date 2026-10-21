# MLX packed-weight / Half-activation graph oracle

The [CPU-only exporter](../tools/reference/mlx_half.py) freezes numerical gates
before composing the resident operators into a complete GPU model. It consumes
the accepted [MLX Qwen fixture](../tests/fixtures/mlx-qwen-v1/manifest.json), admits
its literal manifest SHA and exact config/checkpoint/array bytes, and identifies
the selected oracle sources before numerical imports and execution. Inputs and
sources are checked again before publication. No MLX runtime or GPU is imported.

Independent unsigned-word reconstruction produces the original FP32 packed
matrices. Floating exclusions, normalization/bias weights and stored activations
use Half. Input embedding rows round to Half; the final tied output projection
uses the original reconstructed FP32 embedding parameters. This models one tied
packed allocation with distinct operation rounding, rather than treating its
Half gather results as the output weight table. The exporter reuses the unchanged
scalar Half graph math and replaces its final tied logits accordingly.

Both ordered FP32 and independent FP64 accumulation are evaluated with the same
storage boundaries. The numerical gate is the existing tiny Q4 policy,
`max(1e-4, 2 * independent_FP32_FP64_max_abs) + 1e-3 * abs(reference)`.
Comparison to the canonical FP32 source uses the existing Half storage-envelope
policy, `2 * independent_Half_vs_source_max_abs + 1e-5 + 1e-4 * abs(source)`.
These policies are frozen before target full-model GPU execution. Held-out
quality thresholds remain unchanged; a numerical envelope does not establish
quality or allow changing a gate after seeing a target failure.

The [frozen fixture](../tests/fixtures/mlx-half-v1/manifest.json) contains 199
arrays in 190,336 bytes: every one of the 39 graph stages over five cases and four
greedy-step distributions. Cases cover a changed future token, rotary offset,
single token and 32-token capacity. Greedy steps follow canonical FP32
teacher-forced history, with lowest-ID tie breaking and four fixed steps; both
reconstructed and canonical token IDs are `[47, 47, 47, 47]`.

All [seven portable tests](../tools/reference/test_mlx_half.py) pass. They bind the
complete manifest, gates and greedy metadata; check exact array inventory,
dtypes/shapes/contiguous coverage and finite values; compare saved actual MLX
CPU/GPU logits under the pre-frozen storage envelope; verify causal-prefix bits;
regenerate every fixture byte; and check immutable/source rejection plus
stdlib-only hash verification. Sixteen independently formed output dots use
original accepted FP32 embedding rows, match stored logits exactly and demonstrate
that rounding the tied weights to Half would change results.

The owned exporter and three owned verification commands complete successfully
with unchanged source/protected-binary endpoints. The
[task receipt](../benchmarks/results/2026-10-22-native-mlx-half-oracle-v1.json)
binds raw logs, source identities and the frozen data. This Python/fixture task
changes no Rust runtime, shaders, defaults or historical numerical policies;
the preceding scalar task's Rust checks are not reported as fresh runs here.

This fixture is an independent tiny numerical oracle. Complete GPU graph/state
execution, real-checkpoint quality, performance and phone qualification remain
open in the [M8 plan](mlx-implementation-plan.md).

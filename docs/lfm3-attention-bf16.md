# LFM3 BF16 CPU attention

The CPU attention leaf consumes token-major BF16 Q and up to three borrowed
per-head KV spans. It maps four adjacent query heads to each KV head, supports
full causal and 256-token sliding windows, and allocates no heap storage. The
caller supplies a single FP32 scratch row of `4 × logical_keys` bytes. A view of
a wrapped ring keeps both old spans and the proposed chunk borrowed. The leaf
never publishes or advances cache state.

The numerical target is the pinned modified parent's eager CPU function on the
M1 Pro, with dropout disabled. QK, scale, additive mask, probabilities and
context store BF16 separately. The short BMM path applies FP32 fused accumulation
when `queries × keys ≤ 3`. The large QK path shares the transposed projection
dot. The large probability/value path uses four FP32 partials across logical
keys and an ordered reduction, including masked positions. The softmax uses the
source FP32 reduction and the nonpositive range of the pinned SLEEF exponential.
Stride padding stays outside validation and arithmetic. Reject malformed spans,
nonfinite inputs, and QK or context overflow, including QK overflow at a masked
key. Discard partial output on arithmetic failure.
[Contract](../tests/fixtures/lfm3-attention-bf16-v1/contract.md),
[PyTorch kernel](https://github.com/pytorch/pytorch/blob/5811a8d7da873dd699ff6687092c225caffcf1bb/aten/src/ATen/native/cpu/SoftMaxKernel.cpp),
[SLEEF license](../src/cpu/lfm3_attention/SLEEF-LICENSE.txt).

The independent fixture covers 25 synthetic cases and 17,742,592 BF16 words
across the five stored stages. All words match. The existing FP32 diagnostic
passes 8,128 independent FP64 comparisons per stage for QK, softmax and context
accumulation. A subnormal midpoint witness distinguishes the source short
BMM's fused result from the large path. Tests exercise both arithmetic paths,
ignored padding, admission failures, and three-span ring wrap with unchanged
state. A first deep run with the standard FP32 exponential differed by six
BF16 probability words. The same cases pass with the pinned SLEEF arithmetic.
[Fixture](../tests/fixtures/lfm3-attention-bf16-v1/fixture.json),
[source guard](../artifacts/lfm3-attention-bf16-2026-11-11/fixture-checks.json),
[small BMM witness](../artifacts/lfm3-attention-bf16-2026-11-11/probe-small-bmm.json),
[initial deep-run differences](../artifacts/lfm3-attention-bf16-2026-11-11/targeted-release-deep-1.log).

The installed MLX-LM unquantized attention helper returns 630,021 CPU and
757,040 Metal BF16 context differences out of 1,040,384 words on the same
synthetic operands and logical masks. Both guarded replays reproduce their raw
outputs and stay within the separate 512 MiB audit budget. This is a component
correspondence decision. Full graph, model loading, CLI, and end-to-end
performance, fit, and quality still require qualification.
[MLX CPU](../artifacts/lfm3-attention-bf16-2026-11-11/mlx-guarded/mlx-cpu-report.json),
[MLX Metal](../artifacts/lfm3-attention-bf16-2026-11-11/mlx-guarded/mlx-metal-report.json).

## Release baseline

On the M1 Pro, 24 cohorts cover 1, 32, and 128 queries at a cold prefix, two
sliding-window depths and a deep full prefix. Each cohort retains its first
pass, three warmups and ten timings. The second sweep reverses the first.
The timed boundary includes geometry and finite-input validation of already
prepared operands. It excludes cache seeding and allocation. All input and
output hashes agree across the two sweeps.

| Query chunk | Cold full | Window, prefix 1,023 | Full, prefix 1,023 |
| --- | ---: | ---: | ---: |
| 1 | 0.0073 ms | 0.456–0.463 ms | 1.785–1.809 ms |
| 32 | 1.593–1.638 ms | 13.265–13.476 ms | 48.089–49.025 ms |
| 128 | 23.605–24.176 ms | 70.019–71.569 ms | 210.402–212.748 ms |

These are per-layer medians for synthetic typed operands. The scratch row is
4,604 bytes at the deepest timed full shape. Peak child RSS is 9,371,648
bytes. AC power, Low Power Mode off and nominal thermal state were verified
before and after each cohort.
[Raw samples and hashes](../benchmarks/results/2026-11-11-lfm3-attention-bf16-v1.json).

Fmt, strict CPU and Metal Clippy, 728 CPU tests and 874 Metal tests pass.
The five optimized source checks, debug and release allocation checks, three
native arm64 build/link slices with 46 exports each, and Mac CPU and Metal C
smokes pass. The extra all-examples run encounters a baseline
`gemma4_generate` test fixture missing `Memory::positional_bytes`.
[Logs and results](../artifacts/lfm3-attention-bf16-2026-11-11/checks.json).

The initial source and MLX captures omitted full runtime inventory checks.
Guarded replays verified all 28,982 pinned files at both boundaries, reproduced
151 Torch files and both raw MLX output streams byte for byte, and peaked at
510,754,816 bytes of combined final RSS upper bounds. Astra approved the
component after verifying the replay, with no unresolved findings.
[Runtime replay](../artifacts/lfm3-attention-bf16-2026-11-11/runtime-reconciliation.json),
[Astra review](../artifacts/lfm3-attention-bf16-2026-11-11/astra-review.json),
[Commit](../artifacts/lfm3-attention-bf16-2026-11-11/publication-checks.json).

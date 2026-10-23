# Independent Qwen3 dense oracle

This starts E1 in the [engine expansion plan](engine-expansion-plan.md). The
[producer](../tools/reference/mlx_qwen3.py) defines the numerical contract before
native Qwen3 graph edits. It does not add native Qwen3 loading or generation.

The tiny decoder has hidden width32, four query heads and two KV heads of width16,
two layers, intermediate64, vocabulary64 and capacity32. Query width64 is
deliberately different from hidden width32. Row-major Q/K/V weights are
`[64,32]`, `[32,32]`, `[32,32]`; output projection is `[32,64]`. Q and K each use
a distinct learned width16 RMSNorm shared across their heads, after projection
and before ordinary split-half RoPE. The original bias-free Llama graph cannot
substitute for these semantics.

The independent scalar graph uses ordered FP32 and FP64 reductions. Original
MLX low-bit-first packed words and scale/bias metadata reconstruct weights;
four-bit matrices, an eight-bit override and a floating exclusion exercise mixed
storage. Both tied and independently owned untied output variants are included.
The mixed graph preserves Half boundaries and packed output reconstruction.
Q/K normalization is over the head dimension, never the concatenated query.

The FP32 comparison policy remains `1e-5 + 1e-4 * abs(reference)`. The mixed-Half
policy is frozen from the independent FP32/FP64 envelope as
`max(1e-4, 2 * envelope) + 1e-3 * abs(reference)`, before target evaluation.
Cases cover a prompt, changed future token, nonzero rotary offset, single token
and exact capacity. Saved wrong-graph witnesses bypass or swap Q/K norms; the
untied variant also distinguishes accidental tying. Greedy IDs follow the mixed
graph's own history with lowest-ID tie breaking.

Explicit producer trials use pinned MLX0.32.3, MLX-LM0.32.0 and NumPy2.3.2 on the
Mac CPU and GPU, checking all zero-offset case logits and actual two-plus-two-token
cached execution. The standalone offset case is a scalar positional witness,
not an upstream cache experiment. Package/native producer files and consumed
source arrays are pinned before imports and checked again at publication.

Portable tests cover per-head normalization against a direct formula, an
incorrect whole-query normalization counterexample, zero/epsilon-dominated
heads, GQA mapping, causality, rotary offsets, output ownership, exact fixture
coverage/dtypes and malformed metadata/policy/payload rejection. Fixture check
mode requires an external manifest SHA and runs without importing numerical
packages. Export is exclusive and cannot overwrite a fixture.

The [fixture](../tests/fixtures/mlx-qwen3-v1/manifest.json) contains 891 arrays in
1,281,536 bytes; its manifest SHA256 is
`5dc5b7ec0fccd8ac6876814b88a917107993aab5f5e07dcb222f57ec377a7a4a`.
Actual upstream CPU/GPU full and cached logits pass the fixed FP32 gate. The mixed
graph's four greedy IDs are `[3,3,3,3]` (tied) and `[24,36,11,51]` (untied).
All 15 [portable tests](../tools/reference/test_mlx_qwen3.py) pass, including
stdlib-only fixture admission and mutation rejection. The
[receipt](../benchmarks/results/2026-10-23-qwen3-oracle-v1.json) binds the producer,
fixture, logs and unchanged target sources/protected binaries. No native timing
or quality measurement is claimed.

Native scalar/Metal execution, real checkpoints, tokenizer and
thinking/chat semantics, quality, performance and iPhone qualification remain
open. This oracle does not qualify Qwen3 MoE, rotary scaling or sliding attention.

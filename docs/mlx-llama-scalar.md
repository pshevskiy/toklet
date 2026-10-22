# Native Llama artifact loading and scalar execution

The additive `Source::open_dense` and `reference::load_dense` entry points admit
MLX affine Qwen2 and the supported bias-free Llama graph. The original
`Source::open`, `reference::load_qwen` and existing Metal loader retain Qwen-only
admission. [Implementation](../src/mlx/dense.rs) validates the actual Llama
configuration: SiLU, RMSNorm, full causal GQA and ordinary split-half RoPE.
Attention/MLP bias flags, dropout, scaled/traditional/partial RoPE, sliding
attention, alternate layer graphs and unknown fields reject. Explicit zero bias
tensors also reject; affine `.biases` quantization metadata is distinct.
Declared head dimensions must agree with hidden/query-head geometry, KV mapping
must divide query heads, and admitted context must fit the model's positions.

The source reuses bounded safetensors/index parsing, source-precision floating
storage and the existing packed layout validation. Untied output requires its
own `[vocab,hidden]` matrix with independent quantization/exclusion settings.
Tied output owns one embedding table; any redundant head must match exactly
before being discarded. Packed untied bytes count in source admission and the
expanded head counts in diagnostic FP32 admission before any expansion.

The scalar model adds `Model::new_bias_free` with empty bias vectors and an
optional independently owned output head. Its original constructor still
requires Qwen biases and ties output. Both forward paths and cached execution
resolve the output table at the operation boundary. Bias selection likewise
occurs once per projection, preserving statically dispatched numerical loops.
Allocated memory accounts for head capacity; ordinary decode adds no buffers.

The [CPU tests](../tests/mlx_llama.rs) admit the externally frozen
[Llama oracle](mlx-llama-oracle.md) by literal manifest SHA and verify all390
FP32 stages across both heads and five cases, plus actual upstream CPU/GPU saved
logits. Cached chunks1/3/full, nonzero offsets, exact-prefix reuse, reset/replay,
overflow rejection and load/session memory admission preserve the same result.
Public constructor checks reject missing Qwen biases, nonempty bias-free biases,
malformed/nonfinite heads and verify independent head capacity accounting.

Rust1.90 formatting and strict all-target Metal Clippy pass. The fresh ordinary
CPU suite has140 passes/eight explicitly ignored cases; the Metal-feature suite
has154 passes/74 ignored cases. The [receipt](../benchmarks/results/2026-10-22-llama-scalar-v1.json)
binds raw owned logs and exact checked source copies. An earlier failed Qwen
error-text regression is retained without check credit; its original source
snapshot was not captured. Restoring the original Qwen parser fixes that test.
Astra reviewed the code and completed evidence before commit.

This task qualifies tiny scalar/reference execution. It does not enable Llama
in Metal or qualify real checkpoints, model quality, performance, chat templates
or iPhone execution. The [M8 plan](mlx-implementation-plan.md) keeps those tasks
open.


Subsequent [resident Metal integration](mlx-llama-metal.md) adds tiny Llama with
Reference attention. The scalar-only status above describes this earlier task's
revision; `Source::open` and `reference::load_qwen` remain Qwen-only.

# Immutable real Llama-family checkpoint

The selected candidate is
[mlx-community/SmolLM2-135M-Instruct-8bit](https://huggingface.co/mlx-community/SmolLM2-135M-Instruct-8bit/tree/0f0d9b8218915bc34d401e1a340b8c049d300d5e),
revision `0f0d9b8218915bc34d401e1a340b8c049d300d5e`. SmolLM2 uses the
`LlamaForCausalLM` graph; it is not a Meta Llama-branded checkpoint. Its small
size makes it a candidate for both devices, without establishing mobile fit.
The pinned model file is143,030,401 bytes, SHA256
`ec569c771e56424636be99625aba9031abc525a68a5f4ab16d678eca96fef12c`.
Nine selected model/tokenizer/card files total147,875,570 bytes. Every file passes
its immutable Git blob or LFS SHA identity, bounded HTTPS transfer and exclusive
publication. Three portable size/identity/publication methods pass.

Data-only inspection finds694 physical tensors:211 U32 eight-bit/group64 affine
matrices with F16 scale/bias metadata, plus61 F16 normalization vectors. Payload
spans are contiguous, shapes/index names exact, projection biases absent and
output tied. The graph has30 layers, hidden576, intermediate1536, nine query/three
KV heads of width64, vocabulary49152, ordinary rotary theta100000 and RMSNorm
epsilon1e-5. Its special configuration fields require separate strict admission;
this task does not silently ignore them or widen the engine geometry.

The [receipt](../benchmarks/results/2026-10-22-smollm-checkpoint-v1.json) retains
actual downloads, tests, exact source copies and unchanged numerical/Swift and
protected-binary endpoints. Prior Rust checks are inherited from `ca5c22a`;
this acquisition runs no model inference. Tokenizer IDs/stops, chat compatibility,
independent numerical reference, target execution, phone fit, quality and
performance remain unqualified. Qwen datasets and results do not become SmolLM
quality evidence. The frozen real-model oracle must precede target edits.

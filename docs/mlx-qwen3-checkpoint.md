# Immutable Qwen3 eight-bit candidate

24 October 2026. The public candidate is
[`mlx-community/Qwen3-0.6B-8bit`](https://huggingface.co/mlx-community/Qwen3-0.6B-8bit/tree/11de96878523501bcaa86104e3c186de07ff9068)
at immutable revision `11de96878523501bcaa86104e3c186de07ff9068`.
Ten files, totaling 649,377,414 bytes, are acquired and inspected as data.
The [receipt](../benchmarks/results/2026-10-24-qwen3-checkpoint-v1.json)
binds their exact identities, source, commands, tests and retained diagnostics.
No repository code or chat template was executed.

The 633,442,994-byte weight file has SHA256
`3ad5c96e0a476d4eee48e9f525d0ee0a3f1830b3e5b8e17b441de0661e921397`.
Its 78,506-byte header describes a contiguous 633,364,480-byte payload:
704 physical tensors, comprising 197 U32 packed matrices, 394 BF16 scale/bias
arrays and 113 BF16 norm vectors. These represent 310 logical tensors. Shapes,
offsets, complete coverage and the index agree. This is structural validation;
it does not yet validate every reconstructed numerical value.

The 28-layer graph has hidden width 1024, query width 2048, KV width 1024,
FFN width 3072 and vocabulary 151936. Sixteen query heads and eight KV heads
have width 128. Each layer has separate learned Q/K RMSNorm vectors of width128,
with ordinary RoPE theta 1e6 and epsilon 1e-6. Output weights are tied; projection
biases are absent. Quantization is eight-bit/group64. Sliding windows and rotary
scaling are disabled. The maximum position count 40960 is a configuration value,
not a tested context or device-fit promise.

The tokenizer inventory contains 151643 base tokens and 26 added IDs. Its
template mentions `enable_thinking`; `<think>` and `</think>` have IDs151667
and 151668. Declared EOS 151645 is `<|im_end|>`; these are inspected identities,
not qualified generation/stop rules. Tokenization, template evaluation,
thinking behavior and multi-turn chat need their own probes.

The [acquisition tool](../tools/experiments/m8_qwen3_checkpoint.py) uses four
bounded workers and 38 disjoint ranges of at most 16 MiB, then verifies every
chunk and the assembled weight SHA before exclusive atomic publication.
All 48 owned transfer commands pass: one inventory, 38 ranges and nine smaller
files. Exact status, range, encoding, size and immutable Git/LFS identities are
checked. Cancellation joins the workers and reaps owned processes. Nine portable
methods cover these contracts, including retained diagnostics after an owned
command failure combined with descriptor-close failure.

The [data-only inspector](../tools/reference/mlx_qwen3_checkpoint.py) authenticates
the exact buffer or open descriptor it consumes. Five portable methods pass,
including tensor/index mutations, bounded JSON and duplicate-field rejection,
tokenizer ID ambiguity and descriptor substitution while the pathname remains
unchanged. The actual checkpoint inspection passes. Astra required the
descriptor-identity and independent failure-diagnostic fixes before execution.

Rust, Metal, Swift and protected release binaries are unchanged. Their previous
exact-source checks remain inherited; no new engine or hardware check is claimed.
The [tiny resident Qwen3 geometry](mlx-qwen3-metal.md) remains the only GPU Qwen3
admission. Next, freeze an independent real-model numerical reference that
handles this checkpoint's BF16 affine metadata explicitly, before widening target
geometry. Quality, memory/device fit, throughput, Mac/iPhone execution, app
integration and general Hugging Face admission remain open in the
[active plan](engine-expansion-plan.md).

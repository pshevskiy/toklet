# Independent real SmolLM2 Llama-family oracle

The [CPU exporter](../tools/reference/mlx_smollm_real.py) freezes the
[immutable eight-bit checkpoint](mlx-smollm-checkpoint.md) before native target
changes. Its exact694-tensor reader independently unpacks unsigned octets from
little-endian U32 words, reconstructs F16 scale/bias groups through FP64 to FP32,
and retains one tied table plus the current matrix/stripe. Input gathers,
exclusions, activations and KV use Half; tied output uses original reconstructed
FP32 weights. The [bias-free graph](../tools/reference/llama_fp16.py) reuses
unchanged independent numerical primitives and explicitly handles nine query
heads mapped to three KV heads, RMSNorm epsilon1e-5 and rotary theta100000.

Six authored cases cover lengths1/3/4/17, future changes, offsets0/5 and layers
0/1/29. Four full greedy distributions and three scoring-prefix distributions
bring coverage to349 arrays/13,945,632 bytes. Ordered tiny scalar cross-checks
independently validate the optimized graph's9:3 mapping, causality and absent
biases. All nine portable methods pass, including literal packing/reconstruction,
finite-domain checks, exact schemas/gates, descriptor/path substitution,
exclusive publication and stdlib verification.

The unchanged real-graph numerical formula is
`max(1e-4, 2*pooled_FP32_FP64_max_abs, 2*equivalent_prefix_max_abs) + 1e-3*abs(reference)`.
Pooling includes actual independent scoring-prefix/full-prefix differences.
The maximum absolute allowance is32; the largest logits allowance is
0.207672119140625. These independently observed storage/accumulation envelopes
are frozen before target trials and do not replace held-out quality gates.

The captured tokenizer and tokenization producer are pinned separately:
`Hello, world!` becomes `[19556,28,905,17]` without inserted special tokens;
BOS is1, EOS/PAD is2. FP32 and FP64 greedy histories both select
`[339,3543,719,1891]`. This establishes fixed raw-ID probe identity, not chat
formatting or task quality. Qwen token datasets/baselines are not used.

| Frozen output | SHA256 |
| --- | --- |
| [manifest](../tests/fixtures/mlx-smollm-real-v1/manifest.json) | `3cd3af44e0d4d57cc7cf23c8dd7cc7b60ccf0968e48bfae90adba35517e039bd` |
| arrays | `4f471fe2f15c257c8fbf4169a88b99b7ba28fa1e9bf881eaa91f77a7f8cdcca2` |

All three owned commands pass: portable methods, CPU export and stdlib-only
verification with external manifest SHA. Selected NumPy/tokenizer source/native
packages and numerical producers are bound before import and checked afterward.
Logical reference admission is1,195,800,705 bytes; observed lifetime physical
peak is185,025,592 bytes under the independent6GiB Mac cap. Neither measurement
establishes phone fit. The [receipt](../benchmarks/results/2026-10-22-smollm-real-oracle-v1.json)
binds579 files and58 exact source copies. Rust/shader/Swift/protected binaries
remain unchanged, with prior Rust checks inherited. No target inference,
real-model adoption, held-out quality, performance or device qualification is
claimed by this CPU reference task.

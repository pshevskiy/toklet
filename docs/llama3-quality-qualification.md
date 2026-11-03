# Pinned Llama original-quality qualification

The original-vs-native quality gates pass on Apple M1 Pro for
`mlx-community/Llama-3.2-1B-Instruct-8bit` revision
`d48cdf0a4ea22d893b7c63a99d6a693e24822795`, weights SHA-256
`cd3582a78766654e3e2107e3b4d51d9cb18e6754b659bfb44a34328ef9d7fa52`.
The original reference is `meta-llama/Llama-3.2-1B-Instruct` revision
`9213176726f574b556790deb65791e0c5aa438b6`, authenticated BF16 weights widened
to FP32. The native profile uses MLX affine projections, compensated
normalization, Reference attention, full logits, chunk32 and Llama3 scaled RoPE.
The pinned [plain-text codec policy](llama3-plain-codec.md) applies to tasks.

| Frozen gate | Measured native minus original | Limit | Result |
| --- | ---: | ---: | --- |
| 10,000-target mean NLL | +0.0000197305 nats/token | ≤0.05 | Pass |
| 100-task pass-rate loss | −1 percentage point (23 vs22 passes) | ≤2 | Pass |

The [loss diagnostic](llama3-original-loss.md) uses blocks up to1,025 positions;
[task generation](llama3-task-quality.md) uses capacity566. Their original models,
native revisions, numerical profiles, complete counts and authenticated receipts
are joined explicitly. Historical per-stage receipts retain their then-pending
quality flags; this joined decision supersedes those flags only for these gates.

The managed Mac app remains512 positions. This record does not qualify the
model's declared131,072-position window, another artifact/arithmetic profile,
ordinary app discovery or iPhone execution. The next Mac adoption task is a
fresh isolated large-file ranged download → verify → open → chat → offline
reopen, followed by a separately reviewed catalog change. Existing flagged
cached Chat/cancellation/recovery checks remain applicable. Physical iPhone14Pro
is unavailable and its qualification remains open. No performance claim follows.

Astra approves this scoped decision. All input receipt hashes and identities
were checked; documentation links and whitespace pass.
[Joined evidence](../benchmarks/results/2026-11-03-llama3-quality-qualification-v1.json).

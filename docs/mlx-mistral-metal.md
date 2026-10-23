# Selected native Mistral decoder on Mac

The engine admits the immutable four-bit/group64
`mlx-community/Mistral-7B-Instruct-v0.3-4bit` snapshot at
`a4b8f870474b0eb527f466a03fbc187830d271f5`. Its
[checkpoint identity](mlx-mistral-checkpoint.md) and
[independent frozen CPU reference](mlx-mistral-oracle.md) precede target edits.

Mistral uses the shared statically selected bias-free dense graph, with untied
output weights, Half activations/KV and fused affine packed projections.
Weights remain resident. Row-major projections and separate per-layer KV
`[kv_heads, capacity, head_dim]` preserve the existing ownership and layout.
The strict boundary requires matching architecture/class, explicit
`sliding_window: null`, ordinary rotary positions and qualified geometry.
Missing/non-null windows and Llama-only metadata, including null-valued keys,
are rejected. Public Qwen-only loaders retain their existing contract.

Reference attention is selected outside numerical loops. Online attention is
rejected for this architecture. The real shape is 32 layers, hidden 4096,
intermediate 14336, 32 query/eight KV heads of dimension 128, vocabulary 32768,
theta 1e6 and epsilon 1e-5. Capacity is bounded at 512. A tiny two-layer shape
has independent scalar coverage using the previously frozen untied Llama
payload; its configuration differs only in architecture and explicit window.

## Verification

Final-source formatting, strict all-target Metal Clippy, ordinary CPU tests and
Metal-feature tests pass. The receipt retains an earlier passing baseline with
its historical source copies; Astra then caught the missing/null window
ambiguity, which was fixed before the final baseline and all real target trials.
Release checks additionally pass the preallocation FP32 expansion-budget
rejection, the four-fixture native C ownership/state comparison, and three
real-model GPU tests on **Apple M1 Pro**.

The target compares **816 snapshots / 19,302,400 values** against the frozen
manifest, using its unchanged absolute/relative gates. Maximum stage absolute
error is **0.3125**. Exposed graph stages, offsets and prefill chunks pass;
attention scores/probabilities are not directly exposed by this target harness.
The 349 CPU arrays must not be mistaken for 349 directly compared target arrays.
Four full greedy distributions, three scoring prefixes and eight token
selections pass; maximum logit absolute error is **0.0369415283203125**.

The separate 512-capacity state test verifies exact fresh/retained full
logits, bounded chunk commits, full-capacity rejection and valid-prefix replay.
It does not establish an independent 512-token numerical oracle. Packed
resident weight allocations total **4,077,396,878 bytes**; maximum logical
session admission is **4,267,272,802 bytes**, within the 5 GiB numerical test
budget and 6 GiB Mac plan budget. These are logical allocation figures, not a
measured process physical peak.

The [receipt](../benchmarks/results/2026-10-23-mistral-metal-v1.json) binds exact
sources, retained diagnostics, fixture/reference/checkpoint identities and the
unchanged historical CPU binaries. The 241.71-second GPU suite duration includes
multiple models and diagnostic comparisons; it is not a decode benchmark.
Held-out quality, performance, other Mistral configurations, sliding attention
and Online attention remain unqualified. Native instruction text, app catalog
routing and real app download/chat are subsequent tasks. Physical iPhone work
is paused at the user's request; this task contacts no phone.

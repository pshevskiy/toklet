# Resident tiny Llama on Metal

The explicit native `load_mlx` path now admits the strictly validated bias-free
Llama graph and builds resident packed projections, embedding and output weights.
An untied `[vocab,hidden]` head keeps its own original affine words/scales/biases;
a tied head borrows the single embedding tensor. Final projection selects the
actual head for packed, compensated Half and MPS branches. Model planning counts
all resident head buffers before upload. There is no full expanded model copy.

Q/K/V biases are optional immutable tensors selected during assembly. Bias-free
operations use the existing session zero buffer, selected once per projection;
there is no per-token bias allocation or dynamic numerical dispatch. Existing
Qwen/Half/Q4 assembly still requires its biases and tied head. Leftover tensors
reject rather than being silently dropped. The rotary shader receives the
validated model theta as FP32 bits rather than assuming Qwen's1e6 constant;
both production dispatches and the standalone operator check pass it explicitly.
Existing Qwen/Half/Q4 default execution modes are unchanged.

Architecture-specific geometry checks run before GPU creation. Llama is currently
limited to the independently specified tiny shape: two layers, hidden32,
intermediate64, four query/two KV heads, width8, vocabulary64, capacity at most32,
ordinary split-half RoPE theta10000 and RMSNorm epsilon1e-6. Wider shapes and
scaled/sliding/bias variants remain unsupported. This is a graph implementation
and tiny qualification, not real Llama checkpoint support.

The [explicit GPU tests](../tests/mlx_llama_metal.rs) consume the original
[frozen oracle](mlx-llama-oracle.md) by literal manifest and exact file/array
hashes. Both tied and untied heads run full/three/one-token chunks at all five
cases, including offset5 and capacity32, with Reference attention.
Readback-mode tests require actual full distributions or token-only absence,
and cover exact-prefix reuse, pending sampled history, changed-prefix replay,
partial prefill, boundary overflow/recovery and memory admission. The original
Qwen MLX and ordinary/compensated Half GPU paths are regression checked because
the bias and assembly representation is shared.

A frozen gate fails for the Online candidate at the offset case's second-layer
attention projection. Its precise numerical cause is unproven. Llama therefore
rejects explicit Rust/CLI Online requests before session allocation; no tolerance
is changed and Qwen Online remains available. The initial failed runs also expose
the former hardcoded rotary base; exact failed sources and logs are retained.

The successful release run on Apple M1 Pro checks1,050 diagnostic snapshots and
149,568 values, eight required full decode distributions and16 exact selections.
Resident numerical payloads are17,812/20,372 bytes for tied/untied models, an exact
2,560-byte packed-head difference. Model-load boundary and session-budget checks
pass. The standalone rotary test compares13,824 independent values for both
bases through position32767 under its unchanged operator envelope; this does not
extend the decoder's32-token admission.

All11 explicit GPU/integration tests pass: rotary, two Llama cases, three Qwen
MLX cases, ordinary and compensated Half, the Rust caller of the C ABI, and two
CLI cases. Rust1.90 formatting, strict all-target Metal Clippy,140 ordinary CPU
passes/eight ignored and155 Metal-feature passes/77 ignored also pass. The
[receipt](../benchmarks/results/2026-10-22-llama-metal-v1.json) binds3,639 files,
76 current source copies, exact failed-run correspondence, original gates and
protected binaries. Failed source snapshots are retained; the comment-only C
header is an explicitly identified SHA-verified reconstruction rather than an
original captured file. Astra reviewed code and completed evidence before commit.

The CLI and Rust caller of the C ABI exercise both heads through the existing
explicit native selection. The C ABI has an explicit architecture-specific
Reference default for tiny Llama; Qwen, ordinary Half and Q4 keep Online.
This is selected before session creation, with no retry/fallback. Raw token IDs and statistics are tested; tokenizer
chat compatibility, a newly compiled C client, signed iOS package and physical
iPhone execution are separate qualifications. Historical Qwen package/device
results remain scoped to their original revisions. Real models, held-out
quality, performance and wider device support remain open in the
[M8 plan](mlx-implementation-plan.md).

Example with the synthetic untied fixture (not a pretrained model):

```sh
cargo run --release --features metal -- --model tests/fixtures/mlx-llama-v1/untied \
  --backend metal --metal-projection mlx-affine --metal-attention reference \
  --context-capacity 32 --prefill-chunk 3 --tokens 1,7,13,3 --max-new-tokens 4 --json
```

The four selected IDs are `[3,3,3,3]`; the tied fixture selects `[47,47,47,47]`.

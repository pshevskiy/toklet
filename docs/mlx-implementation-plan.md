# Native Rust support for MLX-format models

Requested 21 October 2026, after the completed M0–M7 audit. The user explicitly
selected **native Rust model support** instead of a Swift MLX runtime backend.
Use Toklet's Rust/Metal engine to execute compatible Hugging Face MLX artifacts;
recreating MLX's tensor framework, autodiff and compiler is outside this choice.
The earlier uncommitted Swift investigation is retained under ignored artifacts.
No Swift MLX runtime dependency is added to the product.

## Contract and scope

Implement one mechanism at a time with scalar references and independent fixtures.
MLX packed tensors are not Toklet Q4, even where both use four-bit affine codes.
Keep bit order, shape, metadata dtype, quantization mode and numerical rounding
explicit. Resolve global/per-layer quantization and exclusions at loading time.
Preserve quantized embeddings/tied output ownership and resident buffers instead
of expanding the full model per token. Preserve existing CPU/Metal paths/defaults
and historical qualification; new representations and architecture semantics need
separate numerical, state, quality, memory and physical-device evidence.

Architecture support is incremental and explicit. The pinned upstream language/
vision registry is a reference inventory, not evidence of native Toklet support.
Each architecture needs its actual graph, positions, normalization, cache/state,
weight mapping and, where applicable, modality processing. Unknown/unimplemented
architectures must reject clearly. The user's wider architecture request remains
open until those implementations and checks exist; a Qwen importer does not
silently close it. Architecture support, compatible checkpoints, estimated device
fit and physically measured configurations remain separate statuses.

The app chooses capabilities/budgets from the device it is running on. File size
alone does not establish fit. Include load conversion, resident weights, KV or
recurrent state, scratch, tokenizer/media/application allocations and temporary
buffers. Mac and phone limits remain independent. Search/downloads stay outside
numerical kernels and follow engine support, as requested.

## Ordered tasks and exits

Astra reviews each code task; commit only after its checks pass and before starting
the next code task. Keep [the improvement record](improvements.md) updated.

- [x] **M8.1 Native artifact primitives.** Validate MLX affine U32 packing, separate
  scales/biases, bit widths/group layouts and metadata conversion without
  allocation. Preserve a scalar reconstruction/matvec oracle, handpacked cases,
  pinned upstream references and malformed-input/overflow checks. Floating-point
  quantization modes remain explicitly unsupported until separately implemented.
- [ ] **M8.2 Local Qwen execution.** Add validated safetensors/index/config loading,
  per-tensor quantization resolution, resident packed projections/embedding/tied
  output, independent tiny/reference parity, session/failure/memory checks and
  explicit CLI/native app selection. Start with supported Qwen semantics; do not
  automatically adopt previously rejected quality configurations.
  - [x] Validated [local artifact source](mlx-qwen-source.md), packed/floating
    tensor inventory, per-module settings, bounded streaming and tied ownership.
  - [x] [Resident affine product/gather operators](mlx-metal-affine.md), explicit
    GPU value domain and independent primitive qualification on M1 Pro.
  - [x] Explicit [full-model scalar reference import](mlx-reference.md) and frozen
    mixed-precision tiny Qwen graph traces/native MLX CPU/GPU logits.
  - [x] Independent [Half-activation graph oracle](mlx-half-oracle.md), including
    original packed tied-output weights and gates frozen before GPU composition.
  - [x] [Complete resident tiny graph/session](mlx-metal-model.md), strided views,
    frozen independent parity and transaction/admission checks on M1 Pro.
  - [x] Explicit [CLI selection](mlx-cli.md), truthful mixed-precision statistics
    and local subprocess generation/state/stop checks.
  - [x] Additive [native C bridge entry point](mlx-native-bridge.md), explicit
    session selection, Mac C/Rust lifetime/state checks and iOS Rust compilation.
  - [x] Explicit [Apple UI/worker selection](mlx-apple-worker.md), Mac worker
    state/replay/cleanup and strict iOS source compatibility checks.
  - [ ] Real-checkpoint quality/memory/performance, linked app packaging and jobs.
- [ ] **M8.3 Architecture expansion.** Record the full language/vision reference
  inventory and implement compatible dense, MoE, recurrent/hybrid and multimodal
  graph families in reviewed tasks. Qualify weight names, bias/tied conventions,
  positions, cache/state and real media processing separately. Each absent family
  remains visibly unsupported; registration in another runtime is insufficient.
- [ ] **M8.4 Device-aware discovery/downloads.** After engine compatibility exists,
  expose Hugging Face model search, truthful support/fit labels and device-local
  memory/storage admission. Pin immutable revisions and exact file inventories;
  stream and hash downloads, cancel safely and publish complete snapshots
  atomically. Ordinary inference then works offline. Test corrupt/truncated data,
  traversal, interruptions, stale budgets and unavailable network/authentication.
- [ ] **M8.5 Mac and iPhone qualification.** Test downloaded compatible models on
  the M1 Pro and iPhone 14 Pro, including different implemented architectures and
  actual image input for vision claims. Record identities, deterministic outputs,
  cancellation/reload/lifecycle, completed timings, memory and thermal behavior.
  Selected checkpoint evidence never qualifies every artifact of an architecture.

Exit: native Rust/Metal executes the implemented MLX representations/architectures
with validated local/offline inference; the app on either tested device can find,
admit and download an eligible immutable snapshot. Architecture and device
coverage must enumerate what actually passed and retain all remaining open work.

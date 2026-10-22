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
- [x] **M8.2 Local Qwen execution.** Add validated safetensors/index/config loading,
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
  - [x] [Fresh native package and linked apps](mlx-apple-package.md), complete
    shader provenance, three platform slices and actual Mac C execution.
  - [x] Explicit [authenticated MLX diagnostic jobs](mlx-apple-jobs.md), legacy
    schema preservation and tiny Mac routing checks.
  - [x] [Lifecycle error-trace cleanup](mlx-lifecycle-cleanup.md), preserving
    original errors and failed traces after verified owner shutdown.
  - [x] [Independent real-reference gates](mlx-real-oracle.md), full selected public-checkpoint CPU arrays and external manifest admission before target trials.
  - [x] [Selected real-checkpoint Mac target parity](mlx-real-metal.md), complete stage/chunk/offset checks and cached/online/token-only session contracts.
  - [x] [Shared sealed jobs and real Mac app-runner integration](mlx-mac-jobs.md), selected short functional checks, observed process charge and signed iPhone preparation.
  - [x] [Guarded physical MLX jobs](mlx-phone-jobs.md), actual selected public-checkpoint generation/scoring/tasks and native process peaks on iPhone 14 Pro.
  - [x] [Full frozen loss collection](mlx-full-loss.md) on Mac and iPhone: 10,000 targets each and exact cross-device scores. The public four-bit candidate fails the unchanged loss gate (+0.19920869 > 0.05) on both and is rejected for adoption; acceptance remains open.
  - [x] [Upstream same-artifact loss diagnosis](mlx-upstream-loss.md): full MLX-LM mean 3.08305608 also fails the frozen gate, with descriptive precision differences retained. No native backend replacement or acceptance waiver.
  - [x] [Immutable public eight-bit candidate acquisition](mlx8-checkpoint.md), exact files and raw-ID metadata inspection; reference/target/quality statuses are tracked separately below.
  - [x] [Independent eight-bit numerical reference](mlx8-real-oracle.md), 346 arrays and unchanged policy frozen before target trials; target/quality evidence is tracked separately below.
  - [x] [Eight-bit native Mac numerical/session parity](mlx8-real-metal.md), all frozen stage gates, eight full distributions and 16 exact selections; app/device/quality/performance remain separate.
  - [x] [Eight-bit shared app jobs and physical short checks](mlx8-app-jobs.md), explicit immutable recipe, real-model storage admission, four Mac jobs and ten signed-phone stages; full quality/performance/state/lifecycle remain open.
  - [x] [Eight-bit full loss acceptance](mlx8-full-loss.md) on Mac and iPhone: 10,000 targets each, exact paired scores and +0.00122855 mean loss against the unchanged 0.05 gate.
  - [x] [Repeated physical-grading idle hook](mlx-task-grading-window.md), mandatory local checks plus optional remote checks at all five existing boundaries, strict fresh outputs and 49 mocked portable methods. Actual native-app response admission/grading remains open.
  - [x] [Eight-bit full task response collection](mlx8-full-task-collection.md), all 100 original full-cap responses on Mac and iPhone with exact paired raw IDs/stops/state; decoding and isolated outcome grading remain open.
  - [x] [Post-exit native-app response admission](mlx-task-responses.md), original tokenizer/runtime binding, two read-only local/remote idle observations, explicit app-event derivatives and exact paired literal text. Isolated outcome grading remains open.
  - [x] [Eight-bit full task quality](mlx8-full-task-quality.md) on Mac and iPhone: 2/100 each versus FP32 2/100, zero drop; one isolated40-program run, exact phone literal equivalence, six idle observations and verified VM restoration. The low baseline remains explicit.
  - [x] [Strict Mac measurement admission](mlx-mac-measurement.md), complete clocks/state/memory/duration checks and one128/128 app trial; full matrix/confirmation/sustained and phone performance remain open.
  - [x] [Eight-bit app performance observations](mlx8-performance.md), all 128/512/1024 primary/confirmation workloads and 600-native-second sustained runs on Mac/iPhone; phone thermal slowdown remains explicit, with no comparative speedup or steady-rate claim.
  - [x] [Eight-bit physical state witnesses](mlx8-phone-state.md) on iPhone 14 Pro: actual 128/384 committed/pending prefill cancellation, exact resume and full1152 boundary/overflow/recovery.
  - [x] [Eight-bit actual physical MLX OS lifecycle](mlx8-phone-lifecycle.md), same live process across Settings/return, exact retained-prefix replay/reference, synthetic warning and natural close.
  - M8.2 closes selected local Qwen execution only: the eight-bit snapshot passes the frozen numerical/quality/state/performance/lifecycle trials; the public four-bit snapshot remains rejected. Raw-ID diagnostics do not qualify downloaded chat templates, every Qwen configuration or wider M8 architecture/discovery/device scope.
- [ ] **M8.3 Architecture expansion.** Record the full language/vision reference
  inventory and implement compatible dense, MoE, recurrent/hybrid and multimodal
  graph families in reviewed tasks. Qualify weight names, bias/tied conventions,
  positions, cache/state and real media processing separately. Each absent family
  remains visibly unsupported; registration in another runtime is insufficient.
  - [x] [Pinned full Python source/reference inventory](mlx-architectures.md), 137 language/1,399 vision modules with static root exports and literal aliases; no new native architecture support.
  - [ ] Dense family expansion, beginning with independently specified Llama/Mistral-style semantics.
  - [ ] Expert routing, recurrent/hybrid state and actual multimodal/media execution, each with separate fixtures and target qualification.
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

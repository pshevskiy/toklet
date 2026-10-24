# Runtime Metal resource planning

`metal::planning::inspect_mlx` is an explicit Rust-only cold inspection for native
MLX affine dense execution. It accepts load/context options, session capacity,
chunk size, position offset, normalization, readback and attention mode. Projection
is fixed to `MlxAffine`. The existing CPU-only compatibility report and C/Swift
interfaces keep their previous contracts.

The actual loader and session constructor consume the same concrete plans as
inspection. Weight planning resolves logical tensors once, queries MPS Half row
strides and includes guard elements for floating exclusions. Packed codes stay
U32 with separately padded scales and biases. Tied output is counted once.
Session planning retains separate Half activations, FP32 intermediate/score
buffers, per-layer Half K/V `[kv_heads, capacity, head_dim]`, guards, token/status
and optional token-selection buffers. It includes host history, current/staged
logits or selected IDs, cache/view tables and the existing 64 MiB allowance.

The report exposes three overlapping admission totals:

- Source payload plus its existing metadata/stream allowance.
- Resident GPU weights plus the existing metadata/stream allowance during load.
- Resident weights plus one session's KV, scratch, history, output, tables and
  metadata allowance.

Resident weights are already in the latter two totals. Do not add those totals
together. Load and session budgets are independent requests; the source and
resident load gates must both fit the load budget. A small budget yields explicit
fit flags while invalid dimensions/options or unsupported modes remain errors.

Inspection checks configuration, tensor headers, geometry, modes and queried
MPS layouts before creating the GPU context. It then initializes the same precise
Metal/MPS pipelines and checks each numerical allocation against the device's
`maxBufferLength`, host indexing and existing shader layout bounds. Allocation
uses the same buffer-limit helper, and loading retains all payload/domain checks.

This process allocates driver, library and pipeline resources. The 64 MiB
allowance is an estimate, not measured driver memory. Inspection allocates no
numerical weights, KV, scratch or logits owners. It does not measure free memory,
reserve a budget, guarantee allocation success, validate all payload values,
qualify a model/device or authorize reuse after files change. Actual loading and
session admission remain mandatory. Diagnostics, profiling, tokenizers, app
owners and other sessions need separate admission.

## Evidence and remaining work

Before extraction, 42 actual allocation reports were frozen in the
[resource baseline](../tests/fixtures/metal-resource-v1/baseline.json): six tiny
Qwen2/Llama/Mistral/Qwen3 artifacts, tied/untied output, mixed affine/floating
storage, independent query width, chunks 1/3/32, both readback modes and Qwen2's
qualified Online path. This is an M1 Pro allocation-accounting baseline, not an
independent numerical oracle or a portable stride promise.

The explicit GPU tests compare each frozen report with both the new plan and
actual owners. Additional checks cover exact and limit-minus-one source/load and
session budgets, nonzero position offsets and invalid/overflowing options.
Ordinary tests exercise every scratch/KV allocation's limit, checked addition,
float/packed weight limits and unqualified Online rejection without a GPU.
The selected real Qwen3 memory test compares planned resident weights and session
admission with actual owners, then retains its independent final-row numerical
and exact diagnostic budget checks.

[The receipt](../benchmarks/results/2026-10-24-metal-resource-planning-v1.json)
records results, source identities, original failed checks and their correction.
Formatting, strict Clippy,180 CPU/203 Metal-feature tests, both iOS compile
targets and16 explicit release GPU tests pass. A test-only Clippy style failure
and an accidental debug Half allocation-assertion failure are retained; the
unchanged allocation gate passes in the established release configuration.
This task does not add timing, quality or physical iPhone qualification. The
[H1 CLI/native runtime consumers](metal-resource-consumers.md) now expose
these plans. The Swift/app runtime consumer and exact qualification-receipt
evaluation remain open, as do the other items in the active expansion plan.

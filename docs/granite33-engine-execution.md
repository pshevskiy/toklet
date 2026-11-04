# Granite3.3-2B MLX8 Mac engine execution

5 November 2026. Real execution and checks pass Astra review. This is diagnostic engine coverage, with managed
Chat, full-model fidelity/quality, performance and iPhone qualification separate.

Artifact: `mlx-community/granite-3.3-2b-instruct-8bit`, revision
`37b98558f3eb3db67899d719fb9a74d2443a2529`. The 2,692,145,735-byte safetensors file
has SHA256 `e56b83c455906865ebcca788b183a1902489b86003f721870aa7543ce7348b99`.
Its 924 tensors comprise 281 U32 and 643 F16 entries, with affine8/group64 metadata
and one tied embedding/head table. Geometry: hidden 2048, intermediate 8192,
40 layers, 32 query/8 KV heads, head dimension 64, vocabulary 49159. The declared
131,072-position window is not the engine's tested window.

The shared dense engine uses the [reviewed scaling blocks](granite-scaling-blocks.md),
Reference attention, compensated normalization, Half activations/KV and an
initial 512-position limit. A typed coarse selector extends the existing shared
Qwen1.5/SmolLM3 diagnostic worker; no numerical hot-loop dispatch is added.

Astra selected an explicit system message to prevent the template's current date
from affecting the prompt. The frozen 27 IDs encode “What is 2 + 2?” with the instruction
to return only the result; rendered bytes and IDs agree through independent
encoding. There is no additional BOS. Stop 0 may occur inside prompt turn
boundaries; only a generated 0 stops output. The eight-slot cap includes stops
and greedy ties select the lowest vocabulary ID.

Strict stock MLX loads the unmodified original F16 export, validates 281 quantized
modules and all four scales, and emits complete 49,159-value rows before each
selected ID. Stock and native choose `[38,0]`: visible “4” and hidden stop.
Complete-row errors are descriptive; no numerical tolerance is widened or
full-fidelity gate inferred from greedy agreement.

Each process has a 6 GiB external peak guard,900 second timeout, owned process-group
cleanup and exit evidence. Native weights load once; sessions are dropped before
replacement. The shared probe checks chunks 1/32, actual 512 KV positions, rejection
of 513 preserving tokens/state/logits, and reset back to the original prompt.

The first stock worker's copied BF16 assertion failed before rows; the pinned
header establishes F16 and the retry keeps that original dtype. The first native
wrapper selected Smol; the input guard rejected it before model planning/loading.
Both source/launch attempts and logs remain in the evidence directory. Frozen
artifact, prompt, stops, arithmetic and acceptance policy were unchanged.

Evidence: [inventory/runs/replay](../artifacts/granite33-inventory-2026-11-05/),
[checks](../artifacts/granite33-real-checks-2026-11-05/).

All seven required checks pass: formatting, strict Clippy/default+Metal, 558 CPU
and 667 Metal tests, release library and all three probe builds. The corrected
selector additionally passes fresh formatting, both strict Clippy variants and
release build. CPU/Metal test bodies are unchanged by that one selector fix.

Resident weights: 2,692,042,734bytes. Native admission: 2,815,063,656bytes;
peak RSS: 2,777,350,144bytes; observed physical high-water: 2,771,896,864bytes.
Stock RSS: 2,497,413,120bytes; observed physical: 3,053,931,616bytes. Both owned
processes exit 0 with no remaining group and stay below 6 GiB. These are separate
process observations, not a whole-app memory or speed benchmark. Full-row maximum
errors are 0.08621645/0.05864096; greedy agreement is the scoped decision criterion.

[Result](../benchmarks/results/2026-11-05-granite33-engine-execution-v1.json).

Astra approves the scoped code and evidence, with no actionable findings.

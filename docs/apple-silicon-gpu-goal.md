# Apple Silicon execution priorities

15 November 2026. Focus for the active [Gemma/LFM3 performance goal](mlx-gpu-performance-plan.md).
Astra researches the next mechanisms while Sol implements; Astra reviews each
combined diff/check packet before commit. Large builds, GPU/model evaluations
and acceptance measurements run serially to preserve memory and conditions.

The immediate target is GPU-resident blocks using unified memory. Toklet already
uses Metal shared, tracked buffers. Keep weights, activations, committed KV and
proposals on those owners; pass checked views between dependent encoders and
wait at the publication boundary. Shared storage still requires synchronization.
[Apple shared-storage contract](https://developer.apple.com/documentation/metal/mtlstoragemode/shared).

Prioritize these mechanisms:

- Compose accurate projections, RMS/rotary, attention and FP32 pointwise work on
  one command. The new [attention encoder](gemma-gpu-attention.md) supports shared
  ring views and caller-owned commands; the [complete-block pilot](gemma-gpu-producer.md)
  now keeps all eight nominal cases on one command/wait. Whole-graph integration
  and request performance remain open.
- Reuse weights across prefill rows with a qualified matrix/tiled kernel. Keep
  single-token decode focused on weight traffic and dispatch cost. Measure
  actual request effects on the named M1 Pro, rather than infer throughput from
  peak bandwidth or core counts.
- Preserve private proposed KV and commit only staged new rows after all status,
  guard and cancellation checks. Avoid full-cache copies and duplicate CPU/GPU
  weight owners. Keep PLE file-backed with bounded row gathers.

The pinned official tensor inventory contains 9,257,138,758 active BF16 bytes,
including a 4,697,620,480-byte PLE table. Removing media and inactive KV tensors
alone cannot fully resident-fit that payload under 6 GiB. Header-only size
analysis is not a current payload authentication or measured runtime footprint.
One hypothetical affine4/group64 layout totals 2,603,981,382 tensor bytes; a
common converted artifact still needs exact provenance and original full quality
and numerical gates before any comparison.
[Header/inventory ledger](../artifacts/mlx-gpu-gap-2026-11-15/g0-active-weight-ledger-1.json).

## Neural Engine feasibility

Use public Core ML APIs for a separately qualified stateless projection or FFN
experiment. FP32-typed MLProgram excludes the Neural Engine; CPU+ANE configuration
also permits CPU fallback. Inspect compiled operation placement and actual trace,
include conversions/partition crossings and retained caches in latency/memory,
and preserve the original numerical/full-quality gates. Model storage dtype or
successful loading alone does not prove ANE execution.
[Typed execution](https://apple.github.io/coremltools/docs-guides/source/typed-execution.html),
[compute-unit selection](https://developer.apple.com/documentation/coreml/mlcomputeunits/cpuandneuralengine),
[compute-plan inspection](https://developer.apple.com/documentation/coreml/mlcomputeplan-85vdw).

Core ML state does not automatically preserve Toklet's FP32 KV and transaction
semantics. Apple's stateful-KV example selects CPU/GPU. Metal 4 provides compiled
model integration on the GPU timeline, but the configured toolchain lacks
`metal-package-builder`; actual GPU/ANE placement remains untested here. Keep
that experiment conditional while accurate Metal work proceeds.
[Stateful models](https://apple.github.io/coremltools/docs-guides/source/stateful-models.html),
[Metal machine-learning passes](https://developer.apple.com/documentation/metal/machine-learning-passes).

[Detailed GPU/block research](../artifacts/mlx-gpu-gap-2026-11-15/astra-gpu-research-1.md),
[hardware/API evidence and bounded ANE experiment](../artifacts/mlx-gpu-gap-2026-11-15/astra-apple-hardware-research-2.md).
No ANE, complete-graph or competitive-performance qualification is claimed.

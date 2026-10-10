# Selected Qwen stored-Half gating, 10 October 2026

The selected stored-Half products pass native qualification and independent
saved-output replay. Astra accepted the completed code and evidence without
findings after an independent replay. The source extension
permits width 4,864 alongside
the previously checked width 64, retaining the 8,192-value aggregate admission
limit. One complete wider row fits; two wider rows or two wider cases fail
admission before native allocation. The shader and arithmetic remain unchanged.

The [driver](../../tools/experiments/rust_metal_qwen_gating.py) reads saved
`gate_silu`, `up_projection` and `gated` arrays for layers 0, 1 and 23 in four
pinned cases. Twelve complete stages contain 36 rows and 175,104 products.
Each complete payload must match its shape, dtype, byte count and SHA before
splitting into independent one-row native inventories. Four full/selected
manifests, model topology, config and original FP32 gates are checked without
opening or hashing the checkpoint. Both input operands, the product oracle and
the FP32 source retain full-array and row identities.

Two immutable contiguous Half rows have layout `[1, 4864]`. A third distinct
owned buffer stores each FP32 product with one Half round-to-nearest-even cast,
preserving the earlier SiLU Half boundary. Every finite Half product has at
most 22 significant bits and lies exactly within FP32 and FP64 range before
the Half cast. An independent FP64-product-to-Half oracle must reconstruct
every frozen product before GPU work; every completed output must match those
bits and the unchanged FP32 source gate.

Each guarded buffer contains 4,871 Half values and occupies 9,742 bytes.
Three shared allocations count once at 29,226 bytes; two host baselines and
three readbacks add 48,710 bytes. Including the existing 64 MiB runtime
allowance, maximum admission is **67,186,800 bytes**. Input buffers, their
guards, output guards, retained resource ownership and completed-command
status are checked before host consumption. Buffers and pipelines are reused
across three warmups and ten samples. A final full saved-payload replay detects
changed references after execution.

The same isolated binary repeats the prior tiny products and rejects the
completed max-Half-times-two overflow with structured validation failure,
exit 1 and discarded output. Kernel launch sizes come from the queried device
and pipeline limits. The observed execution width is 32: both admitted gating
widths are divisible by it, so gating inactive-lane tails are unqualified.
The separate add regression exercises lengths 1, 17 and 257.

The [native record](../../benchmarks/results/2026-10-10-qwen-metal-gating-v1.json)
contains 468 completed Qwen commands, **2,276,352 exact Half output values**
and 3,276 unchanged output guards. Maximum original FP32 gate fraction is
**0.490092755385** and maximum absolute source error is 1.3365683555603027;
the original tolerance policy is unchanged. Tiny products repeat 143 completed
commands, 87,360 exact outputs and 1,001 guards. Including the 39-command add
regression, there are **650 positive completed commands** and three required
negative jobs: invalid shader, insufficient admission and completed product
overflow. Every successful command reports completion without a Metal error
before host output is consumed.

Maximum own-process physical charge across the Qwen jobs is **10,158,728 bytes**;
the largest across all jobs is **13,369,944 bytes**, below conservative admission
and 6 GiB. These are isolated primitive processes, not full-model memory or
end-to-end performance measurements. No latency improvement is claimed.

This run used Apple M1 Pro, MacBookPro18,3, 16 GiB, macOS 26.6.2 (25G83),
Rust 1.90.0 and SDK 26.5. The standalone arm64 release package uses a locked,
offline build and macOS 15 deployment target. Metal language 2.3, safe math
and precise functions are selected explicitly. Start/end records show AC power
and configured AC low-power mode off, with no recorded thermal warning;
continuous thermal/power sampling, energy and a GPU trace are unavailable.

The native binary SHA is
`1370e455bc5afe49557aab51175c22c3c14510426a0293d4668a383e57e4889b`.
Exact original and copied sources, drivers, 38 inputs, four Qwen manifests,
two tiny manifests and config are bound in the record. The unchanged shader
SHA is `a63e219e6385671d4e26675ab2b67d5a6a3f5ba26aa820062241ac1753dd2ebe`.
The versioned record SHA is
`b2386554137a0494a2e6dee39175803e2e0d50a3043384a412fb8fa3b8fe717d`;
its large local native JSONL has SHA
`dfa4230d3adce374d4fe3b6d3c4eebb1fc6cf98914d8b013fdc966f662856c06`.
The [analysis](../../benchmarks/results/2026-10-10-qwen-metal-gating-analysis-v1.json)
and exact [producer](../../benchmarks/results/2026-10-10-qwen-metal-gating-analysis-v1.py.txt)
independently replay every product, source gate, guard, transcript and identity.

The offline tests cover complete-array reconstruction of both operands and
references, exact Half products, unchanged source gates, payload corruption,
finite inputs, post-load change detection, overflow failure propagation and
output path protections. Native Rust tests require the original inventory
ceiling, guarded lengths and exact admission. Root formatting, Clippy with
warnings denied, 76 ordinary Rust tests, 70 experiment Python tests and 35
benchmark Python tests pass in the retained
[task checks](../../benchmarks/results/2026-10-10-qwen-metal-gating-task-checks-v1.json),
with exact [runner source](../../benchmarks/results/2026-10-10-qwen-metal-gating-task-checks-v1.py.txt).
The isolated package passes module/package formatting, Clippy, **19 Rust
tests** and release compilation, with full transcripts in the native record.

All checks and native work follow the append worker's natural exit and
[owned-process idle proof](../../benchmarks/results/2026-10-10-cpu-qwen-gating-check-window-idle.json)
at 08:49:00 UTC. Only orchestration remains held ahead of the next workload;
no numerical worker was stopped. The exact
[idle verifier](../../benchmarks/results/2026-10-10-cpu-qwen-gating-check-window-idle.py.txt)
and [prepared resumption verifier](../../benchmarks/results/2026-10-10-cpu-qwen-gating-check-window-resume.py.txt)
are retained. Root numerical sources and protected release binaries are
unchanged. Astra independently reproduced the retained analysis exactly and
verified outputs, source gates, guards, required failures, transcripts,
source/copied-file/binary identities, ownership, admission and scope; no
findings remain.

The subsequent
[completed idle window](../../benchmarks/results/2026-10-10-cpu-qwen-gating-check-window-completed.json)
records clean reviewed revision `f36ad90`, both task commits, eight verified
model assets and 94 source/artifact bindings. All checks and hashes finish
before resuming only orchestration at 09:01:08 UTC. The fully cached workload
then runs uninterrupted while only its parent is
[held ahead of pending](../../benchmarks/results/2026-10-10-cpu-cached-check-window-hold.json).
No numerical source or frozen executable is changed by this archival update.

Reproduce in a fresh idle window with disjoint paths:

```sh
.venv/bin/python tools/experiments/rust_metal_qwen_gating.py \
  --output benchmarks/raw/qwen-metal-gating-replay.json \
  --build-dir artifacts/qwen-metal-gating-replay \
  --input-dir artifacts/qwen-metal-gating-replay-inputs
```

This task qualifies selected stored-Half products only. Fusion, MLP/decoder/KV
composition, sessions, full quality, end-to-end performance, GPU traces and
M4 completion remain pending.

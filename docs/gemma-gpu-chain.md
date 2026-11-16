# Resident Gemma three-layer GPU chain

16 November 2026. Bounded implementation of the [GPU plan](mlx-gpu-performance-plan.md).
The eight original numerical cells, state/replay controls and allocation checks
pass. Required checks and combined Astra code review pass, with no remaining
findings. This
private test path is not selected by the ordinary model, session or CLI.

## Shapes and ownership

| Layer | Attention | Head width | FF width | KV source |
| --- | --- | --- | --- | --- |
| 13 | Local | 256 | 6,144 | Own committed prefix and private proposal |
| 14 | Global | 512 | 6,144 | Own committed prefix and private proposal |
| 15 | Local | 256 | 12,288 | Layer 13 |

Qualification covers at most four active rows and four active positions. Both
local and global donor caches physically allocate 512 positions. Larger admitted
shapes and deeper active contexts remain unqualified. `Last` contracts only the final
consumer query; both producer proposals still contain every new row. Decode
imports the causal three-row prefill prefix from the same frozen sequence.

The graph borrows immutable BF16 weights, the base pipeline bundle and one
certified GELU table; it owns the dedicated strided/RMS pipelines. A session owns three
layer arenas, two committed caches, private proposals,
entry/PLE checkpoints and bounded reference scratch. One mutable transaction
owns all three nodes; a node frame cannot publish independently. Shapes, device
identity, physical table size, pipeline variant, ranges and aliases are checked
before operator encoding. Strided BF16 output writes directly into the final
row-major destination while retaining the original projection arithmetic.

The nominal chain uses 102 compute dispatches on one completed command/wait,
with zero CPU replay in all eight cells. Global status grammar and integrity
are checked across all nodes before the ordered numerical decision. Uncertainty
replays the entire chain from its entry checkpoint and old committed caches;
it never patches a lane or a later node. A successful replay performs 25 matrix
operations and 27 actual projection commands/waits, which remain separately
reported. Commit copies are prevalidated and both cache generations/history
advance last. Failure/cancellation preserves a valid prefix; queued-command drop
drains work and poisons unsafe reuse.

[Implementation](../src/metal/gemma_affine/bf16/resident/producer/chain.rs),
[shared operation recipe](../src/metal/gemma_affine/bf16/resident/producer/recipe.rs).

## Precision prerequisites

The existing scalar reference and original numerical limits remain unchanged.
A guarded higher-accuracy RMS kernel uses power-of-two scaling and double-float
arithmetic. Exact integer preflight for cancellation replaces a coarse add-grid
rejection without changing its floating arithmetic. Leaf replay verifies 902,736
values, including 198 real cases and 74 integer boundary controls; this is not
an unrestricted all-input RMS proof.

The finer GELU table preserves every approved parent record and adds 90,114
records over the symmetric high-curvature core. Its single GPU payload is
5,480,520 bytes, an increase of 2,162,736 bytes. Independent native verification
covers all 184,549,378 replacement inputs; authenticated unchanged outside
coverage completes the finite-domain composition. Conditional admission and
exact whole-case fallback remain necessary. Astra approves this artifact, not
whole-model numerical or performance adoption.

Original sealed certificates retain their complete historical hostname. Runtime
platform admission omits only mutable nodename while preserving exact kernel,
architecture, OS, compiler/source and qualified release-profile checks. This
corrects an incidental host label; it does not prove physical-machine identity
or recertify another platform.

[RMS proof](../artifacts/mlx-gpu-gap-2026-11-15/g2-rms-precision-guard2-reader-progress-1.json),
[GELU artifact review](../artifacts/mlx-gpu-gap-2026-11-15/astra-gelu-core-review-1/verdict.json).

## Completed-command observability

Two public Metal timestamp getters run after the existing successful wait. Raw
finite endpoints, nullable duration, availability reason and command/epoch
binding describe the completed candidate **whole command**, including any caller
work. They exclude subsequent reference commands. Missing/invalid timestamps do
not alter numerical success. All eight nominal observations have valid native
properties; the independent reader checks schema/arithmetic and binding, without
claiming an independent profiler calibration.

Host encoding, commit-plus-wait, completion/validation/replay and publication
retain their separate scopes. Counts distinguish commands, dispatches, waits,
CPU touches/copies of shared buffers and physical owners. These overlapping
clock scopes cannot establish GPU utilization or idle time by subtraction.
If recipe encoding fails after partial dispatch, the dispatch count becomes
unknown before propagating the original error. Completion stays false and the
existing drain/poison behavior remains. Diagnostic captures also add CPU reads. There are no new commands, waits,
counter resolves, numerical allocations or timers inside operators.

## Verification and limits

| Check | Result |
| --- | --- |
| Original coupled witnesses | Eight cells, 2,375,680 values, 480 exact mask zeros, 64 causal-prefix states, eight atomic dual-cache commits |
| Lifecycle/fallback | 16 controls; three whole-chain replays, 12 failed/canceled transactions and two cancel/reuse controls |
| RMS integration | Eight controls; three replays, five atomic failures and 266,752 independent canonical values |
| Warm reuse | 32 iterations, zero Rust allocations, one candidate command/wait each; native allocations unmeasured |
| Original producer regression | Eight v1 cases and both unchanged numerical readers pass |
| Required checks | Fmt, strict CPU/Metal Clippy, 1,044 Metal tests (393 ignored) and release builds pass |
| Failure-count regression | Fresh release test covers partial errors after prior work, unchanged error identity, sticky unknown and successful 102-dispatch accumulation |
| CPU check reuse | 847 tests; all 4,009 non-Metal source/dependency/test inputs match the tested baseline |

Nominal observed peak is 4,699,607,696 bytes including the 32 MiB helper. The
unchanged 6 GiB guard uses sampled owned-process accounting plus terminal
measurements; it is not a continuous allocator cap. This private test excludes
the ordinary codec and does not establish full-graph/CLI working-set fit.

[Nominal checks/readers](../artifacts/mlx-gpu-gap-2026-11-15/g2-deep-chain-core-nominal-progress-1.json),
[state/RMS/warm checks](../artifacts/mlx-gpu-gap-2026-11-15/g2-deep-chain-core-state-progress-1.json),
[legacy producer checks](../artifacts/mlx-gpu-gap-2026-11-15/g2-deep-chain-pilot-progress-3.json),
[CPU reuse proof](../artifacts/mlx-gpu-gap-2026-11-15/g2-deep-chain-CPU-reuse-proof-2.json),
[source9 required checks](../artifacts/mlx-gpu-gap-2026-11-15/g2-deep-chain-build-progress-9.json),
[C2 regression](../artifacts/mlx-gpu-gap-2026-11-15/g2-deep-chain-C2-release-regression-1.json),
[exact numerical-input/check reuse](../artifacts/mlx-gpu-gap-2026-11-15/g2-deep-chain-C2-reuse-proof-2.json),
[Astra code approval](../artifacts/mlx-gpu-gap-2026-11-15/astra-deep-chain-code-review-1/verdict.json),
[publication/source/CI](../artifacts/mlx-gpu-gap-2026-11-15/g2-deep-chain-publication-1.json),
[current checks and retained failures](../artifacts/mlx-gpu-gap-2026-11-15/g2-deep-chain-progress-1.json).

Source9 changes only failure-count metadata and its pure regression. The actual
GPU numerical/state/allocation captures above are retained from source8 with an
exact inverse-patch and unchanged-input proof; they are not new source9 captures.
CPU Clippy/tests and release CLI checks also reuse their unchanged inputs.

Time the existing RMS aggregate, pool scratch with GPU validation before
overwrite, then qualify the complete
resident graph and ordinary CLI. Full model quality/state, representative
request latency, matched Ollama/vLLM wins and the original eight overhead gates
remain open. Summary defaults remain disabled; no new speedup is claimed here.

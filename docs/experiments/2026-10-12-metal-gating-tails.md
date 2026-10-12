# Stored-Half gating inactive lanes, 12 October 2026

The bounded gating extension passes native qualification and independent
saved-output replay on Apple M1 Pro. It covers inactive lanes in rounded
threadgroup dispatches without changing the product shader or arithmetic.
Astra accepted the completed code and evidence without findings after independent byte-identical replay. This is M4 preparation; MLP,
attention/KV and complete decoder composition remain open.

The [Rust host](../../tools/experiments/rust_metal_gating.rs) admits explicit
widths 1, 31, 32, 33, 63, 64, 65, 127, 129, 4,863 and 4,864, preserving
1–32 rows, at most 32 cases, the 8,192-value aggregate inventory and one-MiB
input ceiling. [Separate tail inputs](../../tools/experiments/rust_metal_gating.py)
contain nine one-row cases and 5,344 products, reconstructed from the existing
preselected normal/subnormal RNE ties, signed zeros and finite maxima.
Independent FP64 multiplication and Half rounding must match the fixed edge
bits before device work. Finite Half products are exact in FP32 and FP64;
the unchanged shader performs a FP32 product and one Half store.

Storage covers the complete rounded launch plus seven additional sentinels.
The pipeline/device limits determine launch width. Checked arithmetic rejects
zero or overflowing launch/storage sizes, including grids outside shader UInt32
indexing. Admission is checked first without padding, then recomputed for every
case after capability discovery and before native operand/output allocations.
Each padded host baseline is allocated at its final size and copied once.
Three distinct shared buffers count once; two host baselines and three
readbacks are accounted separately. The original fixture/JSON/repetition
inventory remains bounded under the 64-MiB runtime allowance.

Each command retains resources through successful completion, status and error
checks. Host access begins after completion. Both full input allocations must
remain unchanged, including inactive lanes and sentinels; every output lane
at or after count must retain its guard. A completed nonfinite Half result is
discarded with structured validation failure and exit 1. Pipelines and buffers
are reused across three warmups and ten samples per case.

The observed queried width is 32. These lengths target its boundaries, but no
capability is hardcoded in storage or dispatch. The qualification driver rejects
an aligned-only result rather than silently claiming inactive-lane coverage.

| Active values | Inactive launched lanes | Checked output guards | Admission bytes |
| ---: | ---: | ---: | ---: |
| 1 | 31 | 38 | 67,109,488 |
| 31 | 1 | 8 | 67,109,488 |
| 32 | 0 | 7 | 67,109,488 |
| 33 | 31 | 38 | 67,110,000 |
| 63 | 1 | 8 | 67,110,000 |
| 65 | 31 | 38 | 67,110,512 |
| 127 | 1 | 8 | 67,111,024 |
| 129 | 31 | 38 | 67,111,536 |
| 4,863 | 1 | 8 | 67,186,800 |

Both final isolated qualification runs use the same binary SHA-256
`6e32569d4dc87c39c0f9b2d65764f2962cfed9d70ca8e03ab261c370dc3705d8`.
Each tail job completes 117 commands, checking 69,472 exact output values and
2,483 guards. The [full primitive run](../../benchmarks/results/2026-10-12-metal-gating-tails-v3.json)
also repeats add, RMSNorm, embedding, residual, SiLU and original tiny gating:
1,157 positive commands and four required negative jobs. Its gating subset
checks 156,832 products and 3,484 guards. The
[saved Qwen run](../../benchmarks/results/2026-10-12-qwen-metal-gating-tails-v2.json)
repeats all 36 complete width-4,864 rows, tiny gating, tails and add: 767 positive
commands and three required negative jobs. Its gating subset checks 2,433,184
products and 6,760 guards. The unchanged Qwen source gate's maximum fraction is
0.490092755385; no tolerance is relaxed. Saved Qwen inputs require no checkpoint
reads. Neither native run is a full-model quality or performance measurement.

Maximum own-child physical charge is 16,974,448 bytes in the full primitive
run and 13,124,160 in the Qwen run. Maximum conservative admission is
67,186,800 bytes, below 6 GiB. The largest padded gating allocation contains 4,871
Half values: 9,742 bytes per allocation, shared total 29,226 and baseline/readback
total 48,710. These are primitive processes; total system/device memory,
energy, continuous thermal telemetry and GPU traces are unavailable.

Hardware is M1 Pro / MacBookPro18,3 / 16 GiB / macOS 26.6.2 (25G83), Rust 1.90.0,
SDK 26.5, macOS 15 deployment target, Metal language 2.3, safe math and precise
functions. Start/end observations retain AC power, configured AC low-power mode
off and no recorded thermal warning. Source hashing warms storage. No latency
or energy benefit is claimed.

The first [build attempt](../../benchmarks/results/2026-10-12-metal-gating-tails-v1.json)
failed before native work because a new test's integer type was ambiguous;
its [stderr](../../benchmarks/results/2026-10-12-metal-gating-tails-v1.json.stderr.txt)
and [copied Rust source](../../benchmarks/results/2026-10-12-metal-gating-tails-build-failed.rs.txt)
are retained. The next tiny and Qwen runs completed correctness checks, but
Astra found that cloning a short baseline and growing it could reserve more
capacity than reported. Their records remain
[superseded tiny](../../benchmarks/results/2026-10-12-metal-gating-tails-v2.json) /
[superseded Qwen](../../benchmarks/results/2026-10-12-qwen-metal-gating-tails-v1.json),
with the [copied implementation](../../benchmarks/results/2026-10-12-metal-gating-tails-superseded.rs.txt).
Final native qualification uses the corrected direct final-size allocation;
superseded runs supply no acceptance measurements.

All final native children finish their completed-work handshake and are reaped
before subsequent checks or replay. The [post-completion idle proof](../../benchmarks/results/2026-10-12-metal-gating-tails-idle-v1.json)
records actual tool completion, absence of prior numerical chains, stopped VM
and unchanged protected model/executable endpoints. The
[archive producer](../../benchmarks/results/2026-10-12-metal-gating-tails-archive-v1.py.txt)
retains exact small records/stderr/source copies. Large native JSONL files remain
local under `benchmarks/raw/`; final record hashes bind them. Original/copied
sources, inputs, drivers, manifests and binaries are checked before and after
native work and again by saved replay.

The [analysis](../../benchmarks/results/2026-10-12-metal-gating-tails-analysis-v1.json)
and exact [producer](../../benchmarks/results/2026-10-12-metal-gating-tails-analysis-v1.py.txt)
replay every native transcript, original validator, exact product, rounded
guard and memory calculation; a second replay is byte-identical. Each isolated
package passes formatting, Clippy with warnings denied, twenty Rust tests and
arm64 release build. Root formatting/Clippy and 76 ordinary Rust tests pass in
an isolated target. Seventy-one experiment Python tests and 193 benchmark
Python tests pass. Three benchmark cleanup tests initially failed under sandbox
restrictions on `ps`; the complete suite passed with owned-process inspection
allowed. No engine numerical source or frozen release executable is changed.

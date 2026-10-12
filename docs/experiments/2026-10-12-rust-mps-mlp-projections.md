# Isolated tiny MLP projections through Rust MPS, 12 October 2026

All thirty frozen tiny gate/up/down projection stages pass unchanged source
numerical gates and match the frozen Half bits in this run. Astra accepted the completed
code and evidence without findings after independent byte-identical replay. These are individual saved-input projections;
GPU propagation through a composed gated MLP remains the next task.

The [driver](../../tools/experiments/rust_metal_mlp_projections.py) exercises
both layers of all five tiny cases at the actual shapes: gate/up `[rows,32]`
by row-major `[64,32]` weights, and down `[rows,64]` by `[32,64]` weights.
Gate/up read saved `post_attention_norm`; down reads saved `gated`. Original
Half input/weight/output identities and independent FP32 source gates remain
pinned. The model has no biases for these projections. Resident positive-zero
Half vectors supply the existing alpha-1/beta-1 MPS interface; native results
qualify that representation at these shapes.

Six layer/stage batches each contain five cases and preserve the existing
sixteen-case / 32,768-value / one-MiB admission bounds. Splitting layers keeps
each completed native protocol line within its unchanged one-MiB ceiling.
No Rust, shader, Cargo, common transport or kernel behavior changes. The
[previous Rust boundary](2026-10-12-rust-mps-projection.md) provides queried
strides, checked mapped inactive storage, full native input/guard checks,
tracked resources and one bias→MPS FP32 result→Half-store command.
Matrices, kernels and buffers remain resident across three warmups and ten
samples per case. Completed nonfinite output must be discarded.

| Stage | Commands | Half outputs | Equal frozen bits | Maximum original source gate fraction | Maximum admission bytes |
| --- | ---: | ---: | ---: | ---: | ---: |
| Gate | 130 | 73,216 | 73,216 | 0.498911408659 | 67,152,466 |
| Up | 130 | 73,216 | 73,216 | 0.495685454354 | 67,152,466 |
| Down | 130 | 36,608 | 36,608 | 0.492990364008 | 67,146,130 |

The [native record](../../benchmarks/results/2026-10-12-rust-mps-mlp-projections-v1.json)
retains **390 MLP projection commands and 183,040 output values**. Every output
is the single Half RNE conversion of its recorded FP32 result. There are no
new Half-bit discrepancies in these thirty stages, and no tolerance changes.
Maximum absolute source errors are gate 0.002560675144, up 0.002335071564 and
down 0.000940904021. MPS internal accumulation precision/order remains
unqualified; exact agreement here does not establish general bit identity.

Same-binary regressions include all query/manual projections, add, original
tiny gating, inactive tails, and required failures. All **858 positive commands**
and four negative jobs pass. The previously investigated query differences
retain their original scope; the manual bias-rounding, cancellation and stride
probes pass. Final binary SHA-256 remains
`9291779eda5338a081bfe3d9f5b00727c8fe2bd6e0c218df3c916d9d700ab583`,
with identical compiled-source hashes to the earlier accepted boundary.

Largest own-child physical charge across all jobs is **13,271,592 bytes**;
maximum conservative projection admission **67,152,466 bytes** is below 6 GiB.
Independent calculations count shared allocations once, immutable host
baselines/readbacks separately and the existing 64-MiB runtime allowance.
These are bounded primitive processes, not full-model or system-memory evidence.
Native code checks complete input/padding/tail storage after completion, but
transcripts retain Booleans and active FP32/Half outputs, not complete guard
readbacks; independent replay does not claim archived guard-byte comparison.

Hardware is M1 Pro / MacBookPro18,3 / 16 GiB / macOS 26.6.2 (25G83), Rust 1.90.0
and SDK 26.5. The isolated locked offline arm64 release build targets macOS 15;
shader language 2.3 uses safe math and precise functions. Start/end observations
retain AC power, configured AC low-power mode off and no thermal/performance
warning. Continuous temperature/clocks, energy, total device/system memory and
GPU traces are unavailable. No latency, fusion or end-to-end speed claim is made.

The [idle proof](../../benchmarks/results/2026-10-12-rust-mps-mlp-projections-idle-v1.json)
and [archive producer](../../benchmarks/results/2026-10-12-rust-mps-mlp-projections-archive-v1.py.txt)
retain actual natural tool completion, absence of prior numerical chains,
stopped VM and protected model/executable endpoints unchanged. Native children
complete their handshake and are reaped before checks/replay. Numerical fixtures
need no checkpoint reads; endpoint hashes run after native completion. Large
local raw JSONL remains hash-bound in the record.

The [analysis](../../benchmarks/results/2026-10-12-rust-mps-mlp-projections-analysis-v1.json)
and [exact producer](../../benchmarks/results/2026-10-12-rust-mps-mlp-projections-analysis-v1.py.txt)
replay every raw event, validator, original source gate, exact Half conversion,
layout and admission calculation. Original/copied sources, drivers, inputs,
manifests and binary identities are verified. A second replay is byte-identical.
Isolated formatting, Clippy with warnings denied, 22 Rust tests and release
build pass. Three new targeted tests and all 78 experiment Python tests pass.
The unchanged root engine's formatting/Clippy, 76 Rust tests and 193 benchmark
Python checks remain as recorded in the preceding projection task. Paths,
links and whitespace checks pass. Composed MLP, other decoder projections,
attention/KV, full Qwen/backend quality, performance and M4 completion remain
open.

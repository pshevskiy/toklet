# Tiny attention projections through Rust MPS, 12 October 2026

All thirty saved-input tiny K/V/output projections pass their unchanged source
numerical gates. K and output match every frozen Half bit; one value-projection
position differs by one normal Half ULP in every repetition. The discrepancy is
investigated below. Astra accepted the completed task without findings after byte-identical independent replay.

The [driver](../../tools/experiments/rust_metal_attention_projections.py)
covers both layers of all five frozen cases. K/V consume `input_norm`
`[rows,32]`, row-major weights `[16,32]` and frozen Half biases `[16]`.
Their contiguous `[rows,2,8]` token/head/dimension references are viewed as
`[rows,16]` without transposition. Output consumes saved `attention_context`
`[rows,32]` with `[32,32]` weights. It has no model bias; a resident positive-zero
Half vector supplies the existing alpha-1/beta-1 MPS interface. Inputs are
independent saved references; attention, rotary and KV do not propagate here.

Six stage/layer batches each contain five cases, preserving sixteen cases,
32,768 aggregate values and one-MiB input/completion-line bounds. The native
Rust, shaders, Cargo dependencies and transport are unchanged from the
[composed MLP task](2026-10-12-rust-metal-mlp.md). They provide queried strides,
checked mapped inactive storage, tracked distinct resources, full native
input/guard checks and one GPU bias→MPS FP32 result→Half-store command per
projection. Resources are reused for three warmups and ten samples. Nonfinite
completed output must be discarded.

| Stage | Commands | Half values | Equal frozen bits | Maximum original source gate fraction | Maximum admission bytes |
| --- | ---: | ---: | ---: | ---: | ---: |
| K | 130 | 18,304 | 18,304 | 0.496994643917 | 67,124,530 |
| V | 130 | 18,304 | 18,291 | 0.489298943779 | 67,124,530 |
| Output | 130 | 36,608 | 36,608 | 0.490160061283 | 67,133,842 |

The [native record](../../benchmarks/results/2026-10-12-rust-mps-attention-projections-v1.json)
retains 390 attention-projection commands and 73,216 Half values. Each is the
single Half RNE conversion of its recorded FP32 result. Maximum source absolute
errors are K 0.002014636993, V 0.001685380936 and output 0.001064658165.
No gates were relaxed.

The sole unique discrepancy is `future_changed`, layer 0 V, flat index 56
(token 3, output 8): GPU `0x0f0d`, frozen `0x0f0e`. Both are normal Half values,
separated by `2^-22` (0.0000002384185791015625). Independent FP64 dot plus
saved bias is 0.00043046940118074417; GPU FP32 is
0.00043044984340667725, a difference of -0.00000001955777406692505.
The adjacent Half midpoint is 0.0004304647445678711, between these results.
Independent feature-order FP32 dot plus bias is 0.00043048523366451263.
All thirteen repetitions retain the same difference. This explains the observed
rounding boundary without qualifying MPS internal accumulation precision/order
or claiming general bit identity.

Same-binary checks include ten composed MLPs, thirty isolated gate/up/down
stages, query/manual projections, add, gating, inactive tails and overflows.
All **1,378 positive commands and five negative jobs** pass. Compiled-source
hashes and binary SHA-256 remain identical to the composed MLP task:
`fe88318ca76bc8e59315d18861da288b46b66595b24b860293bcc80ba1febeda`.
Native transcripts retain active FP32/Half values and guard-check Booleans;
complete guard bytes and all composed intermediate frames are not archived.

Largest own-child physical charge across jobs is **13,369,968 bytes**.
Projection admission counts shared physical allocations once, host baselines
and readbacks separately, plus a 64-MiB runtime allowance. The above projection
limits are below 6 GiB; this is bounded primitive evidence, not full-model or
system-memory qualification.

Hardware remains M1 Pro / MacBookPro18,3 / 16 GiB / macOS 26.6.2 (25G83),
Rust 1.90.0 and SDK 26.5. The locked offline isolated arm64 release build targets
macOS 15; shader language 2.3 uses safe math and precise functions. Start/end
observations retain AC power, configured AC low-power mode off and no
thermal/performance warning. Continuous temperature/clocks, energy, total
system/device memory and GPU traces are unavailable. No speed claim is made.

The [idle proof](../../benchmarks/results/2026-10-12-rust-mps-attention-projections-idle-v1.json)
and [archive producer](../../benchmarks/results/2026-10-12-rust-mps-attention-projections-archive-v1.py.txt)
retain natural tool completion, prior numerical-chain absence, stopped VM and
unchanged protected model/executable identities. Native children were reaped
before endpoint hashing, replay and offline tests.

The [analysis](../../benchmarks/results/2026-10-12-rust-mps-attention-projections-analysis-v1.json)
and [exact producer](../../benchmarks/results/2026-10-12-rust-mps-attention-projections-analysis-v1.py.txt)
replay every event, original gate, exact FP32→Half conversion, layouts,
admission, input/source/copied-source/manifests/binary hashes and regressions.
Eleven build checks pass, including standalone formatting, Clippy with warnings
denied, 25 Rust tests and release build. Three new targeted tests and all 85
experiment Python tests pass. Root engine formatting/Clippy, 76 Rust tests and
193 benchmark Python tests remain recorded in the preceding composed MLP task;
none of those sources changed here. Rotary/cache writes, causal GQA, full
layer/decoder/Qwen, backend quality, scheduling, performance and M4 completion
remain open.

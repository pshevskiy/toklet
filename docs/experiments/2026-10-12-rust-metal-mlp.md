# Resident tiny gated MLP, 12 October 2026

All ten composed tiny MLPs pass the original intermediate gates and match the
frozen Half bits in their first complete frames. Astra accepted the completed code and evidence after independent replay and
a frame-count wording correction. Unlike the [isolated projections](2026-10-12-rust-mps-mlp-projections.md),
these runs propagate GPU outputs through the graph. Full attention/layer/decoder,
KV, Qwen and M4 completion remain open.

The [Rust graph](../../tools/experiments/rust_metal_mlp.rs) encodes gate projection
→ Half store → SiLU → Half store, up projection → Half store, multiplication
of the two stored Half operands → Half store, and down projection → Half store.
All dependent operations share one command commit and checked completion wait.
There is no explicit host wait or readback between operators. Distinct owned
Half intermediates and three FP32 projection buffers preserve the numerical
boundaries, rather than fusing away rounding. Original
[projection](../../tools/experiments/rust_metal_projection.metal),
[SiLU](../../tools/experiments/rust_metal_silu.metal) and
[gating](../../tools/experiments/rust_metal_gating.metal) shader math is unchanged.

A single fixture fixes hidden size 32, intermediate size 64 and 1–32 rows.
At most 7,168 finite Half input/weight values and one-MiB JSON are accepted.
Three immutable row-major weight matrices, normalized input and a zero-bias
vector remain resident; pipelines, matrices, kernels and scratch are reused
across three warmups and ten samples. The shared matrix helper now checks
physical layout, byte width/type and owned buffer size before native
construction, providing a checked private boundary for both projection and MLP.

Matrix row bytes and every shader's launch limits are queried. The common width
respects projection bias/store, SiLU, gating and device limits. Contiguous
hidden-64 Half intermediates are admitted explicitly; an incompatible queried
stride returns a capability error. Rounded mapped inactive storage and seven
sentinels remain allocated and checked. Tracked hazards order all encoders;
owners and retained references live through completion. Host readback starts
after successful command status/error checks. Nonfinite output in any stage
discards the entire graph result with validation failure and exit 1.

FP32 GPU result storage is owned as raw aligned four-byte `u32` words, with
checked Float32 matrix descriptors. Every word bit pattern is valid and can be
read back without a float-to-word conversion allocation. Readback vectors
compact active values in place, retaining their original capacity and checking
padding/guards before overwriting any unread source. Shared physical storage
counts once; immutable input/first-output snapshots and current readbacks are
bounded at two shared-storage copies. The existing 64-MiB allowance covers the
bounded parsed fixture, JSON and repetition records.

The [driver](../../tools/experiments/rust_metal_mlp.py) reads saved
`post_attention_norm` and immutable weights for both layers of all five tiny
cases. It requires all five original stage gates, exact Half conversion of each
FP32 projection result, and independent exact-product reconstruction of gating
from the actual stored SiLU/up operands. It retains one complete active Half /
FP32 intermediate frame per case. Native checks require the twelve subsequent
frames to match that first frame bit for bit, in addition to full input/guard
checks; every final down output is independently retained per repetition.
Full padding/input readback bytes and repeated nonfinal frames are not archived.
Their evidence is the reviewed native comparison and Boolean transcript, not
independent byte replay of every stage on every repetition.

| Stage | First-frame values | Equal frozen Half bits | Maximum original source gate fraction |
| --- | ---: | ---: | ---: |
| Gate projection | 5,632 | 5,632 | 0.498911408659 |
| SiLU | 5,632 | 5,632 | 0.488969387178 |
| Up projection | 5,632 | 5,632 | 0.495685454354 |
| Gating | 5,632 | 5,632 | 0.491684814232 |
| Down projection | 2,816 | 2,816 | 0.492990364008 |

The [final record](../../benchmarks/results/2026-10-12-rust-metal-mlp-v3.json)
contains **130 composed command buffers**, **25,344 first-frame Half values**
and **36,608 independently retained final outputs** across all repetitions.
No tolerance is changed. Exact frozen agreement is observed for this graph and
these inputs; MPS internal accumulation precision/order remains unqualified.
Same-binary regressions cover the thirty isolated MLP projections, original
query/manual projections, add, gating and tails. There are **988 positive
command buffers** and five required negative jobs in total: invalid shader,
insufficient generic budget, completed MLP overflow, completed projection
overflow and completed gating overflow. The previously investigated query
projection differences remain within their original unchanged gates.

Maximum composed MLP admission is **67,269,664 bytes**, including 53,600 shared
bytes, 107,200 host snapshot/readback bytes and the 64-MiB allowance. Largest
own-child physical charge across all jobs is **13,337,152 bytes**, below
admission and 6 GiB. These are tiny primitive/graph processes, not full-model
or system-memory measurements. No latency improvement, fusion benefit,
allocation-free decode or energy claim is made.

Hardware is Apple M1 Pro / MacBookPro18,3 / 16 GiB / macOS 26.6.2 (25G83),
Rust 1.90.0 / SDK 26.5 / macOS 15 deployment target. Locked offline arm64 release
builds use Metal language 2.3, safe math and precise functions. Start/end AC
and policy observations retain low-power mode off and no thermal/performance
warning. Continuous temperature/clocks, energy and a GPU trace are unavailable.
Binary SHA-256 is
`fe88318ca76bc8e59315d18861da288b46b66595b24b860293bcc80ba1febeda`.

Two pre-GPU build attempts remain outside acceptance: [v1](../../benchmarks/results/2026-10-12-rust-metal-mlp-v1.json)
failed an include-file inner comment and redundant-field Clippy checks;
[v2](../../benchmarks/results/2026-10-12-rust-metal-mlp-v2.json)
failed a collapsible-if Clippy check. Their stderr and copied changed sources
are retained. The final v3 run starts fresh and completes naturally. The
[archive producer](../../benchmarks/results/2026-10-12-rust-metal-mlp-archive-v1.py.txt)
and [post-completion idle proof](../../benchmarks/results/2026-10-12-rust-metal-mlp-idle-v1.json)
retain actual completion, prior numerical-chain absence, stopped VM and
unchanged protected model/executable endpoints. Endpoint hashes and checks run
after native completion; fixtures need no checkpoint reads during native work.

The [analysis](../../benchmarks/results/2026-10-12-rust-metal-mlp-analysis-v1.json)
and [exact producer](../../benchmarks/results/2026-10-12-rust-metal-mlp-analysis-v1.py.txt)
independently replay raw first frames, every final output, unchanged source
gates, exact projection casts/product, memory/layout calculations and all
regressions. A second replay is byte-identical. Source/copy, input, driver,
manifest, binary and local raw JSONL hashes are verified. Large raw JSONL remains
local under `benchmarks/raw/`, hash-bound by the record.

Isolated formatting, Clippy with warnings denied, 25 Rust tests and release
build pass. Four targeted graph tests and all 82 experiment Python tests pass.
Root formatting/Clippy, 76 ordinary Rust tests and 193 benchmark Python tests
pass in the supported configurations after native completion. Links and
whitespace checks pass. The root numerical engine and frozen release executables
are unchanged. Attention projections, rotary/cache writes and causal GQA
remain before full-layer/decoder composition and real-model qualification.

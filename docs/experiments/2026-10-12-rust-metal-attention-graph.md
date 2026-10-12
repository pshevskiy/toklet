# Propagated tiny Metal attention branch, 12 October 2026

All ten actual GPU attention branches pass their nine original numerical
gates. Q/K/V projections now feed rotary, cache writes, causal GQA and the
output projection within one command and one completion wait. Astra accepted completed code and evidence without findings after byte-identical independent replay. Saved normalized inputs remain the branch boundary.

The [host](../../tools/experiments/rust_metal_attention_graph.rs) encodes
Q/K/V Half weights/biases → MPS FP32 results → Half stores, split-half Q/K
rotary → Half stores, raw K/V layout writes, stable FP32 causal GQA → Half
context, then the bias-free O projection → FP32 result → Half store. Each
consumer reads actual preceding GPU output. K/V are separate `[2,32,8]`
resident buffers, initially zero; no expanded heads or host intermediate
readback/wait occurs inside the command. Normalized input `[rows,32]`,
weights and Q/K/V biases are immutable resident Half storage. O uses a
positive-zero Half bias vector in the existing alpha-1/beta-1 MPS interface.

Existing [projection and MLP encoding helpers](../../tools/experiments/rust_metal_mlp.rs)
are exposed only within the private standalone Metal module. Numerical bodies
are unchanged; matrix initialization comments now allow resident nonzero bias.
The graph concatenates unchanged qualified projection, rotary, cache and
attention shader sources, caches four secondary pipelines and selects a common
queried execution width. Matrix descriptors, queried row strides, resource
ownership and FP32/Half conversions retain the preceding boundary's checks.
Tiny contiguous Half strides32/16 are required explicitly; incompatible
strides are a capability error.

The loader admits rows1–32 and checked rotary `position+rows<=2048`, with
bounded one-MiB JSON, exact finite UInt16 arrays and fixed tiny topology.
Numerical positions are the frozen0–31 and5–7, not the whole admitted2048
range. Cache placement is offset0 and valid length is rows, including the
absolute rotary-offset5 case. Normalization, residuals and MLP are outside this
branch; layer1 consumes its saved normalized input rather than GPU layer0.
No integrated session-valid-prefix state advances.

Allocation admission precedes numerical buffer allocation. Nine immutable
buffers and fifteen intermediate buffers have checked layouts. Q storage
covers rounded query eight-tuples; K covers rounded rotary pairs as well as
projection/cache dispatch tails. Separate FP32 score/probability rows cover
the rounded query grid. Each full cache has seven guards; output row padding/
inactive/tail storage follows the qualified helpers. All buffers are distinct,
initialized and tracked; retained command/matrix owners survive completion.
Copied shape arrays retain exact slot/size/alignment contracts. CPU buffer
access starts only after checked command completion.

The [driver](../../tools/experiments/rust_metal_attention_graph.py) covers both
layers of all five frozen cases. Seven Half stage gates and two FP32 attention
gates remain unchanged. Frozen cache arrays have no FP32-source gate; their
acceptance is exact lossless head-major copying of the actual GPU K/V outputs,
including all unwritten zero capacity. Frozen cache-bit agreement is diagnostic.
No new numerical cache tolerance is invented.

| First-frame numerical stage | Values | Equal frozen bits | Maximum original gate fraction |
| --- | ---: | ---: | ---: |
| Q projection | 2,816 | 2,814 | 0.498143005408 |
| K projection | 1,408 | 1,408 | 0.496994643917 |
| V projection | 1,408 | 1,407 | 0.489298943779 |
| Q rotary | 2,816 | 2,814 | 0.498128816330 |
| K rotary | 1,408 | 1,408 | 0.496959097521 |
| Half context | 2,816 | 2,815 | 0.494596089993 |
| O projection | 2,816 | 2,815 | 0.490160061283 |
| FP32 scores | 8,528 | 8,523 | 0.495719949277 |
| FP32 probabilities | 8,528 | 7,398 | 0.474988628550 |

The [successful native record](../../benchmarks/results/2026-10-12-rust-metal-attention-graph-v2.json)
retains **130 branch commands and36,608 final O outputs**. First complete
active frames contain25,728 Half and25,504 FP32 values, including MPS results
before Half stores. Native code requires twelve subsequent complete active
frames to equal the first; every final output is independently replayed.
Input/guard checks retain Boolean evidence, not complete archived guard bytes.
FP32 MPS row padding is permitted as previously qualified; its tails remain
guarded. Full caches retain exact actual GPU K/V placement.

Eight first-frame Half positions differ from frozen references: previously
investigated Q/V projection differences propagate unchanged through position0
Q rotary and cacheV; the known context difference persists. A new O output
boundary appears at capacity/layer0/index580 (token18,feature4): GPU `0x24f2`,
frozen `0x24f1`, normal Half spacing `2^-16` (0.0000152587890625).
Independent FP64 dot with actual GPU Half context is0.019310078638227424;
with frozen context it is0.019309797844471177. GPU FP32 result is
0.01931007392704487; adjacent Half midpoint0.01930999755859375 separates the
results. The sole context Half difference changes this dot enough to cross
that boundary. All gates remain unchanged. MPS internal accumulation/order
and general transcendental bit identity remain unqualified.

Every projection's observed FP32 result converts exactly once to its stored
Half bits. All8,528 scores match independent ordered FP32 reconstruction using
actual GPU Q/K. Each context matches independent ordered FP32 reduction using
recorded GPU probabilities/V, followed by Half RNE. All masked probability
bits are zero. Changed future tokens preserve the first three branch-output
tokens exactly in both layers. CacheV's sole frozen-bit discrepancy is exactly
its actual V input discrepancy; it introduces no layout-copy error.

Same-binary full/cached GQA, raw cache layout, rotary and all composed MLP
regressions plus failures total **1,196 positive commands and six negative
jobs**. Finite input causing a Half projection overflow is rejected after
completion and its result discarded. Fifteen build checks include formatting,
Clippy with warnings denied,33 standalone Rust tests and locked offline arm64
release build. Binary SHA-256:
`e4317c9005c8e59e122e6b1c315f6eb24168e84048e26ce9ec37613f2cec1397`.
The [excluded first build](../../benchmarks/results/2026-10-12-rust-metal-attention-graph-v1.json)
stopped before GPU execution on JSON metadata array-expression syntax. Its
stderr and changed copied Rust source are retained. Python fixture preflight
also identified cache arrays as Half-only layout oracles before native launch;
the final policy above reflects their original manifest.

Maximum branch admission is **67,310,380 bytes**: shared physical allocations
once plus conservative immutable baselines/first output snapshots/current
readbacks separately, plus64MiB bounded runtime/fixture/JSON allowance. A fixed
1,038-byte stack cache initializer is covered by that allowance. Peak own-child
physical charge across all jobs is **22,528,624 bytes**. Buffers/pipelines/matrices
are reused; diagnostic caches/guards reset outside command timing. No
steady-state decode, full-model memory or speed qualification is claimed.

Hardware remains M1 Pro / MacBookPro18,3 /16GiB / macOS26.6.2(25G83), Rust1.90.0,
SDK26.5, minimum macOS15, shader language2.3, safe math/precise functions.
Start/end observations retain AC power, configured AC low-power mode off and
no thermal/performance warning. Continuous temperature/clocks, energy,
system/device memory and GPU traces remain unavailable.

The [idle proof](../../benchmarks/results/2026-10-12-rust-metal-attention-graph-idle-v1.json)
and [archive producer](../../benchmarks/results/2026-10-12-rust-metal-attention-graph-archive-v1.py.txt)
retain natural completion, prior numerical-chain absence, stopped VM,
protected endpoints unchanged and excluded pre-GPU evidence. All children are
reaped before checks/hashing. The [analysis](../../benchmarks/results/2026-10-12-rust-metal-attention-graph-analysis-v1.json)
and [exact producer](../../benchmarks/results/2026-10-12-rust-metal-attention-graph-analysis-v1.py.txt)
replay raw events, nine original gates, actual cache copies, projection casts,
ordered scores/context stores, every finaloutput, layouts/admission,
discrepancies, causal preservation and source/copied-source/driver/input/
manifests/binary identities. Three targeted and full offline suites pass:
99 experiment Python,193 benchmark Python, root formatting/Clippy and76
ordinary Rust tests in an isolated debug target. Real-model suites were not
rerun; root release binaries remain unchanged.

Retain the bounded propagated attention branch. Adding GPU normalization,
residuals and MLP completes the next layer composition; full decoder/session,
Qwen/backend quality, scheduling and M4 completion remain open.

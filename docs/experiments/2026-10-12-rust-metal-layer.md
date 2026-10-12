# Complete tiny Metal layer, 12 October 2026

All ten complete GPU layer jobs pass their eighteen original numerical gates.
Every final layer output matches the frozen Half bits across all thirteen
repetitions. Astra accepted completed code and evidence without findings after byte-identical independent replay. Each layer still starts from its
independent saved input; the two-layer decoder is not composed here.

The [host](../../tools/experiments/rust_metal_layer.rs) composes input RMSNorm,
Q/K/V projections, Q/K split-half rotary, head-major K/V writes, causal GQA,
O projection, attention residual, post-attention RMSNorm, gated MLP and final
residual in one command with one completion wait. Every numerical consumer
reads actual preceding GPU output. The unchanged qualified shaders preserve
FP32 reductions/products/sums, precise transcendental functions, epsilon1e-6,
stored Half boundaries, head sharing and causal masks. MPS uses Half operands,
FP32 result storage, alpha1/beta1 and explicit final Half stores. Its internal
accumulation/order remains unqualified.

The two residual kernels update state in place. Two bounded GPU blits copy
incoming state to the first residual buffer, then that residual to final
state, preserving the first residual for diagnostics. The new pinned
`MTLBlitCommandEncoder` Cargo feature enables the public buffer-copy API;
dependency versions and lockfile are unchanged. Copy sizes are checked,
four-byte aligned active Half payloads at offset0, within distinct owned
source/target allocations. Guards and padding are excluded from copies.
Tracked resource dependencies order blits and compute/MPS operations without
intermediate host reads or waits.

The loader admits one bounded case per child: rows1–32 and checked rotary
`position+rows<=2048`, with exact finite UInt16 arrays and one-MiB JSON.
Fixed tiny topology is hidden32/intermediate64/query heads4/KV heads2/
head dimension8/cache capacity32. Frozen positions0–31 and5–7 are numerically
qualified; the full2048 parser bound is not. Layer0 reads saved embedding;
layer1 reads saved layer0 Half output. All weights, norms and biases load once
as resident Half buffers. The [driver](../../tools/experiments/rust_metal_layer.py)
requires frozen normalization epsilon and all original per-stage gates.

Fourteen immutable buffers and twenty-seven intermediates have checked
layouts. Queried Half strides32/16/64 must be contiguous where shaders require
it. Hidden32 storage covers both the rounded RMS row grid and the rounded
attention/projection/rotary grids. Hidden64 scratch covers rounded MLP
scalar grids. Separate score/probability rows and full head-major caches
retain guards. Admission precedes numerical buffer allocation. All initialized
owners, pipelines, matrices and seven MPS kernels remain resident and live
through successful completion; CPU buffer access starts afterward. Shape
arguments retain each qualified shader's exact slot/size/alignment contract.

| Newly composed Half stage | First-frame values | Equal frozen bits | Maximum original source gate fraction |
| --- | ---: | ---: | ---: |
| Input normalization | 2,816 | 2,816 | 0.488425146999 |
| Attention residual | 2,816 | 2,816 | 0.493156774780 |
| Post-attention normalization | 2,816 | 2,816 | 0.490197506633 |
| Gate projection | 5,632 | 5,632 | 0.498911408659 |
| SiLU | 5,632 | 5,632 | 0.488969387178 |
| Up projection | 5,632 | 5,632 | 0.495685454354 |
| Gating product | 5,632 | 5,632 | 0.491684814232 |
| Down projection | 2,816 | 2,816 | 0.492990364008 |
| Final layer residual | 2,816 | 2,816 | 0.495746068936 |

All nine preceding attention numerical gates also pass without modification.
Its eight known Half discrepancies retain the [propagated branch](2026-10-12-rust-metal-attention-graph.md)
scope: Q/V projection differences, their unchanged rotary/cache copies, one
context ULP and the explained O projection midpoint crossing. The first
residual rounds these differences away in this fixture; all post-residual
stages are frozen-bit exact. This observed absorption is not a general
bit-identity or quality guarantee.

The [successful native record](../../benchmarks/results/2026-10-12-rust-metal-layer-v2.json)
retains **130 complete-layer commands and36,608 final output values**.
First active frames contain62,336 Half and39,584 FP32 values across eighteen
Half/nine FP32 arrays per job. Native code requires all twelve subsequent
complete active frames to equal the first. Every final output is independently
replayed. Input/guard checks retain Boolean evidence; complete guard bytes,
FP32 MPS row padding and later nonfinal frames are not archived. MPS row
padding remains permitted within logical rows; FP32 tails stay guarded.

Independent reconstruction verifies all observed projection FP32→Half
stores, lossless actual GPU K/V copies, ordered scores/context reductions,
stored-Half gating products and both FP32-add→Half residual boundaries.
All8,528 scores match ordered reconstruction using actual GPU Q/K. Cache
arrays have no invented FP32 numerical gate: they must copy actual K/V bits
exactly, including zero unwritten capacity. Future-token changes preserve the
first three final-output tokens exactly in each layer.

Same-binary full/cached GQA, cache-layout, rotary and composed MLP regressions
plus required failures total **1,196 positive commands and six negative jobs**.
Finite incoming Half values with extreme normalization weight produce a
completed nonfinite layer, which is rejected and discarded. Sixteen build
checks include formatting, Clippy with warnings denied,35 standalone Rust
tests and locked offline arm64 release build. Binary SHA-256 is
`b1935f5eafb4f829e7da87f206ffb33f6a8c52f4daba4d40568d94fef6fb587d`.
The [excluded first attempt](../../benchmarks/results/2026-10-12-rust-metal-layer-v1.json)
stopped before GPU execution on ambiguous `AsRef` inference in the buffer
identity check; explicit dereferencing corrected it. The stderr and copied
failed source are retained. Fixture preflight was corrected before launch;
no failed preflight numerical results were accepted.

Maximum layer admission is **67,489,972 bytes**: shared physical storage once,
conservative full host baselines/first snapshots/current readbacks separately,
plus64MiB for bounded fixtures/runtime/diagnostic records. A fixed1,038-byte
stack cache initializer is covered by that allowance. Largest own-child
physical charge across all jobs is **22,479,472 bytes**. Cache/guard resets
occur outside command timing. This is resident diagnostic layer qualification,
not allocation-free inference, integrated session memory or a speed claim.

Hardware remains M1 Pro / MacBookPro18,3 /16GiB / macOS26.6.2(25G83), Rust1.90.0,
SDK26.5, minimum macOS15, shader language2.3, safe math/precise functions.
Start/end observations retain AC power, configured AC low-power mode off and
no thermal/performance warning. Continuous temperature/clocks, energy,
system/device memory and GPU traces remain unavailable.

The [idle proof](../../benchmarks/results/2026-10-12-rust-metal-layer-idle-v1.json)
and [archive producer](../../benchmarks/results/2026-10-12-rust-metal-layer-archive-v1.py.txt)
retain natural completion, prior numerical-chain absence, stopped VM,
protected endpoints unchanged and excluded pre-GPU evidence. All children
are reaped before checks/hashing. The [analysis](../../benchmarks/results/2026-10-12-rust-metal-layer-analysis-v1.json)
and [exact producer](../../benchmarks/results/2026-10-12-rust-metal-layer-analysis-v1.py.txt)
replay every validator, original gate, arithmetic/layout boundary, finaloutput,
causal preservation, admission and source/copied-source/driver/input/
manifests/binary identities. Three targeted and full offline suites pass:
102 experiment Python tests,193 benchmark Python tests, root formatting/
Clippy and76 ordinary Rust tests in an isolated debug target. Real-model
checks were not rerun; root release binaries remain unchanged.

Retain the bounded complete layer. Two-layer GPU propagation, embedding,
final normalization/tied logits, decoder/session behavior, Qwen/backend quality,
scheduling and M4 completion remain open.

# LFM3 canonical BF16 CPU projection

Status: Astra approves scoped CPU projection retention, with no findings.
This adds a bias-free CPU linear leaf for LFM3's BF16 activation/weight/output
boundaries. Full attention, graph, model/session/CLI, quality, memory fit and
adoption remain open.

Input is contiguous `[tokens, in]`, immutable weights `[out, in]` and output
`[tokens, out]`, all BF16. Operands widen exactly into source-order FP32 fused
multiply/add partials; reduction precedes one nearest-even BF16 store. The pinned
Torch CPU no-BFDOT profile uses 32 independent partials, a fixed register/lane
reduction tree, four tail partials for remaining eight-element blocks and fused
scalar tails. These constants specify source numerical order, rather than CPU
tuning or a generally assumed SIMD width. The named M1 Pro reports BF16
instruction capability zero; other source runtime/hardware profiles are separate.
[Torch linear dispatch](https://github.com/pytorch/pytorch/blob/5811a8d7da873dd699ff6687092c225caffcf1bb/aten/src/ATen/native/Linear.cpp),
[BF16 GEMM](https://github.com/pytorch/pytorch/blob/5811a8d7da873dd699ff6687092c225caffcf1bb/aten/src/ATen/native/cpu/BlasKernel.cpp),
[dot/reduction profile](https://github.com/pytorch/pytorch/blob/5811a8d7da873dd699ff6687092c225caffcf1bb/aten/src/ATen/native/cpu/ReducedPrecisionFloatGemvFastPathKernel.cpp).

All nonzero dimensions, checked extents, slice lengths and finite operands are
validated before output writes. FP32/BF16 arithmetic overflow rejects; numerical
failure may leave partial output, which the caller must discard. Inputs and
weights remain immutable, and no history or KV state is published. The leaf uses
ordinary safe functions/slices with static arithmetic, caller-owned storage,
and no expanded FP32 weight tensor. Existing FP32 projection/reference bodies
are unchanged. The existing Gemma BF16 adapter's FP32 input/output boundary
does not establish this BF16 activation profile.

The [prospective contract](../artifacts/lfm3-projection-bf16-2026-11-11/prospective-contract-1.json)
precedes new source fixtures and Rust implementation outcomes. The producer
authenticates the complete 3,457,635,448-byte child before and after reading four
physical rows from every one of its 154 rank-two tensors, including tied
embedding/head storage. The selected rows are 0, 1, midpoint and last; input
widths remain 2,048 or 7,680. Retained selected coefficients total 3,604,480 B.
This samples learned rows across the complete inventory; it does not load or
qualify complete matrices or a model graph.

Actual pinned Torch BF16 linear calls produce the independent source outputs;
NumPy FP64 dots supply a separate mathematical reference. No candidate reduction
appears in the producer. All 94,248 learned outputs from 616 cases match the
canonical source exactly, as do all 65,280 finite identity controls. Of 742 corner
outputs, 739 finite words match and three source nonfinite markers reject.
Tail lengths, cancellation, BF16 midpoint neighbors, signed zeros, subnormals
and fused-overflow rescue are explicit. All 154 learned row banks preserve
the 128-token source result when partitioned into 1/7/17/103-token calls.

The candidate's unrounded FP32 dot retains the original independent bound
`abs(actual-reference) <= 1e-5 + 1e-4*abs(reference)`, with zero failures and
maximum absolute error 8.188740431e-6. This is before the BF16 store; it does not
claim access to Torch's unrounded internal result. Every nonfinite BF16 input
pattern is rejected before writes, in both input and coefficient positions.
Shape/overflow, alignment, partial-output discard and successful retry checks
also pass.

Installed MLX `nn.Linear` CPU and Metal were compared on identical selected
rows/operands, without importing remote model code or Torch. Raw-output repeats
verify 156,498,420 typed operand bytes per device and reproduce every initial
output hash. CPU differs at
24,823 of 160,270 words: 24,820 learned words and three corners, with no finite-
grid differences. Metal differs at 280 words: 23 learned, three corners and
254 finite-grid words. Every operation is available. Both fail the frozen exact
component correspondence gate. These are four-row/operator comparisons; they
do not score full-model quality, identify the behavior of every full-matrix
dispatch, or reverse previous correspondence/adoption rejections.
[CPU comparison](../artifacts/lfm3-projection-bf16-2026-11-11/child-cpu-report-2.json),
[Metal comparison](../artifacts/lfm3-projection-bf16-2026-11-11/child-metal-report-2.json).

Source/runtime hashes remain fixed, including 28,982 reference runtime files and
the actual Torch git revision. Source and all four MLX collections close within the
separate 512 MiB audit bound; maximum conservative final controller/child RSS
peak sum is 291,799,040 B. Each final PID peak covers its own sampled RSS. Final
physical peak is unavailable. Initial inspection misses and a sandboxed CPU
capability-query failure are retained; the identical escalated query succeeds.

Fresh formatting, both strict all-target Clippy configurations, 723 CPU tests
(50 ignored), 869 Metal-enabled tests (338 ignored), nine depth-example tests
and eight optimized source tests pass. Debug/release prepared checks each run
24 projections and 96 rejected calls with zero Rust allocations. Four portable
methods pass in ordinary and optimized Python modes; six actual source and ten
compiled console admission controls pass, including a disposable altered-bank
clone. Native77 builds three fresh arm64 slices with 46 exports each, strict
C links and CPU/Metal tiny-model execution smokes. These do not qualify LFM3
GPU execution or iPhone runtime. The application retains its native46 link.

The [frozen release campaign](../artifacts/lfm3-projection-bf16-2026-11-11/performance-contract-1.json)
has 42 cohorts: seven profiles, 1/32/128-token calls and two reversed-order runs,
each with a prepared first call, three warmups and ten raw samples. Six profiles
execute complete representative matrices; the tied-head profile explicitly
executes only rows 0–1,023 of 128,000. First/last calls verify the applicable
canonical sampled coordinates. Inputs follow the synthetic source-fixture
generator, not captured model activations. Bank read/hash/decode and caller
preparation are separately named; their reads may warm storage.

| Profile | Output × input width | 1 token | 32 tokens | 128 tokens |
| --- | --- | --- | --- | --- |
| conv-in | 6144 × 2048 | 9.150–9.220 | 165.027–165.567 | 643.531–649.766 |
| conv-out | 2048 × 2048 | 2.933–2.984 | 54.283–55.463 | 213.759–217.538 |
| mlp-up | 7680 × 2048 | 11.152–11.513 | 205.475–205.532 | 808.084–809.990 |
| mlp-down | 2048 × 7680 | 11.311–11.399 | 204.082–204.406 | 802.274–807.044 |
| attention-gate | 16 × 2048 | 0.024–0.024 | 0.438–0.464 | 1.729–1.770 |
| attention-kv | 512 × 2048 | 0.734–0.764 | 13.488–13.909 | 53.594–54.135 |
| tied-head-tile | 1024 × 2048 | 1.481–1.532 | 27.133–27.310 | 107.306–107.928 |

Times are milliseconds for one prepared public projection including validation
and stores. All AC/Low Power Mode off/thermal-zero/no-compiler endpoint checks
pass on the named M1 Pro with ordinary release Rust 1.90/LLVM 21.1.1. Raw ranges
and median absolute deviations are retained; ten samples do not qualify tails.
Large matrix costs make projections a useful target for later end-to-end
profiling. Independent component medians are not summed into model latency.

Each process owns one BF16 matrix, caller input/output, numerical vector headers
and the separately reported raw loading buffer/embedded fixtures. At most
31,457,280 B of coefficients are resident per complete profile; the seven
retained disk banks total 102,825,984 B. Maximum conservative performance RSS
peak sum is 153,747,456 B; maximum sampled aggregate physical footprint is
64,570,472 B. Final physical peak remains unavailable. No dtype speedup,
whole-model throughput/TTFT, full-head, sustained, quality or memory-fit claim
follows from these operator baselines.

[Measurement index](../benchmarks/results/2026-11-11-lfm3-projection-bf16-v1.json),
[required checks](../artifacts/lfm3-projection-bf16-2026-11-11/required-checks-1/checks.json),
[optimized source checks](../artifacts/lfm3-projection-bf16-2026-11-11/release-source-checks-1.json),
[allocation checks](../artifacts/lfm3-projection-bf16-2026-11-11/allocation-checks-1.json),
[native package](../artifacts/lfm3-projection-bf16-2026-11-11/native77-publication.json),
[source/console controls](../artifacts/lfm3-projection-bf16-2026-11-11/controls-checks-1.json).

Next qualify source BF16 causal/window attention, then compose the graph, loading,
codec, session/CLI and original full-model gates. The 10,000-target/100-task quality
comparisons, output caps, numerical/session rules and 6 GiB ceiling remain fixed;
codec 512 MiB, app 32 MiB, reference audit 512 MiB and disk floor 2 GiB also hold.
The 1% summary overhead bound stays unqualified/opt-in. Screen capture and iPhone
runtime remain deferred; the whole improvement goal remains active.

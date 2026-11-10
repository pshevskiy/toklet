# LFM3 canonical BF16 norm and unary CPU profile

Status: Astra approves scoped CPU component retention, with no findings.
Full LFM3 graph, codec/CLI, model fit, quality and adoption remain open.

## Source contract and arithmetic

The [prospective contract](../artifacts/lfm3-bf16-2026-11-10/prospective-contract-1.json)
predates fixtures and implementation outcomes. The target is the pinned modified
parent's BF16 arithmetic over the authenticated equal child/parent weights;
child-runtime correspondence is audited separately. No remote model code is
executed by the reference producers.

RMS normalization accepts contiguous BF16 [rows,width], width 128 or 2,048,
shared BF16 [width] gains and fixed epsilon 1e-5. Inputs widen to FP32; squared
values, sum/mean, epsilon addition, square root/reciprocal and normalization use
FP32. Normalized values round to BF16 BEFORE multiplication by the gain, followed
by a second BF16 store. No +1 gain is added. The final model.embedding_norm
belongs after all layers; this primitive does not choose its graph position.

The fixed numerical grouping follows the pinned Torch 2.9.1 M1 CPU mean/sum:
16 logical partials, at most 256 input values per group and two nonzero cascade
levels for these widths. Four groups then combine in source order. This
source-specific ordering is independent of the Rust target's hardware SIMD width;
there are no custom intrinsics or unsafe numerical code.
[Torch mean](https://github.com/pytorch/pytorch/blob/5811a8d7da873dd699ff6687092c225caffcf1bb/aten/src/ATen/native/ReduceOps.cpp),
[cascade sum](https://github.com/pytorch/pytorch/blob/5811a8d7da873dd699ff6687092c225caffcf1bb/aten/src/ATen/native/cpu/SumKernel.cpp),
[installed header/inspection](../artifacts/lfm3-bf16-2026-11-10/reduction-source-inspection-1.json).

Sigmoid and SiLU widen BF16 to FP32 and store BF16 once. They retain the source
FP32 exponential/division and x*sigmoid(x) order, including saturation. Exponential
+infinity may produce a valid final zero sigmoid. SiLU's subsequent MLP up-product
is a separate BF16 boundary; binary products/residuals are not qualified here.

Geometry, lengths and all finite input/gain checks precede writes. Empty calls
reject. Nonfinite FP32 norm reductions and BF16 output overflow reject; partial
output from numerical errors is discardable and no state is published. Extreme
finite inputs whose source reduction overflows are outside this qualified norm
domain. The unchanged FP64 mathematical normalization and stable FP32 activation
references remain available.

## Independent numerical evidence

The producer authenticates the complete 3,457,635,448-byte child before and after
extracting all 119 learned norm vectors (402,944 B). These comprise 97 hidden/final
width 2,048 vectors and 22 shared head-width 128 vectors. The canonical modified
parent and child gain bytes were already authenticated equal by the
[source audit](lfm3-source-qualification.md).

The 238 cases cover lengths 1/17 and the source Q16/K4 head geometry. Inputs cover
bounded BF16 magnitudes [-16,16], signed zeros/subnormals and typical-magnitude
lanes. Every one of 4,082,688 final BF16 words and 5,706 mean/inverse rows matches
exactly, including split row batches. Sixteen additional constant/impulse controls
cover epsilon-dominated values, subnormals, gain signs and signed zeros. Both
sigmoid and SiLU match the canonical source across all 65,280 finite BF16 inputs.
Nonfinite unary patterns, geometry/gain/input errors and late numerical overflow
exercise output preservation/discard/retry behavior.

The unrounded FP32 mathematical diagnostic retains 1e-5 + 1e-4 * abs(reference),
with zero failures and maximum absolute error 8.352e-7 against independent FP64
normalization. Both middle/output BF16 cast boundaries occur after this diagnostic.

Pinned Torch 2.9.1 / NumPy 2.3.2 and MLX / MLX-Metal 0.32.3 cover 20,691 runtime files.
Closed reference/correspondence campaigns remain under the separate 512 MiB guard;
maximum conservative final controller+child RSS-peak sum is 260,997,120 B, with each
PID's final peak covering its own sampled RSS. Final physical peak is unavailable.
The 2 GiB disk floor and 6 GiB engine / 512 MiB codec / 32 MiB app limits stay unchanged.

[Canonical fixture campaign](../artifacts/lfm3-bf16-2026-11-10/fixture-checks-1.json),
[epsilon/zero controls](../artifacts/lfm3-bf16-2026-11-10/corners-checks-1.json),
[portable/admission controls](../artifacts/lfm3-bf16-2026-11-10/reference-negative-checks-1.json).

## Child builtins are a distinct numerical profile

Actual installed MLX component probes use the same checked inputs/gains with
completed CPU/Metal evaluation. CPU differs from the canonical parent at 33 norm
words and one word each for sigmoid/SiLU. Metal differs at 90,323 norm words,
three sigmoid words and 511 SiLU words. These fail the frozen exact component
correspondence gate; they do not score whole-model quality.

Using canonical normalized FP32 values isolates cast placement: the CPU two-cast
calculation matches every word; its single-cast control differs at 1,035,846.
The Metal two-cast control still differs at 90,315 words, including zeros where
the canonical source returns subnormal values. The actual child results are not
interpreted as evidence that its norm omits the middle cast.
[CPU correspondence](../artifacts/lfm3-bf16-2026-11-10/child-cpu-report-1.json),
[Metal correspondence](../artifacts/lfm3-bf16-2026-11-10/child-metal-report-1.json).

## Checks and release operator baselines

Fresh fmt, both strict all-target Clippy configurations, 699 CPU tests (50 ignored),
845 Metal-enabled tests (338 ignored), nine depth-example tests and five optimized
source tests pass. The depth-example tests also occur in the Metal total.
Both norm widths perform 16 norm/16 sigmoid/16 SiLU calls and 24 rejections after
warmup with zero Rust allocations. Caller slices/gains stay separately owned;
only 32 FP32 accumulators (128 logical bytes) plus scalars are needed by the norm.
Four portable tests pass ordinarily and with Python optimization; six actual
source admission and ten actual console argument controls reject.

Native73 compiles three fresh arm64 slices, each with 46 C exports and strict C
links. Fresh CPU+Metal C smokes pass on the existing tiny model; these do not
qualify LFM3 Metal execution or iPhone runtime. The app still links native46.

Six frozen release cohorts on Apple M1 Pro/MacBookPro18,3 / 16 GiB, macOS 26.6.2
(25G83), Rust 1.90.0 Homebrew / LLVM 21.1.1 pass. Both runs use a first operator pass,
then three warmups and ten samples; second-run token order is reversed. Power
endpoints report AC, Low Power Mode off, thermal state 0 and no active compilers.
Norm vectors follow fixture order, followed by grouped activation calls; this
is not a complete graph execution order.

| Tokens | Warm component median range | First component pass range |
| --- | --- | --- |
| 1 | 1.285–1.315 ms | 1.238–1.320 ms |
| 32 | 39.048–39.207 ms | 38.580–51.673 ms |
| 128 | 152.448–154.071 ms | 152.508–156.404 ms |

One component pass contains 119 norm calls, 24 SiLU calls of width 7,680 and 11
sigmoid calls of width 16. At 128 tokens, warm norm medians are 97.966–98.360 ms,
SiLU 54.815–55.711 ms and sigmoid 0.0513–0.0525 ms. Raw samples, medians, ranges and
median absolute deviations are retained, including the slower first 32-token
pass. These are operator baselines, without a speedup, model throughput, TTFT,
tail or telemetry-overhead qualification. First-call time excludes process and
fixture/buffer preparation; warm time includes validation and BF16 stores.

Owned gains occupy 402,944 B, descriptors 4,096 B and caller input/output storage
38,976 / 1,247,232 / 4,988,928 B. The probe separately embeds 1,284,559 B of fixture
bytes; those are disclosed rather than treated as live model weights. Maximum
conservative final RSS-peak sum is 38,060,032 B and maximum sampled aggregate
physical footprint 23,642,928 B; final physical peak is unavailable. This does not
prove whole-model memory fit.

[Required checks](../artifacts/lfm3-bf16-2026-11-10/required-checks-1/checks.json),
[optimized source checks](../artifacts/lfm3-bf16-2026-11-10/release-source-checks-1.json),
[allocation counter](../artifacts/lfm3-bf16-2026-11-10/allocation-targeted-1.log),
[native build](../artifacts/lfm3-bf16-2026-11-10/native73-publication.json),
[native smokes](../artifacts/lfm3-bf16-2026-11-10/native-smoke-73/checks.json),
[timing contract](../artifacts/lfm3-bf16-2026-11-10/performance-contract-1.json),
[raw samples/check index](../artifacts/lfm3-bf16-2026-11-10/performance-1/checks.json),
[review/commit](../artifacts/lfm3-bf16-2026-11-10/publication-checks.json).

Next qualify BF16 binary/convolution boundaries and 256-token windows, then
compose the LFM3 graph, weight/codec loading and CLI. Complete original
10,000-target/100-task stock-to-modified and modified-to-native comparisons,
output caps, numerical/session gates and 6 GiB model fit remain mandatory.
Screen capture and iPhone runtime remain deferred.

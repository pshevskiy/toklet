# LFM3 canonical BF16 binary and convolution CPU boundaries

Status: Astra approves scoped CPU component retention, with no findings.
Full LFM3 execution, state
composition, codec/CLI, model fit, quality and adoption remain open.

## Arithmetic and ownership

The [prospective contract](../artifacts/lfm3-conv-bf16-2026-11-11/prospective-contract-1.json)
predates new fixtures and implementation outcomes. It retains the pinned modified
parent BF16 CPU target over authenticated equal child/parent weights; installed
child correspondence is independently audited. No remote model code runs.

Elementwise addition and multiplication widen BF16 operands to FP32 and round
once to nearest-even BF16. These explicit boundaries separate B*x, convolution,
C*conv, SiLU/up products and normalized residual addition. Source attention gates
broadcast [tokens,16,1] directly over [tokens,16,128], without expanded storage.
The existing [norm/unary profile](lfm3-bf16-source.md) stays unchanged.

The convolution reads contiguous BF16 input/output [tokens,channels], model-owned
BF16 weights [channels,7] and immutable BF16 history [6,channels], oldest-to-newest.
The child [2048,7,1] and parent [2048,1,7] coefficient representations have equal
physical order under the prior [complete source audit](lfm3-source-qualification.md).
The new leaf does not mutate history, publish a candidate or implement a session
transaction. Existing FP64/FP32 convolution/history references remain unchanged.

The canonical parent CPU fallback uses BF16 functional.conv1d with seven taps,
groups=2048, padding=6, no bias and the source crop. The pinned nontransposed
BF16 BLAS path uses four FP32 partials: partial zero receives taps 0,4,5,6; the
others receive 1,2,3. They then combine in order 0+1+2+3 before the BF16 store.
This numerical grouping is independent of hardware SIMD width. No activation,
gate, bias, projection or state commit is fused.
[Torch convolution](https://github.com/pytorch/pytorch/blob/5811a8d7da873dd699ff6687092c225caffcf1bb/aten/src/ATen/native/ConvolutionMM2d.cpp),
[BF16 BLAS accumulation](https://github.com/pytorch/pytorch/blob/5811a8d7da873dd699ff6687092c225caffcf1bb/aten/src/ATen/native/cpu/BlasKernel.cpp).

All geometry, checked size arithmetic, lengths and nonfinite input checks precede
writes. Empty calls reject. Late arithmetic overflow rejects with discardable
partial output; all model/history inputs remain immutable. There is no numerical
allocation, dynamic dispatch, unsafe numerical code or target-specific intrinsic.
The source-qualified convolution shape is 2048 channels; negative controls also
exercise small channel counts, without claiming broad model qualification.

## Independent evidence

The producer authenticates the complete 3,457,635,448-byte child before and after
extracting all 13 coefficient banks (372,736 B). All 104 cases use lengths
1/7/17/128 with zero and nonzero histories and deterministic bounded BF16 inputs
in [-16,16], including signed zeros/subnormals. Every one of 8,146,944 source
output words matches exactly. Split calls use the exact six preceding raw BF16
rows and match the same source output.

The separate FP32 candidate-reduction diagnostic against independent FP64
mathematical convolution retains 1e-5 + 1e-4*abs(reference), with zero failures
and maximum absolute error 5.216e-7. This is before the BF16 store; it does not
claim access to an unrounded internal Torch convolution output. Exact canonical
BF16 equality is an additional gate.

Each binary operation crosses all 65,280 finite BF16 left operands with 17 frozen
anchors. Across 34 grids, 2,219,520 output words match the source, with source
arithmetic infinities represented by explicit rejection markers. All 256
nonfinite input patterns reject before writes. Three head-broadcast cases cover
51,200 output words exactly. Shape, late overflow, immutable history and valid
retry controls pass.

Torch 2.9.1 / NumPy 2.3.2 and MLX / MLX-Metal 0.32.3 cover 20,691 pinned runtime
files. The qualified source collection and both correspondence campaigns close
under the separate 512 MiB aggregate audit limit. Maximum conservative final
controller+child RSS-peak sum is 356,106,240 B; each final PID peak covers its own
sampled RSS. Final physical peak is unavailable. The first source collection
completed the same fixture but failed the sandboxed final process-closure query;
it has no bounded-audit qualification. The identical repeat closes successfully.
[Source collection](../artifacts/lfm3-conv-bf16-2026-11-11/fixture-checks-2.json),
[retained initial failure](../artifacts/lfm3-conv-bf16-2026-11-11/fixture-collection-failure-1.json),
[portable/admission controls](../artifacts/lfm3-conv-bf16-2026-11-11/reference-negative-checks-1.json).

Actual installed MLX CPU differs at 24 convolution words, with zero binary/head
gate differences. Metal differs at 2,842 convolution, 79,316 binary and 4,754
head-gate words. No convolution operation is unavailable in either probe. These
fail the frozen exact child-component correspondence gate; they do not score
whole-model quality or justify changing the canonical gate.
[CPU correspondence](../artifacts/lfm3-conv-bf16-2026-11-11/child-cpu-report-1.json),
[Metal correspondence](../artifacts/lfm3-conv-bf16-2026-11-11/child-metal-report-1.json).

## Checks and prepared operator baselines

Fresh fmt, both strict all-target Clippy configurations, 705 CPU tests (50 ignored),
851 Metal-enabled tests (338 ignored), nine depth-example tests and five optimized
source tests pass. The example tests also occur in the Metal total. Four portable
tests in ordinary/optimized modes, six source-admission controls and ten compiled
console argument controls pass. Warmed 1/17/128-token calls at 2048 channels
perform 24 convolutions/24 adds/24 products/24 head gates and 96 rejections with
zero Rust allocations. The original native74 attempt
failed before compilation on a wrongly adapted old-cache path; native75 uses
the correct closed native73 cache and freshly compiles all three slices, each
with 46 exports and strict C links. Fresh tiny CPU/Metal C smokes pass. These do
not qualify LFM3 GPU execution or iPhone runtime; the app still links native46.

The release probe uses all 13 authenticated banks and synthetic inputs/history.
One component pass groups 13 convolutions, 26 conv products of width 2048,
24 MLP products of width 7680, 48 residual adds of width 2048 and 11 head-gate
products. This is not graph execution order. First component time excludes
process/fixture/buffer preparation; warm phases include validation and stores.
No whole-model throughput, TTFT, optimization gain, sustained/tail latency or
1% summary-overhead qualification follows.

All six frozen cohorts pass on Apple M1 Pro/MacBookPro18,3, 16 GiB, macOS 26.6.2
(25G83), Rust 1.90.0 Homebrew / LLVM 21.1.1. Each fresh process records a prepared
first pass, three warmups and ten samples; the second run reverses token order.
AC, Low Power Mode off, thermal state 0 and no active compilers are observed at
all endpoints. Raw samples, ranges and median absolute deviations are retained.

| Tokens | Warm component median range | First component pass range |
| --- | --- | --- |
| 1 | 1.150–1.168 ms | 1.1328–1.1333 ms |
| 32 | 33.547–33.923 ms | 33.701–34.024 ms |
| 128 | 135.099–135.660 ms | 134.942–135.666 ms |

At 128 tokens, warm convolution medians are 27.326–27.355 ms, conv products
16.295–16.375 ms, MLP products 56.666–56.963 ms, residuals 29.647–29.672 ms and
head gates 5.101–5.145 ms. These characterize this subset; independent medians
are not summed into model timing.

Owned coefficient storage is 372,736 B; no descriptors are allocated. Caller
input/output/history/gates total 74,784 / 1,631,232 / 6,451,200 B for 1/32/128
tokens. The probe separately embeds 467,468 B of fixture bytes. Four FP32
convolution partials use 16 logical bytes plus scalars/ABI spills; this is not a
measured stack peak. Maximum conservative final RSS-peak sum is 40,042,496 B and
maximum sampled aggregate physical footprint is 26,379,272 B. Final physical
peak is unavailable, and no whole-model memory-fit claim follows.

[Required checks](../artifacts/lfm3-conv-bf16-2026-11-11/required-checks-1/checks.json),
[optimized source checks](../artifacts/lfm3-conv-bf16-2026-11-11/release-source-checks-1.json),
[allocation checks](../artifacts/lfm3-conv-bf16-2026-11-11/allocation-checks-1.json),
[native build](../artifacts/lfm3-conv-bf16-2026-11-11/native75-publication.json),
[native smokes](../artifacts/lfm3-conv-bf16-2026-11-11/native-smoke-75/checks.json),
[timing contract](../artifacts/lfm3-conv-bf16-2026-11-11/performance-contract-1.json),
[raw samples/check index](../artifacts/lfm3-conv-bf16-2026-11-11/performance-1/checks.json),
[review/commit](../artifacts/lfm3-conv-bf16-2026-11-11/publication-checks.json).

Next qualify transactional BF16 history and 256-token window state, then compose
the graph, weight/codec loading and CLI. Original 10,000-target/100-task quality
comparisons, output caps, numerical/session gates and 6 GiB model fit remain
mandatory. Codec 512 MiB, app 32 MiB and disk floor 2 GiB stay fixed. Screen
capture and iPhone runtime remain deferred.

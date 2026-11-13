# Rejected LFM3 two-token projection experiment

Status: **closed; rejected for code retention after one bounded correction**.
Astra found no numerical or ownership defect, but the original single-token
compatibility gate remains unresolved. All five candidate code files were
restored byte-for-byte to `cf80a168`; the checked experiment survives in its
frozen patch, binaries and raw evidence. No model/request/adoption gain is
qualified. The selected roadmap and original gates remain active.
[Final Astra review](../artifacts/lfm3-tokens2-2026-11-13/tokens2-astra-review-2.json),
[exact code revert](../artifacts/lfm3-tokens2-2026-11-13/tokens2-code-revert-2.json).

## Mechanism and bounded dispatch correction

Two adjacent input rows share each exact BF16 coefficient widening. Inputs stay
contiguous `[tokens,in]`, weights `[out,in]` and output `[tokens,out]`; no packed,
transposed or full FP32 bank is introduced. Each output preserves its own ordered
32 FP32 FMA partials, reduction tree, eight-element/scalar tails and nearest-even
BF16 store. Shape/finite admission precedes writes. Arithmetic failure preserves
the first logical output error and successful prefix; later output is discardable.
No new allocation, thread pool, unsafe access, indirect numerical call or dtype
policy is introduced. Odd tokens use the original scalar reduction.

Astra held v1 retention and requested one bounded dispatch correction. The experimental public
function selected count 1 directly into scalar; other counts called an ordinary
private paired helper with a narrowly justified `inline(never)` boundary. The
original scalar, `dot`, `dot2` and paired validation/arithmetic bytes are unchanged.
[Correction and byte proof](../artifacts/lfm3-tokens2-2026-11-13/tokens2-dispatch-correction-2.json),
[corrected immutable binary](../artifacts/lfm3-tokens2-2026-11-13/tokens2-probe-ready-2.json).

## Corrected v3 operator observations

The M1 Pro release protocol retains three warmups and ten measured public calls
per fresh process, matched banks/inputs/precision and public guards/stores.
The three large 128-token projections reduce balanced geometric-mean latency by
**48.9924% / 49.5323%** in the screen and reverse confirmation. All six complete
profiles improve at 32 tokens (47.43–49.30%) and 128 tokens (45.52–50.63%).
[Screen](../artifacts/lfm3-tokens2-2026-11-13/tokens2-operator-screen-3/result.json),
[reverse confirmation](../artifacts/lfm3-tokens2-2026-11-13/tokens2-operator-confirm-3/result.json).

The complete fixed matrix covers six full profiles at 1/32/128 tokens and a
partial 1,024-row tied-head tile at one token. Its four sets have identical whole
outputs for every configuration: 76 fresh processes / 760 measured samples,
including 12 primary processes reused within this corrected matrix. Zero
historical timing samples are reused. The tile checks only source rows 0/1;
it does not qualify the full 128,000-row head.
[Corrected full matrix](../artifacts/lfm3-tokens2-2026-11-13/tokens2-operator-compat-3/result.json).

One-token results remain mixed. Values below are latency reductions relative to
scalar; negative values indicate candidate slowdown.

| One-token profile | Source1 / candidate1 | Source2 / candidate2 |
| --- | ---: | ---: |
| Conv in | +0.741% | +1.720% |
| Conv out | -0.745% | -0.842% |
| MLP up | -1.373% | -0.701% |
| MLP down | -2.135% | +0.790% |
| Attention gate | -5.023% | +6.225% |
| Attention KV | +0.164% | -1.295% |
| Partial head tile | +7.019% | -7.200% |

Collection/identity passes are not a blanket compatibility or retention pass.
No post-hoc noise threshold excuses these regressions. Astra rejected code
retention because the original compatibility gate remains unresolved. Close
this bounded candidate without model integration or a restricted workload waiver.

The earlier corrected screen2 is excluded conservatively: root launched it
before explicit preparation pause, and exact local-tool overlap was not disproved.
Its passing collection output is preserved; none of its samples enters v3.
[Raw screen2](../artifacts/lfm3-tokens2-2026-11-13/tokens2-operator-screen-2/result.json),
[policy exclusion](../artifacts/lfm3-tokens2-2026-11-13/tokens2-screen-exclusion-2.json).

## Historical v1 observations remain retained

Before the dispatch correction, large-shape balanced reductions were
45.05% / 45.44%, with source/candidate variation 1.11% / 0.37%. Six complete
profiles improved 39.64–45.31% at 32 tokens and 43.83–46.30% at 128 tokens.
Separate-process one-token attention gate slowed 1.45% / 5.66% and attention KV
2.16% / 4.01%. These valid negative measurements are not erased by the correction.
[v1 screen](../artifacts/lfm3-tokens2-2026-11-13/tokens2-operator-screen-1/result.json),
[v1 reverse](../artifacts/lfm3-tokens2-2026-11-13/tokens2-operator-confirm-1/result.json),
[v1 matrix](../artifacts/lfm3-tokens2-2026-11-13/tokens2-operator-compat-1/result.json).

A different same-address follow-up used two counterbalanced processes, three
warmup plus 100 measured pairs per case and 32 public calls per batch. All
original anchors/outputs matched; public wrapper calls remained present.

| Historical one-token case | Process 1 paired-median overhead | Process 2 paired-median overhead | Process 1 paired 95% interval | Process 2 paired 95% interval |
| --- | ---: | ---: | --- | --- |
| Attention gate | 0.043% | 0.055% | -0.013–0.088% | -0.041–0.304% |
| Attention KV | -0.066% | 0.397% | -0.247–0.050% | 0.028–0.808% |

That executable/batching control establishes no cause, blanket noise waiver or
new threshold and does not replace the separate-process results.
[Historical paired follow-up](../artifacts/lfm3-tokens2-2026-11-13/tokens2-fallback-paired-analysis-1.json).

## Machine code, checks and remaining gates

Corrected public dispatch has a 32-byte frame. Count 1 restores that frame and
tail-branches to the same scalar function before entering the private paired
helper's 592-byte frame. `dot2` is inlined into that helper; no standalone symbol
remains. Original scalar/`dot` code sizes and frames are unchanged. These are
per-function static facts, not stack peaks, latency causes or cycle/speed ratios.
[Corrected assembly and raw functions](../artifacts/lfm3-tokens2-2026-11-13/tokens2-assembly-2.json).

Fresh corrected formatting, strict CPU/Metal Clippy, 17 release projection
controls and the allocation control pass. The controls preserve independent
BF16/FP64 fixtures, both unrounded lanes, finite-word coverage, tails, unaligned
views, admission errors, first-error ordering and retry. Fresh corrected full
suites pass 794 CPU / 954 Metal-feature tests, with zero failures (53 / 351
ignored). All seven required check receipts and 383 performance seals reconcile.
[Full check audit](../artifacts/lfm3-tokens2-2026-11-13/tokens2-checks-audit-2.json),
[independent raw audit](../artifacts/lfm3-tokens2-2026-11-13/tokens2-performance-audit-3.json).
[Format](../artifacts/qwen35-2b-payload-2026-11-13/lfm3-tokens2-fmt-2/result.json),
[CPU Clippy](../artifacts/qwen35-2b-payload-2026-11-13/lfm3-tokens2-cpu-clippy-2/result.json),
[Metal Clippy](../artifacts/qwen35-2b-payload-2026-11-13/lfm3-tokens2-metal-clippy-2/result.json),
[release controls](../artifacts/qwen35-2b-payload-2026-11-13/lfm3-tokens2-release-cpu-2/result.json),
[allocation](../artifacts/qwen35-2b-payload-2026-11-13/lfm3-tokens2-allocation-2/result.json).

These checks qualify the rejected experiment's correctness evidence; this
evidence-only commit retains no candidate code. Stop this bounded search. The
1,025-position graph, 62 state/replay, full quality, 6 GiB fit and complete 512/32
request/compatibility/sustained gates remain binding for future mechanisms.
[Unchanged prospective policy](../artifacts/lfm3-tokens2-2026-11-13/tokens2-operator-prospective-1.json).

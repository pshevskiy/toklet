# Toklet improvement index

Updated **22 November 2026**. This is the compact index of retained capabilities,
measured gains and material restrictions. The [full dated history](archive/2026-11-09/improvements.md)
preserves every previous entry, source revision, failed check and evidence link.
Read it only when the current task needs that detail.

Measurements apply to the named hardware, route, workload and timing boundary.
Gains from different experiments cannot be added together. Current task and
qualification scope live in [TODO](../TODO.md); acceptance policy is in
[PLAN](../PLAN.md#validation-and-performance-policy).

### Engine architecture and model decision

The Astra investigation places 70% of the controlled vLLM completion gap
before the first ID. The released decode trace has 1.419 s of GPU time and
0.703 s of host wait outside GPU execution. The next model target is
Nanbeige4.2-3B MLX 4-bit. The plan starts with a direct MLX reference and
proposes a modular Rust and Metal runtime with separate prefill and decode
plans. After AC admission, the same-worker replay verifies 52 requests in
opposite orders. Bypassing the HTTP path saves 5.54 and 2.76 ms at median
completion, with paired 95% intervals including zero. Peak is 5.27 GiB.
The Nanbeige metadata and tokenizer join all 513 tensor entries and identify
156 unaligned packed file spans, plus 37 padded rows outside the tokenizer.
Qualifying the new engine model route remains open.
[Same-worker replay](../artifacts/mlx-gpu-gap-2026-11-15/root-lfm3-same-worker-replay-1/normal.json) ·
[Nanbeige evidence](../artifacts/mlx-gpu-gap-2026-11-15/root-nanbeige-mlx-q4-admission-1/RUN.md).
[Diagnosis and proposal](mlx-gpu-performance-plan.md#rust-and-metal-architecture-proposal).

### Experimental Nanbeige Metal CLI

The explicit affine Q4 profile authenticates the checkpoint, loads aligned
packed banks, executes two passes through 22 shared blocks and retains 44
independent BF16 K/V states. The M1 Pro CLI runs at capacity 2,048. Both
matched partitions reproduce greedy IDs 858 and 461. Thirty independently
checked native and MLX BF16 comparisons retain nonzero error. The final
capacity-129 diagnostic peaks at 2.31 GiB. Astra confirms the experimental
scope and finds that MLX's split-K BF16 stores explain all 64 sampled
layer-zero V values. Formatting, both strict Clippy configurations, 869
default tests, 1,074 Metal tests and both explicit GPU controls pass. The
Metal library rerun uses a symbol-stripped, unoptimized Toklet package.
The real-model measurements use the optimized release binary. Numerical
adoption and comparative performance remain open.
[Native comparisons](../artifacts/mlx-gpu-gap-2026-11-15/root-nanbeige-mlx-q4-admission-1/native-final-normal.json) ·
[Split-K witness](../artifacts/mlx-gpu-gap-2026-11-15/root-nanbeige-mlx-q4-admission-1/astra-splitk-l0-value-1/result.json).

The inline split-K diagnostic matches all 262,144 layer-zero V prefill words
from MLX 0.32.3. Both reversed warmed orders increase GPU time. The identical
MLX 0.32.4 inputs favor the released unsplit projection. Its differences are
26 of 131,072 words at T128, and 16 and eight of 65,536 in the T64 windows.
Astra keeps the BF16 split-K experiment as evidence. The operator result
prioritizes larger FFN tiles and a separately tested FP32 vector bias sum.
[Paired timing and version replay](../artifacts/mlx-gpu-gap-2026-11-15/root-nanbeige-prefill-work-1/RUN.md).

The Mac path uses 32-row tiles for gate, up and down at
64 or 128 prefill rows. Two accumulator tiles share one packed-weight
reconstruction. The 52-request M1 Pro comparison reduces median 512/32
completion by 10.10% in both orders, and TTFT by 12.18% and 12.14%.
Both decode intervals include zero. Identical greedy IDs, all 44 committed
KV prefixes and full terminal logits pass at 2.47 GiB guarded peak.
The API and CLI report the prepared row sizes. Formatting, both strict
Clippy configurations, 1,074 Metal tests and two explicit GPU controls
pass. The unchanged default inputs retain their 869-test result. Astra
approves the experimental scope. Full-request timing covers T128 on
M1 Pro. [Connected results](../artifacts/mlx-gpu-gap-2026-11-15/root-nanbeige-ffn-connected-1/normal.json) ·
[Final checks](../artifacts/mlx-gpu-gap-2026-11-15/root-nanbeige-ffn-connected-1/RUN.md).

The subsequent connected FP32 vector bias sum improves Nanbeige logit
agreement against MLX 0.32.4 on the fixed tape. All 32 greedy decisions
and the rollback controls pass. Decode improves 6.98% and 7.40% in
the two warmed orders. Completion improves 1.31% and 1.39%. Retain
the arithmetic as an experiment while full quality and the 5% request
threshold remain open. Direct MLX finishes in 3.315–3.316 s versus
5.841–5.842 s for Toklet, with about 92% of that gap in prefill.
The matched T128 gate/up panels agree bitwise and identify the next
kernel opportunity. [Decode, direct timing and admission evidence](../artifacts/mlx-gpu-gap-2026-11-15/root-nanbeige-modern-decode-1/RUN.md).

The compact T128 gate and up tile uses 128 threads, 5 KiB of shared
scratch and lane-local output. The optional Mac pipeline tests its
actual compiled layout with a signed Q4 basis at model load. All
52 M1 Pro requests match the previous route's token IDs and all
four terminal diagnostics match their full KV and logits. Median
completion improves 7.01% and 7.07%, with 8.67% and 8.65% shorter
TTFT. Peak guarded charge is 2.47 GiB. Both strict Clippy builds,
1,074 Metal tests and two explicit GPU controls pass. Astra approves
the experimental Mac profile.
[Paired performance and verification](../artifacts/mlx-gpu-gap-2026-11-15/root-nanbeige-ffn-compact-connected-1/RUN.md).

Two direct MLX owners match all 393,216 words of Nanbeige's T128
down projection in about 3.28 ms. The native K64 tile matches that
panel and reduces median GPU time from 5.50–5.58 to 3.73 ms.
The Mac trial checks the GPU buffer address during pipeline and
session preparation. The CLI reports the session's prepared row
count. The 52-request trial shortens median completion by 5.90% and
5.96% in opposite orders, with 7.44% and 7.39% shorter TTFT. Token
IDs, all 44 committed K/V prefixes and terminal logits agree at a
2.47 GiB peak. The exact-source checks pass and Astra approves the
scoped Mac change. A separate compact query experiment matches all
786,432 words in both orders and saves 1.31–1.32 ms per visit.
Its isolated savings fall below the request-level screen.
[Down projection](../artifacts/mlx-gpu-gap-2026-11-15/root-nanbeige-prefill-work-1/sol-down-k64-1/normal.json) ·
[Connected trial](../artifacts/mlx-gpu-gap-2026-11-15/root-nanbeige-down-connected-1/RUN.md) ·
[Query projection](../artifacts/mlx-gpu-gap-2026-11-15/sol62-nanbeige-query-compact-1/RUN.md).

The paired compact attention route reuses K32 for query and K64 for
output on T128 Mac requests. The CLI reports the effective session
route after checking the aligned GPU input. The 52-request trial
reduces median completion by 9.71% and 9.86% and TTFT by 12.43% and
12.60% in opposite orders. IDs, all 44 committed K/V prefixes,
and terminal logits match at 2.46 GiB peak guarded charge. The
exact-source checks pass and Astra approves the three-file route.
[Connected trial](../artifacts/mlx-gpu-gap-2026-11-15/root-nanbeige-attention-connected-1/RUN.md) ·
[Joined attention command](../artifacts/mlx-gpu-gap-2026-11-15/sol62-nanbeige-attention-pair-1/normal.json) ·
[Astra review](../artifacts/mlx-gpu-gap-2026-11-15/root-nanbeige-prefill-work-1/release-attention-candidate-1/REVIEW.md).

The K64 gate/up pair agrees on 2,752,512 BF16 words and runs in
7.26 ms versus 8.51–8.53 ms for K32. Both paired lower savings
fall below 1.311 ms per visit. Astra retains the isolated evidence.
The local vLLM overlay smoke verifies 44 disjoint cache regions
and all 32 output IDs. Its vocabulary head scores 32 rows across
35 forwards. The Toklet route scores 35.
[Gate/up comparison](../artifacts/mlx-gpu-gap-2026-11-15/sol62-nanbeige-gate-up-k64-pair-1/RUN.md) ·
[vLLM workload admission](../artifacts/mlx-gpu-gap-2026-11-15/root-nanbeige-vllm-overlay-smoke-1/RUN.md).

### Prefill attribution and observer timing

The two Nanbeige Gate/Up and SwiGLU fusion variants match all
1,376,256 words of the released Down input. K32's lower savings
of 1.142 and 1.183 ms miss the 1.304 ms visit screen. Fused K64
uses one shared activation tile and the existing Down operand.
It reduces the isolated GPU median from 8.58 to about 7.00 ms
and clears the screen in both warm orders. Both resource guards,
fault controls, dual readers and Astra's operator review pass.
[Exact fusion and review](../artifacts/mlx-gpu-gap-2026-11-15/sol62-nanbeige-gate-up-swiglu-fused-k64-1/REVIEW.md).

The guarded single-encoder Nanbeige trial completes all 52 requests
and checks 44 terminal K/V digests plus logits for each owner.
Median completion improves 5.31% and 5.30%, with lower paired
bounds of 5.25% and 5.12%. TTFT improves 7.03% and 7.07%.
Decode is 0.08% and 0.19% slower at the medians. The reverse-order
interval observes a small slowdown, so the experiment remains
detached pending attribution.
[Connected timings](../artifacts/mlx-gpu-gap-2026-11-15/sol62-nanbeige-gate-up-fused-connected-1/RUN.md) ·
[Astra review](../artifacts/mlx-gpu-gap-2026-11-15/sol62-nanbeige-gate-up-fused-connected-1/REVIEW.md).

The two same-owner factors isolate the T1 host encoder and preceding
prefill. All 104 requests, 3,224 completed T1 command clocks and eight
full terminal states pass at 2.37 GiB peak. Both orders of both factors
give paired decode intervals spanning zero. Sol's outer specialization
removes the per-visit fusion check in the compiled T1 graph. Its two
ordinary cohorts complete 52 exact requests at 2.47 GiB peak with
paired lower completion gains of 5.39% and 5.33%. The reverse decode
median increases by 1.05 ms, with its entire paired interval slower.
Both implementations remain detached under the original decode rule.
[Host and prefill clocks](../artifacts/mlx-gpu-gap-2026-11-15/sol62-nanbeige-fused-t1-crossover-1/RUN.md) ·
[Astra review](../artifacts/mlx-gpu-gap-2026-11-15/sol62-nanbeige-fused-t1-crossover-1/REVIEW.md) ·
[Outer specialization and review](../artifacts/mlx-gpu-gap-2026-11-15/sol62-nanbeige-fused-outer-dispatch-1/REVIEW.md).

Reusing sixteen affine-Q4 coefficients through SIMD shuffles preserves
all 1,769,472 authentic FFN words, both analytical Down pipelines and
the fault control. The isolated T128 two-dispatch chain slows from
10.77–10.78 to 15.50–15.51 ms. It loses all 20 timed pairs and
closes below the 1.240 ms per-visit savings gate.
[Isolated measurements](../artifacts/mlx-gpu-gap-2026-11-15/sol62-nanbeige-affine-codebook-1/normal-1.json) ·
[Astra review](../artifacts/mlx-gpu-gap-2026-11-15/sol62-nanbeige-affine-codebook-1/REVIEW.md).

The dense L0 Gate/Up layout reproduces all 1,769,472 native FFN
words and 16,384 signed prepared coefficients, with shared packed
Down and the independent MLX Down control. GPU medians fall from
10.750 to 8.997 ms and from 10.776 to 9.002 ms. Both paired lower
savings clear 1.240 ms. Metal allocates 132,120,576 bytes and takes
7.44 ms wall time to prepare it after allocation. The strict build
and both readers pass. Full-model memory and ordinary speed remain open.
[Guarded result](../artifacts/mlx-gpu-gap-2026-11-15/sol62-nanbeige-dense-gate-up-1/normal-2.json) ·
[Astra review](../artifacts/mlx-gpu-gap-2026-11-15/sol62-nanbeige-dense-gate-up-1/REVIEW.md).

The full-model dense fit holds 22 disjoint pairs at 5.06 GiB peak.
Cold allocation takes 366 ms and preparation 292 ms wall time,
including 71 ms on the device. The guarded terminal logits, all
44 K/V pairs and T1/T43/T64 fallback controls match the release.
The low-budget loader uses packed weights and the later undersized
session is rejected. The compiled T1 graph bypasses dense selection.
[Guarded fit](../artifacts/mlx-gpu-gap-2026-11-15/sol62-nanbeige-dense-connected-1/fit-normal.json) ·
[Compiled T1 graph](../artifacts/mlx-gpu-gap-2026-11-15/sol62-nanbeige-dense-connected-1/ARM64-REVIEW.md).

The dense connected requests finish in 4.076 and 4.086 s versus
4.618 and 4.611 s. First token arrives about 0.56 s sooner. All 52
requests and four terminal witnesses match, with 5.16 GiB peak.
Decode is slower in all 20 measured pairs, with median increases of
16.47 and 13.23 ms. The candidate remains detached.
[Ordinary results and Astra review](../artifacts/mlx-gpu-gap-2026-11-15/sol62-nanbeige-dense-connected-1/REVIEW.md).

The same-binary follow-up passes all 104 exact requests at 5.17 GiB
peak. At fixed residency, dense prefill adds 16.92 and 20.58 ms of
decode at the paired medians. Both intervals show a slowdown. With
packed prefill in both arms, the residence-only paired median changes
sign between orders. The next isolated change keeps the final prefill
chunk packed before starting the same T1 decode path.
[Timing contrasts and Astra review](../artifacts/mlx-gpu-gap-2026-11-15/sol62-nanbeige-dense-causal-1/REVIEW.md) ·
[Owner custody](../artifacts/mlx-gpu-gap-2026-11-15/sol62-nanbeige-dense-causal-1/custody-normal.json).

Packing the final T128 chunk keeps 22 pairs resident and records
three dense cuts followed by one packed cut. Guarded fit, strict
release Clippy and exact compiled T1 fallback pass. All 52 ordinary
requests match the released state. Completion drops 4.595 to 4.300 s
and 4.624 to 4.291 s. First token is about 295 and 321 ms sooner.
Both decode medians improve at 5.16 GiB peak. The fixed-prompt probe
clears both ordinary gates. Generic opt-in and cold timing are next.
[Ordinary timings and Astra review](../artifacts/mlx-gpu-gap-2026-11-15/sol62-nanbeige-dense-tail-packed-1/REVIEW.md).

The Nanbeige `--dense-prefill` CLI option on Apple M1 Pro uses remaining
prompt work to select the route. Signed startup runs on the compiled
pipelines. It fits 22 sidecars at 5.06 GiB peak, and four alternate
prompt lengths match packed IDs, logits and every K/V owner. The
compiled T1 path and default CLI schema match the released behavior.
A flagged 128-token prompt allocates no sidecars. The single cold
pair has unequal load times and remains descriptive. All 52 warm
requests match. Completion falls from 4.620 to 4.290 s and from
4.616 to 4.289 s. First token arrives about 313–314 ms sooner, and
both median decode times fall. One paired decode interval includes
zero. Guarded peak memory rises from 2.47 to 5.16 GiB. The full
default and Metal suites and all four explicit GPU tests pass.
Waited cancellation or nonfinite status preserves the committed
prefix, and retry matches exactly. A poisoned startup basis leaves
zero resident dense pairs and reproduces the packed terminal state.
Astra approves the exact opt-in. Warm preparation adds about 0.68 s
before the timed requests. Cold performance remains unqualified.
Pushed `0e5f6378`.
[Generic fit](../artifacts/mlx-gpu-gap-2026-11-15/sol62-nanbeige-dense-option-1/fit-normal.json) ·
[Alternate and cold controls](../artifacts/mlx-gpu-gap-2026-11-15/sol62-nanbeige-dense-option-1/aux-normal.json) ·
[Ordinary timings and Astra review](../artifacts/mlx-gpu-gap-2026-11-15/sol62-nanbeige-dense-option-1/REVIEW.md) ·
[All nine release checks](../artifacts/mlx-gpu-gap-2026-11-15/root-nanbeige-dense-release-checks-1/normal-2.json) ·
[Astra release review](../artifacts/mlx-gpu-gap-2026-11-15/root-nanbeige-dense-release-checks-1/RELEASE-REVIEW.md).

The guarded Nanbeige T1 capture retains 3,072 normalized BF16 words
and both 10,752-word projections at its first committed decode input.
The terminal witness matches the release. This furnishes the exact
activation-sum screen with authentic packed-vector operands.
[Capture and Astra review](../artifacts/mlx-gpu-gap-2026-11-15/sol62-nanbeige-t1-sum16-capture-1/REVIEW.md).

The T1 activation-sum experiment reproduces all 21,504 BF16 outputs
and 192 exact sums with passing rounding and fault controls. GPU
paired savings range from 46.13 to 91.27 µs in the first order and
from −17.08 to 70.00 µs in reverse. Astra confirms closure because
the reverse lower bound misses the 14.663 µs screen.
[Standalone operator evidence and Astra review](../artifacts/mlx-gpu-gap-2026-11-15/sol62-nanbeige-t1-sum16-operator-1/REVIEW.md).

Further dense Down coverage would exceed 6 GiB with all 22 banks.
Partial coverage would require about 1.02 ms per visit for a 5%
request gain. The fastest measured exact Down takes 3.28 ms.
Layer-major execution preserves causal attention if positions stay
ordered within each layer. The current evidence gives no measured
GPU budget for the required 215 ms request saving. Astra defers a
prototype until the generic route has current stage attribution.
[Memory and timing analysis with Astra review](../artifacts/mlx-gpu-gap-2026-11-15/sol62-nanbeige-next-prefill-brief-1/REVIEW.md).

The sparse stage profiler reproduces all 27 requests and every state
witness in its complete owner. Splitting the encoders shifts TTFT by
up to 92.23 ms against a 21.451 ms limit. The paired median residual
outside measured GPU time and encoding shifts by 90.82 ms. This
residual combines commit, queue scheduling and host wake time.
Timestamp sampling separately shifts a clock by at most 17.48 ms.
The second owner stops at the disk floor. Close the diagnostic and
keep ordinary completed-command clocks for future request comparisons.
[Capture and Astra review](../artifacts/mlx-gpu-gap-2026-11-15/sol62-nanbeige-dense-attribution-1/CAPTURE-REVIEW.md) ·
[Metal capabilities](../artifacts/mlx-gpu-gap-2026-11-15/sol62-nanbeige-dense-attribution-1/dispatch-capabilities.txt).

All 100 original Nanbeige task cases are frozen under the pinned codec.
The Python template and the existing compiled Rust codec match every
input. The original categories and caps remain, with thinking disabled
for code and prose and raw continuation. Quality scoring remains open.
[Frozen task inputs](../artifacts/mlx-gpu-gap-2026-11-15/root-nanbeige-quality-tasks-1/RUN.md).

The guarded same-Q4 scoring smoke uses 32 frozen held-out targets
and two separate EOS controls. Both Toklet and pinned MLX finish
37 forced T1 steps under 2.66 GiB. The score includes all 166,144
head rows. Mean Toklet loss is 0.02357 nats/token higher and two
greedy choices differ on the held-out sample.
[Held-out smoke and Astra review](../artifacts/mlx-gpu-gap-2026-11-15/sol62-nanbeige-heldout-fidelity-1/SMOKE-REVIEW.md).

The guarded full same-Q4 comparison scores all 10,000 targets across
ten independent windows. Mean NLL is 3.53068 for Toklet and 3.53302 for
pinned MLX. Greedy choices differ at 435 targets. Both engines exactly
repeat their own smoke controls, with guarded peaks of 2.60 and 3.30
GiB. The first attempt stopped on an external compiler. The unchanged
retry supplies the complete evidence. Astra also verifies 21 targets
with absolute loss differences of at least one nat. The original BF16
reference and task acceptance checks remain open.
[Full same-Q4 results and Astra review](../artifacts/mlx-gpu-gap-2026-11-15/sol62-nanbeige-heldout-fidelity-1/FULL-REVIEW.md) ·
[BF16 streaming proposal](../artifacts/mlx-gpu-gap-2026-11-15/astra-nanbeige-bf16-quality-1/PROPOSAL.md).

The current official Gemma 4 E2B BF16 console pair uses one fresh linked
library and binary on both arms. Batched T32 prefill lowers first token
from 11.930 to 6.997 s and Done from 39.274 to 34.402 s. All 128 IDs
and public state match. Decode rises 63 ms and guarded peak is 4.65 GiB.
Formatting, strict Metal Clippy, parser and after-flush cancellation
pass. Independent cohorts and the original quality suite remain open.
[Current pair and Astra review](../artifacts/mlx-gpu-gap-2026-11-15/sol62-gemma-bf16-t32-current-1/ACTUAL-REVIEW.md).

The detached Gemma 1,025-position diagnostic reproduces all 1,024
original BF16 scores, their unscored terminal row and eight raw F32
anchors. It verifies 90 chronological K/V plates across the 512,
1,024 and 1,025 boundaries. The 33 cuts record 240 Metal and 915
CPU visits with 4.45 GiB peak. Formatting, strict Metal Clippy, fresh
release linking and the targeted geometry test pass after the two
detached build fixes. This covers the first of ten held-out blocks.
[First scored block and Astra review](../artifacts/mlx-gpu-gap-2026-11-15/sol62-gemma-bf16-t32-quality-1025-1/ACTUAL-REVIEW.md).

The pinned MLX layer-at-a-time walk matches its stock Q4 graph on all
1,058 scores and the four terminal heads and 176 joint K/V hashes.
The long case reaches 1,025 committed positions at a 2.80 GiB guarded peak.
The separate guarded transfer authenticates both official BF16 shards,
8.34 GB of source data and 201 tensor spans, at 129 MiB peak. It retains
the hash tables and the frozen embedding rows for the paired scorer.
Actual BF16 scoring and the original task suite remain open.
[Layerwise result and Astra review](../artifacts/mlx-gpu-gap-2026-11-15/root-nanbeige-layerwise-oracle-1/REVIEW.md) ·
[Parent acquisition and Astra review](../artifacts/mlx-gpu-gap-2026-11-15/root-nanbeige-layerwise-oracle-1/ACQUISITION-REVIEW.md).

The installed MLX affine PSO matches all three native T128 Q/O/Down
panels plus separate MLX O/Down inputs. Signed basis and sticky fault
controls pass. With three extra status scans charged on one command,
both lower paired GPU savings exceed 2.00 ms. Cold library and PSO
creation total 79 ms. Astra approves a connected trial under the
existing state, working-set and decode gates.
[Guarded result and Astra review](../artifacts/mlx-gpu-gap-2026-11-15/sol62-nanbeige-qmm-three-roles-1/ACTUAL-REVIEW.md).

On two warmed 512/32 Nanbeige cohorts, the connected T128 Q/O/Down
trial reduces completion from 4.311 to 3.933 s and 4.289 to 3.926 s
against Toklet's dense opt-in. Paired 95% lower gains are 8.40% and
8.19%. TTFT falls by over 11% in each order. Both decode medians
improve by less than 4 ms across 31 evaluations. The reverse paired
decode interval includes zero. All 52 requests reproduce the frozen
IDs. Terminal logits and 44 K/V hashes match, including the fallback
controls. Guarded peak across fit and timing is 5.17 GiB. The trial
uses a pinned private MLX pipeline. Its startup hash takes 641–646 ms
and library load takes 21–23 ms before the warm request clock. Product
integration requires a supported shader, capability checks and repeat
qualification for both cold CLI and warm owners.
[Ordinary cohorts and Astra review](../artifacts/mlx-gpu-gap-2026-11-15/sol62-nanbeige-qmm-connected-1/ACTUAL-REVIEW.md) ·
[Fit and linked-path review](../artifacts/mlx-gpu-gap-2026-11-15/sol62-nanbeige-qmm-connected-1/FIT-REVIEW.md).

Building the single MLX QMM specialization from its pinned source
produces a 13,472-byte Metal library. The actual M1 Pro operator test
matches 1,572,864 authentic Toklet Q/O/Down words and 786,432
independent MLX words. The signed basis, nonfinite status and canaries
pass.
[Single-kernel parity and Astra review](../artifacts/mlx-gpu-gap-2026-11-15/sol62-nanbeige-qmm-one-symbol-1/ACTUAL-REVIEW.md).

The connected compact-kernel opt-in passes the four fit guards, the
four cold CLI controls and the linked packed-decode audit. Warm
completion falls from 4.273 to 3.925 s and 4.842 to 3.918 s on the
matched internal tape. Both paired lower completion bounds exceed 5%.
In the first order, however, median decode takes 20.80 ms longer and
all ten pairs are adverse. The released baseline's decode moves by
104.23 ms between owners. Retention waits for a crossed comparison
within each resident owner. Peak across all twelve owners is 5.17 GiB.
A single cold CLI completion rises from 12.570 to 13.738 s while setup
rises 1.546 s. That observation is descriptive.
[Compact-kernel actual checks and Astra review](../artifacts/mlx-gpu-gap-2026-11-15/sol62-nanbeige-qmm-product-1/ACTUAL-REVIEW.md).

The same-resident crossover alternates only T128 QMM selection and
reproduces all 52 requests, terminal logits and 44 K/V prefixes.
Paired median completion savings are 370 and 368 ms. The paired decode
intervals include zero and exclude a 20.80 ms penalty in both owners.
[Crossed timings and Astra review](../artifacts/mlx-gpu-gap-2026-11-15/sol62-nanbeige-qmm-resident-causal-1/ACTUAL-REVIEW.md).

The fixed independent confirmation retains the explicit warm M1 Pro
opt-in. Completion improves from 4.290 to 3.919 s and from 4.284 to
3.922 s. Paired lower savings are 8.30% and 8.27%. Both decode intervals
include zero. Fit, fault, cold console and linked path checks pass. The
exact source passes formatting, strict Clippy in both modes, 870 default
tests, 1,075 Metal tests and seven explicit GPU controls. The packaged
image carries the upstream MIT notice. The 32-head external join and 1%
request-summary qualification remain open.
[Independent confirmation and Astra approval](../artifacts/mlx-gpu-gap-2026-11-15/sol62-nanbeige-qmm-product-confirmation-1/ACTUAL-REVIEW.md) ·
[Packaged console usage](../assets/nanbeige-qmm/README.md).

The detached Ollama publisher-fork raw worker now builds with its pinned
MLX and MLX-C. Xcode's optional Metal toolchain was present but inactive.
Official activation restores preprocessing with the original system Xcode
selection intact. The guarded CMake retry and native link pass.
[Astra source review](../artifacts/mlx-gpu-gap-2026-11-15/sol62-ollama-nanbeige-raw-worker-1/REVIEW.md) ·
[Exact native build](../artifacts/mlx-gpu-gap-2026-11-15/sol62-ollama-nanbeige-raw-worker-1/build-receipt.json).

The private Ollama raw-ID owner completes the 512/32 tape after verifying
513 imported tensors and four codec files. It scores 35 head rows, checks
44 cache owners and 543 committed positions with one pending, and peaks
at 3.47 GiB. All 32 sampled IDs match. The terminal snapshots differ
from the archived direct MLX and Toklet cache hashes.
[Raw replay](../artifacts/mlx-gpu-gap-2026-11-15/sol62-ollama-nanbeige-raw-worker-1/trial-1/normal.json) ·
[Terminal values](../artifacts/mlx-gpu-gap-2026-11-15/sol62-ollama-nanbeige-raw-worker-1/TERMINAL-NUMERIC.md).

The private Ollama matched-work witness executes four T128 bodies,
elides the first three heads and feeds back its own greedy IDs through
31 T1 evaluations. The guarded untimed replay passes at 3.46 GiB. Both
readers agree on all 32 sampled IDs and 44 terminal cache pairs.
Terminal state and logits reproduce the 35-head Ollama witness.
All 32 scored-head digests differ from matched MLX.
[Physical witness](../artifacts/mlx-gpu-gap-2026-11-15/sol62-ollama-nanbeige-raw-worker-1/matched32/run-1/read-normal.json).

The direct MLX prefill replay reproduces all 44 sealed K/V pair hashes
at 3.12 GiB. Against Ollama, slot zero in the first T128 tile differs
in 55,617 K and 67,198 V words out of 131,072 per plane. Independent
raw readers place both first differences at prompt position zero.
This narrows the discrepancy to the first cache write or its inputs.
[Prefill receipt](../artifacts/mlx-gpu-gap-2026-11-15/root-nanbeige-ollama-prefill-cut-1/capture.json) ·
[Independent raw check](../artifacts/mlx-gpu-gap-2026-11-15/root-nanbeige-ollama-prefill-cut-1/check-normal.json).

Both layer-zero probes reproduce their own retained first cache slot.
Embedding and RMSNorm each match all 393,216 words. Both runtimes use
the same T128 activation and differ at 55,567 K and 67,198 V words
before RoPE. All 131,072 Ollama V words match MLX 0.32.3. Direct MLX
peaks at 2.76 GiB and Ollama at 2.41 GiB in the guarded probes. Astra
replays both comparisons and approves the detached companion preflight.
CMake accepts the pinned MLX 0.32.4 wheel. The MLX-C build stops at two
compile-cache API errors before linking.
[Guarded preflight](../artifacts/mlx-gpu-gap-2026-11-15/root-nanbeige-ollama-layer0-1/mlxc-wheel-preflight-1/receipt.json).

The detached MLX 0.32.4 bridge builds with native-thread cache controls
and explicit defaults for the newly added SDPA and gather-QMM arguments.
All four smoke commands pass. The unchanged Ollama Go worker maps the
three sealed native libraries, consumes its own 32 samples and joins
all scored logits, 44 terminal cache pairs and their 512-row prefixes
with MLX and vLLM. The guard peaks at 3.51 GiB. Astra clears the untimed
private-worker result after verification of the complete source diff.
[Replay](../artifacts/mlx-gpu-gap-2026-11-15/root-nanbeige-ollama-mlx324-1/normal.json) ·
[Bridge smoke](../artifacts/mlx-gpu-gap-2026-11-15/root-nanbeige-ollama-mlx324-1/bridge/smoke-receipt.json).

Two reversed orders complete 52 guarded requests on the private
Ollama and vLLM routes. Astra approves the exact timing launch.
Ollama peaks at 3.52 GiB and completes in 3.267/3.291 s. Private vLLM
peaks at 3.92 GiB and completes in 3.621/3.622 s. Ollama's paired
completion savings are 9.75% and 9.21% with 95% lower bounds of
9.52% and 8.88%. The savings primarily occur after the first ID.
The dual readers agree. Setup clocks retain engine-specific boundaries,
and vLLM's private runtime validation remains inside its request clock.
[Cohorts](../artifacts/mlx-gpu-gap-2026-11-15/root-nanbeige-ollama-timing-1/normal.json) ·
[Astra launch review](../artifacts/mlx-gpu-gap-2026-11-15/root-nanbeige-ollama-timing-1/REVIEW.md).

Astra clears the detached Toklet 32-score worker for diagnostic timing.
Six fresh owners return 78 requests with the same 32 sampled IDs in
both reversed orders. Toklet completes in 4.506–4.546 s at 5.17 GiB
peak, versus Ollama's 3.738–3.745 s at 3.52 GiB and vLLM's 4.081 s
at 3.92 GiB. Toklet's first ID arrives 574–596 ms after Ollama's,
about 75% of the completion gap. Its decode is faster than vLLM's
and slower than Ollama's in both orders. The native witness has zero
matching scored-logit and terminal K/V digests against MLX. The native
final prefill head scores one logical row through its tiled kernel.
[Cohorts](../artifacts/mlx-gpu-gap-2026-11-15/root-nanbeige-toklet-clocks-1/normal.json) ·
[Astra launch review](../artifacts/mlx-gpu-gap-2026-11-15/root-nanbeige-toklet-clocks-1/REVIEW.md).

A guarded trace on the same native route captures the first T128
embedding, RMSNorm, projections and live cache writes at 5.06 GiB.
Both inputs to the projection stage match joined MLX. The pre-RoPE
K and V outputs differ in 14 and 26 of 131,072 words each, and the
same counts appear in the cache. V reproduces the archived native
projection on the same operands. Both readers agree. Astra approves
the observed K/V attribution. Q is outside the capture.
[Array comparison](../artifacts/mlx-gpu-gap-2026-11-15/root-nanbeige-native-first-cut-1/normal.json) ·
[Guarded capture and checks](../artifacts/mlx-gpu-gap-2026-11-15/root-nanbeige-native-first-cut-1/NATIVE.md) ·
[Astra review](../artifacts/mlx-gpu-gap-2026-11-15/root-nanbeige-native-first-cut-1/REVIEW.md).

The detached MLX 0.32.4 split-K replay matches all 262,144 combined
BF16 K/V words. The native control reproduces the original 14 and
26 mismatches. The independent FP32 merge, canaries, formatting,
strict Clippy and offline build all pass. The T384 repeated-input
control instead matches native T128 K/V in all three copies. Astra
approves the numerical evidence. Both physical guards pass. This
tests the projection pair and shape choice on one authentic input.
[Replay and checks](../artifacts/mlx-gpu-gap-2026-11-15/root-nanbeige-kv-parity-1/RUN.md) ·
[Evidence seal](../artifacts/mlx-gpu-gap-2026-11-15/root-nanbeige-kv-parity-1/outer.json).

The detached connected split-K witness matches the first live Q, K,
V and committed cache against MLX. The slot-zero prefill prefix and
all 32 IDs join. Five rotated Q words differ. Causal attention differs
in 234 words across 107 rows whose observable inputs are exact. O
matches on equal input rows. Zero scored or terminal cache digests
join MLX. The incumbent control reproduces all archived scores and
terminal pairs. Peak guarded memory is 5.07 GiB.
[Connected report](../artifacts/mlx-gpu-gap-2026-11-15/root-nanbeige-connected-splitk-1/normal.json) ·
[Incumbent control](../artifacts/mlx-gpu-gap-2026-11-15/root-nanbeige-connected-splitk-1/control.json) ·
[Astra review and scoped checks](../artifacts/mlx-gpu-gap-2026-11-15/root-nanbeige-connected-splitk-1/RUN.md).

The first rotary FMA change matches all first-cut Q and K words.
The frequency-only control keeps all five Q differences. Replaying
causal attention on identical Q/K/V leaves 237 of 786,432 output
words different. The dense Gate/Up weight-sharing candidate matches
all 1,376,256 SwiGLU and 393,216 Down words and loses all 40 timing
pairs. Isolated GPU medians rise from 5.25–5.27 to 6.11 ms, and the
chain with Down rises from about 8.99 to 9.86 ms. Astra approves
both detached conclusions.
[Rotary and attention](../artifacts/mlx-gpu-gap-2026-11-15/root-nanbeige-attention-replay-1/RUN.md) ·
[Gate/Up timing](../artifacts/mlx-gpu-gap-2026-11-15/root-nanbeige-dense-tile-reuse-1/RUN.md) ·
[Combined evidence seal](../artifacts/mlx-gpu-gap-2026-11-15/root-nanbeige-attention-replay-1/outer.json).

The connected T128 Steel and corrected rotary candidate matches all
nine first-cut panels and the first cache owner at all four prefill
boundaries. Both modes emit the same 32 IDs. The split-K control
reproduces the archived state. Full-request parity remains open.
The guarded proof GPU clocks are 2.443 s and 2.580 s, with a peak
charge of 5.07 GiB. A separate MLX Gate/Up shape test matches all
4,128,768 words and measures about 18.4 ms for either shape.
[Connected and shape reports](../artifacts/mlx-gpu-gap-2026-11-15/root-nanbeige-steel-connected-1/normal.json) ·
[Astra review and checks](../artifacts/mlx-gpu-gap-2026-11-15/root-nanbeige-steel-connected-1/REVIEW.md).

The guarded FFN witness matches all eight first-cut panels after
applying MLX's BF16 sigmoid rounding. The incumbent differs in
451,662 SwiGLU words on the same Gate and Up operands. Ten cache
owners match at position 128, then only the first matches at the
next three prefill boundaries. Both complete requests keep all 32
IDs. Exact sampled-score and terminal-cache gates remain open.
Both guards, formatting, strict Metal Clippy and release links pass.
[FFN and cache frontier](../artifacts/mlx-gpu-gap-2026-11-15/root-nanbeige-ffn-connection-1/RUN.md).

The native second-cut witness reproduces the archived request. The
post-attention residual matches exactly, and FFN RMSNorm differs
in 20 words over two rows. The T128 MLX RMS reduction joins all
19 second-cut panels and all 176 prefill cache digests. Both guarded
owners complete 32 matching IDs at 5.08 GiB. Formatting, strict
Metal Clippy and release links pass. Score and terminal-cache parity
remain open. The completed GPU proof clocks cover one request each.
[RMS and prefill evidence](../artifacts/mlx-gpu-gap-2026-11-15/root-nanbeige-second-cut-1/RUN.md).

The detached QMV route matches all 166,144 words of the first live
MLX score on the exact final hidden row. Both guarded builds pass
and keep the same IDs, prefill cache and downstream native state.
Same-operand T1 replay explains every Q/K/V difference by the
four-term bias reduction precision. The remaining score and terminal
gates are open. The proof command clocks cover one request per route.
[First score and T1 trace](../artifacts/mlx-gpu-gap-2026-11-15/root-nanbeige-head-decode-1/RUN.md).

The detached T1 FP32 bias sum matches 8,191 of 8,192 first-decode Q/K/V
words, all 176 prefill K/V pairs and all 32 IDs. Native and pinned MLX
attention match on common inputs. Full decode parity remains open.
Scoped checks, both hardware guards and Astra review pass.
[Decode bias and attention replay](../artifacts/mlx-gpu-gap-2026-11-15/root-nanbeige-t1-bias-1/RUN.md).

Separate guarded halves and SwiGLU replays match all 8,192 Q/K/V and 10,752
SwiGLU words in the first decode. The best run matches 34/1,364 appends.
All 31 decode score rows and 44 terminal K/V pairs differ. Astra approves
the detached evidence. Strict checks and both 512/32 Metal replays pass.
[Halves](../artifacts/mlx-gpu-gap-2026-11-15/root-nanbeige-t1-halves-1/native/RUN.md) ·
[SwiGLU](../artifacts/mlx-gpu-gap-2026-11-15/root-nanbeige-t1-swiglu-1/native/RUN.md).

Pinned MLX QMV handles all 9,579 eligible T1 projections on the same
512/32 tape. It matches 58/1,364 appends and one terminal K/V pair.
All IDs and prefill pairs agree, and all decode score rows differ.
Scoped checks, the 5.06 GiB guarded replay and Astra review pass.
[Wheel replay](../artifacts/mlx-gpu-gap-2026-11-15/root-nanbeige-t1-qmv-wheel-1/native/RUN.md).

The first-decode trace isolates attention differences on identical operands.
Pinned one-pass vector SDPA makes all 155 panels and the first two append
rows exact. Overall 215/1,364 appends, 2/31 decode scores and 2/44 terminal
pairs match. Scoped checks, the 5.06 GiB guard and Astra review pass.
[Owner trace](../artifacts/mlx-gpu-gap-2026-11-15/root-nanbeige-t1-owner-trace-1/RUN.md) ·
[Vector SDPA](../artifacts/mlx-gpu-gap-2026-11-15/root-nanbeige-t1-sdpa-wheel-1/native/RUN.md).

The runtime position trace isolates one rotated-Q word at 514. Precise T1
rotary closes all 144 selected panels and matches 675/1,364 appends, 8/31
decode scores and 13/44 terminal pairs. The owner window finds five RMSNorm
differences on identical inputs at position 520. Checks, guards and Astra
review pass, including the offline recovery of the window validator.
[Position trace](../artifacts/mlx-gpu-gap-2026-11-15/root-nanbeige-t1-position-trace-1/RUN.md) ·
[Precise rotary](../artifacts/mlx-gpu-gap-2026-11-15/root-nanbeige-t1-rotary-1/RUN.md) ·
[Owner window](../artifacts/mlx-gpu-gap-2026-11-15/root-nanbeige-t1-owner-window-1/RUN.md).

Pinned T1 RMSNorm joins all 32 scores and IDs, 176 prefill pairs, 1,364
appends and 44 terminal pairs on the detached Nanbeige 512/32 tape.
All 144 selected position-520 panels agree. The 2,790 T1 calls preserve
finite checks and cache publication. Checks, 5.06 GiB guard and Astra
review pass. Ordinary CLI and timing qualification remain open.
[Complete decode comparison](../artifacts/mlx-gpu-gap-2026-11-15/root-nanbeige-t1-rms-wheel-1/RUN.md).

The ordinary detached CLI uses the exact Nanbeige kernels with snapshots
disabled. It matches 32 IDs and preserves the prefix at first-flush SIGINT.
The native 3+10 screen measures 3.391 s median completion at 5.16 GiB.
Scoped checks, same-source witness and Astra review pass.
[Ordinary route checks](../artifacts/mlx-gpu-gap-2026-11-15/root-nanbeige-exact-cli-prep-1/RUN.md) ·
[Timing screen](../artifacts/mlx-gpu-gap-2026-11-15/root-nanbeige-exact-cli-prep-1/SCREEN.md).

Both fresh Nanbeige orders pass the vLLM completion gate with 6.01% and
6.19% lower medians. Ollama leads by 3.87% and 3.82% relative to its own
medians. All 78 requests and six guards pass, at a maximum 5.18 GiB.
The overall completion and request-summary overhead gates remain open.
[Matched cohorts](../artifacts/mlx-gpu-gap-2026-11-15/root-nanbeige-exact-cohorts-1/RESULT.md).

Nanbeige final-T128 fused gate/up preserves the complete MLX tape and all
171 trace panels. Two ordinary orders complete 52 matching requests at a
5.36 GiB peak. Paired completion saves 238/220 ms. Paired decode adds
11/19 ms. Astra approves retaining the detached candidate for further work.
[Tail A/B](../artifacts/mlx-gpu-gap-2026-11-15/root-nanbeige-exact-tail-dense-ordinary-1/RUN.md).

Completed brackets confirm 141 ms less final-chunk GPU work. Total decode
GPU time changes by less than 0.2 ms. Dense first-T1 delay appears before
GPU start. Four guarded owners, 52 requests, four integrity replays and
Astra review pass. The separate ordinary A/B governs latency.
[Command-level profile](../artifacts/mlx-gpu-gap-2026-11-15/root-nanbeige-exact-profile-1/RUN.md).

The detached scan fusion moves three T128 checks into precise rotary.
Scoped build checks and untimed Metal sentinels pass at 5.35 GiB. Nine
connected fault cases and waited cancellation preserve the committed
prefix. Exact retry and the following T1 match. Astra approves this scope.
The frozen numerical and timing runs await AC with Low Power Mode off.
[Scan fusion](../artifacts/mlx-gpu-gap-2026-11-15/root-nanbeige-qkv-scan-fusion-prep-1/RUN.md).

The offline summary diagnostic verifies four disabled owners per model.
Matching request ordinals leaves prefill drift, with inconsistent rank
associations for preceding decode. Both readers and Astra review pass.
The original fixed-N summary qualification remains open.
[Summary diagnostic](../artifacts/mlx-gpu-gap-2026-11-15/root-o4-owner-conditioning-1/RUN.md).

Packed and dense-tail ordinary Nanbeige pass guarded untimed rollback.
The shared fixture verifies cancellation, nonfinite status and synthetic
completion error. Both BF16 score rows, all 44 cache owners and all 32
IDs match after retry or reset and replay. Peak is 5.06 GiB. Scoped checks
and Astra review pass. Actual driver failure remains untested.
[Untimed rollback evidence](../artifacts/mlx-gpu-gap-2026-11-15/root-nanbeige-exact-fault-prep-1/RUN.md).

The sealed scan-fusion candidate matches all scores, cache digests and
171 panels in one untimed 512/32 replay. One guarded owner completes at
5.26 GiB and Astra approves. AC and Low Power Mode off remain required
for the frozen paired witness and latency run.
[Untimed scan-fusion result](../artifacts/mlx-gpu-gap-2026-11-15/root-nanbeige-qkv-scan-fusion-prep-1/untimed-full-1/report.json).

Both scan-fusion arms match the frozen tape. Across 52 guarded requests,
paired completion savings are +4.10 and −7.67 ms, with both 95% intervals
including zero. Paired decode slows by 3.07 and 6.99 ms. Peak guarded
memory is 5.36 GiB. Astra rejects the candidate and retains dense tail.
[Scan-fusion A/B](../artifacts/mlx-gpu-gap-2026-11-15/root-nanbeige-qkv-scan-fusion-prep-1/RUN.md).

Token-first dense Gate/Up preserves full tape parity and passes 52 guarded
requests. Stable reverse paired completion saves 11.66 ms (95% interval
3.04–22.60 ms) and decode adds 4.38 ms. The other order is substantially
slower overall and yields a wide savings interval. Astra retains dense tail.
[Traversal A/B](../artifacts/mlx-gpu-gap-2026-11-15/root-nanbeige-dense-traversal-1/RUN.md).

Prepacked dense Gate/Up matches the full frozen tape. In 52 native
requests, median completion is 3.089–3.094 s. Paired completion savings
have 95% lower bounds of 80.28 and 62.78 ms. Reverse paired decode adds
21.20 ms with unchanged dispatches. Peak is 5.36 GiB. Astra approves fresh
peer qualification.
[Prepack A/B](../artifacts/mlx-gpu-gap-2026-11-15/root-nanbeige-dense-prepacked-1/RUN.md).

The prepacked dense-tail candidate clears all four frozen peer gates.
Completion is 5.87% and 7.15% lower than Ollama and 14.75% and 14.93%
lower than vLLM. Every paired lower bound exceeds 5%. All 78 requests
pass at a maximum guarded 5.37 GiB. Astra approves the detached pinned MLX
lane. The patch applies to the sealed dense-tail baseline.
[Reviewed implementation patch](../artifacts/mlx-gpu-gap-2026-11-15/root-nanbeige-dense-prepacked-1/dense-prepacked.patch) ·
[Prepack matched cohorts](../artifacts/mlx-gpu-gap-2026-11-15/root-nanbeige-prepacked-cohorts-1/RESULT.md).

The normal CLI exposes the experimental `--steel-prefill-metallib` flag.
The fixed route matches 32 scores and IDs, 176 prefill pairs, 1,364 appends
and 44 terminal K/V pairs. SIGINT, rejection and legacy smoke checks pass.
Peak is 5.47 GiB. The 870 default and 1,075 Metal tests pass with strict
all-target Clippy and a fresh release build. Astra approves the opt-in.
[Normal CLI checks](../artifacts/mlx-gpu-gap-2026-11-15/root-nanbeige-prepacked-cli-1/RUN.md).

The integrated Steel library worker clears the frozen private peer tests.
Across both orders, completion falls 6.04% and 6.33% against Ollama and
15.05% and 15.39% against vLLM. Every paired lower endpoint exceeds 5%.
TTFT is 201–205 ms faster than Ollama. Native decode adds 3.73 ms in the
first order and saves 7.71 ms in reverse. Both readers validate 78 exact
requests at 5.58 GiB guarded peak. Astra verifies the raw receipts and timing
from fresh session admission to the first and final sampled IDs. The
original summary gates remain open.
[Integrated timing](../artifacts/mlx-gpu-gap-2026-11-15/root-nanbeige-integrated-cohorts-1/RESULT.md).

The detached decode consumer checks replace 9,548 finite-scan dispatches.
Exact scores and K/V, 21 injected nonfinite cases, cancellation and clean
retry pass. Across 52 requests at 5.37 GiB guarded peak, paired completion
savings are +1.99 and −4.17 ms. Both completion and decode intervals span
zero. The trial misses the 60 ms headroom screen. Astra retains the
integrated dense-tail prepack route with standalone T1 checks.
[Astra review](../artifacts/mlx-gpu-gap-2026-11-15/sol62-nanbeige-t1-consumer-scan-1/ACTUAL-REVIEW.md) ·
[T1 finite-check A/B](../artifacts/mlx-gpu-gap-2026-11-15/sol62-nanbeige-t1-consumer-scan-1/RUN.md).

The additional 60 ms Nanbeige screen prices two detached GPU variants.
Two-head T1 attention matches 6,144 archived BF16 words and the fault
controls. Across the full owner and depth sweep its isolated paired
savings are 9.59 and 10.04 ms. T128 Q/O/Down activation packing matches
1,572,864 corrected Steel words and slows the triplet by 0.57 and 0.53 ms.
Both orders include the three pack dispatches and the original scans.
Retain dense-tail prefill and the qualified Steel kernels.
[Two-head review](../artifacts/mlx-gpu-gap-2026-11-15/sol62-nanbeige-dense-tail-next-1/ACTUAL-REVIEW.md) ·
[QMM review](../artifacts/mlx-gpu-gap-2026-11-15/sol62-nanbeige-qmm-a-pack-screen-1/ACTUAL-REVIEW.md).

The LFM3 all-disabled control passes all 12 original 1% limits. Astra
reconstructs 64 resets and 64 generations over eight fresh owners with zero
exports. The largest upper bound is 0.96339% for reverse-order TTFT.
Peak guarded charge is 3.72 GiB. Native prefill accounts for most of the
nominal TTFT shifts. The prior enabled/disabled marginal TTFT upper is
1.06524%. A real comparison after an evidenced change still needs all gates.
[LFM3 control review](../artifacts/mlx-gpu-gap-2026-11-15/root-lfm3-o4-sham-readiness-1/ACTUAL-REVIEW.md).

The detached late LFM3 collector keeps the startup clock and 1,064,960 B
allowance and allocates after Metal attachment. The original fixed-N
run passes 8 of 12 checks across 64 exact generations, 64 resets and four
exports. ABBA paired/marginal upper bounds are 1.04969%/1.06865% for TTFT
and 1.56739%/1.17209% for completion. BAAB passes all six. The strict
cancelled-work smoke remains failed and the separate exact resume check
passes. Keep the port detached.
[Late collector review](../artifacts/mlx-gpu-gap-2026-11-15/sol62-lfm3-o4-summary-late-owner-2/ACTUAL-REVIEW.md).

Gemma’s original BF16 loss block 9 matches all 784 archived scores and the
terminal row. A same-tape scalar-attention control matches 150 physical and
150 chronological K/V planes. The 25 cuts give 240 Metal and 635 CPU visits,
including the T17 tail, under a 4.48 GiB guarded peak. Formatting, strict
Metal Clippy, locked release linking and the geometry test pass. Astra
approves the block. Blocks 0 and 9 cover 1,808 of 10,000 original targets.
[Gemma block 9 proof](../artifacts/mlx-gpu-gap-2026-11-15/sol62-gemma-bf16-t32-quality-block9-1/RESULT.md).

The remaining Gemma loss trial matches block 1, then stops after the first
cut of block 2. The two-cut witness isolates one masked layer-11 score.
The ordered binary64 sum crosses the FP32 midpoint while the Metal
expansion rounds the exact dot to the other adjacent word. Input, probability,
context and committed K/V checks match. The strict score gate fails.
[Astra diagnosis](../artifacts/mlx-gpu-gap-2026-11-15/sol62-gemma-bf16-t32-quality-block2-audit-1/ACTUAL-REVIEW.md).

The test-only ordered CPU score and Metal context variant reproduces all
64 archived rows, passes 30 all-plane checks and preserves the committed
prefix through two malformed requests. The guarded peak is 4.48 GiB.
Formatting, strict Metal Clippy, locked release, geometry and eight CPU
span tests pass. Astra approves the scoped proof. Public latency and full
quality qualification remain open.
[Astra repair review](../artifacts/mlx-gpu-gap-2026-11-15/sol62-gemma-bf16-t32-quality-block2-repair-1/ACTUAL-REVIEW.md) ·
[Repair proof](../artifacts/mlx-gpu-gap-2026-11-15/sol62-gemma-bf16-t32-quality-block2-repair-1/trial-1/normal.json).

The detached ordered CPU-score and Metal-context CLI clears the public
384/128 point screen. Completion falls from 39.176 s to 36.076 s, first ID
from 11.970 s to 9.027 s and decode from 27.065 s to 26.909 s. Both owners
match all output IDs and final state at a 4.66 GiB guarded peak. Astra
confirms the one-pair result, strict Metal Clippy and fresh release link.
[Public timing result](../artifacts/mlx-gpu-gap-2026-11-15/sol62-gemma-bf16-t32-context-public-screen-1/RESULT.md).

The ten-block rerun rejects the timed Gemma variant. The owner matches
6,240 original target records and six terminal rows, then fails on one
visible context word in block 6. The same 101 products put the exact dot
just below the FP32 midpoint and the ordered binary64 sum on it.
Astra verifies both results and clean process teardown at 4.52 GiB. Hold the
C567 pilot and retain the original score and context gates.
[Corpus review](../artifacts/mlx-gpu-gap-2026-11-15/sol62-gemma-bf16-t32-quality-repaired-corpus-1/ACTUAL-REVIEW.md).

CPU context tiling matches all 196,608 published D256 and D512 words and
the captured block-6 dot. Twelve warmed alternating pairs give medians
of 21.925 to 7.180 ms and 43.685 to 13.990 ms at those shapes. Astra
approves the 81.2 MB guarded operator result. End-to-end timing is open.
[CPU tile review](../artifacts/mlx-gpu-gap-2026-11-15/sol62-gemma-bf16-cpu-context-tile-1/ACTUAL-REVIEW.md).

The integrated tile matches all 128 archived block-6 logit rows and 120
physical K/V planes in independent sessions, including the failed cut.
The guarded proof passes at 4.50 GiB. In a fresh 384/128 pair, Done falls
38.952 to 32.456 s, first ID falls 12.046 to 9.020 s, and engine decode
falls 26.771 to 23.302 s. All output and state match at 4.65 GiB peak.
The frozen reader refuses the expected call split. The separate reader
derives and verifies 180 multirow and 4,685 single-row calls from the
sealed graph. Astra approves only the descriptive one-pair comparison.
[Block 6 proof](../artifacts/mlx-gpu-gap-2026-11-15/sol62-gemma-bf16-cpu-context-block6-1/ACTUAL-REVIEW.md) ·
[Public tile result](../artifacts/mlx-gpu-gap-2026-11-15/sol62-gemma-bf16-cpu-context-public-screen-1/RESULT.md).

The direct tiled CPU route passes Gemma’s full original loss gate. Ten
fresh sessions match all 10,000 archived targets and ten terminal rows.
Mean NLL is 9.964929 versus official 9.995776, a difference of −0.030847
against the +0.05 limit. Both readers verify 322 cuts, 11,270 tiled calls
and 3,540 K/V digests at a 4.51 GiB guarded peak. Astra approves the loss
gate. The same checked executable provides the separate C567 task entry.
[Tiled loss review](../artifacts/mlx-gpu-gap-2026-11-15/sol62-gemma-bf16-cpu-context-quality-1/ACTUAL-REVIEW.md).

The original C567 pilot passes on the checked loss executable. Three task
outputs match the archived trajectories. A fresh scalar owner matches all
66 sampled decisions, including hidden stops. Astra verifies 2,625 tiled
visits and clean guards at 4.52 GiB peak. The remaining 97 generations,
full replay and isolated grade stay open.
[C567 pilot review](../artifacts/mlx-gpu-gap-2026-11-15/sol62-gemma-bf16-cpu-context-quality-1/PILOT-ACTUAL-REVIEW.md).

The same tiled release test binary completes the original 100 cases.
All fresh task trajectories match the scalar archive, and an independent
scalar owner matches all 4,149 sampled decisions. Canonical assembly
recovers from a Python import collision without repeating inference.
The isolated grader gives 51/100 versus official 52/100, within the
two-point limit. Astra verifies all cases and the stopped guest state.
[Task actual review](../artifacts/mlx-gpu-gap-2026-11-15/sol62-gemma-bf16-cpu-context-quality-1/TASK-ACTUAL-REVIEW.md) ·
[C567 task result](../artifacts/mlx-gpu-gap-2026-11-15/sol62-gemma-bf16-cpu-context-quality-1/TASK-RESULT.md).

Four same-binary Gemma owners complete 52 exact warmed 384/128 requests.
Completion falls 38.566 to 31.704 s and 38.497 to 31.835 s. Paired lower
savings bounds of 17.42% and 17.10% clear the 5% target. Both first-token
and decode medians improve at 4.67 GiB guarded peak. Astra verifies the
original launch refusal, corrected binding and all four owner receipts.
[Warmed actual review](../artifacts/mlx-gpu-gap-2026-11-15/sol62-gemma-bf16-cpu-context-public-warmed-1/ACTUAL-REVIEW.md).

The selected LFM3 512/32 diagnostic matches the archived output, state
and reset at 3.45 GiB guarded peak. Exclusive host intervals account for
4.702397 s of its 4.702400 s prefill. Resident FFN takes 1.612 s, ConvOut
0.857 s, ConvIn 0.475 s and norm/residual 0.467 s. Astra verifies the
attribution and nested waits. The original overhead gate remains open.
[LFM3 attribution result](../artifacts/mlx-gpu-gap-2026-11-15/sol62-lfm3-release4b-prefill-attribution-1/RESULT.md).

The ordinary Gemma BF16 CLI selects exact tiled context for both
submission policies. Both 384/128 smokes preserve the tape, cancellation,
reset and replay. The 84-case suite records 76 passes and the eight
original unavailable or inapplicable entries. The reader verifies 28
real exports and all parity controls. Peak native charge is 4.65 GiB.
Formatting, three strict Clippy variants, 871 default and 1,076 Metal
tests pass. The initial missing model path is rebound to verified
identical live files. Astra approves the production patch.
[Final production review](../artifacts/mlx-gpu-gap-2026-11-15/sol62-gemma-bf16-cpu-context-production-1/FINAL-REVIEW.md).

Production Gemma passes all 24 original 1% timing checks over both
submission policies and both owner orders. The 16 owners reproduce all
128 generations and resets. Eight enabled owners export 16 verified
records each, and disabled owners produce zero exports. Maximum paired
and marginal upper bounds are 0.757492% and 0.713923%. Collector capacity
is 1,041,192 logical bytes against 1,064,960 admitted. Guarded peak charge
is 4.68 GiB. Astra verifies the original readers and raw receipts.
[Production Gemma O4 review](../artifacts/mlx-gpu-gap-2026-11-15/sol62-gemma-production-o4-readiness-1/run-1/ACTUAL-REVIEW.md).

A detached Gemma T1 command groups Gate and Up at the two real matrix
geometries. Original FP32 outputs, committed state and late-Up retry
match. The 15:20 weighted paired savings are 0.257/0.224 ms, with lower
bounds of 0.227/0.210 ms against 0.300 ms. Keep the released projection
path. Guarded sampled peak is 4.65 GiB.
[T1 Gate/Up review](../artifacts/mlx-gpu-gap-2026-11-15/sol62-gemma-bf16-t1-gate-up-price-1/ACTUAL-REVIEW.md).

Gemma's detached complete T1 FFN stops on one FP32 ULP before timing.
CPU replay locates the expected Down word at L0 index 0. The guarded
untimed trace matches Gate and Up and first diverges at GELU product
index 2. The tolerance certificate accepts the product with zero status.
Forced replay and integrity-failure checks preserve exact state.
[CPU Down replay](../artifacts/mlx-gpu-gap-2026-11-15/root-gemma-t1-down-cpu-witness-1/RESULT.md) ·
[FFN chain review](../artifacts/mlx-gpu-gap-2026-11-15/sol62-gemma-bf16-t1-ffn-chain-attribution-1/ACTUAL-REVIEW.md).

Exact dyadic replay of the certified table bounds yields zero unique
mode-0 products and zero of two full calls. It predicts 1,635 L0 and
2,907 L15 product mismatches. One L15 constant lane qualifies. Astra
verifies the offline result and keeps the table path detached.
[Exact rounding review](../artifacts/mlx-gpu-gap-2026-11-15/sol62-gemma-bf16-t1-ffn-exact-rounding-1/ACTUAL-REVIEW.md).

One T1 command for Q, K and V matches both real Gemma producer shapes.
All 104 calls, late-V faults and exact retries pass at 4.64 GiB sampled
peak. The 12:3 weighted paired lower bounds are 0.437/0.460 ms. Astra
advances a connected grouping screen with Gate/Up using the original
384/128 workload and direct completed-request clock.
[Q/K/V review](../artifacts/mlx-gpu-gap-2026-11-15/sol62-gemma-bf16-t1-qkv-grouped-price-1/ACTUAL-REVIEW.md).

Connected T1 grouping matches 139 checkpoints, 128 IDs and exact physical
K/V at capacity 512 and chunk 32. It removes 8,255 decode waits. Four
directed faults preserve committed state after the fixture correction.
The first ordinary pair saves 1.645 s to Done and adds 37.5 ms to first
token at 4.68 GiB guarded peak. Astra advances the next warm screen after
small-chunk eligibility fallback. The result remains a point screen.
[Connected review](../artifacts/mlx-gpu-gap-2026-11-15/sol62-gemma-bf16-t1-projection-group-connected-1/ACTUAL-REVIEW.md).

The capacity-safe Gemma grouping passes both exact native witnesses.
Chunk 1 falls back for 20 wide Gate/Up pairs and chunk 32 takes all 50
groups. All 52 warmed requests match at 4.69 GiB guarded peak. Paired
Done lower bounds of 1.134/1.711 s miss the both-order 1.3 s gate.
Decode improves in both orders and prefill direction varies. Keep
the grouping candidate detached.
[Warmed review](../artifacts/mlx-gpu-gap-2026-11-15/sol62-gemma-bf16-t1-projection-group-warmed-1/ACTUAL-REVIEW.md).

Gemma BF16 compact single-row staging uses two guarded 12,288-word
banks for non-head decode projections. The 98,360 B admission retains
349 dispatches and 277 waits per decode. The exact witness matches 139
checkpoints, all 128 IDs and four fault controls. The two warmed orders
save 1.945/2.093 s at the median with 1.893/1.945 s paired lower bounds.
All 52 requests match at 4.68 GiB peak. The six required Rust checks pass.
C567 matches 4,149 natural decisions and the archived grader bytes for
51/100 versus 52/100. An ordinary compact summary export passes schema
and reset checks. Timing qualification covers the original local workload.
[Exact and point result](../artifacts/mlx-gpu-gap-2026-11-15/sol62-gemma-bf16-t1-compact-1/RESULT.md) ·
[Adoption review](../artifacts/mlx-gpu-gap-2026-11-15/sol62-gemma-bf16-t1-compact-warmed-1/ACTUAL-REVIEW.md) ·
[Task carry-forward](../artifacts/mlx-gpu-gap-2026-11-15/sol62-gemma-bf16-t1-compact-task-test-1/RESULT.md).

Production compact Gemma passes all 12 original 1% timing checks. The
eight owners match 64 generations and 64 resets. Four enabled owners
export 16 verified records each, with zero disabled exports. Maximum
paired and marginal upper bounds are 0.508181% and 0.527368%.
Collector capacity is 1,041,192 B within 1,064,960 B admitted. Peak
guarded charge is 4.70 GiB. The compact policy has its own receipts.
[Compact O4 review](../artifacts/mlx-gpu-gap-2026-11-15/sol62-gemma-compact-o4-1/ACTUAL-REVIEW.md).

Keeping the last T128 ConvOut cut scalar preserves all 14 LFM3 output
tapes and terminal states. The first three cuts use 30 ordinary and nine
exception panels at 3.49 GiB peak. Prefill saves 629/621 ms. Engine decode
adds 58/311 ms and client decode adds 59/312 ms. Both orders fail the
decode hold, so the standalone partition stays detached.
[Partition review](../artifacts/mlx-gpu-gap-2026-11-15/sol62-lfm3-release4b-convout-partition-1/ACTUAL-REVIEW.md).

The four-route ConvOut and sampled ConvIn FMA4 test matches all 28
output and state records. Adding ConvOut to FMA4 improves both decode
clocks in every owner. Adding FMA4 to ConvOut raises TTFT in three of four
owners, with median engine/client increases of 34.5/33.9 ms. Astra
verifies the 8-byte accounting correction and original first-owner
receipt. The mixed TTFT keeps composition detached.
[Factorial review](../artifacts/mlx-gpu-gap-2026-11-15/sol62-lfm3-convout-convin-factorial-1/ACTUAL-REVIEW.md).

[Layer-zero probe](../artifacts/mlx-gpu-gap-2026-11-15/root-nanbeige-ollama-layer0-1/normal.json).

The full untimed vLLM dense substitution exercises 1,540 visits and all
32 IDs at 3.78 GiB. It reproduces all 44 direct MLX prefill and terminal
cache pairs. The 31 decode heads are exact. The selective prefill head
differs in 36,106 words, with 0.125 maximum absolute error. A separate
guarded replay at 3.76 GiB confirms the cause for this row. The live
and shadow one-row heads agree. The full 128-row head uses the same
retained hidden row and exactly matches all 166,144 oracle words.
The installed paged primitive differs on about 2.35 million of the
146.79 million output words evaluated on identical dense-trajectory
inputs. External timing qualification remains open.
[Guarded vLLM result](../artifacts/mlx-gpu-gap-2026-11-15/sol62-nanbeige-vllm-full-dense-attention-1/normal-1.json) ·
[Head shape witness and Astra review](../artifacts/mlx-gpu-gap-2026-11-15/sol62-nanbeige-vllm-first-head-shape-1/ACTUAL-REVIEW.md) ·
[Astra external numerical review](../artifacts/mlx-gpu-gap-2026-11-15/sol62-nanbeige-qmm-connected-1/ACTUAL-REVIEW.md).

Independent direct MLX and private vLLM owners complete all four T128
bodies and 31 decode evaluations with exactly 32 one-row heads. Both
match every sampled ID and both sets of 44 cache hashes. Peak is
3.18 GiB for direct MLX and 3.86 GiB for private vLLM. The vLLM witness
uses real scatter and sampling with one dense attention result per visit.
The lean timed path retains that route independently of recording and
removes the inherited head observer. The 52 requests complete in
3.312–3.316 s for direct MLX and 3.631–3.640 s for private vLLM.
Peak rises from 3.18 GiB to 3.92 GiB across these two runtime modes.
The medians differ about 45 ms at TTFT and 275 ms after the first ID.
Toklet's exact 32-head route still needs to join this timing protocol.
[Matched work and Astra review](../artifacts/mlx-gpu-gap-2026-11-15/sol62-nanbeige-mlx-matched-32-1/ACTUAL-REVIEW.md) ·
[Matched private clocks and Astra review](../artifacts/mlx-gpu-gap-2026-11-15/sol62-nanbeige-mlx-matched-32-timing-1/ACTUAL-REVIEW.md).

The first same-weight LFM3 FFN replay changes 27 of 2,048
canonical residual words on the first affected row and fails
the block's exact admission. The separate canonical T1 ConvIn
four-output route matches all 6,144 words, passes eleven fault
and fallback checks and yields isolated lower estimates of
0.443 and 0.510 s over the original 403 visits. Connected
confirmation is required because the isolated baseline is slower
than the release's mean wait.
[Exact numerical replay](../artifacts/mlx-gpu-gap-2026-11-15/astra-lfm3-external-arithmetic-row-1/RESULT.md) ·
[ConvIn evidence](../artifacts/mlx-gpu-gap-2026-11-15/root-lfm3-t1-convin-fma4-1/RUN.md).

The connected LFM3 ConvIn trial passes all 52 exact requests
at 3.56 GiB peak. Decode medians improve from 2.376 to 2.020 s
and from 2.338 to 1.996 s. Complete requests improve from
7.057 to 6.651 s and from 6.955 to 6.633 s. The paired lower
bounds are 4.48% and 4.05%, closing the standalone candidate
below the 5% retention threshold.
[Connected ConvIn results](../artifacts/mlx-gpu-gap-2026-11-15/sol62-lfm3-t1-convin-connected-1/RUN.md).

Nanbeige's first-decode vLLM witness matches the complete incoming
slot-zero cache and post-RoPE query for the replay. Installed paged
SDPA repeats its own 6,144 words exactly. Dense MLX on the same
inputs changes 1,543 words and in a private replay restores all
2,048 slot-one append words from the independent reference.
The live overlay still reproduces all 32 IDs and 44 terminal owners
at 3.79 GiB. This isolates one decode boundary. Astra verifies all
28 raw panels and both operators' controls.
[First-decode attention review](../artifacts/mlx-gpu-gap-2026-11-15/sol62-nanbeige-vllm-first-decode-attention-1/REVIEW.md).

Two independent local vLLM Metal overlay owners complete 26
Nanbeige requests and both untimed work witnesses. First ID takes
2.489 s and completion 3.539–3.540 s at 3.76 GiB peak.
The measured fused Toklet experiment runs at about 4.36 s and
2.47 GiB. These diagnostics use the original pinned tape and
different head work and attention arithmetic. The external
comparison remains open.
[Raw clock and work evidence](../artifacts/mlx-gpu-gap-2026-11-15/root-nanbeige-vllm-direct-clocks-1/RUN.md) ·
[Astra review](../artifacts/mlx-gpu-gap-2026-11-15/root-nanbeige-vllm-direct-clocks-1/REVIEW.md).

The real detached Nanbeige CLI reduces head work from 35 to 32
completed scores on the pinned 512/32 request. Both modes match all
32 IDs, the original JSON schema, all 44 terminal K/V pairs and the
full BF16 logit digest. Cancellation after a completed unscored body,
an injected status error and real broken stdout pass. The guarded
peak is 2.39 GiB. Qualification covers this exact CLI work and state.
[Real CLI validation and review](../artifacts/mlx-gpu-gap-2026-11-15/sol62-nanbeige-head-cli-validation-1/REVIEW.md).

On Nanbeige, the detached 512/32 stage run passes all 12 requests,
14,928 sampled stages and four exact terminal state witnesses at
2.52 GiB. Two one-encoder controls measure about 3.50 s of GPU
prefill. Splitting encoders for timestamps increases the GPU clock
2.89–2.97%, with 98.01% coverage. In both orders the FFN projections
take about 2.16 s, query and attention output take 0.85 s, GQA takes
0.205 s and all four vocabulary projections take 50 ms. Astra
confirms the stage ranking. Seven offline clock and layout controls
pass. The ranked work determines the next kernel comparison.
[Stage report](../artifacts/mlx-gpu-gap-2026-11-15/root-nanbeige-prefill-profile-1/RUN.md) ·
[Astra review](../artifacts/mlx-gpu-gap-2026-11-15/root-nanbeige-prefill-profile-1/ACTUAL.md).

Direct MLX query and output reproduce all 1,179,648 archived
Nanbeige BF16 outputs. The combined isolated clock difference
extrapolates to 0.167 s at 176 visits. That falls below the
0.229 s development screen. Prior exact MLX FFN clocks leave a
larger estimated interval of about 0.45 s, making gate and up
the next connected optimization target.
[Matched projection clocks](../artifacts/mlx-gpu-gap-2026-11-15/root-nanbeige-prefill-profile-1/price-1/checked-normal.json).

The detached Nanbeige one-shot path scores 32 rows across 35 forwards.
It reproduces all 32 IDs and all 44 native terminal cache digests,
including 543 committed positions and one pending token. The vLLM
overlay records all 88 K/V plane digests at 3.75 GiB peak guarded
memory. The two engines' full cache digests differ. Across the
31 captured append rows, vLLM and independent MLX reproduce both
slot-zero planes exactly and diverge from slot one. Use that boundary
for the next arithmetic comparison. The captures establish matching
work and state coverage for these samples. Native model qualification
and external performance remain open.
[One-shot and terminal checks](../artifacts/mlx-gpu-gap-2026-11-15/sol62-nanbeige-head-state-1/RUN.md) ·
[Sealed vLLM report](../artifacts/mlx-gpu-gap-2026-11-15/sol62-nanbeige-head-state-1/live-normal.json).

The Nanbeige M64 gate/up experiment halves the logical Q4 reconstruction
count and matches all 2,752,512 BF16 words in each arm. Completed paired
GPU medians rise from 8.38–8.39 to 8.92–8.93 ms, with all 20 pairs
slower. Strict Clippy, release build, both analytical bases, nonfinite
and canary controls, dual readers and Astra review support closing
the experiment.
[Wider tile screen](../artifacts/mlx-gpu-gap-2026-11-15/sol62-nanbeige-gate-up-m64-1/RUN.md).

The LFM3 same-owner crossover records 64 requests, 6,144 command
brackets and exactly 64 enabled summary records at 3.67 GiB. Active
collection shifts mean TTFT by −18.62 and +2.98 ms in the two orders.
Completion shifts +8.03 and +16.02 ms. After first token, native
evaluation accounts for +26.71 and +12.95 ms. Completed decode FFN
GPU time contributes +14.01 and +10.37 ms. The engine's other
after-first-token work differs by at most 0.13 ms. This
attribution keeps the original 1% request-summary gate open.
[Balanced crossover](../artifacts/mlx-gpu-gap-2026-11-15/sol62-lfm3-o4-observer-crossover-1/run-1/normal.json) ·
[Decode components](../artifacts/mlx-gpu-gap-2026-11-15/sol62-lfm3-o4-observer-crossover-1/run-1/post-first-normal.json) ·
[Astra review](../artifacts/mlx-gpu-gap-2026-11-15/sol62-lfm3-o4-observer-crossover-1/REVIEW.md).

The Nanbeige T128 layer-zero GQA baseline verifies 2,621,440 K/V
words and all 104 timed outputs. The four-cut median sum falls
from 6.203 ms to 4.654 ms when run in reverse order. The reverse
misses the 5.250 ms zero-cost screen for the L0×44 projection.
Astra approves retaining the baseline and closing the isolated trial.
[Four-depth baseline](../artifacts/mlx-gpu-gap-2026-11-15/sol62-nanbeige-gqa-offsets-1/RUN.md).

The resident LFM3 convolution handoff matches all 1,050,624
gated BF16 words and 61,440 private history words across five
adjacent cuts. The best conditional lower prefill savings with the
same exceptional ConvOut panel are 0.112–0.121 s over 52 eligible
visits. Both miss the 0.452 s development screen. The corresponding
released scalar comparison is slower in both orders, and the T1
comparison gives mixed timings. Astra approves closing the trial.
[Resident handoff](../artifacts/mlx-gpu-gap-2026-11-15/sol62-lfm3-connected-conv-chain-2/RUN.md).

The detached LFM3 O4 diagnostic passes all 64 requests and
6,144 Metal command brackets at 3.72 GiB. After removing resident FFN
brackets and serial T128 Q/K/V waits, remaining engine prefill
contributes 52.95% and 77.09% of observed TTFT variance. Mean TTFT
with summary enabled shifts +40.54 and −46.05 ms across the two orders.
The FFN submit interval shifts +11.17 and +4.89 ms. Astra verifies
the physical and wire attribution. The original 1% overhead gate
remains open. The detached build, two offline readers and five
negative clock controls pass. Retiring two authenticated public
parent shards of a closed Qwen4B investigation recovered 8.45 GiB.
[Diagnostic and checks](../artifacts/mlx-gpu-gap-2026-11-15/root-lfm3-o4-brackets-1/RUN.md) ·
[Astra review](../artifacts/mlx-gpu-gap-2026-11-15/root-lfm3-o4-brackets-1/REVIEW.md).

### LFM3 Metal sampled decode

The GPU Q/K/V and dedicated FFN kernels run on sampled single-token
LFM3 decode. On the M1 Pro, 52 warmed BF16 512/32 requests reduced
median client completion by 21.96% and 21.46% in opposite orders.
Engine decode saved about 1.95 s in each order. Peak guarded charge
was 3.683 GiB. Marginal first-token time improved by 32.84 and 3.12 ms.
The paired changes were −26.50 and +2.23 ms.

Formatting, strict CPU and Metal Clippy, 865 CPU tests and 1,070 Metal
tests pass. Physical state, rollback, budget and public SIGINT checks
pass. The 10,000-target loss check has zero greedy differences against
the modified reference, and the 100-task result matches its 37/100.
Astra approves the exact change. Cross-engine and request-summary
qualification remain open.
[Warmed evidence](../artifacts/mlx-gpu-gap-2026-11-15/sol62-lfm3-t1-gpu-qkv-ffn-product-warm-1/normal.json) ·
[Task gates](../artifacts/mlx-gpu-gap-2026-11-15/sol62-lfm3-t1-gpu-qkv-ffn-s2-tasks-4/run-1/normal.json).

The new [competitive MLX GPU plan](mlx-gpu-performance-plan.md) starts an active
Gemma/LFM3 goal. Static/profile diagnosis identifies CPU-only LFM3, hybrid Gemma,
277 batched-decode command/wait boundaries, compensated dot projections and missing
cross-token prefill tiling. Summary callbacks and pre-terminal JSON work are
measurement targets; observed drift prevents a causal 10.72% collector claim.
G0–G6/O4 freeze matched artifacts/work/cache/clocks, qualify resident GPU blocks
and matrix kernels, reduce telemetry work and require actual 5% wins plus the
original 1% overhead gates. No new implementation or speedup is claimed. All
440 relative links/26 anchors and whitespace pass; Astra approves the planning
scope with no findings. Prior numerical/model rejections and UI/iPhone deferrals
remain binding. [Evidence](../artifacts/mlx-gpu-gap-2026-11-15/planning-checks-1.json).

The [SenseNova-Loop-9B text plan](../PLAN.md#sensenova-loop-9b-acceptance)
pins the official release and public MLX conversion. Twenty targeted checks
pass, including second-pass rollback, BF16 rotary and 24,576 exact layer-4
router words. The CPU scalar CLI builds and passes six console preflight
cases. The optimized native test authenticates both original shards,
passes real prefill, cached decode, cancellation and exact reset replay,
and peaks at 5.11 GiB for capacity 4. Same-weight stock MLX uses 5.62 GiB
and all three greedy IDs match across the 744,960-logit diagnostic. Both
independent readers agree. Astra confirms the binding seal and the
same-weight diagnostic. The CLI reports first flushed output, committed-token
throughput and KV, recurrent, convolution and scratch bytes. Strict checks,
864 CPU and 1,069 Metal tests, the optimized CLI build, and five final CLI
preflights pass after a disk-floor retry. Astra approves experimental use. The
successful whole-file text runtime awaits a complete on-disk snapshot.
[Checks and limitations](../artifacts/sensenova-loop-9b-2026-11-18/native-progress-15-experimental-reviewed.json).

The Gemma per-tile BF16 disabled/disabled ABBA control completes 32
requests across four guarded owners at 4.69 GiB peak. Both readers agree.
Marginal 95% upper bounds reach 22.63% TTFT and 28.44% completion.
In the second block, median TTFT falls from 8.314 s to 6.821 s and
completion from 45.570 s to 35.832 s across identical routes. Native
prefill and decode move with the public clocks. Hold the O4 campaign.
[Control results](../artifacts/mlx-gpu-gap-2026-11-15/sol62-gemma-bf16-per-tile-o4-readiness-1/run-1/read-normal.json) ·
[Native phase split](../artifacts/mlx-gpu-gap-2026-11-15/sol62-gemma-bf16-per-tile-o4-readiness-1/run-1/phase-localization.json).

The [current Gemma cost map](gemma-current-cost-map.md) records four selected
GPU frames and two main-thread sample windows from guarded 384/128 work.
CPU attention has 46.68%/28.77% sample share before first token/during decode;
completion waits include GPU execution and retain 278 prefill / 277 decode
boundaries. The diagnostic fix exposes an explicitly committed-prefix view
without changing ordinary sampling eligibility, kernels or scheduling. Required
fmt/strict CPU+Metal Clippy, 847 CPU / 1,019 Metal tests (53/365 ignored) and
release build pass; actual first-prefill parity checks 262,144 logit values.
The failed attempt is preserved. Astra approves the scoped fix/evidence with
no findings; [publication](../artifacts/mlx-gpu-gap-2026-11-15/g1-publication-1.json)
is tracked separately. Summary cost attribution and the original overhead/
competitive gates remain open.

The genuine-record [summary component probe](gemma-current-cost-map.md#summary-component-measurements)
retains 280 measured/84 warmup samples across two processes and reproduces the
captured summary/manifest exactly. Confirming full insertion medians are
3.569/3.555 us; token accounting is about 133 ns per token, excluding decoder and
I/O bodies. Logical collector/ring storage stays 9,000/1,032,192 bytes; guard and
expected overwrite accounting pass. No production code changes or new overhead
qualification follow. Defer the typed-ring rewrite pending a useful request-scale
effect and prioritize accurate GPU attention. All three original overhead
failures remain binding. All 422 relative links and whitespace pass; Astra
approves the component evidence and decision with no findings.

The artifact-only [G2 attention feasibility probe](gemma-current-cost-map.md#g2-attention-feasibility-follow-up)
passes 932,968 frozen and 376,320 saved real original-BF16 graph input values at
unchanged bounds, with exact masks and bitwise segment-layout agreement. Four
GPU fault injections and eight raw-reader corruption controls pass; all guards
close under 512 MiB. Four dependent dispatches use one command/wait without
intermediate readback. Production code is unchanged; reusable ownership,
integration, full quality/state and request-speed qualification remain open.
Astra identified a non-isolating overflow injection; its corrected masked-only
case and 42-pair certificate pass. Astra approves scoped feasibility after the fix, with no remaining findings.

The [reusable GPU attention workspace](gemma-gpu-attention.md) adds bounded
buffers, checked shared-ring views and caller-owned encoding, one command/wait,
guarded publication and permanent poisoning after execution failure. Required
fmt/strict CPU+Metal Clippy, 847 CPU/1,022 Metal tests and release pass. Explicit
GPU checks and 2,844,296 independent value comparisons pass; 32 warmed operations
allocate zero Rust memory. Astra approves scoped retention with no findings; the ordinary model graph,
full-block quality/state, overhead and external-speed gates remain open.
[Apple hardware focus](apple-silicon-gpu-goal.md) records parallel Astra research /
Sol implementation and keeps ANE conditional on precision, placement and costs.

The [full-finite GELU experiment](gemma-gpu-gelu.md) replaces the rejected direct
FP32 tanh candidate with a fixed 3,317,784-byte native-bound table. Independent
exhaustive scans cover all 4,278,190,080 finite FP32 gates; original cases, GPU
controls, exact whole-case replay and corruption checks pass. None of 24 saved
model cases requires normal fallback. Astra approves artifact-only retention;
production integration, full-model quality/state and performance remain open.

Shared RMS/rotary/attention encoders are published as `b5437a76`, with required
checks, explicit GPU/raw replay/allocation checks and Astra approval. All seven
code-head workflows/15 jobs pass. The ordinary graph and request speed remain
unchanged. [Implementation/evidence](gemma-gpu-attention.md).

The [resident producer-block pilot](gemma-gpu-producer.md) keeps all eight nominal
original cases on one completed command/wait with zero CPU replay. The max-only
prefills previously took ten commands/waits. Independent replay passes 847,360
original values, exact masking/prefix and transaction/fault controls under 6 GiB.
Fmt/strict CPU+Metal Clippy, 1,034 Metal tests and release builds pass; 847 unchanged
CPU tests are reused. Astra approves the combined scoped change with no findings.
The ordinary CLI graph, whole graph, allocation/request speed, summary overhead
and external wins remain unqualified.
[Review/publication](../artifacts/mlx-gpu-gap-2026-11-15/g2-producer-publication-1.json).

Checked multi-tile BF16 output is the next prerequisite. The frozen six-file
Sol draft preserves the original shader and proofs; targeted build/GPU and
independent replay pass 29 cases, failure controls and 32 zero-allocation warmed
chains. Complete-task checks/Astra remain pending while the deeper chain is
composed. This test-gated prerequisite does not change ordinary generation. [Progress](../artifacts/mlx-gpu-gap-2026-11-15/g2-bf16-strided-progress-1.json).

The old deeper-chain failure is retained: native replay isolated table GELU
interpolation for a one-key token, with separate later-token softmax drift.
The finer core passes exhaustive native/independent proof, GPU original/extra
cases, four raw readers and all corruption controls. Astra approves artifact
retention, with no coupled or production approval. The source8 integration and
optional completed-command GPU timestamps pass the original eight coupled
cells (2,375,680 values), state/RMS controls and 32 warmed zero-Rust-allocation
iterations. The source9 failure-count correction passes fresh fmt/Metal Clippy,
1,044 Metal tests, release and a pure error regression; CPU/CLI and numerical GPU
evidence reuse is explicitly bounded. Astra approves the combined bounded
engine change with both findings resolved.
[Publication/source/CI](../artifacts/mlx-gpu-gap-2026-11-15/g2-deep-chain-publication-1.json).
[Artifact review](../artifacts/mlx-gpu-gap-2026-11-15/astra-gelu-core-review-1/verdict.json),
[current engine checks](../artifacts/mlx-gpu-gap-2026-11-15/g2-deep-chain-progress-1.json).
The combined common-Q4 source5 owner/CLI passes release/lint, 1,051 Metal tests,
state/59 codec cases and CLI diagnostics; 850 CPU tests reuse 4,015 exact inputs.
Fresh original 10,000-target loss and independent full-vocabulary anchors pass
under 6 GiB; 100 tasks/replay/grade, stock precision and final engine review remain.
[Full-loss result](../artifacts/mlx-gpu-gap-2026-11-15/g0-common-q4-full-loss-analysis-prep-1/loss-result.json).

Pooled scratch source3 passes all new GPU numerical/lifecycle/warm/resident checks.
Old regressions and required fmt/strict Clippy, 1,047 Metal tests/release pass;
847 unchanged CPU tests are reused. Astra approves scoped code retention with
no required fixes. Pushed `14fbb1b5`; all 7 workflows/15 jobs pass.
[Scope](gemma-gpu-pool.md). A disk-floor interruption is
retained and 8.16 GB owned compiler cache reclaimed without deleting evidence.
[Hardware summary](../artifacts/mlx-gpu-gap-2026-11-15/g2-scratch-pool-hardware-summary-1.json),
[continuation](../artifacts/mlx-gpu-gap-2026-11-15/g2-scratch-pool-final-prep-1/continuation-handoff-v1.json).

The two-phase RMS scheduling experiment passes 336 cases with exact current-kernel
bit/status/reason parity and the original scalar gates. Two reversed aggregate
runs lower 22-norm GPU medians from 147.73/149.12 to 37.93/37.77 ms prefill and
131.67/131.43 to 38.08/37.50 ms decode. Astra approves artifact retention with no
required fixes. The source4 private graph integration passes 16 numerical cells,
28 lifecycle controls, 64 warm zero-Rust-allocation operations and all 12 guarded
independent readers. Both three-layer GPU timing runs show about 69% lower
prefill /68% lower decode time, adding 14,232 session bytes. All 32 final stages pass, including strict CPU/Metal Clippy, 1,050 Metal
tests (403 ignored), release builds and old regressions; 847 CPU tests reuse exact
unchanged inputs. Astra approves private graph retention with no required fixes.
Whole-model/request, quality, overhead and external wins remain unqualified.
[Retained graph scope](gemma-gpu-rms-scheduling.md),
[final checks](../artifacts/mlx-gpu-gap-2026-11-15/g2-rms-two-phase-code-review-prep-1/final-check-completion-1.json),
[combined code review](../artifacts/mlx-gpu-gap-2026-11-15/astra-rms-two-phase-code-review-1/verdict.json).
Published `0bfd68c3`; all seven workflows/15 jobs pass and all 4,209 compiled
inputs match committed bytes. [Publication/source/CI](../artifacts/mlx-gpu-gap-2026-11-15/g2-rms-two-phase-graph-publication-1.json).
[Current build](../artifacts/mlx-gpu-gap-2026-11-15/g2-rms-two-phase-graph-build-progress-4.json),
[hardware](../artifacts/mlx-gpu-gap-2026-11-15/g2-rms-two-phase-graph-hardware-progress-3.json),
[independent replay](../artifacts/mlx-gpu-gap-2026-11-15/g2-rms-two-phase-graph-readers-run-1/result.json).
[Combined artifact evidence](../artifacts/mlx-gpu-gap-2026-11-15/g2-rms-two-phase-artifact-review-checks-1.json),
[Astra approval](../artifacts/mlx-gpu-gap-2026-11-15/astra-rms-two-phase-artifact-review-1/verdict.json).

The next private borrowed-Q4 projection source3 passes fmt/strict CPU+Metal
Clippy/certified release unit and all 35 targeted GPU/reader/regression stages.
Warm32 allocates zero Rust memory; the observed peak is 505,348,096 bytes with
helper. Full Metal1052pass/409ignored and releaseCLI also pass, with847CPUtests
reused against4009 exact inputs. Astra approves the scoped private prerequisite
with no required fixes; no model/graph/route/performance adoption is qualified.
[Scope/review](gemma-gpu-q4-projection.md).
[Current checks/failure](../artifacts/mlx-gpu-gap-2026-11-15/g0-q4-borrowed-encode-progress-5.json),
[retained failure](../artifacts/mlx-gpu-gap-2026-11-15/g0-q4-borrowed-encode-progress-1.json).

The RMS reduction/root/output split passes original numerical/fault gates and
eight readers, but prefill is 6.657%/7.528% slower and decode improves less than
1%. Astra approves rejecting it as a performance candidate. The two-phase graph
remains; no additional pipeline/dispatch complexity is integrated.
[Measured rejection](../artifacts/mlx-gpu-gap-2026-11-15/astra-rms-three-phase-review-1/verdict.json).

The [LFM3 external request observations](lfm3-engine-comparison.md) record
78 requests across two 3+10 cohorts per engine, with preserved raw observations,
model/executable identities and clock/memory scope. Ollama lacks authoritative
IDs; vLLM matches exact prompt/cache input but only 3/13 output-ID/text rows
match CPU per cohort. No speedup ratio, quality or adoption is qualified.
All 26 genuine vLLM timing captures replay consistently. Source/guard/import
controls and scoped content checks pass; Astra approves scoped publication with
no findings. [Checks/review/commit](../artifacts/lfm3-engine-comparison-2026-11-11/comparison-publication-checks.json).

The [optional vLLM request timeline](lfm3-vllm-timeline.md) replays sealed captures
and separates server queue, scheduled-to-first, first-to-last and overall-rate
boundaries. Missing fields remain unavailable; inconsistent arithmetic withholds
derived decode rate. Twenty-four portable controls in both modes, five pinned-source
cases and nine guarded offline CLI cases pass under 512 MiB after correcting
audit/helper errors. Required checks bind 5,177 unchanged inputs and both 52-test
CI commands pass. Astra's empty-entry and stop-metadata findings are fixed;
follow-up approves scoped retention with no remaining findings; actual engine, quality
and performance qualification remain separate.

The [external HTTP benchmark observer](lfm3-http-metrics.md) validates bounded
Ollama/vLLM streams and separates client clocks, API counters and unavailable
state. Twenty-eight portable controls in both modes and twelve synthetic HTTP/CLI
cases pass within 512 MiB. All 26 genuine saved Ollama replies preserve counters,
text and original event clocks. Input mismatch and truncated streams cannot
qualify a comparison. Required checks bind to 5,177 unchanged inputs. Astra's
rejected-byte, counter-overflow and stop-ID findings are fixed, and scoped
retention is approved. [Review/commit](../artifacts/lfm3-engine-comparison-2026-11-11/http-observer-publication-checks.json).
No model startup,
engine overhead, quality, display or performance qualification follows.

The real LFM3 BF16 comparison reaches 1,025 positions with 68,054,016
exact source words and two late-abort retries. The source and native runs
stay within their respective memory caps. Ordinary one-token continuation
at the verified depth has a 1.285-second median on M1 Pro.
Astra approves scoped numerical retention. The complete original quality
results below pass; result review and managed adoption remain open.
[Scope and measurements](lfm3-context1025.md).

The original LFM3 quality campaign completes both 10,000-target loss pairs and
both 100-task comparisons under unchanged gates. Stock-to-modified NLL increases
0.00781 nats per token; task passes rise from 35 to 37. Native retains all
37 passes, all 10,000 greedy loss IDs and all 5,094 sampled task decisions, with
3.66e-15 maximum absolute NLL delta. Both isolated grading windows confirm the
guest stopped. The guarded loss retry stays within 6 GiB; interrupted output
remains unqualified. Astra approves scoped CPU qualification and retention;
managed adoption remains open.
[Complete decision and evidence](lfm3-quality.md).

The isolated LFM3 finite-widening experiment removes redundant BF16 conversion
work after complete public validation, preserving FMA/reduction/store order.
Fmt, strict CPU/Metal Clippy, 734 CPU/880 Metal tests, targeted release/allocation
checks, three native package slices and both C smokes pass. Full parity matches
854 observations/68,054,016 words, thirty wrapped views and two late retries.
Earlier failed checks, interrupted output and runner corrections remain in the
linked evidence. Full primary ABBA matches all 52 requests, including 40 measured;
medians 349.54→248.15 s and 349.58→246.60 s improve 29.0%/29.5%, above 0.63%
maximum median variation. The primary gate passes; final CPU retention is approved.

The full 156-request 128/512/1,024-input × 128-output matrix matches all work,
output/state and 4,992 sampled decisions per cohort, including 120 measured
requests. Paired median reductions are 21.3%/20.6%, 25.8%/25.9% and
27.8%/27.9%. All twelve sustained windows pass; no candidate median regresses.
Maximum separate RSS/physical guard charges including reserve are
3.82/3.74/3.74/3.74 GB; differences do not establish allocation savings.
Request-performance gates pass; full CPU retention is approved after quality/state checks.
CPU phase observations show 29.9–30.7% lower prefill medians and 16.2–17.2%
lower decode medians; operator-level attribution remains unqualified. Supplemental
observer metadata/unit errors are corrected with failures retained. The corrected
serial retry passes 55 portable controls per mode. Fresh compiler identity/new
loss build pass; native loss collection completes all 10,000 frozen targets and
ten blocks. Both guards pass; maximum observed outer guard charge is 4.15 GB
including reserve, within 6 GiB. Both loss gates pass; native matches all greedy
loss IDs with maximum NLL delta 3.66e-15 against the modified reference. All 100
tasks and decoding complete, matching all 5,094 sampled decisions and stops.
Isolated grading first stopped on an owned setuid `ps` inspection error. The native
process-group replacement and concurrent-working-set monitor pass 156 shared
tests and nine accounting controls in both Python modes, SDK/live checks and a
260-command inert stress check. All 14 isolation and 28 authored evaluator
controls then pass. The fresh retry completes all 80 isolated program runs,
retains 37/100 task passes for modified reference and native, verifies guest
shutdown and passes both complete quality comparisons. Maximum retry guard
charge is 2.48 GB including the 2 GiB guest reservation; CPU policy remains four
seconds. All 62 state checks and exact event/memory replay pass, including
cancellation/resume to the same 16 IDs/text and all six hidden-stop cases.
Maximum outer guard charge is 3.90 GB, with no survivors; these are functional
checks. Nineteen recovery-pipeline controls pass in both modes.
Astra approves CPU retention and supporting host changes with no actionable
findings. Commit/push references are recorded with publication. Build/assembly
limits and failures remain.
All 441 portable tests across 60 local modules pass in both modes under 512 MiB
after correcting interpreter/module-isolation invocations. A clean publication
worktree preserves root UI/workflow/history edits and adds ownership controls
to CI. Astra approves scoped retention; managed adoption remains open.
Fresh publication formatting and all 59 tests in the changed workflow pass;
all 444 numerical inputs match the 12 required Rust check receipts/logs.
The measured baseline is byte-exactly reproduced by a fresh build from clean
`2da16a9e` after state exit, within the 2 GiB build budget. The original cache
and binary are preserved; the original build's historical commit remains unproven.
The portable record preserves all 208 scalar samples and exact source replay;
initial report power-label/trial-identity assumptions are corrected with failures retained.
[Experiment scope / request measurements](lfm3-finite-widening.md),
[consolidated checks / retained failures](../artifacts/lfm3-finite-guard-2026-11-11/widen-checks-consolidated-1.json),
[primary analysis](../artifacts/lfm3-finite-guard-2026-11-11/native-primary-analysis-2.json),
[matrix execution](../artifacts/lfm3-finite-guard-2026-11-11/compatibility-execution-plan-3.json),
[full matrix analysis](../artifacts/lfm3-finite-guard-2026-11-11/compatibility-analysis-3.json),
[CPU phase observations](../artifacts/lfm3-finite-guard-2026-11-11/compatibility-phase-observations-1.json),
[closed receipts / fresh loss-build binding](../artifacts/lfm3-finite-guard-2026-11-11/compatibility-closure-checks-1.json),
[fresh quality execution](../artifacts/lfm3-finite-guard-2026-11-11/fresh-quality-execution-plan-3.json),
[closed 10,000-target collection](../artifacts/lfm3-finite-guard-2026-11-11/fresh-loss-collection-closure-checks-1.json),
[retained grading failure](../artifacts/lfm3-finite-guard-2026-11-11/fresh-quality-grading-failure-audit-1.json),
[guard stress qualification](../artifacts/lfm3-finite-guard-2026-11-11/grading-inert-guard-1/checks.json),
[fresh complete quality](../artifacts/lfm3-finite-guard-2026-11-11/fresh-quality-recovery-4/original-quality-decision.json),
[host requalification](../artifacts/lfm3-finite-guard-2026-11-11/host-requalification-1/result.json),
[recovery and regression checks](../artifacts/lfm3-finite-guard-2026-11-11/grading-recovery-checks-1.json),
[baseline byte-reproduction preparation](../artifacts/lfm3-finite-guard-2026-11-11/baseline-byte-reproduction-preparation-1.json),
[exact baseline reproduction](../artifacts/lfm3-finite-guard-2026-11-11/baseline-byte-reproduction-result-1.json),
[state execution](../artifacts/lfm3-finite-guard-2026-11-11/state-execution-plan-4.json),
[complete 62-check replay](../artifacts/lfm3-finite-guard-2026-11-11/state-analysis-4.json),
[complete pre-review qualification](../artifacts/lfm3-finite-guard-2026-11-11/finite-qualification-closure-1.json),
[review/commit](../artifacts/lfm3-finite-guard-2026-11-11/finite-publication-checks.json),
[portable measurements](../benchmarks/results/2026-11-12-lfm3-finite-widening-v1.json),
[current experiment](../artifacts/lfm3-finite-guard-2026-11-11/active-experiment-state-1.json).

Granite configuration admission is implemented and under qualification:
canonical default RoPE shares the strict Llama declaration parser; equal supported
dtype aliases are normalized only for Granite. Existing scale, shape, context,
quantization and index checks remain binding. Three pinned raw configs and frozen
CPU/Metal graphs cover the new dialects. Required fmt/both strict Clippy,
737 CPU/883 Metal tests, explicit GPU and 44 CLI cases pass; eight monitor
controls in both modes and an owned live control pass. Six failed attempts
remain retained; compiler-only admission is 3 GiB within the 6 GiB Mac ceiling.
Astra approves, no findings. [Scope](granite-config.md),
[portable checks](../benchmarks/results/2026-11-12-granite-config-v1.json).

Both pinned Granite tensor packages authenticate without model loading or index
repair. Full bounded affine4/8 conversion and saved-byte lineage audit pass:
282 matrices/81 norms, with all 3,659,530,240 matrix values compared exactly to
upstream FP32 during conversion. The offline audit replays hashes and exact
norms; it does not repeat floating-point reconstruction. All 48 CPU/GPU controls,
33 portable tests per Python mode and six actual-artifact CLI cases pass.
Observed conversion charge is 455,131,136 bytes within the 512 MiB audit cap.
Existing Metal metadata accepts 128/512 and rejects 1,024 positions; scalar FP32
exceeds 6 GiB. Earlier memory, index-field and CLI-expectation failures remain
retained. Evidence closure and Astra review pass, no findings. Legacy MiniCPM tools
remain unchanged; full-model fit, inference, quality and performance are open.
[Controls](../artifacts/granite-conversion-2026-11-12/controls-closure-2.json),
[full conversion](../artifacts/granite-conversion-2026-11-12/full-conversion-1/manifest.json),
[saved-byte audit](../artifacts/granite-conversion-2026-11-12/full-audit-1.json),
[CLI cases](../artifacts/granite-conversion-2026-11-12/child-inspection-2.json),
[review/commit](../artifacts/granite-conversion-2026-11-12/publication-checks.json).

The bounded mixed affine4/8 mechanism passes checks and Astra review.
The installed MLX default requests group-64 8-bit output/down projections and
untied head. The existing MiniCPM5 converter remains affine8/6,144-column only;
Granite requires 8,192 columns. No full payload, model inference, quality, physical
fit, request performance or managed adoption is qualified by config admission.
[Conversion constraints](../artifacts/granite-source-planning-2026-11-12/conversion-reuse-inspection-1.json).

Granite conversion `eb14db30` is pushed and both exact-head CI workflows pass.
Both complete tokenizer files authenticate; only the modified source's explicit
BatchLongest left-padding declaration differs from stock. All other tokenizer
fields match, with 100,352 IDs and 96 added-token IDs also in BPE vocabulary.
The first counting assumption fails and remains retained; corrected bounded
inspection passes. Five cold planning cases preserve the existing 512-position
Metal cap and admit the 128/128 request on Apple M1 Pro. The actual raw CLI smoke
passes with all 128 outputs, 255 committed positions and one pending token at
2,899,741,912 bytes observed charge within 6 GiB. Source/binary/input hashes close
unchanged and owned cleanup is empty. Debug clocks do not qualify performance;
full numerical, quality, codec, all-workload fit and adoption gates remain open.
[Codec preparation](../artifacts/granite-codec-2026-11-12/codec-inspection-2.json),
[cold planning](../artifacts/granite-runtime-plan-2026-11-12/runtime-plan-1.json),
[raw native smoke](../artifacts/granite-native-smoke-2026-11-12/native-smoke-1.json).

Granite's complete graph diagnostic rejects the current affine4/8 Metal route:
212/20,268,711,936 profile-coordinate comparisons fail unchanged per-array gates.
All 30 native runs/304 observed state checks pass, with exact schedule, reset and
causal-prefix bytes; the observed peak is 5.03 GB under 6 GiB. Independent
reference capture passes within 512 MiB. The first failure is reproduced from
native Q/K/V, pointing upstream; other failures remain unlocalized. Required
fmt/strict CPU+Metal Clippy, 737 CPU/887 Metal tests, 58 portable controls per mode,
1,120 fixture comparisons, the GPU fixture and admitted numeric controls pass.
Failed reference/launcher attempts are retained. Tooling retention and the
explicit non-adoption decision have Astra follow-up approval, with no remaining
findings. Published as `8a90db86`; both exact-head CI workflows pass. Its two required-
evidence findings are fixed: deletion controls and fresh genuine replay verify
all 304 checks/10 resets/40 receipts with byte-identical numerical records. The
genuine capture remains numerically rejected. Quality, performance, overhead
and managed adoption remain unqualified.
[Scope](granite-graph.md),
[decision](../benchmarks/results/2026-11-13-granite-graph-v1.json),
[strict comparison](../artifacts/granite-graph-2026-11-12/offline-comparison-2/manifest.json).

Ministral source preparation records the existing configuration's independent
3,072 hidden/4,096 query widths and the selected adapter's unsupported text tag.
Its source combines YaRN with absolute-position query scaling cast before SDPA;
independent fixtures must cover the 16,384-token boundary and cache offsets.
The pinned Q4 index rejection is retained; payload, conversion and runtime
qualification remain open. This inspection launches no model or compiler.
[Source inspection and remaining gates](../artifacts/ministral3-source-planning-2026-11-12/static-source-inspection-1.json). Current source pins are
revalidated after retaining/explaining Granite's changed Rust hash. At least
1.15 GB more than observed free space is needed even for complete parents plus a
theoretical all4 text child and disk floor, before format/exclusion overhead;
rebuildable compiler-cache cleanup precedes payload work. No model is launched.
[Current preflight](../artifacts/ministral3-source-planning-2026-11-13/input-revalidation-2.json).

Ministral preparation reclaims 1.66 GB observed disk space from 12,277 retired
Cargo intermediates; measured binaries, current caches and raw evidence remain
available. A frozen 11-case profile now precedes independent Rust/fixture work
on cached CPU Half YaRN and post-rotation query gain. The 1.21 MB independent
fixture, 24 portable controls per mode, six targeted Rust tests, fmt and strict
CPU/Metal Clippy and all 744 CPU/894 Metal-feature tests pass. Six release
primitive tests and zero-allocation key/query/rejection controls pass; optimized
Python regenerates identical bytes. Astra approves scoped retention with no
findings. Published as `7ea011a6`; both exact-head CI workflows pass. This mechanism
admits no model or Metal route; original downstream gates remain. Next work splits
Metal implementation and source/conversion planning, with exclusive file ownership
and one root-owned heavy-check queue.
[Execution plan](../artifacts/ministral3-yarn-2026-11-13/parallel-work-plan-1.json).
[Review/commit](../artifacts/ministral3-yarn-2026-11-13/publication-checks.json).
[Cache receipt](../artifacts/ministral3-source-planning-2026-11-13/cache-cleanup-result-1.json),
[prospective profile](../artifacts/ministral3-yarn-2026-11-13/prospective-primitive-protocol-2.json).

Ministral selected-config binding and test-integrated borrowed Metal YaRN
pass 755 CPU/907 Metal-feature tests, fmt/strict Clippy and three release GPU
diagnostics under unchanged gates. GPU output comparisons, exact replay/chunk/
rebase, numeric status/reuse and zero measured Rust encode allocations pass.
Four 3+10 operator cohorts are diagnostic only. A Cargo-kind receipt-finalization
error is retained and corrected without repeated checks. Astra approves scoped
retention with no findings. Published as `5c0eee96`; both exact-head CI workflows
pass. Model/source/graph/quality/request and O1-overhead qualification remain open.
[Checks/review](../artifacts/ministral3-metal-2026-11-13/publication-checks.json).

Ministral source acquisition authenticates four complete containers within the
512MiB setup policy after6.55GB observed compiler-cache reclamation. Stock's
optional index count remains rejected; modified-only mixed conversion passes
all3,428,843,520 matrix reconstructions exactly. All144 genuine controls and86
portable tests perPythonmode pass. The saved-byte auditor now validates rich
metadata identities and all4,252 tiles; failed checks stayretained. ActualCLI
preservesunknown_architecture. RequiredRust checks reuse4,641 unchangedinputs.
Astra approves scoped retention with no findings. Published as `83a8617a`;
all three exact-head CI workflows pass. Graph/codec/quality/fit/request/O1/
adoption remain open. The shared tokenizer JSON authenticates separately; raw
codec and differing chat profiles remain unqualified.
[Next codec preflight](../artifacts/ministral3-codec-2026-11-13/preflight-1.json).
[Conversion/checks/review](../artifacts/ministral3-conversion-2026-11-13/publication-checks.json).

The [parallel delivery plan](parallel-delivery.md) now assigns the active
Qwen3.5-2B milestone to typed-source, hybrid-graph and independent-reference lanes.
One integrator owns shared APIs, early integration, the single heavy-job queue,
combined Astra review and publication. The
[dispatch receipt](../artifacts/qwen35-2b-source-planning-2026-11-13/parallel-work-plan-1.json)
records file ownership, resource bounds and remaining gates. Track integration,
queue/review wait and repeated-check reasons in one ledger. The typed-source/tiny-graph slice passes formatting,
strict CPU/Metal Clippy, 781 CPU/940 Metal-feature tests, twelve portable controls
per Python mode and its explicit GPU storage check. Astra approves scoped
diagnostic retention with no findings. No workflow speedup is measured;
full-model qualification gates remain open.
[Review/publication](../artifacts/qwen35-2b-source-planning-2026-11-13/publication-checks.json).
[Diagnostic scope](qwen35-2b-typed-source.md).

The next 2B slice implements the pinned raw-ID hybrid CLI, complete phase
capture/reference protocols and bounded source/parent/conversion audits. Early
compilation, strict CPU/Metal Clippy and 786 CPU tests pass; real qualification
and combined Astra review remain pending. Fresh selected-source byte verification
passes after a retained monitor-interpreter failure. Parent history and the full
metadata directory are captured; tensor values are being compared under the
unchanged gates. Pinned-CA TLS retains certificate/hostname verification. Source
and reference controls pass 34/23 tests per explicit optimization mode; prior
mock/fixture failures and journal-reservation correction remain linked in their
receipts. No conversion, model, quality or performance result is inferred.
[Current source evidence](../artifacts/qwen35-2b-payload-2026-11-13/source-setup-qualification-1.json).

Ministral graph/CLI diagnostic code passes fmt, strict CPU/Metal Clippy,
763 CPU/922 Metal-feature tests and all five release GPU controls, including
eight zero-Rust-allocation tiny decode steps. The debug allocation failure is
retained. Full capture completes 22 runs/9,218 arrays and exact state/causal
checks, but 78/244,809,728 profile coordinates fail the unchanged numerical gate.
Console mode/phase/memory, request/error/cancel and custom-stop controls pass;
the private pre-child reader failure and corrected retry are retained. Current
route remains unqualified and does not advance to performance cohorts. Targeted
native-input replay traces the earliest projection failure to propagated inputs;
sampled operators pass their local gates without changing the full rejection.
Astra approves scoped diagnostic retention with no blocking code findings.
[Review/publication](../artifacts/ministral3-graph-2026-11-13/publication-checks.json),
[scope](ministral3-graph.md),
[decision](../artifacts/ministral3-graph-2026-11-13/graph-comparison-1/manifest.json).

Qwen3.5-2B needs a distinct 24-layer/16-value-head source profile. Its MLX
headers change eighteen gated-norm vectors and convolution layouts; normalization
offsets also depend on layout and source key prefixes. A stale historical Rust
hash fails the first static check; the corrected inspection pins current source
without changing existing 4B/9B gates. Declared-parent/payload authentication,
numerical/quality and runtime qualification remain open. No model or compiler
was launched.
[Source inspection and retained failure](../artifacts/qwen35-2b-source-planning-2026-11-12/static-source-inspection-2.json).

Completed LFM3/MiniCPM mechanism reports and prior disk-cleanup receipts
are preserved in the [12 November reviewed snapshot](archive/2026-11-12/README.md).
Exact inverse link reconstruction and local links pass; only 32 reviewed checked
entries and 106 narrative lines move out of working documents, reducing their
combined bytes 38.4%. Pending work, original user history and live PLAN stay intact.
[Validation](../artifacts/lfm3-finite-guard-2026-11-11/working-history-archive-checks-1.json).

## Retained performance decisions

| Change | Measured result and scope | Evidence |
| --- | --- | --- |
| LFM3 validated finite BF16 widening | Retained for CPU BF16: primary median requests fall 29.0%/29.5%, full matrix 20.6–27.9%; full original quality and 62 state checks pass within 6 GiB. Exact source/reduction order preserved; Astra approves with no findings. | [Scope](lfm3-finite-widening.md), [208 samples](../benchmarks/results/2026-11-12-lfm3-finite-widening-v1.json), [review/commit](../artifacts/lfm3-finite-guard-2026-11-11/finite-publication-checks.json) |
| MiniCPM5 exact-reduction RMS stores | Explicit Mac opt-in; two loaded-model CLI comparisons reduce warm requests **21.51% / 21.50%** (500 to 393 ms), TTFT about14%, and raise decode about35%. All 2,614 native arrays remain bitwise equal; required checks, native65 and Astra pass. Extra64MiB allowance, 6GiB and all model/default gates remain explicit. | [Scope](minicpm5-rms-store.md), [review/commit](../artifacts/minicpm5-rms-store-2026-11-10/publication-checks.json) |
| Ordered CPU projection tile | Warm 128/128 TTFT improves about 77.6% and request time about 52% in two pairs. Qualified opt-in; scalar reference preserved. | [Retention](experiments/2026-10-12-cpu-tiled-retention.md), `0abbe60` |
| Exact committed-prefix reuse | On Q4 0.5B, editing position 128 of a 512-token prompt reduces first-token latency about 21.6%; editing only position 511 about 93%, in paired runs. No KV copying; not Q4 quality qualification. | [Continuation](experiments/2026-10-13-prefix-continuation.md), `340c965` |
| Selected Half/Q4 and Online attention | M5/M6 retained only their documented configurations and quality/memory/context scopes. FP16 KV retained after the I8 evaluation. | [M5 release](experiments/2026-10-15-m5-release.md), [M6 release](experiments/2026-10-19-m6-release.md) |
| Four-row native affine8 prefill | Pinned Qwen2.5-0.5B MLX8 on M1 Pro: 1,024-token prefill improves 13.69%, complete fixed-token request 6.22%. Mac opt-in; managed adoption remains separate. | [M9 decision](m9-mac-affine-decision.md) |
| Gemma BF16 G2 last-row prefill | Official BF16 Mac route: both primary runs improve prefill about 61% and request time about 28%; retained in native45. | [Retention](../benchmarks/results/2026-11-06-gemma-bf16-g2-retention-v1.json) |
| Exact guard validation reduction | Same complete guard coverage with automatic integer vectorization. Two Mac BF16/G2 primary runs reduce request time25.0–26.6%;626 sustained requests confirm26.3%,512 deeper decode pairs bit-identical. Required checks/Astra pass ; 6 GiB preserved. No UI, memory-saving or broader-model claim. | [Scope](gemma-guard-validation.md), [checks/review/commit](../artifacts/gemma-guard-validation-2026-11-08/publication-checks.json) |
| Qwen2 prompt lookup | Repeated contract prompt gains about 53% decode, but the code prompt misses the 5% gate. Explicit nonstreaming API retained; default adoption rejected. | [Timing](../benchmarks/results/2026-11-06-m9-qwen2-lookup-timing-v1.json) |

## Existing correctness, memory and observability foundations

LFM3 bias-free BF16 CPU projections match 94,248 source outputs across four
selected rows of all 154 rank-two tensors, 65,280 finite controls and finite
corner/rejection-marker cases. The original FP32 diagnostic passes; no expanded
FP32 weights or numerical allocations. Fresh fmt/both strict Clippy/723 CPU/
869 Metal/nine example, eight optimized source tests and native77/C smokes pass.
All 42 release operator cohorts pass; 128-token representative MLP calls cost
802–810 ms. Component baselines only; actual child CPU/Metal exact correspondence
rejects. Astra approves scoped retention, no findings; attention/full graph/CLI/
quality/fit/adoption remain
open, with original gates unchanged.
[Scope/evidence](lfm3-projection-bf16.md),
[measurements](../benchmarks/results/2026-11-11-lfm3-projection-bf16-v1.json).

LFM3 transactional BF16 state is implemented with 13 six-row histories, six full
and five 256-token KV owners. Independent fixtures and actual declared-parent/
MLX CPU/Metal logical-state comparisons pass. All-owner validation precedes any
publication; stale/foreign proposals and failures retain the committed prefix.
Fresh fmt/both strict Clippy/714 CPU/860 Metal/nine example checks, eight release
state tests, native76 three 46-export slices and CPU/Metal C smokes pass. Debug
and release transitions allocate nothing. All 24 original FP32 5% compatibility
pairs pass; 18 BF16 state cohorts measure 0.338–0.365 / 7.119–7.254 / 27.706–28.644 ms at
1/32/128 tokens. State owns 28,108,984 B at capacity 2,048. Astra approves scoped retention, no findings;
complete LFM3 graph/CLI/fit/quality/adoption remain open. Existing
arithmetic correspondence rejections and original gates remain unchanged.
[Scope/evidence](lfm3-bf16-state.md).

Canonical LFM3 CPU BF16 convolution/binary/head-gate leaves match 8,146,944
convolution words, all 34 binary grids and broadcast source fixtures. Chunk
splits and the original FP32 diagnostic pass; all four operators allocate
nothing. Fresh required/native75 checks and six release component cohorts
pass: medians 1.150–1.168 / 33.547–33.923 / 135.099–135.660 ms at 1/32/128
tokens. Operator baselines only; actual child CPU/Metal exact correspondence
rejects. Astra approves scoped retention, no findings; transactional BF16
history/window/graph and full model fit/quality/adoption remain open.
[Scope/evidence](lfm3-binary-conv-source.md),
[review/commit](../artifacts/lfm3-conv-bf16-2026-11-11/publication-checks.json).

Canonical LFM3 CPU BF16 norms/unary gates match 4,082,688 norm outputs,
5,706 FP32 statistic rows, 16 corner cases and both full finite unary grids.
Both widths have zero Rust allocations; fresh required/native73 checks pass.
Six frozen release component cohorts measure medians 1.285–1.315 / 39.048–39.207 /
152.448–154.071 ms at 1/32/128 tokens. These are operator baselines; actual child
CPU/Metal exact component correspondence rejects. Astra approves scoped
retention, no findings; full LFM3 graph/fit/quality/adoption and remaining
BF16/window composition stay open.
[Scope/evidence](lfm3-bf16-source.md),
[review/commit](../artifacts/lfm3-bf16-2026-11-10/publication-checks.json).

Source-profile LFM3 BF16 rotary passes exact frequency/trig/460,800 output
fixtures, sparse rows and chunk schedules with zero steady-state allocations.
Two shared admitted owners total 1,049,152 B. Fresh required/native72 checks
pass; 18 release operator cohorts on M1 Pro measure eleven-layer Q/K medians
of 0.0665–0.0703 / 2.112–2.226 / 8.437–8.695 ms for chunks 1/32/128. These
are operator baselines; full model execution/fit/quality/adoption remain open.
Astra approves scoped retention, with no findings.
[Scope/evidence](lfm3-rotary-source.md),
[review/commit](../artifacts/lfm3-rotary-2026-11-10/publication-checks.json).

Seven-tap FP32 convolution/history passes independent exact/tensor fixtures,
288 schedules and zero-allocation checks, preserving concrete three-tap APIs.
The first generic public prepare boundary is rejected at its performance screen;
revised concrete entrypoints pass all six original 5% comparisons (−2.33% to
+1.26%). New seven-tap operator baselines are about 0.357/9.3/37 ms for chunks
1/32/128 across 13 width-2,048 owners. Fresh required/native71 checks pass;
Astra approves scoped retention, with no findings. Full LFM3/BF16 activation/
fit/quality/adoption remain open.
[Scope/evidence](lfm3-seven-tap-history.md),
[review/commit](../artifacts/lfm3-seven-tap-2026-11-10/publication-checks.json).

LFM3's complete BF16 child and modified parent authenticate, with exact equality
of all 286 tensors/1.729B parameters, 13 singleton-axis convolution transforms
and 119 norms. Independent child replay covers 1,014 blocks; no parent payload
is retained. Nineteen portable tests and 28 actual offline CLI rejections pass.
Closed stream/replay audits stay below a separate 512 MiB aggregate guard;
maximum conservative final RSS sum is 247.6 MB. Required engine inputs/outputs
are unchanged and fresh fmt passes. Astra's RSS finding is fixed: each final
peak must cover the same PID's sampled RSS, with fresh replay/negative controls;
Astra approves scoped retention. Next qualify
seven-tap scalar convolution/history fixtures and then the remaining graph;
quality, runtime, fit, performance and adoption stay open.
[Scope](lfm3-source-qualification.md),
[review/commit](../artifacts/lfm3-source-2026-11-10/publication-checks.json),
[parent audit](../artifacts/lfm3-source-2026-11-10/parent-checks-1.json),
[offline replay](../artifacts/lfm3-source-2026-11-10/offline-checks-2.json).

The MiniCPM5 native precision audit finds no output-grid obstruction across all
442,863,360paired original CPU coordinates. A complete replay agrees; a separate
constructive method verifies native witnesses everywhere. Both methods preserve
all original bounds and native 50 Half/1 FP32 domains, with 172.4 MB maximum final RSS
under a separate 512 MiB audit guard. Eight new/25 total portable, 11 CLI and one
actual source-byte negative pass. This does not fix the native 82 failures:
current managed adoption remains rejected, with the research route retained.
Next qualify LFM3's pinned BF16 source/payload and exact graph. Astra approves
scoped diagnostic retention, with no findings;
quality, precision, memory and performance gates stay unchanged.
[Scope](minicpm5-precision-audit.md),
[review/commit](../artifacts/minicpm5-representability-2026-11-10/publication-checks.json).

MiniCPM5 numerical isolation rejects the cached rotary model experiment.
All eight sampled projection coordinates match exact dots of their own inputs;
local rotary fidelity improves to zero failures in 7,084,800 original primitive
values, but the original short model suite fails six of 52,896,768 comparisons.
The required original gates remain binding. Candidate required/native69 checks,
two explicit GPU tests, 36 portable, six raw witnesses and 12 resealed negatives pass;
the deeper run is deliberately stopped and remains unqualified. The rejected
engine/CLI/reader patch is archived outside the proposed documentation-only
commit. Astra approves documentation-only retention with no findings; no
performance, quality or adoption claim.
[Scope/next audit](minicpm5-rotary-rejection.md),
[review/commit](../artifacts/minicpm5-projection-isolation-2026-11-10/publication-checks.json).

MiniCPM5 deeper diagnostics complete with a numerical rejection: 82 failures in
1.773B original-gate comparisons, reproduced by a fresh complete cold replay.
All 87,624 arrays/1,718 chunks and recovery controls complete within the 6 GiB
guard. No disjoint CPU bounds occur across 442.9M coordinates; isolate one
projection mismatch next. Fresh required/native68 checks, 17 portable tests,
15 CLI/51 metadata-sample negatives and 48 independent raw witness replays pass.
Astra's clock/ledger findings are fixed; 13 resealed datasets and all former
zero-ledger/repeated-window counterexamples reject. Both full replays preserve
the numerical result; Astra approves scoped retention, no remaining findings.
The ordinary route stays at 512/chunk32; full
quality, context/adoption and overhead gates remain open.
[Scope/results](minicpm5-context1025.md),
[review/commit](../artifacts/minicpm5-context1025-2026-11-10/publication-checks.json).

MiniCPM5 complete quality inputs now preserve all original 10,000 targets/100
tasks and their caps. Cold recipe replay, seven portable tests and 28 altered-
input rejections pass. Astra independently replays all payload hashes and
approves scoped retention, no findings. Original stock metadata is pinned,
without its weight download; all corpus/task IDs and task renders match the
selected child. Full scoring requires 1,025 positions (largest task 559), beyond
the unchanged 512 diagnostic limit. Numerical/quality/adoption gates remain
open. [Scope/evidence](minicpm5-quality-inputs.md),
[review/commit](../artifacts/minicpm5-quality-inputs-2026-11-10/publication-checks.json).

MiniCPM5 matrix reproduction is retained with Astra approval. The frozen
MLX 0.32.3 GPU/BF16 affine8 recipe reproduces all 296 matrices/888 arrays
exactly while authenticating both full parent objects. All 85 copied norms also
match; 4,128 tiles retain no parent payload. The separate 512 MiB audit bound
passes. This proves a reproducible conversion relation, without authenticating
the publisher's exact binary or qualifying inference/quality/adoption. All 35
portable tests, 34 actual-output negatives, exact seal replay and input-based
Rust/native check reuse pass. Astra independently replays the seal, no findings.
[Scope](minicpm5-matrix-lineage.md),
[review/commit](../artifacts/minicpm5-matrix-lineage-2026-11-10/publication-checks.json).

MiniCPM5 bounded basic text/CLI is retained with Astra approval. Exact98source fixtures/66streams,679CPU/821Metal, native67/C smokes,
36portable and83actual-report negatives pass. The unchanged22k-word/660-phase
cache stress stays below 512 MiB after a preserved/fixed ASCII decode-bound bug.
Two14/38-token native histories compare4.18Mlogits bitwise; eight cohorts retain
80measured/24warm requests, with RMS opt-in gains18.85%/18.32% and12.29%/11.79%.
6GiB owners/observations pass in named scopes; final physical peak unavailable.
Default allocator, original quality/deeper numerical,
managed/default adoption and1%summary-overhead gates remain open. Astra's sole
first-text enclosure finding is fixed; unchanged raw cohorts revalidate.
[Scope/evidence](minicpm5-text-codec.md),
[review/commit](../artifacts/minicpm5-text-codec-2026-11-10/publication-checks.json).

MiniCPM5 bounded source streaming verifies both complete BF16 parent hashes
(5.033GB),381tensors and85exact copied norms, retaining1,349block digests and no
payload. Fifteen portable tests and offline witness sealing pass; engine/native
inputs are unchanged. Astra approves scoped source retention. Matrix reproduction
is covered above; original task quality and adoption remain open.
[Scope](minicpm5-parent-auth.md).

- Readable scalar CPU implementations and independent numerical fixtures remain
  correctness references. Real-model, reduced-precision and quality results have
  separate frozen scopes; [model coverage](model-coverage.md).
- Immutable weights and session-owned KV/scratch preserve committed versus pending
  token history, exact-prefix reuse, capacity admission and cancellation/replay;
  [session evidence](../benchmarks/results/2026-10-09-qwen-session.md),
  [Metal engine](experiments/2026-10-13-metal-engine.md).
- Resident weights, streamed loading, explicit layouts and bounded scratch avoid
  redundant full-model ownership. Zero steady-state Rust numerical allocations
  are measured only for their recorded routes; native/application allocations
  remain separately scoped; [Metal engine](experiments/2026-10-13-metal-engine.md).
- Generation reports, completed Metal timings, bounded calibrated GPU profiles,
  Swift callback metrics and runtime planning already exist;
  [current source inventory](observability-performance-proposal.md#evidence-already-available).
- Memory reporting separates model/session owners, allocated KV capacity,
  planning/admission allowances and measured footprint/conditional loading peaks;
  [owner evidence](../benchmarks/results/2026-11-02-session-owner-memory-v1.json),
  [KV limits](../benchmarks/results/2026-11-03-kv-preview-apply-v1.json),
  [load checkpoints](../benchmarks/results/2026-11-03-load-memory-checkpoints-v1.json).
- Managed Mac profiles and official Gemma public download/offline Chat have scoped
  app evidence; [journeys](mac-hub-journeys.md),
  [official Gemma](../benchmarks/results/2026-11-07-gemma-public-resume-mac-v1.json).
  Further iPhone work remains deferred.

## Restrictions that remain binding

| Result | Current decision |
| --- | --- |
| Historical failed quantization/precision trials | Keep failed evidence and gates; a later passing configuration does not rehabilitate an earlier candidate. [Full history](archive/2026-11-07/improvements.md#rejected-or-inconclusive-experiments). |
| Qwen3 fidelity and Gemma3 coupled intermediates | Preserve rejected trajectories and distinguish native conformance from original adoption quality. [Qwen3](qwen3-native-profile.md), [Gemma3](gemma3-native-conformance.md). |
| Split-KV and Llama Online | Do not adopt rejected arithmetic. [Split-KV](../benchmarks/results/2026-11-05-m9-split-kv-session-v1.json), [Llama](todo-implementation-plans.md#llama-online-correctness). |
| Gemma QAT task quality | Full-task loss exceeds policy despite passing loss/state/fidelity checks. [Decision](../benchmarks/results/2026-11-06-gemma-qat-full-quality-v1.json). |
| Gemma G3 assistant | Longer guard regresses about 21%; keep test-only, retain ordinary G2. [Decision](../benchmarks/results/2026-11-06-gemma-assistant-mtp-retention-v1.json). |
| New device/context/default adoption | Requires its own applicable checks. Mac execution, metadata support or admission estimates do not establish unrestricted support. |

## Console reference comparison

The [LFM3 external configuration preflight](lfm3-external-config.md) is reviewed
in its metadata-only scope. Follow-on setup authenticates all 286 BF16 payloads
and four JSON files. A separate Ollama view changes thirteen singleton shape
entries for its pinned loader, preserving payload bytes and the original import.
All four fresh 3+10 external cohorts complete under 6 GiB with owned cleanup and
source authentication. Ollama lacks authoritative prompt IDs; vLLM matches exact
prompt/cache admission but matches native output IDs/text on only 3/13 prompts
per cohort. Client medians and explicit limitations are in the
[collection summary](../artifacts/lfm3-engine-comparison-2026-11-11/external-collection-summary-1.json).
Twenty-six genuine vLLM captures replay through the reviewed optional-timing
CLI with consistent arithmetic. No cross-engine speedup, quality or managed
adoption is qualified; full comparison/Astra review remains open. Setup and
failed/passing guards, source preservation and the removed redundant 3.46 GB
upload copy are retained in the [configuration evidence](lfm3-external-config.md).

The earlier Gemma comparison completes the following separate scope.
All78 requests pass across two3warm+10 cohorts for each of3engines on M1Pro16GiB.
Official BF16/G2 Toklet request medians2.184/2.202s; verified uncensoredMLX4bit
Ollama121/119ms and vLLM-Metal147/149ms. Precision, weights, cache hits and timing
boundaries differ; no engine speedup ratio or adoption qualification. All observed
memory guards and observed external lifetime sums stay below6GiB. Ollama0.40.15 and
local vLLM0.33.0/Metal0.33.0 are installed/verified. Original BF16 external runs
fail tested admission. The ordinary native affine4 inspector remains strict;
the mixed artifact uses the separate diagnostic below.
Offline report/document checks pass; Astra approves scoped publication.
[Comparison / limitations](console-engine-comparison.md),
[full evidence](../artifacts/engine-comparison-2026-11-07/console-comparison-report.json).

## Observability status

[O1 summaries](o1-gemma-request-summaries.md), [O2.A phase reporting](o2-gemma-cost-report.md),
[O2.B1 repeated workloads](o2-gemma-workloads.md), [O2.B2.A publication](o2-ui-publication.md),
[O3 attribution](o3-gemma-runtime.md) and [R1 screening](gemma-regression-job.md)
are delivered in their reviewed scopes. Full counts, failures, raw measurements
and commit references are in the [completed history](archive/2026-11-09/improvements.md#current-observability-work).
Summary collection stays opt-in/default-disabled: the 1% overhead bound remains
unqualified. Controller publication does not establish visible text.

The user has deferred screen-capture qualification to concentrate on engine/CLI
work. Ready9 finishes13exact trials but rejects433/1721future-display callbacks,
with zero occlusion;32numeric samples are retained. The session is closed, and
no further sharing requests are planned. Prior failures, isolated UI code and
all raw evidence remain preserved, unreviewed and uncommitted. First-visible
metrics remain unqualified; console metrics and automated checks lead active work.
[Deferred status](../artifacts/o2-ui-frames-2026-11-07/status-audit-1/current-state.json).

## Selected model decisions

Order remains Gemma 4 E2B, Qwen3.5-4B, Qwen3.5-9B, then the five under-4B
candidates. Prefer authenticated uncensored MLX artifacts, retaining each
artifact's independent numerical, quality, memory and request gates. The 6 GiB
Mac ceiling, 512/32 pilot and 512 MiB codec budget stay fixed; iPhone is deferred.
[Model/metric plan](model-observability-plan.md).

| Scope | Retained result or restriction | Evidence |
| --- | --- | --- |
| Gemma Heretic affine4 | Cold planning, authenticated mixed-format loading and independent fidelity are complete; task-quality loss rejects adoption. Keep official BF16/G2 selected. | [Artifact](gemma-heretic-mlx.md), [quality rejection](gemma-heretic-quality-probes.md) |
| Qwen3.5-4B | Separate pinned owners, scalar/hybrid graph, codec and generation state bridge are delivered. Full 128/256 fidelity fails: 13,505 failures/3,132 disjoint intervals. No managed adoption; unchanged gates bind later work. | [Model plan](model-observability-plan.md), [decision](../artifacts/qwen35-context256-2026-11-09/decision.json) |
| Qwen3.5-9B recipe/store/head | Separate strict source and untied-head support retain original packed owners and authenticate the complete 5.038 GB stream. These mechanisms alone do not qualify execution or fit. | [Recipe](qwen35-9b-cold.md), [store](qwen35-9b-storage.md), [head](qwen35-untied-head.md) |
| 9B one-input oracle/Metal | Both complete dual-profile captures pass. Hybrid Metal has zero retained CPU matrix bytes and 5.667 GB peak; 64.72×/64.42× applies only to the raw one-input baseline. | [Oracle](qwen35-9b-oracle.md), [Metal](qwen35-9b-metal.md) |
| 9B context17 | Both 33.85M-value gates, partitions, nonzero-prefix failure/repair/retry and zero-allocation checks pass. Release medians 4.093/4.121s are descriptive; no overhead, request, tail, sustained or adoption claim. | [Context17](qwen35-9b-context17.md), `41287c6c` |
| Exact recurrence reference | All 67.17M short-context FP32/FP64 values match scalar references bitwise. Two cohorts lower helper step time 60.02/60.25% FP64 and 95.55/95.23% FP32. No engine/deeper-context gain follows. | [Reference helper](qwen35-reference-vector.md), `88950b11` |
| LFM3 CPU BF16 | Authenticated graph/CLI, 1,025-position parity, both complete original quality comparisons and 62 state checks pass; finite widening retained after matched request measurements and Astra approval. Managed/GPU adoption stays open. | [Scope and decision](lfm3-finite-widening.md) |
| Remaining under-4B qualification | MiniCPM5, converted Granite/Ministral and Qwen3.5-2B routes retain their numerical rejections. Payload/conversion/diagnostic results have advanced beyond metadata; downstream quality, fit, request and adoption gates remain unqualified where blocked. LFM3 retains its separate qualified CPU scope. | [Candidates and receipts](model-observability-plan.md#five-candidates-strictly-below-4b-total-parameters) |

The [Huihui4B source audit](qwen35-alternative-correspondence.md) passes complete
conversion correspondence. Its original128/256 inference-reference collection
then rejects372 disjoint tolerance intervals across38,491,136 paired coordinates;
independent wider replay certifies every failure. Both source/runtime endpoints,
6GiB guard and cleanup pass; peak1,365,065,728 bytes includes helper. Seventeen
new controls plus six recurrence controls pass per Python mode. The partial-line
monitor and copied-digest replay failures are retained and corrected.
[Decision/evidence](qwen35-huihui4b-envelope.md). Astra approves scoped diagnostic/
rejection retention, no findings. [Publication](../artifacts/qwen35-alternative-admission-2026-11-15/envelope-publication-1.json). The user chose explicit Qwen rejection decisions; original4B/9B
rejections stay closed and the newer9B candidate remains metadata-only, not pursued.
Positive Qwen performance/adoption stays unqualified. [All outcomes](improvement-outcomes.md). A legacy Linux assertion against captured
Mac trace bytes requires a CI follow-up: keep the exact Mac probe and add dense/
packed parity against the unchanged local scalar graph on every platform. Seven
Mac controls pass per Python mode; Astra approves the fix with no findings.
The fix is published as `60415234`; all seven exact-head workflows/15 jobs pass,
including exact local scalar/vector parity in both Linux modes. The captured
Mac archive probe remains exact locally and explicitly unavailable on Linux.
The [final requirement audit](../artifacts/qwen35-alternative-admission-2026-11-15/whole-goal-final-audit-1.json)
reconciles the delivery, accepted rejections and deferrals.

Original individual check counts, source revisions, measured cohorts, rejected
experiments and Astra/commit receipts remain in the [reviewed history](archive/2026-11-09/improvements.md#latest-documentation-work)
and linked scope reports. Metadata and narrowly passing probes do not qualify
whole-GPU execution, default allocator/codec/application fit or managed requests.

## Current implementation

The Qwen3.5-2B real-source/CLI diagnostic passes full prospective parent lineage,
bounded Q4 conversion and saved-byte audit for 187 matrices/133 exact small
owners. Required fmt/strict CPU+Metal Clippy, 786 CPU/946 Metal tests, 70 portable
controls per Python mode, four GPU/state/repair and twelve CLI cases pass.
The short graph passes 159,187,008 comparisons. Context17 rejects three of
680,841,024 comparisons; both native traces match, all schedule/reset/causal
controls and complete journals/failing bytes audit. Conditional fixed255,
quality and performance do not advance. Astra approves scoped diagnostic
retention with no blocking findings. Local-input investigation, codec/stop
witnesses, original full quality, representative fit/performance/overhead and
adoption remain open; no performance gain follows. Failed monitor, reader,
allocator, import, binary-selection and audit-wrapper attempts stay retained.
[Scope](qwen35-2b-real-source.md),
[review/commit](../artifacts/qwen35-2b-payload-2026-11-13/publication-checks.json).

The bounded follow-up reproduces all 34,119 native arrays and recovers all
17,394 staged prefix arrays exactly. None of the 24 precision policies or two
declared FP32 reduction schemes passes both selected cases under both original
profiles. This variant search stops with no engine change; Astra approves
the evidence closure with no findings. [Final local rejection](../artifacts/qwen35-2b-attention-2026-11-13/two-f32-qk-check-1/result.json).

Delivery overlaps the Gemma CLI producer, independent controls and offline
reporting through exclusive file ownership. One integrator keeps the serial
heavy queue, required checks, combined Astra review and push. The earlier LFM3
two-token experiment is closed below; no workflow speedup is measured.
[Delivery plan](parallel-delivery.md),
[checks/review/publication](../artifacts/qwen35-2b-attention-2026-11-13/velocity-publication-checks.json),
[source preflight](../artifacts/qwen35-2b-attention-2026-11-13/lfm3-next-performance-preflight-1.json).
Fresh preparation now completes the cached release-probe build and three full
projection baseline cohorts, with 30 measured samples and six passing power/
thermal endpoints. The sandbox-denied initial launch remains failed; the retry
passes. Whole-request attribution and any candidate implementation/retention
remain open. [Current baseline](../artifacts/qwen35-2b-attention-2026-11-13/lfm3-current-operator-baseline-1.json).
Current request attribution completes with exact output/state and 22 matching
input endpoints under 6 GiB. Projection dot accounts for 801/828 active-main
samples in the initial prefill window; this warrants an exact projection
experiment, with no speedup claim. A report-wrapper line-index correction
requires no model rerun. [Profile closure](../artifacts/lfm3-tokens2-2026-11-13/request-profile-closure-1.json).
Before a new kernel integration, the official Gemma observer-overhead gate is
next: its current clean-source Mac archive build passes and binds 462 input
hashes. Historical archives are not substituted for this artifact.
[Archive binding](../artifacts/lfm3-tokens2-2026-11-13/gemma-current-staticlib-1.json).
The original fresh ten-pair TTFT bounds remain unqualified and retained. Bounded
10..100-pair support now preserves default behavior, the worker ring and timed
boundaries. Forty dependent portable controls pass in both modes; default
real-data statistics replay exactly. Strict Swift/conformance and both fixed
100-pair cohorts pass all original headless gates; independent audit reconciles
412 IDs, 206 raw summaries, state/status/clocks and 489 input endpoints. Combined
Astra approves scoped retention with no findings. App/UI/render qualification
stays separate and opt-in.
[Scope, results and evidence](gemma-summary-overhead.md),
[checks/review/publication](../artifacts/lfm3-tokens2-2026-11-13/gemma-stronger-publication-checks.json).

The bounded two-token LFM3 experiment is **closed, rejected for code retention**.
Correctness/allocation controls, fmt, strict CPU/Metal Clippy and 794 CPU/954 Metal
tests pass. Corrected fixed operator cohorts preserve whole outputs and reduce
large-shape latency 48.99/49.53%, but eight slower one-token pairs across six
configurations leave the original compatibility gate unresolved. Astra approves
evidence-only closure; all five candidate code files are restored exactly to
`cf80a168`. No further variant search or model/request integration follows.
Original negatives, the excluded preparation-overlap screen and full quiet-host
rerun are retained. Exclusive file ownership, one heavy queue and explicit pause
acknowledgements govern [parallel delivery](parallel-delivery.md); no workflow
speedup is measured yet.
[Scope](lfm3-two-token-projection.md),
[review](../artifacts/lfm3-tokens2-2026-11-13/tokens2-astra-review-2.json),
[revert](../artifacts/lfm3-tokens2-2026-11-13/tokens2-code-revert-2.json),
[checks/review/publication](../artifacts/lfm3-tokens2-2026-11-13/tokens2-publication-checks-2.json).

Headless Gemma workload identity/admission has complete fixed-cohort evidence.
Astra approves scoped code retention and exact workload qualification, no findings. Strict Swift, 57 portable
controls per Python mode, fmt, strict CPU/Metal Clippy and 786 CPU/946 Metal tests
pass. All 52 requests/40 measured/26 summaries verify exact384input/128output,
state512/511/1, reset admission, clocks, native work and 6 GiB fit. Both original
1% median-ratio and paired-median 95% upper gates pass for both metrics/cohorts;
independent replay reproduces all statistics and verifies489 current endpoints.
The cold writer bound and launch dependency fixes are independently checked;
original power/refusal failures remain recorded. Only this headless BF16/G2 M1
Pro boundary is qualified; production stays disabled/opt-in. No numerical512-input,
UI/render, tail or default-adoption qualification is transferred.
[Scope](gemma-workload-identity.md),
[checks](../artifacts/gemma-workload-identity-2026-11-13/checks-final-3.json),
[audit](../artifacts/gemma-workload-identity-2026-11-13/independent-workload-audit-3.json),
[review/commit/CI](../artifacts/gemma-workload-identity-2026-11-13/publication-checks-3.json).

Completed native decode rows are implemented through additive ABI1 metrics V2;
the old96B record/symbols remain unchanged. Required package/native/C/Swift and
complete65 portable controls per mode pass. The fresh pair and two fixed cohorts
reconcile54 requests/27 native records, exact384/128/state5125111/count127 and526
endpoints; independent replay reproduces all statistics. TTFT95uppers1.230/1.313%
leave the original1% bound unqualified; no samples/gates change. Functional
retention is approved by Astra, no findings, under the unchanged disabled/opt-in
policy. Later
output failure can retain a completed native report; native failures stay unknown.
[Scope](gemma-decode-evaluations.md),
[final checks](../artifacts/gemma-decode-rows-2026-11-13/checks-final-1.json),
[replay/rejection](../artifacts/gemma-decode-rows-2026-11-13/independent-decode-audit-1.json),
[review/commit/CI](../artifacts/gemma-decode-rows-2026-11-13/publication-checks-1.json).

Gemma console cost attribution now has a bounded officialBF16/G2 producer and
strict offline stable-operation report. Required fmt/strict CPU+Metal Clippy,
815 CPU/974 Metal tests, final28 CLI unit controls (2 ignored) and11 portable controls per
mode pass. Both full384/128 first/lastdecode captures preserve exact outputs,
state and2x262144 selected logit bits;349tile/2gather coverage and clocks replay.
Observed charge<=4.803GB includes helper; declared5.932GB includes all reserves.
Host waits128–135ms and GPU unions~52ms overlap inside226–236ms frames; submission
batching is a next hypothesis, not a measured speedup. The early private-selector
and root optimized-wrapper failures are fixed/retained. Astra approves scoped
functional/diagnostic retention, no findings; ordinary V2 TTFT1% remains
unqualified/opt-in. [Review/commit/push/CI](../artifacts/gemma-cost-attribution-2026-11-13/publication-checks-1.json).
[Scope](gemma-profile-cli.md),
[checks](../artifacts/gemma-cost-attribution-2026-11-13/checks-final-1.json).

Gemma single-row submission is retained as an opt-in source4 route after combined
Astra approval with no actionable or blocking findings; publication follows. Required checks and exact source reuse cover CPU816/Metal976,
original45,000,704 numerical values,full10k loss/100-task quality,state/GPU/profile
and strict optimized zero-allocation controls. All6 ordinary collections validate
557 requests/431 measured,including both600s sustained arms,peak4.820GB. Complete
caller medians improve7.4005/7.1650%; paired95%lower6.9263/6.9062% exceeds5% in both
cohorts. All20 total-clock comparisons improve; independent65-pin replay matches
4720 report values. One secondary cached TTFT point is+2.313microseconds in
confirmation,CIincludes0; no universal TTFT claim or new noise threshold follows.

Source4099/binaries are recovered exactly into persistent storage and archived;
full canonical model hashes authenticate only a device-number remount. Completed
primaries replay;178 interrupted matrix requests and failed report/path/adapter
attempts stay retained/unqualified. Default per-tile/native behavior and disabled/
opt-in summaries persist. Combined [Astra review](../artifacts/gemma-head-batch-2026-11-14/astra-review-1/verdict.json)
passes; code `e75a9c30` is pushed with all6 exact-head workflows/11 jobs passing.
[Publication/CI](../artifacts/gemma-head-batch-2026-11-14/publication-checks-1.json);
UI/iPhone/larger contexts remain unqualified. [Scope](gemma-single-row-submission.md),
[checks](../artifacts/gemma-head-batch-2026-11-14/checks-final-1.json),
[report](../artifacts/gemma-head-batch-2026-11-14/actual-submission-report-2/report.md),
[independent replay](../artifacts/gemma-head-batch-2026-11-14/independent-submission-replay-1.json).

Ordinary Gemma CLI/metrics is implemented in an isolated persistent worktree;
Astra approves scoped functional retention with no findings; code is published as
`4f2e6236`,with all6 exact-head workflows/11 jobs green. Defaults stay per-tile/disabled. One official
BF16 owner/codec/G2 session supports generation, cache/reset, cancellation and
output failure, with bounded summaries/export and explicit unavailable probes.
Logical collector fixed/ring/peak storage is9,000/1,032,192/1,041,192 bytes,
separate from admission and process memory; signal latency remains unavailable.
Source3 passes fmt, strict CPU/Metal/probe Clippy,57CPU/72Metal affected controls
(two ignored each),36 reader controls per Python mode and both fresh policy
header smokes. Source2's84-entry conformance keeps its exact scope and original
whole-attempt timeout; unchanged numerical/quality/codec evidence is not relabelled.
[Scope and checks](gemma-cli.md),
[required checks](../artifacts/gemma-cli-metrics-2026-11-14/required-checks-3.json).

The full fixed campaign completes16 owners,128 generations (48warm/80measured),
128 resets and eight complete16-record exports. The6GiB guard passes6,264.96s,
peak5,004,482,128 bytes including helper; source/model remain fixed and cleanup
is clean. Original1% gates pass5/8 comparisons and reject per-tile cohort1 TTFT/
complete and batched cohort2 TTFT. Summary remains opt-in, overhead/adoption
unqualified under the existing exit policy. Both earlier zero-measured attempts
and their runner/power failures stay evidence; noN/gate/exclusion rescue occurs.
Independent raw replay passes with zero discrepancies across128 generations,
128 resets,eight exports and all original statistics; within-arm drift is
descriptive, with no causal attribution or sample exclusion. Astra independently
reproduces all eight gate results and approves scope,with no findings. Committed
source mapping verifies all4,116 measured inputs. [Publication](../artifacts/gemma-cli-metrics-2026-11-14/code-publication-1.json).
[Measurements and gate table](gemma-cli.md#fixed-overhead-measurement),
[capture](../artifacts/gemma-cli-metrics-2026-11-14/cli-overhead-3/result.json).

Delivery overlaps exclusive producer, independent reader/tests and qualification
preparation; root owns integration and the heavy queue. All lanes pause during
acceptance measurements. Compatible functional smokes share loaded sessions;
benchmark validators replay genuine wire envelopes before long captures.
Delivery speedup is unmeasured. [Assignments and policy](parallel-delivery.md).

The ordinary CLI R1 extension implements a published baseline, uncommitted
candidate snapshot and later exact-content publication binding. All 28 independent
controls and nine legacy controls pass in normal/-O modes. Whole-owner resampling
retains correlated uncertainty; shared 5% policy helpers preserve legacy arithmetic.
Both sealed G17 policies admit two cohorts/four disabled owners each, without
model reads or child launches. Eight CLI controls and required-check reuse pass;
the initial verification-helper inventory error is retained and corrected.
Three Astra boundary findings are fixed and covered by six new regression
methods, first demonstrated against the old code. Fresh checks pass; Astra
approves scoped retention with no remaining findings. Code is published as
`ae984fda`; the [live publication receipt](../artifacts/gemma-cli-regression-2026-11-15/publication-status.json)
tracks the final head and exact-head CI result.
[Astra approval](../artifacts/gemma-cli-regression-2026-11-15/astra-review-1/verdict-2.json). Baseline admission establishes
no optimization or observer gate.
[Contract](../artifacts/gemma-cli-regression-2026-11-15/implementation-contract-1.json),
[portable checks](../artifacts/gemma-cli-regression-2026-11-15/affected-checks-2.json),
[real admission](../artifacts/gemma-cli-regression-2026-11-15/real-baseline-admission-2.json).

The coupled 13→14→15 chain now passes all eight original cells, state/replay
controls, 32 zero-Rust-allocation warm iterations and the original producer
regression. The old coupled failures remain linked. Higher-accuracy RMS and the
approved finer-core GELU table preserve the original gates; optional completed-
command GPU timestamps add no submissions or waits. Required checks pass with
1,044 Metal tests and 847 unchanged CPU tests reused. A failure-count correction
passes the release regression; Astra approves with no remaining findings. The common-Q4 conversion/owner/CLI checks also pass, with original
full quality and stock precision matching still open. No ordinary resident
route change, full-model fit or new speed gain is claimed.
[Chain scope/checks](gemma-gpu-chain.md),
[chain status](../artifacts/mlx-gpu-gap-2026-11-15/g2-deep-chain-progress-1.json),
[quantizer checks](../artifacts/mlx-gpu-gap-2026-11-15/g0-q4-prepare-progress-3.json).

# Toklet progress checklist

Updated **22 November 2026**. The competitive GPU implementation goal is active.
The new model admission target is Nanbeige4.2-3B, MLX affine 4-bit.
Read the current task and [Rust and Metal proposal](docs/mlx-gpu-performance-plan.md#rust-and-metal-architecture-proposal).
Previous delivery and model decisions are in [outcomes](docs/improvement-outcomes.md);
completed history remains in the [dated archive](docs/archive/2026-11-12/README.md).

## Current and next task

- [x] **Current** — Prepacked dense tail matches the exact tape and clears
  both 5% completion gates in both fresh Nanbeige orders. Completion is
  3.081–3.091 s across 78 requests at 5.37 GiB peak.
- [ ] **Next** — Qualify normal CLI integration and the original 1% summary
  bounds. The original summary run needs 5 GiB free at admission.
  [Prepack matched cohorts](artifacts/mlx-gpu-gap-2026-11-15/root-nanbeige-prepacked-cohorts-1/RESULT.md) ·
  [Prepack A/B](artifacts/mlx-gpu-gap-2026-11-15/root-nanbeige-dense-prepacked-1/RUN.md) ·
  [Matched cohorts](artifacts/mlx-gpu-gap-2026-11-15/root-nanbeige-exact-cohorts-1/RESULT.md) ·
  [Summary diagnostic](artifacts/mlx-gpu-gap-2026-11-15/root-o4-owner-conditioning-1/RUN.md) ·
  [Untimed rollback](artifacts/mlx-gpu-gap-2026-11-15/root-nanbeige-exact-fault-prep-1/RUN.md).

## Recent implementation and evidence

- [x] Diagnose the LFM3 gap with Astra. The controlled comparison measures
  9.63× first-ID and 6.02× request completion. Decode GPU work alone is
  1.419 s. The proposal prioritizes tiled prefill, packed decode and keeping
  dependent operations resident across a complete GPU block.
- [x] Select Nanbeige4.2-3B as the next intelligence-first admission target.
  The pinned MLX package is about 2.35 GB for weights. The 2,048-position
  BF16 K/V payload estimate is 352 MiB. Direct MLX supplies reference evidence
  for the Rust/Metal engine. Existing numerical rejections retain their scope.
- [x] Run the matched vLLM worker replay. AC and Low Power Mode admission
  pass on retry. Both orders verify all 52 requests, the same live worker per
  owner and the frozen zero-reuse 512/32 work. Median direct completion saves
  5.54 and 2.76 ms at 5.27 GiB peak. Both paired completion ranges include zero
  and stay below 5%. Both readers agree.
  [Actual replay](artifacts/mlx-gpu-gap-2026-11-15/root-lfm3-same-worker-replay-1/normal.json).
- [x] Add the experimental Nanbeige Rust and Metal CLI.
  The full shard and all 513 entries authenticate. Two passes reuse 22 aligned
  weight blocks with 44 independent cache owners. The M1 Pro CLI generates text
  at capacity 2,048. Both partitions match MLX's greedy IDs and preserve cache
  state through cancellation, retry and reset. Both readers agree on 30 raw
  comparisons. Formatting, strict Clippy in both configurations, 869 default
  tests, 1,074 Metal tests and both explicit GPU controls pass. The library
  rerun passes after bytecode cleanup freed 210 MB. Astra’s review conditions
  for the experimental profile are satisfied. Pushed `d4b485bf`.
  [Admission and native evidence](artifacts/mlx-gpu-gap-2026-11-15/root-nanbeige-mlx-q4-admission-1/RUN.md).
- [x] Price inline BF16 split-K on the frozen layer-zero V case.
  All 262,144 prefill values match MLX 0.32.3. Both warmed orders show a
  slowdown. The guarded MLX 0.32.4 replay keeps FP32 subtotals and matches
  the released T128 path at 131,046 of 131,072 words. Astra approves retaining
  the old split-K experiment as evidence. No engine variant is added.
  [Matched inputs](artifacts/mlx-gpu-gap-2026-11-15/root-nanbeige-mlx-q4-admission-1/astra-splitk-panel-invariants-1/result.json) ·
  [Operator replay](artifacts/mlx-gpu-gap-2026-11-15/root-nanbeige-mlx-q4-admission-1/sol-inline-splitk-l0-value-1/summary-normal.json).
- [x] Price 32-row FFN tiling on authentic prefill operands.
  The guarded MLX 0.32.4 run freezes gate, up and down inputs at T128, both
  T64 windows and T43. The packed banks are authenticated ranges in the
  existing shard. A 32-row tile halves the FFN weight reconstruction count
  at T128. All 7,790,592 native output words and all canaries pass. GPU
  medians improve at T64 and T128. T43 slows in both orders. Astra approves
  testing the connected gate, up and down path at exact T64 and T128.
  [FFN inputs](artifacts/mlx-gpu-gap-2026-11-15/root-nanbeige-prefill-work-1/ffn-inputs-1/input.json) ·
  [Native comparison](artifacts/mlx-gpu-gap-2026-11-15/root-nanbeige-prefill-work-1/sol-ffn-m32-1/summary-normal.json) ·
  [Tile census](artifacts/mlx-gpu-gap-2026-11-15/root-nanbeige-prefill-work-1/normal.json).
- [x] Qualify 32-row FFN tiling on complete requests. Keep the released tile
  for other row counts and all other operators. Compare two owner orders
  on the frozen 512/32 tape and require matching IDs, committed KV and full
  terminal logits. The 52 requests and separate terminal diagnostics pass.
  Completion improves 10.10% in both orders at a 2.47 GiB peak. TTFT improves
  12.18% and 12.14%. Both decode intervals include zero. The full Metal
  suite, archived partition replay, GPU rounding and rollback, and CLI
  stats pass. Astra approves the scoped Mac change. Pushed `13086f54`.
  [Connected result](artifacts/mlx-gpu-gap-2026-11-15/root-nanbeige-ffn-connected-1/normal.json) ·
  [Checks and review](artifacts/mlx-gpu-gap-2026-11-15/root-nanbeige-ffn-connected-1/RUN.md).
- [x] Price modern MLX vector arithmetic at wider decode scopes. The old
  BF16 four-term bias sum explains every changed word of the T1 V fixture.
  The single-operator modern Rust and Metal candidate matches all 1,024
  reference words in 52 guarded calls. GPU medians improve 7.24% and 7.95%.
  The full fixed-tape MLX 0.32.4 reference authenticates 35 logit rows
  and all 44 K/V owners under 6 GiB. All 32 raw argmax decisions match
  the released tape. The native overlays match all 32 IDs and prefill
  boundaries. Across 31 decode rows, RMS logit error against MLX falls
  from 0.421082 to 0.274795, and the rollback controls pass. Both
  warmed orders save 6.98–7.40% decode and 1.31–1.39% completion.
  Retain the experiment under the original whole-request and quality gates.
  [Connected MLX reference](artifacts/mlx-gpu-gap-2026-11-15/root-nanbeige-modern-decode-1/mlx-normal.json) ·
  [Native numerical join](artifacts/mlx-gpu-gap-2026-11-15/root-nanbeige-modern-decode-1/sol-native-connected-1/normal-2.json) ·
  [Connected decode timings](artifacts/mlx-gpu-gap-2026-11-15/root-nanbeige-modern-decode-1/sol-native-connected-1/price-3/normal.json) ·
  [Modern MLX replay](artifacts/mlx-gpu-gap-2026-11-15/root-nanbeige-prefill-work-1/mlx32.4-normal.json) ·
  [T1 arithmetic replay](artifacts/mlx-gpu-gap-2026-11-15/root-nanbeige-prefill-work-1/astra-t1-affine-bias-versions-1/result.json) ·
  [T1 native operator](artifacts/mlx-gpu-gap-2026-11-15/root-nanbeige-prefill-work-1/modern-t1-metal-1/normal-3.json).
- [ ] Confirm Gemma 4 E2B official BF16 batched T32 in longer quality
  cases and independent ordinary cohorts. The freshly linked current
  console pair returns all 128 IDs with exact public state. First token
  falls from 11.930 to 6.997 s and completion from 39.274 to 34.402 s.
  Decode rises 63 ms in this pair. Peak is 4.65 GiB. Formatting, strict
  Metal Clippy, the parser test and after-flush cancellation/replay pass.
  The original 1,025-position quality runner needs actual early-path
  counts and the original fallback past 512 visible positions.
  Astra clears a detached first-block diagnostic with 240 early Metal
  calls, 915 expected CPU calls and the original raw state references.
  The first start could not read the system guard in the sandbox.
  The guarded retry passes formatting and stops at an omitted tracked
  example protocol in the detached Clippy inputs. Astra verifies the
  append-only repair restores 16 exact tracked inputs. The next build
  reaches the library test and stops because Serde needs a slice for
  the new 35-visit diagnostic. After the reviewed slice conversion,
  formatting, strict Clippy, fresh release linking and the targeted
  geometry test pass. The first 1,024 targets, terminal row and eight
  full anchors match bitwise. All 90 archived chronological K/V
  plates match. The route census gives 240 Metal and 915 CPU visits
  at 4.45 GiB peak. The remaining nine blocks, task suite, public
  prefill cancellation and independent ordinary cohorts are open.
  [Public pair and Astra review](artifacts/mlx-gpu-gap-2026-11-15/sol62-gemma-bf16-t32-current-1/ACTUAL-REVIEW.md) ·
  [Longer-context source review](artifacts/mlx-gpu-gap-2026-11-15/sol62-gemma-bf16-t32-quality-1025-1/REVIEW.md) ·
  [Initial guard exit](artifacts/mlx-gpu-gap-2026-11-15/sol62-gemma-bf16-t32-quality-1025-1/build-1/fmt/result.json) ·
  [Detached Clippy failure](artifacts/mlx-gpu-gap-2026-11-15/sol62-gemma-bf16-t32-quality-1025-1/build-2/clippy/stderr.log) ·
  [Input repair and Astra review](artifacts/mlx-gpu-gap-2026-11-15/sol62-gemma-bf16-t32-quality-1025-1/REPAIR-REVIEW.md) ·
  [Library-test Clippy failure](artifacts/mlx-gpu-gap-2026-11-15/sol62-gemma-bf16-t32-quality-1025-1/build-3/clippy/stderr.log) ·
  [Compiler fix and review](artifacts/mlx-gpu-gap-2026-11-15/sol62-gemma-bf16-t32-quality-1025-1/COMPILE-FIX-REVIEW.md) ·
  [First scored block and Astra review](artifacts/mlx-gpu-gap-2026-11-15/sol62-gemma-bf16-t32-quality-1025-1/ACTUAL-REVIEW.md).
- [ ] Measure bounded request-summary overhead through the original
  protocol. A same-binary replay of the two published Nanbeige owners
  gives a maximum paired or marginal upper drift bound of 0.07581%
  for warmed engine TTFT and completion. On LFM3, all 64 requests
  and 6,144 FFN command brackets pass at 3.72 GiB. After accounting
  for the serial T128 Q/K/V waits, remaining engine prefill contributes
  52.95% and 77.09% of observed TTFT variance. FFN submit shifts
  +11.17 and +4.89 ms with summary enabled across owners. The
  within-owner submission shift changes sign between orders.
  Focus on native evaluation timing before changing the collector.
  The original 1.06524% marginal TTFT upper exceeds the 1% gate.
  The Gemma per-tile T32 disabled/disabled control passes all 32 requests
  across four owners at 4.69 GiB. Baseline marginal 95% upper bounds are
  22.63% for TTFT and 28.44% for completion. Native phase clocks contain
  the owner drift. Hold the full O4 campaign pending a stable baseline.
  [Control results](artifacts/mlx-gpu-gap-2026-11-15/sol62-gemma-bf16-per-tile-o4-readiness-1/run-1/read-normal.json) ·
  [Native phase split](artifacts/mlx-gpu-gap-2026-11-15/sol62-gemma-bf16-per-tile-o4-readiness-1/run-1/phase-localization.json) ·
  [Timing control](artifacts/mlx-gpu-gap-2026-11-15/root-nanbeige-modern-decode-1/timing-control-normal.json) ·
  [LFM3 live diagnostic](artifacts/mlx-gpu-gap-2026-11-15/root-lfm3-o4-brackets-1/RUN.md) ·
  [Astra review](artifacts/mlx-gpu-gap-2026-11-15/root-lfm3-o4-brackets-1/REVIEW.md) ·
  [Gemma disabled-owner protocol](artifacts/mlx-gpu-gap-2026-11-15/sol62-gemma-bf16-per-tile-o4-readiness-1/RUN.md).
- [x] Run the within-owner LFM3 observer crossover. All 64 requests,
  6,144 FFN brackets and 64 selected summary records pass at
  3.67 GiB. Active collection shifts TTFT by −18.62 and +2.98 ms,
  and completion by +8.03 and +16.02 ms. The after-first-token
  difference falls almost entirely within native decode evaluation.
  The guarded release build, two schedule tests and dual raw readers
  pass.
  [Crossover results](artifacts/mlx-gpu-gap-2026-11-15/sol62-lfm3-o4-observer-crossover-1/run-1/normal.json) ·
  [After-first-token attribution](artifacts/mlx-gpu-gap-2026-11-15/sol62-lfm3-o4-observer-crossover-1/run-1/post-first-normal.json) ·
  [Astra review](artifacts/mlx-gpu-gap-2026-11-15/sol62-lfm3-o4-observer-crossover-1/REVIEW.md).
- [x] Evaluate the faster same-weight BF16 LFM3 FFN at its first
  affected output row. Four changed Down inputs on row zero produce
  27 unequal residual words out of 2,048. Canonical controls and
  independent scalar FMA checks pass. The substitution fails exact
  block admission.
  [Astra numerical replay](artifacts/mlx-gpu-gap-2026-11-15/astra-lfm3-external-arithmetic-row-1/RESULT.md).
- [x] Screen four-output T1 ConvIn for canonical LFM3 decode.
  All 6,144 outputs and eleven controls pass. The four-output
  route handles 6,132 rows, with nine scalar-native and three
  ordered rows. Both lower saving estimates clear 0.361 s.
  Strict Swift compilation passes after two driver warning fixes.
  [Screen and timing evidence](artifacts/mlx-gpu-gap-2026-11-15/root-lfm3-t1-convin-fma4-1/RUN.md).
- [x] Measure sampled T1 ConvIn on the published LFM3 route.
  All 52 requests reproduce exact terminal logits and history.
  Median completion improves 5.75% and 4.64%. Both paired lower
  bounds miss 5%. Keep the measured decode gains and close the
  standalone retention trial.
  [Connected results](artifacts/mlx-gpu-gap-2026-11-15/sol62-lfm3-t1-convin-connected-1/RUN.md) ·
  [Astra review](artifacts/mlx-gpu-gap-2026-11-15/sol62-lfm3-t1-convin-connected-1/REVIEW.md).
- [x] Rank completed Nanbeige T128 GPU work across all four chunks.
  Across both guarded orders, 12 requests, 14,928 stage pairs and four
  terminal state witnesses pass at 2.52 GiB. The sampled device clock grows
  by 2.89–2.97%, with 98.01% named coverage. Projections account for
  2.16 s in the FFN and 0.85 s for query and attention output. All GQA
  costs 0.205 s. Seven trace controls and Astra review pass.
  [Profile and verification](artifacts/mlx-gpu-gap-2026-11-15/root-nanbeige-prefill-profile-1/RUN.md).
- [x] Price the authentic Nanbeige T128 query and output operands
  in pinned direct MLX. Both orders reproduce all 1,179,648 output
  words. The 0.167 s Q/O extrapolation falls below the 0.229 s
  development screen.
  [Matched Q/O operands and timing](artifacts/mlx-gpu-gap-2026-11-15/root-nanbeige-prefill-profile-1/price-1/checked-normal.json).
- [x] Price a 64-token tile for Nanbeige T128 gate/up. Both arms
  reproduce all 2,752,512 output words with the original K8 order.
  The wider tile slows all 20 measured pairs. Both orders miss the
  1.303864 ms development screen and close the experiment.
  [Static checks and native screen](artifacts/mlx-gpu-gap-2026-11-15/sol62-nanbeige-gate-up-m64-1/RUN.md).
- [x] Screen fused Nanbeige Gate/Up and SwiGLU on the same M32 input
  tile. The exact K32 and K64 variants each reproduce 1,376,256
  Down-input words. K32 lower paired savings miss the 1.303864 ms
  screen. K64 lower savings of 1.532 and 1.534 ms pass. Native
  guards, fault controls, dual readers and Astra review pass.
  [Fusion screen and review](artifacts/mlx-gpu-gap-2026-11-15/sol62-nanbeige-gate-up-swiglu-fused-k64-1/REVIEW.md).
- [ ] Qualify fused K64 on the frozen Nanbeige 512/32 workload.
  Both orders clear the paired 5% completion threshold and reproduce
  the full published K/V and terminal logits. TTFT improves about 7%.
  The reverse-order decode interval observes a small slowdown.
  Both same-owner factors pass 104 requests and eight exact terminal
  witnesses in total. All four decode intervals span zero. The guarded
  outer specialization preserves the released T1 graph. All 52 ordinary
  requests pass, with paired lower completion gains of 5.39% and 5.33%.
  The reverse order still shows 1.05 ms slower median decode and a
  paired saving interval of −0.190% to −0.053%. The variant stays detached.
  For a standalone decision, compare true prefill switching with a
  released/released sham in the same uninstrumented session. Close
  that line if the sham shows drift or the true effect changes sign.
  [Connected results](artifacts/mlx-gpu-gap-2026-11-15/sol62-nanbeige-gate-up-fused-connected-1/RUN.md) ·
  [Astra review](artifacts/mlx-gpu-gap-2026-11-15/sol62-nanbeige-gate-up-fused-connected-1/REVIEW.md) ·
  [Same-owner diagnosis](artifacts/mlx-gpu-gap-2026-11-15/sol62-nanbeige-fused-t1-crossover-1/RUN.md) ·
  [Astra crossover review](artifacts/mlx-gpu-gap-2026-11-15/sol62-nanbeige-fused-t1-crossover-1/REVIEW.md) ·
  [Outer specialization](artifacts/mlx-gpu-gap-2026-11-15/sol62-nanbeige-fused-outer-dispatch-1/RUN.md) ·
  [Astra specialization review](artifacts/mlx-gpu-gap-2026-11-15/sol62-nanbeige-fused-outer-dispatch-1/REVIEW.md).
- [x] Screen affine-Q4 weight reconstruction across the complete T128
  Nanbeige FFN. The SIMD shuffle matches 1,769,472 authentic words
  and 4,096 analytical one-hot coefficients. It slows all 20 pairs,
  with GPU medians rising from 10.77–10.78 to 15.50–15.51 ms.
  Strict Clippy, release build, native guard and both readers pass.
  Close this mapping and keep the original loader.
  [Astra proposal](artifacts/mlx-gpu-gap-2026-11-15/astra-nanbeige-g64-shuffle-1/PROPOSAL.md) ·
  [Isolated measurements](artifacts/mlx-gpu-gap-2026-11-15/sol62-nanbeige-affine-codebook-1/normal-1.json) ·
  [Astra results review](artifacts/mlx-gpu-gap-2026-11-15/sol62-nanbeige-affine-codebook-1/REVIEW.md).
- [x] Screen a prefill-only BF16 Gate/Up layout on one authentic T128
  FFN visit. It reproduces all 1,769,472 native output words. Every
  measured pair is faster, with paired lower savings of 1.696 and
  1.720 ms against the 1.240 ms threshold. Actual dense allocation is
  132,120,576 bytes. The corrected strict Clippy, release build and
  native guard pass after an ambiguous Metal trait call in the first build.
  [Detached layout and checks](artifacts/mlx-gpu-gap-2026-11-15/sol62-nanbeige-dense-gate-up-1/RUN.md) ·
  [Guarded result](artifacts/mlx-gpu-gap-2026-11-15/sol62-nanbeige-dense-gate-up-1/normal-2.json) ·
  [Astra review](artifacts/mlx-gpu-gap-2026-11-15/sol62-nanbeige-dense-gate-up-1/REVIEW.md).
- [x] Qualify the connected dense T128 prefill graph. Charge all 22
  physical Gate/Up pairs before allocation and retain packed Q4 for
  decode and fallback. The guarded fit passes at 5.06 GiB with exact
  terminal and fallback state. All pairs allocate 2,906,652,672 bytes.
  The 3 GiB control uses packed weights. A later undersized session is
  rejected on the dense owner. The compiled T1 path bypasses the dense
  selector and dispatch. Both ordinary orders pass all 52 requests
  at 5.16 GiB peak. Completion improves 11.72% and 11.38%, with
  paired lower savings of 11.11% and 10.72%. Decode slows by
  16.47 and 13.23 ms, with both paired intervals entirely slower.
  Hold adoption. All 104 same-binary attribution requests pass. The
  residence-only decode effect changes sign. With 22 pairs resident in
  both arms, dense prefill adds 16.92 and 20.58 ms at the paired medians.
  The final-chunk-packed candidate builds and fits at 5.06 GiB.
  All four committed routes and terminal state match the plan. The
  compiled T1 path keeps the released packed encoder. The two ordinary
  orders pass all 52 requests at 5.16 GiB. Completion improves 6.42%
  and 7.19%, with paired lower gains of 6.24% and 6.88%. Both median
  decode latencies improve. The generic opt-in passes below.
  [Guarded model fit](artifacts/mlx-gpu-gap-2026-11-15/sol62-nanbeige-dense-connected-1/fit-normal.json) ·
  [Compiled T1 graph](artifacts/mlx-gpu-gap-2026-11-15/sol62-nanbeige-dense-connected-1/ARM64-REVIEW.md) ·
  [Ordinary timings](artifacts/mlx-gpu-gap-2026-11-15/sol62-nanbeige-dense-connected-1/normal.json) ·
  [Astra connected review](artifacts/mlx-gpu-gap-2026-11-15/sol62-nanbeige-dense-connected-1/REVIEW.md) ·
  [Decode attribution and Astra review](artifacts/mlx-gpu-gap-2026-11-15/sol62-nanbeige-dense-causal-1/REVIEW.md) ·
  [Final-chunk fit](artifacts/mlx-gpu-gap-2026-11-15/sol62-nanbeige-dense-tail-packed-1/fit-normal.json) ·
  [Compiled final-chunk test](artifacts/mlx-gpu-gap-2026-11-15/sol62-nanbeige-dense-tail-packed-1/ARM64-REVIEW.md) ·
  [Ordinary final-chunk review](artifacts/mlx-gpu-gap-2026-11-15/sol62-nanbeige-dense-tail-packed-1/REVIEW.md).
- [x] Qualify the generic Nanbeige dense opt-in on Apple M1 Pro. Signed startup,
  6 GiB fit, packed fallback and four alternate prompt lengths pass.
  The compiled library and CLI retain the packed T1 path. The real
  CLI preserves its default wire schema and charges no dense memory
  without a later prompt chunk. One cold CLI pair is descriptive.
  Both warm orders pass all 52 exact requests. Completion improves
  7.14% and 7.08%, with paired lower savings above 6.55%. Both
  decode medians improve at 5.16 GiB peak. Astra approves the candidate
  for final checks. Two initial strict build failures and a runner
  JSON key mismatch were corrected before hardware qualification.
  The fault-only build omitted a tracked test-support module. The
  complete checkout passes formatting, strict Clippy in both modes,
  869 default tests, 1,074 Metal tests and four explicit GPU tests.
  Waited cancellation and status preserve exact committed state.
  Failed startup parity allocates zero pairs and falls back exactly.
  Astra approves the exact bounded opt-in. Pushed `0e5f6378`.
  [Generic fit](artifacts/mlx-gpu-gap-2026-11-15/sol62-nanbeige-dense-option-1/fit-normal.json) ·
  [Alternate and cold controls](artifacts/mlx-gpu-gap-2026-11-15/sol62-nanbeige-dense-option-1/aux-normal.json) ·
  [Compiled T1 proof](artifacts/mlx-gpu-gap-2026-11-15/sol62-nanbeige-dense-option-1/ARM64-REVIEW.md) ·
  [Ordinary timings and Astra review](artifacts/mlx-gpu-gap-2026-11-15/sol62-nanbeige-dense-option-1/REVIEW.md) ·
  [All nine release checks](artifacts/mlx-gpu-gap-2026-11-15/root-nanbeige-dense-release-checks-1/normal-2.json) ·
  [Astra release review](artifacts/mlx-gpu-gap-2026-11-15/root-nanbeige-dense-release-checks-1/RELEASE-REVIEW.md).
- [x] Capture the authentic first-decode Gate/Up operands for the exact
  activation-sum experiment. The guarded owner captures all 24,576
  BF16 words at position 512, plus exact terminal state. The proposed
  shared sum must keep the original rounding and exceed 14.663 µs
  paired lower savings per visit in both operator orders before integration.
  [Capture and Astra review](artifacts/mlx-gpu-gap-2026-11-15/sol62-nanbeige-t1-sum16-capture-1/REVIEW.md).
- [x] Price exact T1 Gate/Up activation-sum reuse. All 21,504 BF16
  outputs, 192 sums and fault controls match. The two paired savings
  intervals are 46.13 to 91.27 µs and −17.08 to 70.00 µs. The reverse
  order misses the 14.663 µs lower bound. Astra confirms closure.
  [Operator result and Astra review](artifacts/mlx-gpu-gap-2026-11-15/sol62-nanbeige-t1-sum16-operator-1/REVIEW.md).
- [x] Price the next exact prefill change against the 4.29 s request.
  Standalone dense Down exceeds the resident-bank budget at full coverage.
  Partial coverage would need roughly 1.02 ms per visit against the best
  measured exact 3.28 ms. Astra defers layer-major execution until the
  current route shows enough removable GPU time. Keep the final packed
  chunk and the existing reusable session publication semantics.
  The next attribution design caps encoder splits and prices sampling
  overhead separately before using its phase timings.
  [Prefill analysis and Astra review](artifacts/mlx-gpu-gap-2026-11-15/sol62-nanbeige-next-prefill-brief-1/REVIEW.md) ·
  [Prospective attribution design](artifacts/mlx-gpu-gap-2026-11-15/sol62-nanbeige-dense-attribution-1/PLAN.md).
- [x] Calibrate sparse GPU stage timestamps. One detached build linter
  warning was fixed. The completed owner matches all request and state
  witnesses. Encoder splits shift TTFT by up to 92.23 ms against a
  21.451 ms limit. Timestamp sampling adds a separate shift up to
  17.48 ms. The second owner stops at the disk floor. Astra rejects
  the timing as a basis for ranking kernels. M1 Pro exposes stage-boundary
  timestamps and reports dispatch-boundary sampling unavailable.
  [Capture and Astra review](artifacts/mlx-gpu-gap-2026-11-15/sol62-nanbeige-dense-attribution-1/CAPTURE-REVIEW.md) ·
  [Metal capabilities](artifacts/mlx-gpu-gap-2026-11-15/sol62-nanbeige-dense-attribution-1/dispatch-capabilities.txt).
- [ ] Qualify the original 100 tasks and 10,000 held-out Nanbeige
  targets. The 100 execution inputs now use the pinned Nanbeige codec
  and preserve original task content, categories, output caps and grading.
  Code and prose disable thinking. Continuation remains raw. The
  shipped Rust codec independently reproduces all 100 inputs.
  The guarded same-Q4 scoring smoke passes 32 held-out targets and
  two separate EOS controls. Toklet loss is 0.02357 nats/token higher
  than pinned MLX on this sample with two greedy disagreements.
  The first full run stopped after 4,512 scores when another compiler
  entered the guard. The unchanged retry scores all 10,000 targets.
  Mean Toklet loss is 0.00234 nats/token lower than pinned MLX, with
  435 greedy disagreements. Both reproduce the smoke scores and
  finish under 3.31 GiB. Astra independently verifies the full corpus.
  The original tasks and unquantized-reference gates remain open.
  Astra verifies that the BF16 parent requires 7.77 GiB for weights
  alone. A guarded layer-at-a-time Q4 walk exactly reproduces 1,058
  scores and all terminal state across four cases, including 1,025
  committed positions, at 2.80 GiB peak. An 8 MiB read verifies direct
  BF16 range transport. The guarded transfer authenticates both official
  shards, 8.34 GB of source data, and 201 tensor spans at 129 MiB peak.
  It retains only hash tables and the frozen embedding rows.
  The paired scorer stages the full corpus and smoke as independent
  cache cases in one weight traversal. Actual BF16 scoring remains open.
  [Frozen task inputs](artifacts/mlx-gpu-gap-2026-11-15/root-nanbeige-quality-tasks-1/RUN.md) ·
  [Frozen loss windows](artifacts/mlx-gpu-gap-2026-11-15/root-nanbeige-mlx-q4-admission-1/heldout-normal.json) ·
  [Held-out smoke and Astra review](artifacts/mlx-gpu-gap-2026-11-15/sol62-nanbeige-heldout-fidelity-1/SMOKE-REVIEW.md) ·
  [Full same-Q4 results and Astra review](artifacts/mlx-gpu-gap-2026-11-15/sol62-nanbeige-heldout-fidelity-1/FULL-REVIEW.md) ·
  [Interrupted guard receipt](artifacts/mlx-gpu-gap-2026-11-15/sol62-nanbeige-heldout-fidelity-1/full-1-interrupted-external-compile/toklet-guard/result.json) ·
  [BF16 reference proposal](artifacts/mlx-gpu-gap-2026-11-15/astra-nanbeige-bf16-quality-1/PROPOSAL.md) ·
  [Layerwise Q4 and Astra review](artifacts/mlx-gpu-gap-2026-11-15/root-nanbeige-layerwise-oracle-1/REVIEW.md) ·
  [Parent acquisition and scoring staging](artifacts/mlx-gpu-gap-2026-11-15/root-nanbeige-layerwise-oracle-1/RUN.md) ·
  [Paired BF16 source review](artifacts/mlx-gpu-gap-2026-11-15/root-nanbeige-layerwise-oracle-1/PARENT-PAIRED-REVIEW.md) ·
  [Verified parent acquisition](artifacts/mlx-gpu-gap-2026-11-15/root-nanbeige-layerwise-oracle-1/ACQUISITION-REVIEW.md).
- [x] Price the pinned MLX affine T128 PSO for Q, O and Down together.
  The guarded build, signed basis, archived panels and both corruption
  controls pass. With all three status scans charged, the paired lower
  GPU savings are 2.009 and 2.004 ms against a 1.219 ms screen. Astra
  approves a detached connected trial.
  [Pinned ABI and inputs](artifacts/mlx-gpu-gap-2026-11-15/astra-nanbeige-qmm-three-roles-1/PROPOSAL.md) ·
  [Guarded result and Astra review](artifacts/mlx-gpu-gap-2026-11-15/sol62-nanbeige-qmm-three-roles-1/ACTUAL-REVIEW.md).
- [x] Retain the compact Nanbeige T128 Q/O/Down opt-in on M1 Pro.
  The 13,472-byte image authenticates from the same owned Metal input.
  Fit, fallback, CLI and linked packed T1 checks pass at 5.17 GiB peak.
  The first ordinary run observes 20.80 ms slower decode in one order.
  The resident crossover excludes that magnitude in both owners. The
  fixed confirmation improves completion by 8.63% and 8.45%. Both decode
  intervals cross zero. One cold CLI case takes 13.738 s versus 12.570 s
  for dense. The repaired check archive passes all 12 groups, including
  870 default tests, 1,075 Metal tests and seven explicit GPU controls.
  Astra approves the scoped warm opt-in and exact packaged image. The
  packaged relative path admits the image and commits 256 prompt rows
  before the disk guard cancels its additional smoke run.
  [Usage and packaged Metal image](assets/nanbeige-qmm/README.md) ·
  [Initial product results](artifacts/mlx-gpu-gap-2026-11-15/sol62-nanbeige-qmm-product-1/ACTUAL-REVIEW.md) ·
  [Source archive repair](artifacts/mlx-gpu-gap-2026-11-15/sol62-nanbeige-qmm-product-checks-1/REPAIR-REVIEW.md) ·
  [Archived Metal disk-floor stop](artifacts/mlx-gpu-gap-2026-11-15/sol62-nanbeige-qmm-product-checks-1/tests-1/test-metal-disk-floor-1/result.json) ·
  [All 12 release checks](artifacts/mlx-gpu-gap-2026-11-15/sol62-nanbeige-qmm-product-checks-1/checks-final-normal.json) ·
  [Packaged path admission and guard stop](artifacts/mlx-gpu-gap-2026-11-15/sol62-nanbeige-qmm-product-1/cli-packaged-1-summary.json) ·
  [Final confirmation and Astra approval](artifacts/mlx-gpu-gap-2026-11-15/sol62-nanbeige-qmm-product-confirmation-1/ACTUAL-REVIEW.md).
- [ ] Verify the native 32-head work schedule before external timing.
  The detached composition retains packed T1 and the qualified T128
  graphs. The staged witness checks all retained heads and cache state,
  actual QMM and finite scans, and failures after an unscored T128.
  After retiring download and compiler caches, the guarded release build
  and five untimed owners pass. Both readers agree on the 32 retained
  scores, sampled IDs and 44 terminal K/V pairs at 5.10 GiB peak.
  Astra verifies the actual run and scope. All 32 IDs agree with MLX,
  while 0 of 32 logit digests and 0 of 44 terminal pairs agree. The
  detached timing build passes release and three tape/CLI rejection
  checks after a sandbox guard retry. Both readers verify 78 guarded
  requests at 5.17 GiB peak. Toklet completes in 4.506–4.546 s,
  behind Ollama by 0.768–0.801 s. TTFT accounts for about 75% of
  that gap. The guarded first T128 matches embedding and RMSNorm.
  Before RoPE, K differs in 14 words and V in 26. Both readers agree
  and the captured V exactly matches the archived native operator.
  The detached build passes after the guard and two compile retries.
  Targeted rustfmt passes. Whole-package formatting needs the omitted
  test fixture. Astra approves the trace. The MLX 0.32.4 split-K replay
  now matches both T128 panels bitwise, and both readers reproduce
  the ordered FP32 merge. Formatting, strict Clippy and the guarded
  offline build and replay pass after the sandbox retry. Astra clears
  both probes. Ordinary T384 MLX matches today's Toklet output on
  repeated input and changes the T128 reference. Connected split-K
  now matches the live first-cut Q/K/V and K/V cache and the complete
  slot-zero 512-prefix. All 32 IDs agree. Zero scored rows and zero
  terminal pairs join MLX. Five rotated-Q words differ. Causal
  attention differs in 234 words on rows with exact visible inputs.
  The same-build control reproduces the incumbent exactly. Astra
  approves both runs. Restoring the detached fixtures clears format,
  strict all-target Metal Clippy and the targeted topology test.
  Reordering the first rotary FMA fixes all five Q differences on the
  frozen T128 cut. Same-operand attention still differs in 237 words.
  The matched Gate/Up reuse candidate is slower in all 40 pairs, so
  keep the incumbent. Both readers, hardware guards and Astra review
  pass. Connected Steel and corrected rotary match all nine first-cut
  panels, the first cache owner at all four prefill lengths and all 32
  sampled IDs. The full request still matches 0/32 scored logits and
  0/44 terminal caches. Two proof owners report 2.443 s and 2.580 s
  on the GPU. The Gate/Up shape control shows no clear gain. The
  guarded release, scoped format and strict Metal Clippy pass after
  the sandbox guard retry. Astra approves the detached evidence.
  The FFN trace localizes 451,662 differences to SwiGLU. Explicit
  BF16 sigmoid rounding fixes all eight traced T128 stages through
  the next raw K/V. Ten cache owners match at 128 tokens, then only
  slot zero matches at 256, 384 and 512. All 32 IDs agree. Astra approves.
  The second cut first differs at FFN RMSNorm, by 20 BF16 words.
  The pinned T128 RMS reduction fixes all 19 traced panels and all
  44 cache owners at each of four cuts. Astra approves. All IDs agree.
  The final prefill hidden row matches. Selective QMV closes the first
  score and preserves all four prefill cuts and 32 IDs. On the first T1
  exact input, FP32 bias sums predict all 8,192 MLX Q/K/V outputs. The
  remaining 31 score rows and terminal parity stay open. Next apply
  and qualify the grouped-vector sum in T1. Astra approves the trace.
  The detached FP32 bias sum leaves one Q word. MLX attention matches all
  6,144 words on the same Q/K/V. Check the halves reduction before replacing
  attention for numerical parity.
  [Decode bias and attention replay](artifacts/mlx-gpu-gap-2026-11-15/root-nanbeige-t1-bias-1/RUN.md).
  [First score and T1 trace](artifacts/mlx-gpu-gap-2026-11-15/root-nanbeige-head-decode-1/RUN.md).
  [RMS and complete prefill cache evidence](artifacts/mlx-gpu-gap-2026-11-15/root-nanbeige-second-cut-1/RUN.md).
  [FFN and cache frontier](artifacts/mlx-gpu-gap-2026-11-15/root-nanbeige-ffn-connection-1/RUN.md).
  [Steel and FFN shape results](artifacts/mlx-gpu-gap-2026-11-15/root-nanbeige-steel-connected-1/normal.json).
  [Rotary and attention](artifacts/mlx-gpu-gap-2026-11-15/root-nanbeige-attention-replay-1/RUN.md) ·
  [Gate/Up timing](artifacts/mlx-gpu-gap-2026-11-15/root-nanbeige-dense-tile-reuse-1/RUN.md).
  [Connected comparison and checks](artifacts/mlx-gpu-gap-2026-11-15/root-nanbeige-connected-splitk-1/RUN.md).
  [K/V and shape evidence](artifacts/mlx-gpu-gap-2026-11-15/root-nanbeige-kv-parity-1/RUN.md).
  [First native divergence](artifacts/mlx-gpu-gap-2026-11-15/root-nanbeige-native-first-cut-1/normal.json) ·
  [Astra review](artifacts/mlx-gpu-gap-2026-11-15/root-nanbeige-native-first-cut-1/REVIEW.md).
  [Three-engine diagnostic](artifacts/mlx-gpu-gap-2026-11-15/root-nanbeige-toklet-clocks-1/normal.json).
  [Verified witness](artifacts/mlx-gpu-gap-2026-11-15/sol62-nanbeige-qmm-head32-1/witness-normal.json) ·
  [Disk recovery](artifacts/mlx-gpu-gap-2026-11-15/root-head32-storage-recovery-2.json) ·
  [Detached source review](artifacts/mlx-gpu-gap-2026-11-15/sol62-nanbeige-qmm-head32-1/REVIEW.md) ·
  [Physical witness plan](artifacts/mlx-gpu-gap-2026-11-15/sol62-nanbeige-qmm-head32-1/WORK-WITNESS.md) ·
  [Astra witness review](artifacts/mlx-gpu-gap-2026-11-15/sol62-nanbeige-qmm-head32-1/WITNESS-REVIEW.md).
- [x] Price the remaining Nanbeige gap against the direct MLX GPU graph.
  Both owners reproduce the 32 released IDs on all 26 requests.
  Direct MLX finishes in 3.315–3.316 s versus Toklet's 5.841–5.842 s.
  Prefill accounts for about 92% of the gap in median completion.
  Exact T128 gate/up operands match 2,752,512 words. Direct MLX takes
  3.21–3.22 ms per projection through completion. Toklet's device
  time is 5.36–5.37 ms.
  [Direct timing](artifacts/mlx-gpu-gap-2026-11-15/root-nanbeige-modern-decode-1/direct-normal.json).
- [x] Qualify a narrower-threadgroup T128 gate/up tile using fewer shared
  bytes and lane-local output. Preserve BF16 reconstruction, FP32
  accumulation order and the matched output panel. Require isolated
  and connected timing under the existing memory and numerical gates.
  A guarded native fragment probe verifies FP32 and FP16 normal and
  transposed layouts on this M1 Pro. The candidate keeps independent
  panel and analytical controls. Both compact variants match all
  2,752,512 words. The direct GPU pair falls from 10.72 ms to 8.53–8.54 ms,
  and Astra approves the connected T128 trial. The 52 matched requests
  pass at 2.47 GiB with 7.01% and 7.07% completion gains. Both paired
  intervals clear 5%. Full Metal, both targeted GPU controls, strict
  Clippy, archived parity and CLI readiness pass. Astra approves the
  exact five-file experimental Mac scope.
  [Fragment probe](artifacts/mlx-gpu-gap-2026-11-15/root-metal-fragment-map-1/RUN.md) ·
  [Matched operator evidence](artifacts/mlx-gpu-gap-2026-11-15/root-nanbeige-modern-decode-1/direct-ffn-normal.json) ·
  [Compact panel and timing](artifacts/mlx-gpu-gap-2026-11-15/root-nanbeige-prefill-work-1/sol-ffn-compact-128-1/summary-normal.json) ·
  [Connected trial](artifacts/mlx-gpu-gap-2026-11-15/root-nanbeige-ffn-compact-connected-1/RUN.md) ·
  [Exact checks](artifacts/mlx-gpu-gap-2026-11-15/root-nanbeige-prefill-work-1/release-compact-candidate-1/final-check-summary.json).
- [x] Qualify compact K64 down on the same Nanbeige 512/32 requests.
  Direct MLX matches all 393,216 BF16 output words at 3.28 ms.
  K64 reduces native down GPU medians from 5.50–5.58 to 3.73 ms.
  Both paired lower savings exceed 1.74 ms per visit. Astra approves
  a connected T128 Mac trial with an eight-byte input alignment check.
  Both connected orders clear 5% with all 52 requests, all 44 K/V
  prefixes and terminal logits exact. Formatting, both strict Clippy
  runs, 1,074 Metal tests, both GPU controls and archived parity pass.
  The library retry passes the same disk guard after cache cleanup.
  Astra approves the exact experimental Mac change.
  [Matched MLX](artifacts/mlx-gpu-gap-2026-11-15/root-nanbeige-down-reference-1/RUN.md) ·
  [K64 operator](artifacts/mlx-gpu-gap-2026-11-15/root-nanbeige-prefill-work-1/sol-down-k64-1/normal.json) ·
  [Connected trial](artifacts/mlx-gpu-gap-2026-11-15/root-nanbeige-down-connected-1/RUN.md) ·
  [Exact checks](artifacts/mlx-gpu-gap-2026-11-15/root-nanbeige-prefill-work-1/release-down-candidate-1/final-check-summary.json) ·
  [Astra review](artifacts/mlx-gpu-gap-2026-11-15/root-nanbeige-prefill-work-1/release-down-candidate-1/REVIEW.md) ·
  [Disk recovery](artifacts/mlx-gpu-gap-2026-11-15/root-nanbeige-prefill-work-1/release-down-candidate-1/disk-floor-cache-recovery.json).
- [x] Price the T128 query projection using the existing compact tile.
  Both guarded orders match all 786,432 BF16 words and save 1.31–1.32 ms
  per visit at the paired GPU median. Astra approves the isolated evidence.
  Its savings fall below the request screen.
  [Token and weight spans](artifacts/mlx-gpu-gap-2026-11-15/root-nanbeige-query-input-1/normal.json) ·
  [Query timing](artifacts/mlx-gpu-gap-2026-11-15/sol62-nanbeige-query-compact-1/normal-1.json).
- [x] Price query and attention output together in one T128 layer-zero
  command. Both native variants match 3,276,800 BF16 words, including
  the original K/V prefixes and causal controls. Both paired GPU lower
  savings exceed 2.70 ms per visit and clear the 1.452 ms screen.
  Astra approves the scoped connected trial. Both O variants also
  match all 393,216 words on the separately frozen MLX O input.
  [Pinned attention banks](artifacts/mlx-gpu-gap-2026-11-15/root-nanbeige-attention-input-1/banks-normal.json) ·
  [Captured attention operands](artifacts/mlx-gpu-gap-2026-11-15/root-nanbeige-attention-input-1/normal.json) ·
  [Joined command](artifacts/mlx-gpu-gap-2026-11-15/sol62-nanbeige-attention-pair-1/normal.json).
- [x] Qualify the paired T128 query and output route on the frozen
  512/32 requests. Admit both existing shaders and validate the actual
  attention buffer's GPU address once during session creation.
  All 52 matched requests pass with exact terminal state and 9.71%
  and 9.86% lower median completion. Strict Clippy, CLI stats and both
  archived model partitions pass. The guarded retry passes all
  1,074 Metal tests and both GPU controls after cache recovery.
  Astra approves the exact experimental Mac change.
  [Connected trial](artifacts/mlx-gpu-gap-2026-11-15/root-nanbeige-attention-connected-1/RUN.md) ·
  [Exact checks](artifacts/mlx-gpu-gap-2026-11-15/root-nanbeige-prefill-work-1/release-attention-candidate-1/final-check-summary.json) ·
  [Astra review](artifacts/mlx-gpu-gap-2026-11-15/root-nanbeige-prefill-work-1/release-attention-candidate-1/REVIEW.md) ·
  [Cache recovery](artifacts/mlx-gpu-gap-2026-11-15/root-nanbeige-prefill-work-1/release-attention-candidate-1/cache-recovery-after-disk-stop.json).
- [x] Price K64 for Nanbeige's T128 gate and up projections together.
  All 2,752,512 BF16 outputs match. Both paired lower savings
  of 1.196 and 1.225 ms miss the 1.311 ms development screen.
  Astra approves retaining the evidence and closing the standalone trial.
  [Gate and up comparison](artifacts/mlx-gpu-gap-2026-11-15/sol62-nanbeige-gate-up-k64-pair-1/RUN.md).
- [x] Measure T128 grouped-query attention at past lengths 0, 128, 256
  and 384. The guarded layer-zero baseline matches all 2,621,440 K/V
  words and all 104 timed outputs. Four-cut medians sum to 6.203
  and 4.654 ms. The reverse order falls below the 5.250 ms
  zero-cost screen. Astra approves closing the isolated experiment.
  [Four-depth baseline](artifacts/mlx-gpu-gap-2026-11-15/sol62-nanbeige-gqa-offsets-1/RUN.md).
- [ ] Admit external Nanbeige engines on exactly matching work.
  The installed vLLM Metal source replay takes only 22 of the 44 model
  visits. Its paged KV wrapper also aliases physical indexes if given
  44 arguments. The publisher's Ollama fork preserves packed weights
  and cache ownership, with a different public prefill/drain schedule.
  The Astra-reviewed vLLM source replay verifies 44 distinct cache
  owners and offsets, norm after visits 22 and 44, and a scheduler
  charge of 180,224 bytes per BF16 position. The guarded local
  overlay smoke verifies the 44 regions, all 32 output IDs, zero
  reuse, and 4×128 plus 31×1 scheduled rows at 4.27 GiB peak.
  vLLM scores 32 retained rows across 35 forwards. The published
  Toklet route scores 35. The Astra-reviewed detached route scores
  32 rows for fresh one-shot sessions.
  The guarded 32-row head trial matches all 32 IDs and 44 published
  Toklet KV prefixes. The vLLM terminal witness retains all 44
  committed KV sets under 6 GiB. vLLM and direct MLX match both
  slot-zero cache exactly. At position 512, installed paged SDPA
  differs from direct MLX on 1,543 of 6,144 outputs on the same
  Q and 513 K/V positions. A private dense replay restores the
  entire slot-one append. The other positions and numerical gates
  remain open before external performance cohorts.
  Astra clears an untimed private dense replay for all 1,540 visits
  with the original sampler and 32 scored rows. It retains the exact
  first divergent prefix and all observed cache-pair hashes. The replay
  completes at 3.78 GiB. All 44 prefill and 44 terminal K/V pairs match
  direct MLX. The 31 decode heads match exactly. The same-head replay
  explains all 36,106 differences at the first score. Projection after
  selecting one row reproduces the live difference. Projection on the
  same 128-row panel reproduces all 166,144 independent oracle words.
  The shadow-free direct MLX and private vLLM witnesses now match all
  32 scores and both 44-owner cache snapshots at 3.18 and 3.86 GiB.
  The lean worker keeps dense selection independent of its observers.
  Both timing orders verify all 52 requests. Direct MLX completes in
  3.312–3.316 s at 3.18 GiB peak. Private vLLM takes 3.631–3.640 s
  at 3.92 GiB. Most of that difference occurs during decode. Qualify
  the native 32-head route before using these clocks as a Toklet gate.
  The separately guarded local vLLM timing diagnostic completes
  26 requests at 3.539–3.540 s median completion, 2.489 s first
  ID, and 3.76 GiB peak. The tested Toklet owners score more rows
  and use different attention arithmetic.
  The detached CLI now verifies 35 versus 32 scored rows on physical
  Metal with identical output IDs, JSON schema, 44 K/V pairs and
  terminal logits. Post-completion cancellation, rejected body status
  and broken stdout pass their guarded controls. The three omitted
  projections and reductions account for about 38 ms of an earlier
  device trace. Timing and external attention qualification remain open.
  The pinned stock Ollama MLX registry lacks Nanbeige. The publisher
  fork has 44 cache owners, and its public path reserves a prompt seed
  and evaluates one extra decode position. An untimed raw-ID probe on
  its existing worker can check exact imported Q4 weights and work.
  Astra clears the detached worker source. The exact checkout and private
  Go and CMake tools are staged. Native configure stops because Xcode's
  optional Metal compiler was inactive. The official activation restores
  actual Metal preprocessing and retains the active developer directory.
  The guarded retry builds the exact patched Go binary and native Metal
  companions. Their hashes and CMake and Go identity are sealed. Measured
  free disk clears the private import preflight after cache retirement.
  Retiring the private Go download cache recovers another 427 MiB
  after rehashing the five retained native build files.
  The private import verifies all 513 tensor spans and four codec files.
  The untimed raw replay checks all 32 greedy IDs, 35 scored rows and
  44 cache owners at 3.47 GiB. Full terminal K/V hashes differ in all 44
  pairs from both archived references. The raw comparison checks the
  available decode slices and terminal head along each engine's own
  calculation.
  The guarded 32-head Ollama replay uses its own sampled IDs, preserves
  the earlier Ollama terminal cache and logits and peaks at 3.46 GiB.
  All 32 scored-head digests differ from matched MLX. A guarded direct
  prefill rejoins all 44 reference pairs. The first Ollama cache owner
  already differs at position 0 in 429 K and 502 V words. Both raw
  readers agree at 3.12 GiB. The sandbox denied the first guard start.
  Live embedding and RMSNorm match bitwise. The shared T128 activation
  reproduces 55,567 K and 67,198 V differences before RoPE. All V words
  join archived MLX 0.32.3. Sol's detached build passes on retry after
  the sandbox guard denial. Both GPU guards and paired readers pass.
  Astra approves the probe. The detached MLX 0.32.4 bridge configures
  and hits two compile-cache API changes. Sol adapts callers to the
  calling-thread cache and forwards the two new operator defaults.
  The private bridge links after the sandbox and API retries. Four
  smoke commands pass. The unchanged Go worker loads the sealed
  modern companion and joins all 32 scores, IDs and 44 prefill and
  terminal cache pairs at 3.51 GiB. The prefill check uses the 512-row
  prefixes of the terminal cache. Astra approves private timing after
  the complete source-diff check. Both raw readers agree.
  Sol's timing worker builds after a sandbox guard retry and Astra
  approves its clock and plan. All 52 requests pass in two orders.
  Ollama completes in 3.267/3.291 s and vLLM in 3.621/3.622 s.
  Guarded peaks are 3.52 GiB and 3.92 GiB. Next time the current
  Toklet 32-head path against these private routes.
  [Matched private clocks](artifacts/mlx-gpu-gap-2026-11-15/root-nanbeige-ollama-timing-1/normal.json).
  [Timing admission](artifacts/mlx-gpu-gap-2026-11-15/root-nanbeige-ollama-mlx324-1/timing-admission.json).
  [Modern Ollama replay](artifacts/mlx-gpu-gap-2026-11-15/root-nanbeige-ollama-mlx324-1/normal.json).
  [Bridge preflight](artifacts/mlx-gpu-gap-2026-11-15/root-nanbeige-ollama-layer0-1/mlxc-wheel-preflight-1/receipt.json).
  [Shared-input layer zero](artifacts/mlx-gpu-gap-2026-11-15/root-nanbeige-ollama-layer0-1/normal.json).
  [First prefill difference](artifacts/mlx-gpu-gap-2026-11-15/root-nanbeige-ollama-prefill-cut-1/check-normal.json).
  [32-head Ollama witness](artifacts/mlx-gpu-gap-2026-11-15/sol62-ollama-nanbeige-raw-worker-1/matched32/run-1/read-normal.json) ·
  [Local vLLM clocks](artifacts/mlx-gpu-gap-2026-11-15/root-nanbeige-vllm-direct-clocks-1/RUN.md) ·
  [Astra clock review](artifacts/mlx-gpu-gap-2026-11-15/root-nanbeige-vllm-direct-clocks-1/REVIEW.md) ·
  [Detached CLI proposal](artifacts/mlx-gpu-gap-2026-11-15/sol62-nanbeige-head-cli-1/RUN.md) ·
  [Astra CLI review](artifacts/mlx-gpu-gap-2026-11-15/sol62-nanbeige-head-cli-1/REVIEW.md) ·
  [Real CLI validation and review](artifacts/mlx-gpu-gap-2026-11-15/sol62-nanbeige-head-cli-validation-1/REVIEW.md).
  [Head work and terminal custody](artifacts/mlx-gpu-gap-2026-11-15/sol62-nanbeige-head-state-1/RUN.md) ·
  [Astra first-decode isolation review](artifacts/mlx-gpu-gap-2026-11-15/sol62-nanbeige-vllm-first-decode-attention-1/REVIEW.md) ·
  [Full dense diagnostic source review](artifacts/mlx-gpu-gap-2026-11-15/sol62-nanbeige-vllm-full-dense-attention-1/REVIEW.md) ·
  [Guarded dense diagnostic](artifacts/mlx-gpu-gap-2026-11-15/sol62-nanbeige-vllm-full-dense-attention-1/normal-1.json) ·
  [Head shape diagnostic and Astra review](artifacts/mlx-gpu-gap-2026-11-15/sol62-nanbeige-vllm-first-head-shape-1/ACTUAL-REVIEW.md) ·
  [Matched MLX and vLLM work and Astra review](artifacts/mlx-gpu-gap-2026-11-15/sol62-nanbeige-mlx-matched-32-1/ACTUAL-REVIEW.md) ·
  [Matched private clocks and Astra review](artifacts/mlx-gpu-gap-2026-11-15/sol62-nanbeige-mlx-matched-32-timing-1/ACTUAL-REVIEW.md) ·
  [Ollama terminal numeric comparison](artifacts/mlx-gpu-gap-2026-11-15/sol62-ollama-nanbeige-raw-worker-1/TERMINAL-NUMERIC.md) ·
  [Astra external numerical review](artifacts/mlx-gpu-gap-2026-11-15/sol62-nanbeige-qmm-connected-1/ACTUAL-REVIEW.md).
  [External admission proof](artifacts/mlx-gpu-gap-2026-11-15/astra-nanbeige-external-admission-1/result.json) ·
  [Logical loop replay](artifacts/mlx-gpu-gap-2026-11-15/sol62-nanbeige-double-loop-1/RUN.md) ·
  [Reviewed overlay](artifacts/mlx-gpu-gap-2026-11-15/sol62-nanbeige-vllm-live-overlay-1/RUN.md) ·
  [Live admission](artifacts/mlx-gpu-gap-2026-11-15/root-nanbeige-vllm-overlay-smoke-1/RUN.md) ·
  [Ollama worker proposal](artifacts/mlx-gpu-gap-2026-11-15/sol62-nanbeige-ollama-matched-plan-1/DRAFT.md) ·
  [Ollama source review](artifacts/mlx-gpu-gap-2026-11-15/sol62-ollama-nanbeige-raw-worker-1/REVIEW.md) ·
  [Actual configure stop](artifacts/mlx-gpu-gap-2026-11-15/sol62-ollama-nanbeige-raw-worker-1/configure-guard-1/result.json) ·
  [Verified Metal compiler](artifacts/mlx-gpu-gap-2026-11-15/sol62-ollama-nanbeige-raw-worker-1/xcode-metal-compiler-verified.json) ·
  [Exact guarded fork build](artifacts/mlx-gpu-gap-2026-11-15/sol62-ollama-nanbeige-raw-worker-1/build-receipt.json) ·
  [Disk fit and scoped reclaim](artifacts/mlx-gpu-gap-2026-11-15/sol62-ollama-nanbeige-raw-worker-1/go-cache-retired.json) ·
  [Post-build download cache reclaim](artifacts/mlx-gpu-gap-2026-11-15/sol62-ollama-nanbeige-raw-worker-1/go-module-cache-retired-after-build.json) ·
  [Verified private import](artifacts/mlx-gpu-gap-2026-11-15/sol62-ollama-nanbeige-raw-worker-1/import-1/verified.json) ·
  [Actual raw replay](artifacts/mlx-gpu-gap-2026-11-15/sol62-ollama-nanbeige-raw-worker-1/trial-1/normal.json) ·
  [Private import and raw replay](artifacts/mlx-gpu-gap-2026-11-15/sol62-ollama-nanbeige-raw-worker-1/IMPORT.md).
- [x] Price a resident T128 LFM3 convolution handoff on the original
  matched tape. The previous ConvOut tile saved about 0.83 s of prefill
  and added 0.10–0.12 s to decode. Start with the authenticated gated
  outputs and private history. Keep the dense BF16 reduction and the
  signed-zero fallback. The guarded CPU reconstruction matches all
  1,050,624 gated words across four adjacent prefill cuts and T1/512,
  including 342 negative zeros. Astra verified history custody.
  The native handoff reproduces all five cuts and proposed histories.
  Holding the output panel fixed gives hypothetical lower prefill
  savings of 0.112–0.121 s over 52 visits, below the 0.452 s screen.
  The separate T1 test gives mixed timing. Astra approves closing
  the isolated experiment.
  [Native handoff](artifacts/mlx-gpu-gap-2026-11-15/sol62-lfm3-connected-conv-chain-2/normal.json) ·
  [Exact operands](artifacts/mlx-gpu-gap-2026-11-15/root-lfm3-conv-old-history-1/normal.json) ·
  [Consecutive prefix](artifacts/mlx-gpu-gap-2026-11-15/root-lfm3-convolution-prefix-1/RUN.md) ·
  [Prior crossover](artifacts/mlx-gpu-gap-2026-11-15/sol62-lfm3-release4b-convout-crossover-1/recovery-normal.json).

### Paused LFM3 implementation work

Native carrier orchestration stays paused after the resident
convolution handoff misses the request-level development screen.
[Native trial](artifacts/mlx-gpu-gap-2026-11-15/sol62-lfm3-connected-conv-chain-2/RUN.md).

### Earlier task records

- [ ] **SenseNova-Loop-9B text CLI.** Twenty targeted checks and six console
  preflight checks pass. The explicit CPU scalar profile builds. State
  rollback, BF16 routing and rotary, tokenizer, and packed metadata are
  covered. The optimized scalar engine authenticates both whole shards
  and passes real prefill, cached decode, cancellation and exact reset
  replay at capacity 4. The guarded peak is 5.11 GiB. The pinned MLX
  comparison matches all three greedy IDs over 744,960 logits. The MLX
  peak was 5.62 GiB. Maximum absolute logit difference was 0.328125. Astra
  confirms the binding seal and the diagnostic comparison. Post-seal
  strict checks, 864 CPU and 1,069 Metal tests, the optimized CLI build and
  five final CLI controls pass after a disk-floor retry. The CLI reports
  first flushed output, committed-work rates and state and scratch bytes.
  Astra approves the experimental profile. The successful whole-file text
  CLI and wider qualification remain open.
  [Model gates](PLAN.md#sensenova-loop-9b-acceptance) ·
  [Check results](artifacts/sensenova-loop-9b-2026-11-18/native-progress-15-experimental-reviewed.json).

- [x] Create the competitive GPU goal and audit existing profiles/source. LFM3
  remains CPU; Gemma is hybrid with 277 batched decode waits, CPU attention/unary
  work, compensated projection and no tiled-prefill GEMM. These findings do not
  establish additive cost shares or a qualified external speedup ratio.
- [x] Draft G0–G6/O4 implementation and benchmark gates, preserving numerics,
  full quality/state, 6 GiB and the original 1% overhead bound. All 440 relative
  links/26 anchors and whitespace pass; Astra approves the planning scope with
  no findings. No new engine code or win is claimed.
- [x] **G1 diagnostics and prefix-witness fix:** CPU sampling and four GPU
  frames pass exact output/state/selected-logit checks under 6 GiB. Attention
  accounts for 46.68% pre-first-token / 28.77% decode main-thread samples;
  completion waits account for 46.57% / 59.76%, including GPU execution.
  These are diagnostic observations. [Cost map](docs/gemma-current-cost-map.md).
- [x] The old early-prefill failure is retained. A borrowed committed-prefix
  view fixes that diagnostic while ordinary sampling logits remain hidden for
  pending input. Fmt, strict CPU/Metal Clippy, 847 CPU / 1,019 Metal tests and
  release build pass; fresh early-prefix parity passes all 262,144 logit values.
  Astra approves the scoped fix/evidence with no findings. Code `42efe9c8` is
  pushed; all seven exact-head workflows/15 jobs pass. Publication is tracked
  in the [receipt](artifacts/mlx-gpu-gap-2026-11-15/g1-publication-1.json). No new performance win.
- [x] Summary component probe: genuine manifest/summary parity and all 280
  measured/84 warmup samples replay. Confirming full insertion is 3.569/3.555 us;
  callback clock accounting is about 133 ns/token, excluding decoder/I/O bodies.
  Expected overwrites and guards pass; sandbox/build preparation failures remain.
  [Evidence](artifacts/mlx-gpu-gap-2026-11-15/g1-summary-component-analysis-1.json).
  All 422 relative links/whitespace and Astra evidence review pass, no findings.
  Production inputs are unchanged; no new 1% pass.
- [x] G1 component decision published as `1aaf1601`; Astra approves, 422 links
  and whitespace pass. Production inputs remain unchanged. [Receipt](artifacts/mlx-gpu-gap-2026-11-15/g1-summary-publication-1.json).
- [x] **G2 feasibility:** artifact-only GPU attention passes 932,968 frozen
  values plus 376,320 saved original-BF16 graph own-input values. Both layouts
  agree bitwise, exact masks pass, four GPU faults reject, and normal/optimized
  raw replay passes eight corruption controls. All guards close under 512 MiB.
  Astra found a non-isolating overflow injection; the corrected masked-only
  case and independent 42-pair certificate pass. Follow-up Astra review approves scoped feasibility, no remaining findings;
  no production integration or speed claim. [Review/publication](artifacts/mlx-gpu-gap-2026-11-15/g2-feasibility-publication-1.json).
  [Scope/evidence](docs/gemma-current-cost-map.md#g2-attention-feasibility-follow-up).
- [x] **Reusable GPU attention workspace.** Sol implements fixed
  buffers, checked shared-ring views/caller-command encoding, guarded publication
  and poisoning. Fmt/strict CPU+Metal Clippy, 847 CPU/1,022 Metal tests, release,
  three explicit GPU tests and zero Rust allocations over 32 warm operations pass.
  Independent normal/optimized replay passes 2,844,296 values/136,568 exact masks.
  Astra combined review approves scoped retention with no findings.
  [Publication](artifacts/mlx-gpu-gap-2026-11-15/g2-workspace-publication-1.json). [Implementation/evidence](docs/gemma-gpu-attention.md).
- [x] Parallel Astra GPU/Apple-hardware research supplies block/admission/commit
  guidance; Sol prepares the next FP32 GELU experiment independently. Focus on
  GPU residency, unified-memory reuse and fewer CPU handoffs; ANE remains a
  separately qualified public-API experiment. [Hardware focus](docs/apple-silicon-gpu-goal.md).
- [x] G0 header ledger: active BF16 tensors alone total 9.257 GB including
  4.698 GB PLE; media removal alone does not establish 6 GiB full residency.
  No conversion, new artifact hunt or runtime/quality claim.
- [x] G0 source research proposes one BF16-native Q4 conversion/two authenticated
  views with identical active bytes; Ollama needs 10.6 MB unused extras. No
  conversion/import/precision/runtime qualification. App plist is now 0.40.16;
  fresh server identity is required. [Research](artifacts/mlx-gpu-gap-2026-11-15/astra-common-q4-import-research-1.md).
- [x] G0 precision audit: stock controls do not establish matched FP32 execution.
  Exact-widening adapters require separate qualification/disclosure; stock
  observations stay distinct. No conversion/run. [Precision findings](artifacts/mlx-gpu-gap-2026-11-15/astra-common-q4-precision-research-1.md).
- [x] Workspace `5e886fd1` is pushed; all seven exact-head workflows/15 jobs
  pass and all 4,170 compiled inputs match committed Git bytes.
- [x] **Full-finite GELU feasibility:** a fixed 3,317,784-byte table passes
  exhaustive native/independent scans of all 4,278,190,080 finite FP32 gates,
  original 42 cases, GPU controls, forced whole-case replay and corruptions.
  Five boundary/adversarial cases replay; none of 24 saved model cases does.
  Astra approves artifact-only retention, no findings. Direct-tanh rejection
  remains; production/block/speed qualification is open. [Scope/checks](docs/gemma-gpu-gelu.md).
- [x] **Shared GPU encoders.** Sol freezes checked RMS/rotary encoding
  on the existing GPU/command and one-command attention chains up to 32 queries.
  Fmt/strict CPU+Metal Clippy, 847 CPU/1,024 Metal tests, release and seven
  explicit GPU tests pass. Independent normal/optimized replay checks 622,632
  values, 9,025 exact masks and 98,308 inactive copies; 33 corruptions reject.
  Attention still allocates zero Rust memory over 32 warmed operations.
  Astra approves the scoped encoder/composition change with no findings. [Progress](artifacts/mlx-gpu-gap-2026-11-15/g2-next-mechanism-progress-2.json).
- [x] Sol freezes 92 untouched producer-block states, eight cases and 34 BF16
  tensor descriptors; normal/optimized independent readers and eight corruption
  controls pass. Metadata/fixture evidence only, no new block execution.
  [Handoff](artifacts/mlx-gpu-gap-2026-11-15/g2-producer-block-fixture-1/handoff.json).
- [x] G0 synthetic exact-widening mechanism and independent normal/optimized
  replay pass; Astra reader findings are fixed and final review approves.
  No full-model adapter, stock precision equivalence or speed win is qualified.
  [Review/evidence](artifacts/mlx-gpu-gap-2026-11-15/astra-fp32-widen-review-1/verdict.json).
- [x] **Resident producer-block pilot:** all eight nominal original cases
  complete on one command/wait with zero CPU replay; initial max-only prefills
  took ten. Independent normal/optimized replay passes 847,360 original values,
  192 exact masks, 46 causal-prefix pairs and transaction/fault controls under
  6 GiB. Required fmt/strict CPU+Metal Clippy, 1,034 Metal tests and release builds
  pass; 847 unchanged CPU tests are reused. Astra combined review approves with
  no findings. [Pilot/scope](docs/gemma-gpu-producer.md),
  [review](artifacts/mlx-gpu-gap-2026-11-15/astra-producer-block-review-1/verdict.json),
  [publication](artifacts/mlx-gpu-gap-2026-11-15/g2-producer-publication-1.json).
- [x] Producer pilot `c93d5794` is pushed; all seven exact-head workflows/15
  jobs pass. [Publication/source/CI](artifacts/mlx-gpu-gap-2026-11-15/g2-producer-publication-1.json).
- [x] **Finer-core GELU artifact:** native/independent proof covers all
  184,549,378 replacement inputs with unchanged parent coverage. GPU normal/
  forced runs and four raw readers pass 2,106,831 comparisons each, 14 raw
  corruptions per reader and 14 native semantic controls plus both positive
  baselines. Astra approves artifact-only retention; failures remain linked.
  [Review/evidence](artifacts/mlx-gpu-gap-2026-11-15/astra-gelu-core-review-1/verdict.json).
- [x] **Resident Gemma chain:** qualify the 13→14→15 chain at unchanged
  limits. Higher-accuracy RMS/host controls pass, but the old coupled candidate
  failed. The approved core table and optional completed-command GPU timing are
  implemented in source8. Optimized build and all eight coupled cells pass
  2,375,680 values with one command/wait each and no replay. State/RMS controls,
  32 zero-Rust-allocation warm iterations and the old pilot pass. Fmt, strict CPU/Metal Clippy, 1,043 Metal tests and release builds pass;
  847 unchanged CPU tests are reused. Astra's failure-metric finding is fixed;
  fresh source9 fmt/Metal Clippy, 1,044 Metal tests, release and the pure error
  regression pass. Numerical GPU evidence is reused from unchanged source8 work.
  Astra approves the final combined code change with both findings resolved.
  Pushed `1f687d9f`; all seven workflows/15 jobs pass.
  [Publication/source/CI receipt](artifacts/mlx-gpu-gap-2026-11-15/g2-deep-chain-publication-1.json).
  [Current checks, research and failures](artifacts/mlx-gpu-gap-2026-11-15/g2-deep-chain-progress-1.json).
- [ ] **Parallel G0:** combined source5 passes fmt/strict CPU+Metal Clippy,
  release, 1,051 Metal tests (400 ignored), actual state/all 59 codec cases,
  projection readers and CLI diagnostics; 850 CPU tests are reused against 4,015
  unchanged non-Metal inputs. Fresh original 10,000-target loss validation passes
  under 6 GiB; 100 tasks, replay, isolated grade, stock precision and final engine
  review remain. The cold controller pauses cleanly after loss. The Nov4 native
  baseline was already packed affine4; equal mean loss establishes no new gain.
  Read-only resume prechecks reject the current inherited-environment digest;
  no new model work/campaign bytes changed. Astra approves a separate phase
  authority retaining old/new environment identities and every original stage/gate;
  actual admission/tasks/replay/grade have not run.
  [Authority review](artifacts/mlx-gpu-gap-2026-11-15/astra-common-q4-phase-authority-review-1/verdict.json). [Precheck](artifacts/mlx-gpu-gap-2026-11-15/g0-common-q4-resume-environment-check-2.json),
  [resume research](artifacts/mlx-gpu-gap-2026-11-15/astra-common-q4-environment-resume-research-1/RESEARCH.md).
  [Functional checks](artifacts/mlx-gpu-gap-2026-11-15/g0-common-q4-required-cli-run-4/result.json),
  [full loss](artifacts/mlx-gpu-gap-2026-11-15/g0-common-q4-full-loss-analysis-prep-1/loss-result.json),
  [baseline trace](artifacts/mlx-gpu-gap-2026-11-15/astra-common-q4-loss-identity-research-1/FINDING.md).
  Idle Ollama 0.40.16 PID/binary/version is pinned; stock startup remains unrun.
  [Read-only receipt](artifacts/mlx-gpu-gap-2026-11-15/g0-stock-idle-metadata-preflight-1.json).
- [x] **Pooled GPU scratch:** source3 optimized build/Metal Clippy, eight
  numerical cells, 23 lifecycle controls, 32 warm zero-Rust-allocation operations
  and resident-entry 387,584-value checks pass. The queued-Drop reader fix is
  isolated. All old numerical/lifecycle/RMS/warm/producer regressions pass.
  Fmt/strict CPU+Metal Clippy, 1,047 Metal tests (397 ignored), release CLI/unit
  pass; 847 CPU tests reuse 4,009 exact inputs. Astra approves scoped retention
  with no required fixes. [Scope](docs/gemma-gpu-pool.md),
  Pushed `14fbb1b5`; all 7 workflows/15 jobs pass.
  [Review/publication/source/CI](artifacts/mlx-gpu-gap-2026-11-15/g2-scratch-pool-publication-1.json).
  A reader was interrupted at the original 2 GiB disk floor; 8.16 GB regenerable
  Cargo debug cache was reclaimed, with sources/models/evidence/actual binaries
  preserved. Passed native work is authenticated and reused by the continuation.
  [Hardware summary](artifacts/mlx-gpu-gap-2026-11-15/g2-scratch-pool-hardware-summary-1.json),
  [continuation/failure](artifacts/mlx-gpu-gap-2026-11-15/g2-scratch-pool-final-prep-1/continuation-handoff-v1.json),
  [cleanup](artifacts/mlx-gpu-gap-2026-11-15/owned-debug-target-cache-reclaim-1.json).
- [x] **RMS diagnosis:** two original 22-operation GPU measurements pass the
  unchanged CPU gates; isolated accurate RMS takes 131 ms decode /148–151 ms
  prefill. The original native compile failure and corrected child remain linked.
  [Attribution](artifacts/mlx-gpu-gap-2026-11-15/g2-rms-aggregate-timing-root-run-2/result.json).
- [x] **Two-phase RMS artifact:** fresh native/CPU execution and eight independent
  readers pass 336 cases, 14 GPU faults, 9 preflights and original 74 integer controls.
  Two reversed 3-warmup/10-sample aggregate runs give 147.73/149.12→37.93/37.77 ms
  prefill and 131.67/131.43→38.08/37.50 ms decode, with exact old-kernel bit/status/
  reason parity. Astra approves artifact retention, with no required fixes.
  Sol drafts per-operation storage/epoch/preflight integration outside frozen
  worktrees. [Review](artifacts/mlx-gpu-gap-2026-11-15/astra-rms-two-phase-artifact-review-1/verdict.json).
  No chain/request/quality/overhead/adoption win is qualified.
  [Combined checks/review packet](artifacts/mlx-gpu-gap-2026-11-15/g2-rms-two-phase-artifact-review-checks-1.json),
  [integration design](artifacts/mlx-gpu-gap-2026-11-15/g2-rms-two-phase-integration-design-1/handoff.json).
- [x] **Two-phase RMS graph:** source4 passes fmt/strict CPU+Metal Clippy,
  certified release unit, 1,050 Metal tests (403 ignored), release CLI, all old
  regressions and all 12 paired independent readers. The 16 numerical cells,
  28 lifecycle controls and 64 warm zero-Rust-allocation operations pass; selected
  source2 evidence and 847 CPU tests reuse exact unchanged inputs. Two three-layer
  completed GPU-command runs lower prefill about 69% /decode about 68%, adding
  14,232 session bytes. Astra approves scoped retention with no required fixes;
  earlier Clippy/test-helper failures remain retained. No ordinary CLI/full-model,
  request, quality, overhead or external win is claimed.
  [Scope/measurements](docs/gemma-gpu-rms-scheduling.md),
  [final 32-stage checks](artifacts/mlx-gpu-gap-2026-11-15/g2-rms-two-phase-code-review-prep-1/final-check-completion-1.json),
  [combined Astra review](artifacts/mlx-gpu-gap-2026-11-15/astra-rms-two-phase-code-review-1/verdict.json).
  Pushed `0bfd68c3`; all seven workflows/15 jobs pass and all 4,209 compiled
  inputs match committed Git blobs. [Publication/source/CI](artifacts/mlx-gpu-gap-2026-11-15/g2-rms-two-phase-graph-publication-1.json).
- [x] **Borrowed Q4 GPU projection:** source3 passes fmt/strict CPU+Metal Clippy,
  certified release unit, all 35 targeted GPU/reader/regression stages, full Metal
  1,052 tests (409 ignored) and release CLI; 847 CPU tests reuse 4,009 exact inputs.
  Whole-matrix preflight borrows the existing packed owner with all-16-code
  general-FP32 bounds; product math is unchanged. Warm32 allocates zero Rust
  memory; observed peak 505,348,096 bytes includes the helper. Astra approves
  the private prerequisite with no required fixes. No model/graph/route/speed
  adoption is qualified. Failed capture/guard/compile attempts remain linked.
  [Scope](docs/gemma-gpu-q4-projection.md),
  [all actual checks/failures](artifacts/mlx-gpu-gap-2026-11-15/g0-q4-borrowed-encode-progress-5.json),
  [combined Astra review](artifacts/mlx-gpu-gap-2026-11-15/astra-q4-borrowed-code-review-1/verdict.json).
- [ ] **Next resident Q4 composition:** bridge the authenticated packed owner,
  small norm views and same-owner reference into a bounded resident block before
  full graph/CLI adoption. Preserve 16,384-row Q4 chunks, original numerics,
  shared KV/prefix/failure semantics and 6 GiB without a second full model copy.
  Original common-Q4 full quality and stock precision/identity remain separate
  gates. Get Astra design/check/review for each concrete composition mechanism.
  [Bridge rationale](artifacts/mlx-gpu-gap-2026-11-15/astra-common-q4-resident-bridge-research-1.md).
- [x] Fresh RMS A/B diagnostics and all four independent readers pass unchanged
  leaf gates and exact guard2 bits. A takes about 29–32 ms and B 5–8 ms in the
  separate schedules; these intervals are not additive graph cost shares.
  Astra selects a bounded reduction→denominator→output scheduling experiment,
  with unchanged arithmetic and distinct intermediate receipts.
  [Actual attribution](artifacts/mlx-gpu-gap-2026-11-15/g2-rms-two-phase-attribution-root-run-1/result.json),
  [next experiment](artifacts/mlx-gpu-gap-2026-11-15/astra-rms-reduction-root-split-research-1/RESEARCH.md).
- [x] **RMS three-phase rejection:** fresh compilation, 336 normal/forced cases,
  original controls, 12 stage faults and all eight independent readers pass.
  Whole-command prefill is 6.657%/7.528% slower; decode improves 0.471%/0.508%,
  below the 5% retention gate. Astra approves rejecting it as a performance
  candidate; artifacts remain and the published two-phase graph is retained.
  [Actual checks](artifacts/mlx-gpu-gap-2026-11-15/g2-rms-three-phase-root-run-1/result.json),
  [final Astra decision](artifacts/mlx-gpu-gap-2026-11-15/astra-rms-three-phase-review-1/verdict.json).
- [ ] Stock startup remains unrun. Astra requires lifecycle-wide aggregate log
  limits and sealed guard plan/ledger/log transport; Sol prepares an artifact-only
  child before any model launch. Loaded-content/all-activation precision and
  matched performance remain unqualified.
  [Consolidated review](artifacts/mlx-gpu-gap-2026-11-15/astra-stock-startup-review-1/execution-approval.json).
- [x] G3 research proposes bounded four-token weight reuse with unchanged
  compensated order; no implementation/speed claim. Fresh resident-graph
  profiling precedes selection. [Experiment brief](artifacts/mlx-gpu-gap-2026-11-15/astra-prefill-weight-reuse-research-1.md).
- [x] Static next-graph prep identifies layers 13→14→15, strided multi-tile
  output and remaining CPU validation/commit touches. Astra supplies the checked
  GPU layout recipe; Sol prepares an isolated draft while final checks run.
  [Graph handoff](artifacts/mlx-gpu-gap-2026-11-15/g2-resident-graph-integration-prep-1.md),
  [projection brief](artifacts/mlx-gpu-gap-2026-11-15/astra-bf16-multitile-research-1.md).
- [ ] Then G2 complete resident graph; G3 prefill/decode matrix kernels; G4 output/weight
  traffic; G5 LFM3 GPU; G6 confirmed external wins and all original overhead gates.
  Each code task needs required checks, measurements and Astra before commit.
- [x] Shared encoders `b5437a76` are pushed; all seven workflows/15 jobs pass.
  [Publication/source/CI](artifacts/mlx-gpu-gap-2026-11-15/g2-shared-encoders-publication-1.json).

Previous Gemma 7.4005%/7.1650% opt-in gains and LFM3 CPU retention remain valid;
summary overhead still fails 3/8 comparisons and defaults stay disabled. Qwen 4B/9B
rejections remain closed; no artifact hunt is reopened. Preferred rejected
Heretic Q4 is not a qualified shared benchmark. The new goal stays active until
actual matched wins and all original overhead gates pass.
[Planning evidence](artifacts/mlx-gpu-gap-2026-11-15/planning-checks-1.json).

## Recent completed milestones

- [x] 2026-11-15 performance recap: reconciled Gemma/LFM3 before/after,
  Ollama/vLLM observations, sampled memory and all eight CLI overhead outcomes.
  [Gemma](docs/gemma-single-row-submission.md), [external Gemma](docs/console-engine-comparison.md),
  [LFM3](docs/lfm3-engine-comparison.md), [overhead](docs/gemma-cli.md#fixed-overhead-measurement).
  Existing evidence only; no new benchmark or qualified cross-engine ratio.

- [x] Ordinary official BF16/G2 Gemma CLI and bounded request summaries are
  published as `4f2e6236`. Required fmt/strict CPU+Metal+probe Clippy,57CPU/72Metal
  affected tests (2ignored each),36reader controls per mode,scoped84-entry source2
  conformance and both source3 header smokes pass. Full16-owner collection
  validates128 generations/48warm/80measured/128resets/eight exports; independent
  raw replay and Astra match all results with no findings. Original1% gates pass
  5/8; three remain failed/unqualified and summary stays disabled/opt-in. NoN,
  threshold or sample-exclusion rescue. Source/model/6GiB/cleanup checks pass,
  peak5,004,482,128 bytes including helper. All6 code-head workflows/11 jobs pass;
  every4,116 committed input matches the measured freeze. Failed attempts and
  unavailable observations retain their scope. [Behavior/measurements](docs/gemma-cli.md),
  [checks/review/publication](artifacts/gemma-cli-metrics-2026-11-14/code-publication-1.json).

- [x] Gemma opt-in single-row submission: required checks, exact relevant-input
  reuse,45,000,704 numerical values,full10k loss/100-task quality and optimized
  zero allocations pass. All6 collections validate557 requests/431 measured;
  complete caller medians improve7.4005/7.1650%,both paired95%lower bounds exceed5%.
  Both600s sustained arms pass,peak4.820GB. Independent replay matches4720 values.
  Cached TTFT+2.313microseconds stays disclosed; no threshold/samples change.
  Astra approves scoped retention,no findings. Code published as `e75a9c30`;
  all6 exact-head workflows/11 jobs pass. Defaults stay per-tile and other
  adoption/overhead/device gates stay unqualified. Source/recovery failures and
  all178 excluded partial requests remain evidence.
  [Scope](docs/gemma-single-row-submission.md),
  [checks/review/publication](artifacts/gemma-head-batch-2026-11-14/publication-checks-1.json).

- [x] S1.L external request observations: scoped publication of all
  four fresh Ollama/vLLM 3+10 cohorts and both frozen CPU baseline cohorts.
  Exact vLLM inputs pass, but only 3/13 output-ID/text rows match CPU per cohort;
  Ollama IDs are unavailable. Source/authentication, ownership/runtime, bounded
  lifecycle and shape-only import controls pass; all 26 genuine vLLM timing
  captures replay consistently. No cross-engine speedup, quality or adoption
  qualification. Content/raw-binding checks pass; Astra approves with no findings.
  [Report](docs/lfm3-engine-comparison.md),
  [checks/review/commit](artifacts/lfm3-engine-comparison-2026-11-11/comparison-publication-checks.json).

- [x] S1.L optional vLLM request timing: offline report binds and replays sealed
  captures, distinguishes queue/TTFT/decode/overall-rate boundaries and preserves
  missing or inconsistent fields. Twenty-four portable controls in both modes,
  five pinned-source cases and nine guarded offline CLI cases pass; helper
  namespace/call failures are retained and corrected. Required checks reuse
  5,177 unchanged inputs; both 52-test CI commands pass. Astra's empty-entry
  and stop-metadata findings are fixed and covered by regressions; follow-up
  approves scoped retention with no findings. Actual runtime/model/performance qualification
  remains separate. All 26 genuine vLLM captures also replay with consistent
  timing arithmetic. Both pushed GitHub Actions jobs pass.
  [Actual replay](artifacts/lfm3-engine-comparison-2026-11-11/actual-vllm-timelines-2/checks.json),
  [remote CI](artifacts/lfm3-engine-comparison-2026-11-11/vllm-timeline-remote-ci-1.json).
  [Scope](docs/lfm3-vllm-timeline.md).
  [Review/commit](artifacts/lfm3-engine-comparison-2026-11-11/vllm-timeline-publication-checks.json).

- [x] S1.L HTTP benchmark observations: bounded Ollama/vLLM stream client,
  exact prompt/cache gates and explicit missing/partial metrics. Twenty-eight
  portable controls in both modes and twelve guarded synthetic HTTP/CLI cases
  pass after correcting a malformed negative fixture. All 26 saved genuine
  Ollama replies replay with unchanged counters/text and original event clocks.
  Required checks reuse 5,177 unchanged input files. Astra's rejected-byte,
  counter-overflow and stop-ID findings are fixed; scoped retention is approved.
  [Review/commit](artifacts/lfm3-engine-comparison-2026-11-11/http-observer-publication-checks.json).
  Actual engine
  startup, quality and performance are separate. [Scope](docs/lfm3-http-metrics.md).

Previous roadmap delivery is reconciled in [outcomes](docs/improvement-outcomes.md).
LFM3 CPU code and portable measurements are published as `612771d7` on
`codex/console-engine-comparison`.
[Publication / CI](artifacts/lfm3-finite-guard-2026-11-11/finite-publication-checks.json).

- [x] Qwen3.5-2B real-source/CLI diagnostic: all 320 parent tensors,
  187 Q4 matrices/133 exact small owners, 3,238 conversion tiles and saved-byte
  audits pass. Fmt, strict CPU/Metal Clippy, 786 CPU/946 Metal tests, 70 portable
  controls per optimization mode, four explicit GPU/state/repair and twelve CLI
  cases pass. Short graph passes 159,187,008 comparisons. Context17 rejects
  3/680,841,024 comparisons: ideal FP64 fails three, staged FP32 passes; native
  traces, nine schedule/reset controls and 10,704 causal-prefix arrays per
  profile agree. Full journal/failure replay passes. Conditional fixed255,
  quality and performance do not advance. Astra approves scoped diagnostic
  retention, with no blocking code/evidence findings; model adoption stays
  unqualified. Failed setup/audit attempts remain retained.
  [Scope](docs/qwen35-2b-real-source.md),
  [checks/review/commit](artifacts/qwen35-2b-payload-2026-11-13/publication-checks.json).

- [x] Workload-bound headless Gemma summary evidence: bounded descriptor,
  authoritative rendered-input identity/admission and untimed output-state
  preflight. Strict Swift 6, 57 portable controls per Python mode, fmt, strict
  CPU/Metal Clippy and 786 CPU/946 Metal tests pass. Both fixed 3+10-pair cohorts
  complete: 52 requests/40 measured/26 summaries match input384/output128,
  state512/511/1 and reset admission. Both original 1% median-ratio and paired-median
  95% upper gates pass for TTFT and complete caller time in both cohorts. Independent
  raw replay reproduces every statistic and verifies 489 current endpoints.
  Astra approves scoped retention with no findings; earlier failed attempts remain
  recorded. Only this headless official BF16/G2 M1 Pro workload is qualified;
  production summaries remain disabled/opt-in. UI/render/default adoption stays
  separate. [Scope](docs/gemma-workload-identity.md),
  [checks/review/commit/CI](artifacts/gemma-workload-identity-2026-11-13/publication-checks-3.json).

- [x] Completed native decode-row export: additive ABI1 metrics V2 preserves
  legacy96B/symbols and copies only the existing completed counter. Native13,
  65 portable controls per mode, fmt/strict CPU+Metal Clippy,794 CPU/954 Metal,
  C CPU/Metal and strict Swift pass. Count metadata was corrected before models.
  All54 fresh requests/27 native records reconcile exact384/128/state5125111/
  count127 and526 source endpoints;516 old requests replay exactly. Both
  TTFT95uppers1.230/1.313% fail the unchanged1% bound. Astra approves functional
  retention only, no findings; summaries remain disabled/opt-in and overhead
  unqualified. No more samples or gate change follows. [Scope](docs/gemma-decode-evaluations.md),
  [checks/review/commit/CI](artifacts/gemma-decode-rows-2026-11-13/publication-checks-1.json).
- [ ] The V2 1% TTFT bound and default adoption remain unqualified.

- [x] Existing profiler readiness passes: France15+decode1,349tiles+2gathers,
  bitwise logits and fault/overflow/cancel/reset controls;56.18s/4.808GB observed
  engine charge including helper. Release compile210.66s/3.185GB stayswithin3GiB.
  [Actual readiness](artifacts/gemma-cost-attribution-2026-11-13/readiness-result-1.json).
- [x] Bounded officialG2 `toklet profile` and offline per-operation report are
  implemented. Fmt, strict CPU/Metal Clippy,815 CPU/974 Metal tests, final28 CLI
  unit controls (2 ignored) and11 portable controls per mode pass. Both full384/128 captures
  (evaluation13/139) pass exact outputs/state,2x262144 bitwise logits and349tile/
  2gather coverage; observed charge<=4.803GB, declared5.932GB includes all reserves.
  First Metal integration failure and optimized-wrapper failure are retained/fixed.
  Astra approves scoped functional/diagnostic retention, no findings; source
  and raw evidence are frozen. Commit/push and exact-head CI status are recorded
  in the [publication receipt](artifacts/gemma-cost-attribution-2026-11-13/publication-checks-1.json).
  [Scope](docs/gemma-profile-cli.md),
  [checks](artifacts/gemma-cost-attribution-2026-11-13/checks-final-1.json),
  [actual cost replay](artifacts/gemma-cost-attribution-2026-11-13/actual-report-replay-1.json).
Keep one heavy queue, separate3GiB compiler/6GiB model phases and pause all
preparation for acceptance timing. Host/GPU fields in ordinary summaries remain
unavailable; the new diagnostic is opt-in. Keep closed LFM3/Qwen searches closed;
capacity512/chunk32 is separate from numerical512-input; UI/iPhone stay deferred.
The stronger short headless Gemma overhead milestone is published as
`cf80a168`; all five exact-head CI workflows/ten jobs pass.
[Publication](artifacts/lfm3-tokens2-2026-11-13/gemma-stronger-publication-checks.json).
The bounded 2B investigation is closed. All 34,119 native arrays replay
exactly; all 17,394 recovered staged arrays match their original hashes. None
of the 24 precision policies or two declared FP32 reduction schemes passes both
selected cases under both unchanged profiles. Stop this variant search; no
engine change or deeper qualification follows. Astra approves the bounded
evidence closure with no findings. [Final local decision](artifacts/qwen35-2b-attention-2026-11-13/two-f32-qk-check-1/result.json),
[original full rejection](artifacts/qwen35-2b-payload-2026-11-13/context17-paired-decision-1.json).

- [x] Bounded two-token LFM3 operator experiment closed: 17 release controls,
  120 success/retry plus 360 zero-allocation rejection controls, fmt, both strict
  Clippy configurations and 794 CPU/954 Metal tests pass. Corrected fixed cohorts
  preserve whole outputs and improve large-shape operator latency 48.99/49.53%.
  Eight slower one-token pairs across six configurations leave compatibility
  unresolved; Astra rejects code retention and approves evidence-only closure.
  All five candidate files are restored exactly to `cf80a168`. No model/request
  integration follows. Original negatives, excluded overlapped screen and complete
  quiet-host rerun remain evidence; no gate or accepted scope changes.
  [Scope](docs/lfm3-two-token-projection.md),
  [review](artifacts/lfm3-tokens2-2026-11-13/tokens2-astra-review-2.json),
  [exact revert](artifacts/lfm3-tokens2-2026-11-13/tokens2-code-revert-2.json),
  [checks/review/publication](artifacts/lfm3-tokens2-2026-11-13/tokens2-publication-checks-2.json).
Implementation, evidence and documentation have exclusive owners; the integrator
runs one heavy queue and waits for explicit preparation-pause acknowledgements.
[Delivery plan](docs/parallel-delivery.md).

- [x] Stronger overhead evidence: bounded 10..100 pair configuration preserves
  the ten-pair default, fixed ring and original clocks/1% gates. Forty dependent
  portable checks pass in both modes; default real-data statistics replay exactly.
  Strict Swift, pre-model controls, ten conformance records and both fixed
  100-pair cohorts pass. All 412 IDs/400 measured requests, 206 raw records,
  state/status/clocks, 489 input endpoints and original statistics reconcile.
  Both headless gates pass; Astra approves scoped retention with no findings.
  Application/UI/render qualification stays separate/opt-in.
  [Scope](docs/gemma-summary-overhead.md),
  [full audit](artifacts/lfm3-tokens2-2026-11-13/gemma-stronger-independent-audit-1.json),
  [checks/review/publication](artifacts/lfm3-tokens2-2026-11-13/gemma-stronger-publication-checks.json).
The authenticated official/selected tokenizer metadata correspondence is
preparation only; runtime codec/caller/stops, >=1,025 quality capacity, both
original full quality pairs, representative fit/performance/overhead and adoption
remain open. [Next gates](artifacts/qwen35-2b-payload-2026-11-13/codec-quality-next-gate-1.json),
[tokenizer metadata](artifacts/qwen35-2b-payload-2026-11-13/codec-tokenizer-auth-and-JSON-1.json).
[Remaining-goal audit](artifacts/qwen35-2b-attention-2026-11-13/remaining-goal-requirements-1.json).
Use independent lanes, one active model and one heavy-job queue;
[delivery plan](docs/parallel-delivery.md). Preserve Gemma/4B/9B priority,
original numerical/quality and 6 GiB gates, earlier model decisions and isolated
UI work. Engine/CLI leads; interactive capture and iPhone runtime stay deferred.

- [x] S1.L original and fresh candidate quality: both complete 10,000-target/
  100-task comparisons pass unchanged gates. Stock passes 35 tasks; modified
  reference and native pass 37. All greedy loss IDs and 5,094 sampled task
  decisions/stops/decodes match; maximum NLL delta is 3.66e-15. Guest shutdown
  is verified; failed/interrupted output stays unqualified. Astra approves
  scoped result/retention review; managed adoption remains open.
  [Campaign](docs/lfm3-quality.md),
  [review/commit](artifacts/lfm3-finite-guard-2026-11-11/finite-publication-checks.json).

- [x] S1.L external comparison preflight: authenticated metadata-only views
  preserve numerical configuration and source bytes. Installed AutoConfig and
  MLX-VLM built-in config admission pass; the publisher module's incompatible
  interface is recorded. Eight portable controls and the 512 MiB guarded probe
  pass after correcting a synthetic fixture byte count and bounding actual reads
  against concurrent growth. Astra approves with no findings.
  This preflight does not qualify model startup, inference or performance.
  [Scope](docs/lfm3-external-config.md).

- [x] S1.L finite BF16 widening retained for the CPU route after full public
  validation, with unchanged FMA/reduction/store order. Required fmt/strict
  CPU+Metal Clippy, 734 CPU/880 Metal tests, release/allocation/native/C smokes
  pass. All 854 observations/68,054,016 words, 30 wrapped views and two late
  retries match. Primary ABBA: 29.0%/29.5% median request reduction; complete
  156-request matrix: 20.6–27.9%, no regression, all 12 sustained windows pass.
  Fresh full quality and all 62 state/replay checks pass within 6 GiB.
  Baseline bytes reproduce from clean `2da16a9e`; original build history stays
  unproven. The evaluator fix passes SDK/live, 14 isolation/28 authored cases,
  nine accounting controls and a 260-command stress check. All 441 local portable
  tests pass in both modes; fresh publication fmt/59 changed-workflow tests pass.
  Astra approves with no findings. Failed runs remain retained. No tail,
  observer-overhead, GPU/mobile, allocation-saving or managed-adoption claim.
  [Scope](docs/lfm3-finite-widening.md),
  [208 scalar samples](benchmarks/results/2026-11-12-lfm3-finite-widening-v1.json),
  [complete qualification](artifacts/lfm3-finite-guard-2026-11-11/finite-qualification-closure-1.json),
  [review/commit](artifacts/lfm3-finite-guard-2026-11-11/finite-publication-checks.json).

- [x] S1.G strict Granite configuration: canonical default RoPE and identical
  supported dtype aliases preserve existing scales, shapes and context limits.
  Required fmt/strict CPU+Metal Clippy, 737 CPU/883 Metal tests, the explicit GPU
  fixture and 44 CLI cases pass. Three pinned configs and 936 independent CPU
  arrays are checked; six failed attempts stay retained. The exec-aware private
  monitor passes eight controls in both Python modes and an owned live test;
  historical charges remain conservative. Compiler-only cap is 3 GiB, engine
  ceiling 6 GiB and audit/setup cap 512 MiB. Astra approves, no findings.
  [Scope](docs/granite-config.md),
  [portable checks](benchmarks/results/2026-11-12-granite-config-v1.json),
  [full qualification](artifacts/granite-config-2026-11-12/qualification-1.json),
  [review/commit](artifacts/granite-config-2026-11-12/publication-checks.json).

All four Heretic/stock tensor containers authenticate (14,639,034,320 bytes)
with pinned SHA256/header checks within the 512 MiB setup policy. No model is
loaded and source indexes remain unchanged; stock's understated total still
rejects. Codec, full graph/quality/fit/request gates remain open.
[Source authentication](artifacts/granite-payload-2026-11-12/result-1.json).

- [x] S1.G bounded affine4/8 conversion: full producer and saved-byte audit pass
  for 282 matrices/81 norms. All 3,659,530,240 matrix values reconstruct exactly
  against upstream FP32 during conversion; the offline audit replays actual
  source/child hashes and exact norms without recomputing floating-point values.
  All 48 CPU/GPU controls, 33 portable tests per Python mode and six actual-artifact
  CLI cases pass. Observed conversion charge is 455,131,136 bytes within the
  512 MiB audit cap. Metal metadata accepts contexts 128/512; 1,024 correctly
  rejects at the existing 512-position limit. Scalar FP32 exceeds 6 GiB.
  Earlier memory, index-field and CLI-expectation failures remain retained;
  legacy MiniCPM tools are unchanged. Evidence closure and Astra review pass, no findings.
  Full-model fit, inference, quality, performance and adoption are open.
  [Controls](artifacts/granite-conversion-2026-11-12/controls-closure-2.json),
  [full conversion](artifacts/granite-conversion-2026-11-12/full-conversion-1/manifest.json),
  [saved-byte audit](artifacts/granite-conversion-2026-11-12/full-audit-1.json),
  [CLI cases](artifacts/granite-conversion-2026-11-12/child-inspection-2.json),
  [review/commit](artifacts/granite-conversion-2026-11-12/publication-checks.json).

The conversion is published as `eb14db30`; both exact-head CI workflows pass.
Granite text preparation authenticates both complete tokenizer files. Their BPE,
added-token, normalization and decoder definitions match; only the modified
source's BatchLongest left-padding policy differs. All 100,352 IDs are covered,
including 96 added tokens also present in BPE vocabulary. An initial counting
assumption failed and is retained; the corrected bounded source check passes.
No encoding/template or managed-codec qualification follows.
[Codec inspection](artifacts/granite-codec-2026-11-12/codec-inspection-2.json).

Five cold device/session planning cases pass on Apple M1 Pro. The 128-input/
128-output request plans within 6 GiB; 512/1,024-input and capacity-boundary
requests remain unsupported at the existing 512-position Metal limit. No
numerical buffers are allocated by planning. The actual raw 128/128 CLI smoke
passes: 128 outputs, 255 committed positions and one pending token; observed
charge 2,899,741,912 bytes within 6 GiB. Source/binary/input hashes close unchanged
and owned cleanup is empty. Debug clocks remain diagnostic; this one request
does not qualify full-model numerical/quality, all-workload fit or performance.
[Cold planning](artifacts/granite-runtime-plan-2026-11-12/runtime-plan-1.json),
[native smoke](artifacts/granite-native-smoke-2026-11-12/native-smoke-1.json).

- [x] S1.G full-model graph diagnostic: required fmt/strict CPU+Metal Clippy,
  737 CPU/887 Metal tests, 58 portable tests per mode, 1,120 frozen-fixture
  comparisons, the explicit GPU fixture and admitted checker controls pass.
  All 30 native runs/304 observed state checks pass at a 5.03 GB observed peak;
  schedule/reset/causal-prefix bytes match exactly. The complete comparison
  rejects 212/20,268,711,936 profile coordinates under unchanged gates. The first
  failure localizes to propagated native-input differences; other failures remain
  unlocalized. Current route stays unqualified. Astra's completeness/receipt
  findings are fixed: 34 checker tests per mode and fresh numeric controls pass;
  genuine replay verifies 304 checks/10 resets/40 receipts with byte-identical
  numerical records. Astra follow-up approves scoped tooling/rejection retention,
  no remaining findings. Published as `8a90db86`; both exact-head CI workflows
  pass. Failed checks remain retained.
  [Scope](docs/granite-graph.md),
  [scalar decision](benchmarks/results/2026-11-13-granite-graph-v1.json),
  [required checks](artifacts/granite-graph-2026-11-12/required-checks-closure-2.json),
  [native capture](artifacts/granite-graph-2026-11-12/native-capture-1.json),
  [strict replay](artifacts/granite-graph-2026-11-12/strict-replay-closure-2.json),
  [localization](artifacts/granite-graph-2026-11-12/attention-localization-result-1.json),
  [review/commit](artifacts/granite-graph-2026-11-12/publication-checks.json).

Next: qualify the selected Qwen3.5-2B source/graph route; the reviewed
Ministral diagnostic and its unchanged rejection are published as `8b6220e9`. Ministral
codec and original full-quality/fit/request/O1 gates remain unqualified. Source
authentication/conversion pass; stock index admission remains rejected. Root
owns integration and the single heavy-check queue.
[Parallel execution plan](artifacts/ministral3-yarn-2026-11-13/parallel-work-plan-1.json).
Granite's current affine4/8 Metal route does not advance to adoption or quality/
performance qualification. Any future candidate must pass the original graph,
10,000-target/100-task quality, codec, fit and request/overhead gates. The 6 GiB
ceiling and 512-position Metal cap remain unchanged. Continue the selected
under-4B track after publication.

- [x] S1.M cache headroom: removed 12,277 retired Cargo compilation intermediates;
  observed free space increased by 1.66 GB. Both measured executable hashes and
  369 retained cache leaves are unchanged; current caches/models/raw evidence
  remain available. Recheck exact recipe/disk admission before payload work.
  [Cleanup](artifacts/ministral3-source-planning-2026-11-13/cache-cleanup-result-1.json).

- [x] S1.M cached CPU YaRN: selected source profile and 11 boundary/shape cases
  precede outputs. Independent fixture generation passes within 512 MiB;
  24 portable controls per Python mode, six targeted Rust tests, fmt and strict
  CPU/Metal Clippy pass. The immutable Half-table primitive and producer are
  implemented; all 744 CPU/894 Metal-feature tests and six release primitive
  tests pass. Warmed release key/query and rejected calls allocate zero Rust
  objects on the measured thread. Optimized Python regenerates identical bytes.
  Astra approves scoped retention with no findings. Committed/pushed as
  `7ea011a6`; both exact-head remote CI workflows pass. Model/source/Metal admission, graph,
  quality, fit and performance gates remain open.
  [Frozen profile](artifacts/ministral3-yarn-2026-11-13/prospective-primitive-protocol-2.json),
  [scope](docs/ministral3-rotary.md),
  [scalar checks](benchmarks/results/2026-11-13-ministral3-rotary-v1.json),
  [full checks](artifacts/ministral3-yarn-2026-11-13/rust-checks-1.json),
  [review/commit](artifacts/ministral3-yarn-2026-11-13/publication-checks.json).

- [x] S1.M selected metadata/Metal YaRN: all 755 CPU/907 Metal-feature tests,
  fmt/strict CPU+Metal Clippy and three release GPU diagnostics pass. Unchanged
  dual-profile gates cover 124,928 GPU-output coordinates; replay/chunk/rebase
  and status/reuse checks pass. Measured Rust encode allocations are zero; four
  3+10 operator cohorts retain raw clocks. A post-build Cargo-kind collector
  mismatch is corrected with unchanged-source check reuse; failed finalization
  is retained. Astra approves scoped retention with no findings. Published as
  `5c0eee96`; both exact-head CI workflows pass. Graph/model/quality/fit/request
  and O1 overhead remain open; foreign-device negative is unavailable.
  [Required checks](artifacts/ministral3-metal-2026-11-13/check-queue-result-1.json),
  [GPU/scalar evidence](artifacts/ministral3-metal-2026-11-13/gpu-scalar-closure-1.json),
  [review/publication](artifacts/ministral3-metal-2026-11-13/publication-checks.json).

- [x] S1.M bounded conversion: both complete sources authenticate; stock's
  optional index count still rejects under the strict physical-count policy.
  Modified-only conversion passes183matrices/53norms/222exclusions/4,252tiles,
  with all3,428,843,520 independent/upstream FP32 values exact. All144 genuine
  controls and86 portable tests perPythonmode pass within512MiB. The saved-byte
  audit passes after validating rich metadata identities; its first failure is
  retained. Wrong environment-root and CLI-status expectations are corrected
  without changing policies or loader behavior. ActualCLI remainsunknown/
  unknown_architecture;4,641 unchangedRust inputs reuse requiredchecks. Combined
  Astra approves scoped retention with no findings. Published as `83a8617a`;
  all three exact-head CI workflows pass. Graph/codec/quality/fit/request/O1/
  adoption stay open.
  [Full conversion](artifacts/ministral3-conversion-2026-11-13/full-conversion-1/manifest.json),
  [saved-byte audit](artifacts/ministral3-conversion-2026-11-13/full-audit-2/audit.json),
  [checks/review](artifacts/ministral3-conversion-2026-11-13/publication-checks.json).

- [x] S1.M real graph/CLI diagnostic: source, Metal and CLI integration is built.
  Fmt, strict CPU/Metal Clippy, 763 CPU/922 Metal-feature tests, 28 affected
  CLI reruns and 34 portable controls per Python mode pass. All five release
  GPU tests pass, including eight zero-Rust-allocation decode steps; the debug
  allocation failure stays retained. Full-model capture passes 22 runs/9,218
  arrays and exact state/causal checks, but the numerical gate rejects
  78/244,809,728 profile coordinates. Native-input replay localizes the first
  projection failure to propagated inputs; sampled RMS/rotary checks pass
  locally without overriding that rejection. All 19 console functional requests
  pass after a retained pre-child reader failure/fix. Performance cohorts remain
  blocked. Astra approves explicit diagnostic tooling/retention with no blocking
  code findings; the stale integration-status sentence is corrected. Published
  as `8b6220e9`; all four exact-head CI workflows/eight jobs pass.
  [Review/publication](artifacts/ministral3-graph-2026-11-13/publication-checks.json).
  [Current checks/decision](artifacts/ministral3-graph-2026-11-13/current-phase-1.json).
  [Ownership/invariants](artifacts/ministral3-graph-2026-11-13/parallel-dispatch-1.json).
  Ordinary inspection stays unknown_architecture.
  Source names/nested config, positional ownership and raw CLI are integrated
  only through the explicit diagnostic profile.

- [ ] S1.M full-model/codec qualification: the numerical route remains rejected;
  no full-context, codec, quality, all-workload fit, request/O1 or adoption
  qualification follows. A future candidate must pass the original gates.
  [Next graph interfaces](artifacts/ministral3-graph-planning-2026-11-13/integration-plan-1.json).
  The shared17,078,110-byte tokenizer JSON authenticates against both pinned
  APIs; its bounded structural check passes. Stock/modified configurations and
  templates differ, so rawcodec/chatprofiles require separate qualification.
  [Codec preflight](artifacts/ministral3-codec-2026-11-13/preflight-1.json).

Qwen3.5-2B source inspection records a distinct 24-layer/16-value-head profile
and joint normalization/layout requirements. The earlier preflight's stale Rust
hash is retained as a failed check; current source is inspected and pinned
separately. Parent/payload authentication, numerical/quality and runtime gates
remain open. No model or compiler was launched.
[Source inspection](artifacts/qwen35-2b-source-planning-2026-11-12/static-source-inspection-2.json).

## Completed work index

Full check counts, failed attempts, limitations, reviews and commit references
are preserved in the [reviewed history](docs/archive/2026-11-12/TODO.md). Use the
linked scope reports for individual results; a passed narrow gate never expands
model, context, device or adoption qualification.

| Completed scope | Current decision and evidence |
| --- | --- |
| O1, O2.A, O2.B1, O2.B2.A | Bounded summaries, native phases, repeated numerical workloads and controller publication delivered; summary mode stays opt-in because the 1% overhead bound is unqualified. [O1](docs/o1-gemma-request-summaries.md), [phases](docs/o2-gemma-cost-report.md), [workloads](docs/o2-gemma-workloads.md), [publication](docs/o2-ui-publication.md). |
| O3 and R1 | Scoped runtime explanation/profile and comparable offline regression screening delivered. [O3](docs/o3-gemma-runtime.md), [R1](docs/gemma-regression-job.md). |
| Console engines | Ollama/vLLM installed and 78 requests collected; unmatched weights/precision/cache prohibit engine ratios. [Comparison](docs/console-engine-comparison.md), `34db5778`. |
| Gemma guard and preferred MLX artifact | Exact guard optimization retained. Heretic affine4 candidate rejected on task quality; official BF16 remains selected. [Guard](docs/gemma-guard-validation.md), [artifact](docs/gemma-heretic-mlx.md). |
| Q1.A–I: Qwen3.5-4B mechanisms | Pinned source, bounded owners, scalar/hybrid kernels, codec and generation bridge delivered only in their recorded scopes. [Model plan](docs/model-observability-plan.md). |
| Q1.J: 4B 128/256 | Rejected: 13,505 failures and 3,132 disjoint dual-profile intervals; matching greedy IDs do not qualify adoption. [Decision](artifacts/qwen35-context256-2026-11-09/decision.json), `9d270f3a`. |
| Q2.A–D: separate 9B support | Pinned recipe/store, distinct head and independent one-input oracle delivered without loosening 4B APIs or gates. [Recipe](docs/qwen35-9b-cold.md), [store](docs/qwen35-9b-storage.md), [head](docs/qwen35-untied-head.md), [oracle](docs/qwen35-9b-oracle.md). |
| Q2.E: 9B hybrid Metal | Both one-input captures pass; 64.72×/64.42× applies only to the raw one-input baseline. [Scope](docs/qwen35-9b-metal.md), `3ccce99f`. |
| Q2.F: 9B context17 | Both 33.85M-value gates and prefix/repair/zero-allocation witnesses pass; descriptive release medians 4.093/4.121s. [Scope](docs/qwen35-9b-context17.md), `41287c6c`. |
| Q2.G0: exact reference helper | All 67.17M short-context profile values bitwise match scalar references. Helper gains are separate from engine speed or deeper-context qualification. [Scope](docs/qwen35-reference-vector.md), `88950b11`. |
| LFM3 CPU BF16 | Authenticated graph/CLI, 1,025-position parity, complete original quality and 62 state checks pass; finite widening retained after Astra with 29.0–29.5% primary gains. Managed/GPU adoption remains open. [Scope](docs/lfm3-finite-widening.md). |
| Five under-4B candidates | LFM3 CPU retained; MiniCPM/Granite/Ministral/Qwen2B current routes have scoped numerical rejection decisions. Dependent adoption gates do not advance; closed searches stay closed. [Current requirement audit](artifacts/gemma-head-batch-2026-11-14/remaining-goal-audit-2.json), [selected order](docs/model-observability-plan.md#five-candidates-strictly-below-4b-total-parameters). |

## Remaining observability gates

- [ ] Display-frame qualification **deferred by user, engine/CLI priority**.
  Ready9 reuses checked43source/native46 inputs and completes13exact trials;
  fresh viewport/crop visually inspected, new receipt bound to its own UUID.
  Coverage rejects433/1721future-display callbacks, zero occlusion;32numeric
  samples retained. The owned session is closed; no further sharing requests.
  Earlier7/8failures and isolated UI WIP remain preserved/unreviewed/uncommitted.
  [Deferred status](artifacts/o2-ui-frames-2026-11-07/status-audit-1/current-state.json).
- [ ] Keep first-visible/final-answer and ordinary summary-overhead qualification
  explicit. Controller publication is not rendering; fixed-answer frame matches
  provide upper bounds only. iPhone work remains deferred.

## Working method

- Update current/next work, checkboxes, blockers and check/review outcomes after
  every iteration, including research, failed checks and interrupted work.
- Keep entries concise; link full logs in artifacts and evidence/commit records.
- Use the [parallel delivery plan](docs/parallel-delivery.md): source, Metal and
  independent-reference lanes with one CLI integrator, exact file ownership and
  one heavy-check queue. Delivery timestamps separate queue/review/rework time;
  the last milestone's 34.8-minute acquisition dominates its 95-second conversion.
  This is workflow planning, with no measured speedup claim or gate changes.
  Documentation passes 279 local links, content and staged whitespace checks;
  Astra approves with no findings. Published as `b774d58f`; all three exact-head CI
  workflows pass. Original user work is preserved.
  [Checks/review/publication](artifacts/ministral3-graph-planning-2026-11-13/parallel-delivery-publication-1.json).
- Reviewed `7524640e` history is archived with byte-exact inverse link checks;
  routine TODO/improvement bytes fall 38.4%. Pending work, original user history,
  live PLAN and numerical inputs are preserved.
  [Archive validation](artifacts/lfm3-finite-guard-2026-11-11/working-history-archive-checks-1.json).
- Code completion requires applicable checks and one combined Astra review of
  diff, invariants and results before commit. Keep [improvements](docs/improvements.md)
  current. Documentation-only work needs link/content and whitespace checks.
- Push reviewed milestone commits to `origin/codex/console-engine-comparison`,
  as authorized on 11 November. Reviewed tip `af100419` publishes the finite
  LFM3 retention metadata; exact-head push and both CI jobs pass. Granite config
  checks and Astra review pass; publication is recorded in its own receipt.
  Original UI/workflow work is preserved.
  [Latest recorded push verification](artifacts/lfm3-finite-guard-2026-11-11/finite-publication-checks.json).
- Read [PLAN.md](PLAN.md) for binding criteria, the
  [design rationale](apple_llm_performance_plan.md) when relevant, and targeted
  source before assuming a feature exists. Read archives only on demand.
- Preserve historical failures and scope limits. Mac observations do not qualify
  iPhone; planning/admission estimates do not establish allocation or execution.

## Current delivery and retained decisions

| Area | Current scope and evidence |
| --- | --- |
| Foundation M0–M7 | Complete in recorded scopes; [plan](PLAN.md), [M7 closure](docs/experiments/2026-10-21-m7-release.md). Later iPhone work remains deferred. |
| M8 shared blocks, model breadth and capacity controls | Selected Mac engine/app scopes delivered; [coverage index](docs/model-coverage.md), [shared-block contract](docs/shared-block-model-plan.md), [historical checklist](docs/archive/2026-11-07/TODO.md#m8--architecture-breadth-and-usable-context-controls). No blanket family/full-window qualification. |
| Managed Mac journeys | Six earlier profile journeys reviewed; [report](docs/mac-hub-journeys.md). Qwen1.5B Chat/native41 complete; [evidence](benchmarks/results/2026-11-06-qwen15-mac-app-v1.json). |
| Official Gemma BF16 Mac | Full 10.28 GB public snapshot, six-file SHA, managed publication, Chat/Stop and network-denied reopen pass; [journey](benchmarks/results/2026-11-07-gemma-public-resume-mac-v1.json). Native45 Settings/context/KV checks pass; [evidence](benchmarks/results/2026-11-06-gemma-bf16-live-window-v1.json). |
| G2 prefill | Retained on official BF16 Mac route; both primary runs reduce prefill about 61% and request time about 28%; [evidence](benchmarks/results/2026-11-06-gemma-bf16-g2-retention-v1.json). |
| M9 affine tile | First delivery complete as pinned Mac opt-in, with full quality, state and repeated timing. Managed adoption remains separate; [decision](docs/m9-mac-affine-decision.md). |
| Memory reporting | Owner snapshots, physical KV ceiling in Preview/Apply and conditional load-process peaks complete; [owner](benchmarks/results/2026-11-02-session-owner-memory-v1.json), [KV](benchmarks/results/2026-11-03-kv-preview-apply-v1.json), [loading](benchmarks/results/2026-11-03-load-memory-checkpoints-v1.json). Planned/admitted, owned and measured memory remain distinct. |

## Open or deferred scope

- [ ] Record separate iPhone Settings runtime evidence.
- [ ] Independent physical iPhone14Pro journeys/Settings; latest connection query times out.

The user deferred this device work on 5 November. The paired iPhone 14 Pro is
unavailable; a different Pro Max is not a substitute.
[Availability evidence](artifacts/m8-phone-availability-2026-11-04/availability.json).

Full-Qwen3-window expansion and unselected research techniques remain deferred.
The [detailed plans](docs/todo-implementation-plans.md) preserve task-specific
acceptance; older queue statements and historical pending notes do not override
later accepted decisions. Use the [archived M8 closure record](docs/archive/2026-11-07/TODO.md#m8-closure)
for individual model/adoption limitations.

Preserve these rejected or restricted results:

- Gemma3 coupled-intermediate rejections remain failed under the recorded
  [conformance policy](docs/gemma3-native-conformance.md); full adoption is distinct.
- Qwen3 historical fidelity rejections remain separate from scoped native
  qualification; [native profile](docs/qwen3-native-profile.md).
- Llama Online and split-KV candidates remain rejected/deferred under unchanged
  gates; [Llama decision](docs/todo-implementation-plans.md#llama-online-correctness),
  [split-KV evidence](benchmarks/results/2026-11-05-m9-split-kv-session-v1.json).
- Gemma QAT full task quality fails the two-point loss limit; passing loss/wrap
  fidelity does not override it; [evidence](benchmarks/results/2026-11-06-gemma-qat-full-quality-v1.json).
- G3 assistant remains test-only after the longer guard regressed about 21%;
  [decision](benchmarks/results/2026-11-06-gemma-assistant-mtp-retention-v1.json).
- Prompt lookup remains an explicit nonstreaming API; default adoption rejected.
  [Plan and evidence](docs/todo-implementation-plans.md#p5-strict-greedy-verifier-and-prompt-lookup).

## Latest iteration

| Date | Result and checks | Evidence / next action |
| --- | --- | --- |
| 2026-11-08 | Exact guard reduction retained:25.0–26.6% primary/26.3% sustained request reduction;512 deeper bitwise pairs and626 sustained requests pass. Required CPU/Metal/GPU/native checks and Astra approval pass;6GiB preserved. | [Scope](docs/gemma-guard-validation.md), [checks/review/commit](artifacts/gemma-guard-validation-2026-11-08/publication-checks.json); next preferred Gemma MLX format/quality evaluation. |
| 2026-11-07 | O2.B2.A bounded UI publication complete: six real Gemma controls, 104 paired requests across UI/headless tracks, fresh strict Swift/app checks and 29 portable tests pass. Astra approves state/count join fix; timing captures recompute unchanged. Both 1% overhead bounds remain unqualified/default-disabled. | [Scope](docs/o2-ui-publication.md), [result](benchmarks/results/2026-11-07-o2-ui-publication-v1.json), [checks/commit](artifacts/o2-ui-publication-2026-11-07/checks.json); visible-frame/O3 remain open. |
| 2026-11-07 | O2.B1 numerical coverage complete: strict Swift/new Python and reused production gates pass; two batches plus 601 seconds sustained, max lifetime peak 4.924 GB. Astra approves. Earlier counter-assumption/interference run is superseded by fresh captures; no quality gate changed. | [Scope](docs/o2-gemma-workloads.md), [result](benchmarks/results/2026-11-07-o2b-gemma-workloads-v1.json), [checks/commit](artifacts/o2b-gemma-workloads-2026-11-07/checks.json); next UI/O3. |
| 2026-11-07 | R1 offline regression screening complete: comparable frozen identity/work, independent cohorts, explicit noisy/failure receipts and named portable CI. Local17 tests/Astra pass; no runtime or optimization qualification. O2.B1 corrected suite running. | [Job](docs/gemma-regression-job.md), [checks/commit](artifacts/r1-gemma-regression-2026-11-07/checks.json); next O2.B1 completion, then UI/O3. |
| 2026-11-07 | O2.A native46 API/schema2 phases and strict matched warm reports complete. Required checks and Astra approval pass after parser clock/enclosure fixes. Decode dominates this workload; overhead remains unqualified/default-disabled. | [Report](docs/o2-gemma-cost-report.md), [receipt](benchmarks/results/2026-11-07-o2a-gemma-phase-report-v1.json), [checks/commit](artifacts/o2-gemma-metrics-2026-11-07/checks.json); next O2.B. |
| 2026-11-07 | O1 Gemma opt-in summaries complete; Cargo, strict Swift/Xcode, real output/state/failure/admission controls pass. Corrected full-boundary overhead remains unqualified; Astra approves after three fixes. Earlier archive mismatch, compiler sandbox/type failures and superseded timings are retained in check notes/logs. | [Pilot](docs/o1-gemma-request-summaries.md), [result](benchmarks/results/2026-11-07-o1-gemma-summary-v1.json), [checks/commit](artifacts/o1-gemma-summary-2026-11-07/checks.json); next O2 within full-roadmap goal. |
| 2026-11-07 | Observability-first plan complete; five exact under-4B counts verified. Preferred uncensored MLX formats/lineage and conversion/memory gaps recorded. Initial metadata DNS/Python certificate failures resolved with system curl; receipts retained. Documentation checks pass; binding policy unchanged. No runtime checks/Astra review claimed. | [Plan](docs/model-observability-plan.md), [checks/commit](artifacts/model-observability-plan-2026-11-07/checks.json); next implementation O1 remains stopped. |
| 2026-11-07 | Researched ten current under-30B candidates from primary sources; capability/local-efficiency shortlist, not common-harness qualification. O1 stopped; unfinished code uncommitted and unchecked. Documentation checks pass. | [Research](docs/models-under-30b-2026-11-07.md), [checks](artifacts/models-under-30b-2026-11-07/checks.json); model selection next. |
| 2026-11-07 | Archived full TODO/PLAN/improvement snapshots at `338c4e5b`; routine document bytes reduced about 97%. All snapshot text, 32 milestone criteria, four binding policy sections and both open items preserved; link/content/whitespace checks pass. Documentation-only. | [Archive manifest](docs/archive/2026-11-07/manifest.json), [checks](artifacts/context-archive-2026-11-07/checks.json); next proposed O1 pilot. |

Older iterations and all completed checklists remain in the
[archived TODO](docs/archive/2026-11-07/TODO.md). Add only a concise recent record
here; archive completed history periodically rather than loading it by default.

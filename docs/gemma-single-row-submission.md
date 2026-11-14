# Gemma single-row submission

The opt-in official Gemma 4 E2B BF16/G2 candidate lowers complete-request median
latency by 7.40% and 7.17% in two fixed paired cohorts on the named M1 Pro host.
Numerical, state, quality, memory and measurement checks pass. Combined Astra
review and publication remain pending; existing session and native defaults
continue to use per-tile submission.

## Mechanism and ownership

The earlier selected-evaluation profile records 349 projection dispatches,
commands and waits. Independent tiles within one single-row projection do not
consume each other's output. The candidate encodes those tiles in one command,
then waits, validates and copies the completed result once. It preserves the
precise Metal shader, per-row reduction order and BF16/FP32 representations.

Eligibility requires multiple tiles, one input row and a full output that fits
both the initialized CPU tile length and existing GPU output length. Other shapes
retain per-tile execution. No capacity growth, second model, new numerical
scratch or cross-operation dependency batching is introduced. Immutable weights
stay shared; session KV and scratch remain session-owned.

Each encoder writes its checked, disjoint FP32 row range. The shared atomic-OR
status word is cleared once and inspected after completed GPU work. The command
and buffers remain alive until completion, including failure/drop paths. Failure
is sticky and existing prefix/replay rules remain in force. Backend/policy
selection stays outside numerical inner loops.

For the fixed chunk32 graph, decode retains 349 dispatches but uses 277 commands
and waits; full-row prefill uses 278. Selected captures12/13/139 pass output,
history, work-count and all262144 selected-logit bits per capture. There are41
prefill or42 decode submission groups. Gemma has35 layers and15 KV owners; those
counts are different concepts.

## Explicit use and attribution

The immutable owner exposes `new_batched_g2_session`,
`new_batched_profiled_session` and `runtime_inventory_batched_single_row`.
Legacy constructors retain their existing behavior. On the supported release
Mac route, select the candidate diagnostic explicitly:

```sh
toklet profile --model /absolute/pinned/model \
  --workload /absolute/frozen-workload.json --evaluation 13 \
  --submission batched-single-row --output /absolute/new-profile-directory
python3 tools/bench/gemma_profile_report.py /absolute/new-profile-directory \
  --json /absolute/new-report.json --markdown /absolute/new-report.md
```

The legacy CLI default is `--submission per-tile`. Candidate profile schema2
adds bounded, sparse submission groups; the per-tile wire record is unchanged.
Each shared command/wait/copy/host interval has one first-tile counter owner.
Every tile keeps its own encoder interval and dispatch. The consumer validates
membership/ownership and recomputes unions without double-counting. Legacy
schema1 is still accepted. These diagnostic clocks remain separate from ordinary
request acceptance clocks; logical copy bytes are not measured DRAM traffic.

## Ordinary release measurements

Source4 and copied opt3 binaries are sealed against base7f3b1f2e. The host is an
Apple M1 Pro, MacBookPro18,3,16GiB, macOS26.6.2/build25G83; rustc1.90.0. The source
is `google/gemma-4-E2B-it` revision
`70af34e20bd4b7a91f0de6b22675850c43922a03`, with its pinned codec, BF16 weights,
FP32 activations/KV, capacity512 and chunk32. AC power, Low Power Mode off and
nominal thermal admission are checked ten times before each launch and during
execution. All preparation is paused during measurement.

The primary resets the session for384 prompt tokens and128 emitted tokens;
state is512 history/511 committed/1 pending and actual decode evaluations127.
Each cohort uses three warmup pairs and ten measured pairs, alternating order;
confirmation reverses order. Loading, tokenization, setup, checks, hashing and
export are outside the completed caller interval. Profiling and selected-logit
witnesses are disabled. All557 requests/431 measured requests in the six
collections pass exact output/state/work and resource checks; four golden
preparation requests and sixteen matrix controls are accounted separately.

| Primary cohort | Per-tile median seconds | Batched median seconds | Marginal reduction | Paired-median reduction95% interval |
| --- | ---: | ---: | ---: | ---: |
| First |41.839254|38.742930|7.4005%|6.9263–7.7860%|
| Reversed confirmation |41.588692|38.608854|7.1650%|6.9062–7.4706%|

Both marginal reductions exceed the declared5% retain threshold, with paired
uncertainty above5% in both cohorts. Raw samples and unscaled MAD remain in JSON;
bootstrap uses the existing fixed seed1/10000 policy, without new samples or a
new noise threshold. Primary TTFT medians improve2.78/2.28%; callback-interval
decode rate improves10.33/10.35%. The latter uses `(N-1)/(last-first)` and remains
distinct from core decode evaluations per second and prefill tokens per second.

Both nine-case matrices cover reset, append, fully cached and pending tokens,
edited-prefix replay,128+128 generation,512-token prefill-only boundary, and
128+128/384+128 teacher forcing. All twenty primary/matrix total-clock point
comparisons improve. Teacher clocks cover completed numerical evaluations, with
no invented generation TTFT. Unsupported512/1024/1920+128 requests are explicitly
unavailable under the unchanged512-position route, not truncated or qualified.

One secondary estimate is adverse: reversed fully-cached first-callback median
1.0233955→1.0257085ms (+2.313microseconds,+0.226%). Its paired reduction interval
is[-1.028%,0%], candidate MAD4.146microseconds; the first cohort improves. This is
retained for review, without claiming universal TTFT improvement or introducing
a new acceptance threshold.

Each sustained arm uses three warmups and then at least600 completed measured
caller seconds:15 baseline requests/627.886s and16 candidate requests/619.030s.
The candidate's sustained complete-request median is7.428% lower. These are
separate chronological, descriptive series; no paired or pooled-IID interval is
claimed. Observed peak charge is4,820,374,040B including32MiB helper reserve;
admission and observations stay within6GiB. SDK sampling is not a continuous
allocator cap and can miss undiscovered short-lived peaks.

## Correctness, recovery and decision scope

Fmt, strict CPU/Metal/probe Clippy,816 CPU/976 Metal tests and affected
model/session/GPU controls pass or retain exact relevant-input reuse. The strict
optimized allocation control records zero Rust allocations; debug binding
allocations do not qualify that result. Original short/wrap/holdout comparisons
pass45,000,704 values under unchanged tolerances. Full10000-target loss passes
(delta-0.03085nats versus the+0.05 bound);100 tasks score52 original/51 candidate,
a1percentage-point loss within2points. These are fidelity gates, not absolute
model-quality claims.

An interrupted matrix lacks its terminal manifest/guard result; all178 partial
requests remain unqualified. The lost temporary worktree is restored exactly,
with all4099 frozen source inputs and three binary hashes matching. Source now
lives in persistent storage with a frozen content archive. A device-number-only
remount is bridged by full canonical six-file hashes and equal named hardware,
OS and toolchain fields; old/new actual identities remain recorded. Only complete
primary cohorts are reused, after independent raw replay. Fresh matrices and
sustained arms retain their original warmups, repetitions and gates.

Independent replay authenticates65 pins,557 requests and431 measured requests,
and reproduces4720 report numeric/scalar comparisons with zero discrepancies.
Astra approves retention of this explicit BF16/G2 single-row route on the named
host and admitted workloads, with no actionable or blocking findings. The review
covers the combined diff, invariants, checks, evidence and recovery bridge.
[Review](../artifacts/gemma-head-batch-2026-11-14/astra-review-1/verdict.json).
Commit/push and exact-head CI follow. Default summary overhead remains
unqualified/disabled or opt-in; no UI, tail, energy, larger-context, other-model
or iPhone qualification is transferred.

[Actual report](../artifacts/gemma-head-batch-2026-11-14/actual-submission-report-2/report.md),
[raw/statistical replay](../artifacts/gemma-head-batch-2026-11-14/independent-submission-replay-1.json),
[check/reuse ledger](../artifacts/gemma-head-batch-2026-11-14/checks-final-1.json),
[quality decision](../artifacts/gemma-head-batch-2026-11-14/quality-model-decision-1.json),
[capture completion](../artifacts/gemma-head-batch-2026-11-14/diagnostic-captures-completion-5.json),
[source recovery/archive](../artifacts/gemma-head-batch-2026-11-14/source-recovery/source4-restoration-1.json),
[device content bridge](../artifacts/gemma-head-batch-2026-11-14/model-device-rebind-1.json).

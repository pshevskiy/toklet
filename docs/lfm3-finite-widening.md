# LFM3 finite BF16 widening experiment

Status: **retained for the CPU BF16 route after required checks and Astra
approval with no findings**. Primary and full compatibility/sustained request
gates, fresh full quality and all 62 capacity/state checks pass. Managed
application adoption and GPU/mobile qualification remain open.

The [portable measurement record](../benchmarks/results/2026-11-12-lfm3-finite-widening-v1.json)
retains all 208 scalar samples, including 160 measured requests, their state,
phase and prompt/output identity fields. Exact source projection and per-trial
matching pass; generated text and token-clock vectors remain in the sealed raw
artifacts. Initial report power-label and within-length identity assumptions
are corrected with failures retained. OS version is captured at publication;
same-boot association is explicitly inferred rather than claimed as a per-request
OS capture. Astra approves the scoped CPU retention.

The [prospective contract](../artifacts/lfm3-finite-guard-2026-11-11/prospective-widen-contract-1.json)
freezes the hypothesis, numerical boundary, primary workload and acceptance
before measurements. Sampling identifies the source-order BF16 dot reduction
as the largest active leaf. Its existing general BF16-to-FP32 conversion
handles NaNs on every operand, although both public numerical callers have
already rejected nonfinite operands. A private finite-only widening places
the BF16 bits in the high half of the FP32 word, with no rounding.

The changed operation is shared by linear projection and the ordinary
attention QK reduction. Projection validates complete input and weight slices;
attention validates all query and logical K/V rows, including masked values,
before entering its reduction. Shapes and matching extents are checked at
those public boundaries. The small attention reduction remains separate.
All 32 FMA partials, reduction tree, eight-element tail blocks, scalar tail
order and final nearest-even BF16 stores remain fixed. The change adds no
unsafe code, allocation, dispatch, full FP32 weight copy, dtype, state or ABI
change. The separately prepared finite-predicate experiment remains uncompiled
and unmeasured.

The retained Arm64 disassembly contains the same two four-lane vector FMA
instructions per eight coefficients in both versions. In that selected loop,
the baseline has 24 instructions and the candidate 12: NaN mask/quiet-bit/select
preparation is removed while loads, widening and fused arithmetic remain.
Other scalar, tail, reduction and error paths are outside this count. Static
instruction counts do not establish cycles or a speed ratio; the prologue's
64-byte stack-frame difference does not establish model-memory savings.
[Static assembly comparison and limits](../artifacts/lfm3-finite-guard-2026-11-11/widen-assembly-audit-1.json).

The new exhaustive control compares every one of the 65,280 finite BF16 words
with the independent half 2.7.1 conversion, including signed zeros and
subnormals. Existing nonfinite admission, projection FP64/source fixtures,
attention fixtures, overflow/retry and allocation checks remain binding.
Fresh required checks pass: formatting, strict CPU and Metal all-target Clippy,
734 CPU tests (52 ignored) and 880 Metal-enabled tests (340 ignored). Explicit
release component/model/CLI and allocation checks, all three native slices and
both CPU/Metal C execution
smokes pass. These checks do not qualify LFM3 GPU execution or iPhone runtime.
The consolidated receipt binds twelve successful commands, full logs and all
444 native-build source files. Failed enclosing attempts remain failed; the
smoke retry supplies the successful native result.
[Checks, counts and retained failures](../artifacts/lfm3-finite-guard-2026-11-11/widen-checks-consolidated-1.json).

The complete 1,025-position comparison passes all 854 observations and
68,054,016 BF16 words, thirty wrapped views and two late-abort retries within
6 GiB. It preserves the original source graph, numerical gates and committed
state semantics. No tolerance is widened.
[Parity result](../artifacts/lfm3-finite-guard-2026-11-11/depth-parity-2/verify-checks.json).

## Primary request observations

Hardware is the named M1 Pro Mac with 16 GiB unified memory. The release build
uses Homebrew Rust 1.90.0/LLVM 21.1.1, matching the baseline. The source checkpoint
is the authenticated mchen04 LFM3 Heretic BF16 MLX repackaging at revision
88636d45d403e7c8ab6a40e85c9d0fa90cc5a253, with SHA-256
b6ff17ac993dddf0f6f51bd872b12ae1437f4453a54a0b4878cedc7864fbc07e.
The [native plan](../artifacts/lfm3-finite-guard-2026-11-11/native-cohort-plan-2.json)
binds exact executable hashes, flags, tokenizer identity and raw workloads.

Retained Cargo metadata matches compiler, release profile, target/configuration,
Rust flags and all nine non-Toklet dependency build fingerprints. The baseline
records the root default-feature marker, while the candidate disables default
features. That alias is empty and the current 416 tracked Rust files contain
no explicit conditioning on it; the difference remains recorded. The depfile
points to the clean earlier quality worktree at 2da16a9e. Its current numerical
source matches the candidate base, but paths alone do not attest historical
source content. The original build's historical commit remains unproven;
no rebuild or compiler invocation runs during measurement.
[Build-metadata audit and limits](../artifacts/lfm3-finite-guard-2026-11-11/baseline-build-metadata-audit-1.json).

After every state model exits, a fresh Toklet build from clean
`2da16a9e3eb67173d545c5f4f9efe9b8b0a5bada` reproduces the entire measured
baseline binary byte for byte. Its full SHA-256 is unchanged. All 444 numerical
inputs differ from the frozen candidate only in the intended two files.
The dependency cache is independently copied and Toklet outputs are cleaned
before the observed fresh compile. The 47.36-second build peaks at a guarded
747,094,016 bytes, within 2 GiB; setup stays within 512 MiB. Full original-cache
content checks pass. This qualifies a matching clean source revision; it does
not establish the historical original build commit.
[Exact baseline reproduction](../artifacts/lfm3-finite-guard-2026-11-11/baseline-byte-reproduction-result-1.json).

Each fresh process collects three warmups and ten measured requests at
512 input tokens and 32 sampled output tokens, context capacity 2,048 and
prefill chunk size 128. Every request resets state and reuses zero prefix
tokens. The order is baseline1, candidate1, candidate2, baseline2. AC power,
Low Power Mode off, thermal state zero, disk floor and absence of competing
named model/compiler work are observed. Only the birth-verified, empty default
Ollama service is permitted. The primary clock runs through parsed/logged CLI
completion; loading, engine phases and first nonempty generated text retain
their distinct boundaries.

All 52 requests match work, token IDs, text and stops, including forty measured
requests. The following values are independent medians over ten samples, in
seconds; phase medians need not sum to the request median.

| Cohort | Client request | Engine prefill | Engine decode | Engine sampling |
| --- | ---: | ---: | ---: | ---: |
| Baseline 1 | 349.5412 | 310.2928 | 39.2597 | 0.01620 |
| Candidate 1 | 248.1452 | 215.6951 | 32.4351 | 0.01614 |
| Candidate 2 | 246.5960 | 214.2091 | 32.4295 | 0.01609 |
| Baseline 2 | 349.5760 | 310.2854 | 39.3005 | 0.01622 |

Paired request reductions are 29.0% and 29.5%, exceeding the frozen 5% gate
and both observed median variations: approximately 0.0100% for baseline and
0.6282% for candidate. Prefill is the largest measured request phase.
Observed conservative memory charges plus helper reserve stay below 6 GiB;
variation in RSS is not claimed as an allocation reduction. Ten samples do
not qualify tail latency, and this primary workload does not establish the
original full compatibility or sustained scope.
[Raw-bound primary analysis](../artifacts/lfm3-finite-guard-2026-11-11/native-primary-analysis-2.json).

## Compatibility measurements

The complete frozen matrix contains 156 requests: 36 warmups and 120 measured.
All 39 requests and 4,992 sampled decisions per cohort match work, output/state,
profile, logical memory and the qualified power/thermal boundaries. Inputs are
128, 512 and 1,024 tokens with 128 output steps, reset state and zero prefix
reuse. EOS is ignored; capacity remains 2,048 and prefill chunk size 128.

The table reports completed client request medians in seconds, ten measurements
per cell. Pair 1 compares baseline 1/candidate 1; pair 2 compares baseline
2/candidate 2. The original 512-input/32-output primary remains separate.

| Input tokens | Baseline 1 | Candidate 1 | Candidate 2 | Baseline 2 | Paired median reductions |
| --- | ---: | ---: | ---: | ---: | ---: |
| 128 | 237.3716 | 186.8545 | 188.4067 | 237.2947 | 21.3% / 20.6% |
| 512 | 471.1929 | 349.4861 | 348.8390 | 471.0352 | 25.8% / 25.9% |
| 1,024 | 785.4317 | 566.8864 | 567.1185 | 786.5983 | 27.8% / 27.9% |

| Input tokens | Baseline median variation | Candidate median variation |
| --- | ---: | ---: |
| 128 | 0.032% | 0.831% |
| 512 | 0.033% | 0.185% |
| 1,024 | 0.149% | 0.041% |

CPU prefill medians fall 29.9–30.7% across the two pairs; decode medians fall
16.2–17.2%. In the candidate runs, prefill accounts for about 76% of engine
generation time at 1,024 inputs; decode accounts for about 71% at 128 inputs.
Phase shares are computed per request before taking medians. Sampling-phase
medians range from 64.6 to 65.5 ms; all paired phase changes remain in the data.
These observations guide further profiling; operator-level attribution remains
unqualified.
[CPU phase observations](../artifacts/lfm3-finite-guard-2026-11-11/compatibility-phase-observations-1.json).

All four complete-cohort resource receipts and twelve actual measured sequences
exceeding 600 seconds pass. No candidate median regresses. The maximum of the
separate RSS and sampled physical guard charges, including helper reserve, is
3.82/3.74/3.74/3.74 GB in cohort order. This variation does not establish
allocation savings. Raw samples and analysis retain minima, maxima, median
absolute deviations, phase/first-token clocks and every sustained interval.
Ten samples per group do not qualify tail latency or observer overhead.
[Full matrix analysis](../artifacts/lfm3-finite-guard-2026-11-11/compatibility-analysis-3.json),
[closed receipts and source/build bindings](../artifacts/lfm3-finite-guard-2026-11-11/compatibility-closure-checks-1.json).

The request-performance gates pass; full CPU retention is now approved. Fresh
compiler identity/new loss build pass, then all 10,000 frozen loss targets and
ten blocks complete. Both collection guards pass; maximum observed outer guard
charge is 4.15 GB including reserve, within 6 GiB. Both loss gates pass; all
10,000 native greedy IDs match the modified reference, with maximum NLL delta
3.66e-15. All 100 tasks and decoding complete with the original caps/EOS;
all 5,094 sampled decisions, stops and decoded rows match the reference.
The isolated grader initially stopped on an owned setuid `ps` inspection error. Native
process-group queries and concurrent-working-set accounting pass 156 shared
tests and nine accounting controls in both Python modes, SDK/live checks and a
260-command inert stress check. All 14 isolation and 28 authored evaluator
controls then pass. The fresh four-second grading retry completes all 80
isolated program runs, preserves 37/100 task passes for both modified reference
and native, verifies guest shutdown and passes both complete quality comparisons.
Its maximum guard charge is 2.48 GB including the full 2 GiB guest reservation.
An initial recovery-plan binding failure starts no VM/model and is retained.
All 62 state checks and exact event/memory replay pass. Both capacity runs
match cache/error/pending behavior and cancellation/resume to the same 16 IDs
and text; all six hidden-stop cases pass. Maximum outer guard charge is 3.90 GB,
with no survivors. These checks qualify functionality, not performance.
Nineteen recovery-pipeline controls per Python mode pass. Exact baseline byte
reproduction passes; Astra approves with no actionable findings. Earlier partial
snapshots and supplemental observer metadata/unit-constant failures remain
retained; the corrected observers preserve the original 6 GiB/512 MiB limits.
[Fresh loss-build binding](../artifacts/lfm3-finite-guard-2026-11-11/fresh-quality-1/candidate-loss-binding.json),
[Closed 10,000-target collection](../artifacts/lfm3-finite-guard-2026-11-11/fresh-loss-collection-closure-checks-1.json),
[Retained grading failure](../artifacts/lfm3-finite-guard-2026-11-11/fresh-quality-grading-failure-audit-1.json),
[Guard stress qualification](../artifacts/lfm3-finite-guard-2026-11-11/grading-inert-guard-1/checks.json),
[Fresh complete quality](../artifacts/lfm3-finite-guard-2026-11-11/fresh-quality-recovery-4/original-quality-decision.json),
[Host requalification](../artifacts/lfm3-finite-guard-2026-11-11/host-requalification-1/result.json),
[Recovery and regression checks](../artifacts/lfm3-finite-guard-2026-11-11/grading-recovery-checks-1.json),
[Complete 62-check replay](../artifacts/lfm3-finite-guard-2026-11-11/state-analysis-4.json),
[Complete pre-review qualification](../artifacts/lfm3-finite-guard-2026-11-11/finite-qualification-closure-1.json),
[retained observer corrections](../artifacts/lfm3-finite-guard-2026-11-11/compatibility-512-four-cohort-observation-2.json).

## Required scope and retained failures

The full frozen matrix uses 128/512/1,024 input tokens and 128 output steps,
three warmups and ten measured requests per length in each of four fresh
processes: 156 requests total. EOS is ignored to preserve all 128 steps.
Every measured sequence must actually reach 600 seconds; every slower
candidate median requires investigation. The numerical clients, workloads,
binaries and acceptance policy remain unchanged.
[Current matrix execution](../artifacts/lfm3-finite-guard-2026-11-11/compatibility-execution-plan-3.json).

Fresh quality builds a new candidate loss binary and runs all 10,000 original
loss targets and 100 original tasks, with original output caps, ordinary EOS,
the four-second isolated grader and unchanged loss/task gates. Original
stock-to-modified and modified-to-baseline comparisons pass, but they do not
replace candidate evidence. Fresh candidate quality now passes as well.
The subsequent 62 functional checks pass and cover capacity,
one-beyond rejection, valid cache/pending state, replay, hidden stops and
cancellation/resume. Astra approves retaining the CPU optimization and host
support. Managed adoption remains unqualified.
[Fresh quality execution](../artifacts/lfm3-finite-guard-2026-11-11/fresh-quality-execution-plan-3.json),
[state execution](../artifacts/lfm3-finite-guard-2026-11-11/state-execution-plan-4.json).

The combined review independently verifies source/diff/evidence hashes, all
444 numerical inputs, required Rust logs/counts, 208 scalar projections and
medians, evaluator bindings, archived snapshots and full baseline byte equality.
Approval excludes managed adoption, GPU/mobile, tails, observer overhead,
allocation savings and cross-engine ratios. Commit/push references are recorded
with the publication receipt.
[Astra review](../artifacts/lfm3-finite-guard-2026-11-11/astra-finite-review-1.json),
[publication](../artifacts/lfm3-finite-guard-2026-11-11/finite-publication-checks.json).

Failed memory/build/path attempts, the first unqualified warmup and all failed
queue receipts are retained. The runtime guard initially missed macOS's
capitalized Python executable; corrected controls cover it. External preparation
then required scoped private-manifest handling, an empty output parent and
conservative accounting for independently read macOS footprint counters.
A bounded owned allocation probe reproduces the counter lag. Its correction
retains both raw values and charges the maximum without reducing the cap.
The native matrix's JSON string-to-path error stops before model work; the
corrected retry passes its complete fixture pipeline. Fifty-five portable
controls per Python mode and three inert command lifecycle cases qualify the
current serial wiring, without claiming model results.
[Retry bindings](../artifacts/lfm3-finite-guard-2026-11-11/serial-retry-preparation-3.json),
[current state](../artifacts/lfm3-finite-guard-2026-11-11/active-experiment-state-1.json).

The external collection separately completes all four fresh 3+10 Ollama/vLLM
cohorts under its guards. Ollama lacks authoritative prompt/output IDs; vLLM
matches exact prompt/cache admission but matches CPU output IDs/text on only
3/13 prompts in each cohort. External MLX GPU/API timing does not establish
equivalent numerical execution, quality or a cross-engine speedup ratio.
All 26 genuine vLLM timing captures replay with consistent arithmetic through
the reviewed offline tool, preserving server/client clock boundaries.
[External collection and limits](../artifacts/lfm3-engine-comparison-2026-11-11/external-collection-summary-1.json),
[actual timing replay](../artifacts/lfm3-engine-comparison-2026-11-11/actual-vllm-timelines-2/checks.json).

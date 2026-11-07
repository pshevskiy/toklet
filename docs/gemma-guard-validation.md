# Exact guard validation reduction on Gemma BF16/G2

Retained after required checks, sustained trials and Astra approval. The frozen primary comparisons
reduce complete generation-call time25.0–26.6% on Apple M1 Pro16GiB, with exact
work, output IDs/text and state. This is a selected engine measurement; UI,
first-visible text, energy, tail and broader-model qualification are separate.

The O3 diagnostic found roughly77ms of host validation/copy in a223ms decode
frame. Input/output inactive-capacity checks cover about907MB of array words per
frame; that estimate describes coverage rather than measured DRAM traffic.
The old sentinel equality loop branches per word. An OR of each word's XOR with
the sentinel returns zero exactly when every same tail word matches. OR cannot
cancel two corruptions. Optimized Rust1.90 output uses vector integer loads and
reduction without unsafe, intrinsics or inline-always directives.

Every existing weight/input/output/status guard and finite-output check remains.
Slice bounds, errors, allocation sizes, shaders, precision, token/stop semantics,
KV state and numerical operations are unchanged. Scalar test oracles cover all
bits/positions, empty and active boundaries, equal twin corruptions, real-capacity
last words and word-alignment offsets. GPU fixtures cover affine L1/FP32 reuse,
rejections, BF16 residual/midpoint/guard behavior and original source rows. Full
plain/profile logit-bit, state/fault/overflow/reset controls also pass.

| Warm generation, median | Baseline | Candidate | Reduction |
| --- | --- | --- | --- |
| Request, run1 | 2.219s | 1.629s | 26.6% |
| Request, run2 | 2.215s | 1.660s | 25.0% |
| First-token event, run1 | 400ms | 333ms | 16.7% |
| First-token event, run2 | 396ms | 337ms | 14.8% |

Each variant has three warmups and ten measured requests in two independently
loaded, counterbalanced release processes. Same official revision
`70af34e20bd4b7a91f0de6b22675850c43922a03`, original BF16 weights/G2/FP32 session,
context512/chunk32,15 prompt tokens,8 visible plus stop106, Reset/no prefix reuse.
Generation caller time includes callbacks and excludes loading, tokenization
and final text decoding. State remains24 history/23 committed/1 pending token.
Raw samples, median/IQR and conditions are preserved; no reliable tails inferred.

Deeper numerical guards use128 and384 raw-document input tokens followed by128
teacher-forced steps, with the exact512-token public corpus shared by both builds.
Across two independent trajectories per variant/case, completed-prefill and all
512 matched decode-state pairs have identical full-logit SHA256 bit digests and
greedy IDs. Prefill time improves6.2–9.7%, decode22.5–26.7%, with no observed phase
regression. These are supplementary fixed-work guards; timers exclude digest
collection and establish no chat task/loss score or standalone tail qualification.

Fresh formatting, strict all-targets CPU/Metal Clippy,604 CPU tests (32ignored)
and729 Metal tests (288ignored) pass. Three arm64 native slices export46 C symbols
each and pass strict C linking; Mac CPU/Metal C smoke passes on the authenticated
tiny fixture. The generated app link stays on native46; no app/device runtime qualification.
The only differing production source is the guard predicate; the benchmark
harness/source/dependency pins match between baseline and candidate.

Sustained trials pass: baseline268 requests/601.50s and candidate358/600.54s.
Median caller time falls2.217s to1.633s (26.3%); every output/state check passes.
Both use AC power and report nominal thermal state at request boundaries and
ten periodic host checkpoints. Lifetime physical peaks4,920,676,648 and
4,922,085,696bytes stay below6GiB. Existing system swap is recorded; these
samples establish no swap-free or memory-saving claim. Astra approves retention
under the unchanged >=5% primary gate. No numerical or
quality tolerance is loosened. Initial summaries/profile overhead remain opt-in;
this optimization does not qualify their1% bounds.

[Checks/review/commit](../artifacts/gemma-guard-validation-2026-11-08/publication-checks.json),
[Predeclared acceptance](../artifacts/gemma-guard-validation-2026-11-08/acceptance-contract.json),
[primary raw/dispersion](../artifacts/gemma-guard-validation-2026-11-08/primary-report.json),
[deeper bitwise guards](../artifacts/gemma-guard-validation-2026-11-08/secondary-checks.json),
[sustained checks](../artifacts/gemma-guard-validation-2026-11-08/sustained-checks.json),
[complete evidence](../artifacts/gemma-guard-validation-2026-11-08/final-report.json),
[required checks](../artifacts/gemma-guard-validation-2026-11-08/required-checks-1/checks.json),
[GPU controls](../artifacts/gemma-guard-validation-2026-11-08/hardware/checks.json),
[native package](../artifacts/gemma-guard-native49-2026-11-08/build.json),
[microbenchmark/assembly](../artifacts/gemma-guard-validation-2026-11-08/micro-summary.json).

# Qwen3.5-9B context17 qualification

Q2.F extends the pinned 9B hybrid diagnostic to the fixed 17-token workload at
position offset 37. Both captures pass 33,852,928 comparisons each under the original
`ideal_fp64` and `staged_fp32` gates: zero failures, maximum gate ratio 0.4141 and
matching lowest-ID greedy results. Their complete native capture bodies are
bitwise identical. Required checks and release baselines pass; Astra approves
scoped retention with no actionable findings.

Each capture observes all 1,051 final-token stages, recurrent states, complete
committed K/V caches and the full final logit row. Token 16 has position 53. The
native body is 67,705,856 bytes; both reference profiles occupy 203,117,568 bytes.
The original one-input plan, graph/row arithmetic and separate-head binding are
preserved. The explicitly named context17 plan adds only committed-cache/score
extents. Both old CPU/Metal one-input results recompute unchanged through the
expanded strict reference reader; broader contexts receive no implicit eligibility.

Two release cohorts on Apple M1 Pro 16 GiB / macOS 26.6.2 use three warmups
and ten measured repetitions each. Completed context17 forward medians are
4.092809 / 4.120841 s, IQRs 42.162 / 46.724 ms and sample standard deviations
42.231 / 59.866 ms. This is 4.154 / 4.125 completed input tokens/s, including a
full logit row for every input. A separate one-input control has a 240.546 ms
median and reproduces Q2.E's output bits. All 30 measured samples and nine warmups
are retained; 20 measured samples and six warmups belong to context17.

Only completed ordinary forward is timed. Reset, setup, loading, compilation,
upload, comparisons, logging and output writes are outside the clock. Each
process loads fresh, then resets/replays with resident weights after warmup.
Context17 loading takes 22.566 / 22.248 s and upload 9.244 / 9.349 s, recorded
separately. Both complete 17 tokens, 4,233 projections, 17 gathers, 408 convolution
steps, 408 recurrence steps and 2,176 attention heads per sample. Complete
17-row output vectors match across cohorts; their final rows match the
independently compared captures. AC power, nominal thermal state and Low Power
Mode off are observed at both boundaries of every cohort. These snapshots do
not establish continuous stability. No optimization, cross-workload speedup,
request TTFT, reliable tail, sustained-throughput or 1% overhead claim follows.

The native test also checks every one of the 17 logit rows against partitions 17,
1 and 3, with identical committed state and zero warmed Rust numerical allocations.
Repeated reset is allocation-free. From a committed five-token prefix it checks
partially completed observer failure, in-flight cancellation, stale nonfinite
future K/V masking, late GPU-head poisoning after 248 projections, blocked execution
and overlap-minus-one rejection. Successful repair preserves that prefix; the
12-token retry reproduces every original remaining logit row and final state
bitwise, with zero warmed allocations. This adds an explicit real 9B nonzero-prefix
retry witness to Q2.E's reset/replay check.

All production numerical, ABI and dependency sources are unchanged. The Rust
change adds an ignored test module behind the existing test/Metal/macOS guards.
A fresh optimized native binary and fresh ABI observation are used for captures.
The native62 package, its three 46-export slices and Csmokes are reused only after
checking all original inputs/outputs and the four old 190-input probe closures.
The new standalone probe uses compile-time workload selection; its original
one-input default and the context17 variant compile with strict Rust warnings.
The app remains linked to native46; physical iPhone execution remains deferred.

The independent runtime is admitted before numerical imports with fresh source,
Python/native manifests and a new owned bytecode cache. Its two-profile collection
takes 1,542.669 s and peaks at 613,909,656 physical bytes. These are diagnostic costs,
not an engine performance comparison. Both native diagnostic peaks are at most
5,666,951,680 bytes, with session admission 5,538,789,612 and repair overlap
5,580,214,680 bytes plus the reserved 128 MiB diagnostic allowance, within 6 GiB.
No retained full CPU matrix copy remains. Owner accounting, policy allowances and
measured physical footprints stay separate.

Formatting, both strict all-target Clippy configurations and 667 CPU / 803 Metal tests
pass (46 / 329 ignored), as do 46 Python tests. Three valid copied native/reference
closures pass; 25 damaged/resealed controls reject source, scope, work, memory,
phase/partition/recovery, timing and runtime-origin errors. Rewriting a finite
embedding value and updating its hashes still fails the independent gate. All
original files remain unchanged.

Initial runtime admission rejected reuse across worktrees. A local environment
clone was admitted with the guard unchanged; an earlier preparation ran before
that clone finished and also failed before numerical execution. Metal Clippy
caught a private enum import in the new test; the corrected full suite passes.
Audited cleanup removes only completed generated compiler outputs, preserves
906 pinned files and recovers 1.22 GB. The 7 GB target was missed; the 4.5 GB minimum and
2 GiB live disk guard passed. Failed attempts and raw logs remain in artifacts.

This qualifies only source-specific hybrid context17/offset37, with child-only
`MallocLargeCache=0` and Nano/Aggressive unset. Keep 6 GiB Mac, 512 capacity / 32 chunk,
512 MiB codec policy and the binding [4B 128/256 rejection](qwen35-context256.md).
Deeper contexts, whole-GPU execution, default allocator/codec/application fit,
requests, summary overhead, sustained/tail metrics, quality and managed adoption
remain separate gates. Next is explicit 9B 128/256 evaluation under the same
profiles and tolerances.

[Contract](../artifacts/qwen35-9b-context17-2026-11-09/acceptance-contract.json),
[checks](../artifacts/qwen35-9b-context17-2026-11-09/required-checks-2/checks.json),
[first fidelity](../artifacts/qwen35-9b-context17-2026-11-09/fidelity-1.json),
[second fidelity](../artifacts/qwen35-9b-context17-2026-11-09/fidelity-2.json),
[evidence controls](../artifacts/qwen35-9b-context17-2026-11-09/evidence-controls-1.json),
[reuse audit](../artifacts/qwen35-9b-context17-2026-11-09/production-reuse-audit.json),
[performance contract](../artifacts/qwen35-9b-context17-2026-11-09/performance-contract-1.json),
[raw performance/summary](../artifacts/qwen35-9b-context17-2026-11-09/performance-summary-1.json),
[checks/review/commit](../artifacts/qwen35-9b-context17-2026-11-09/publication-checks.json).

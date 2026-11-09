# Qwen3.5 context checkpoints at128 and256 inputs

Q1.J rejects128/256-input numerical qualification for the selected4B trajectory
under the original dual-reference gate. Astra approves scoped diagnostic and
evidence-reader retention with no actionable findings; no managed route or
larger context is admitted.
The full selected improvement goal remains active.

The frozen workload repeats the prior17raw fixture IDs through256 causal inputs
at absolute rotary offset0. Full checkpoints at128and256 cover last-input
stages, recurrent/convolution state, retained K/V, attention scores/probabilities
and padded full logits. The unchanged scalar Python graph runs both idealFP64
and stagedFP32 with authenticated source and closed Python/native runtime
endpoints. The binding gate remains `abs(actual-reference) <= 1e-5 + 1e-4*abs(reference)`
for every value in both profiles, with exact lowest-ID greedy selection.

| Inputs | Reference | Values compared | Outside gate | Maximum gate ratio |
| --- | --- | ---: | ---: | ---: |
| 128 | idealFP64 | 18,180,608 | 2,580 | 22.8316 |
| 128 | stagedFP32 | 18,180,608 | 286 | 5.8754 |
| 256 | idealFP64 | 20,310,528 | 4,277 | 22.8316 |
| 256 | stagedFP32 | 20,310,528 | 6,362 | 34.0655 |

The complete comparison covers4,204 arrays/76,982,272 values, with
13,505 failed numerical comparisons. Native/reference greedy IDs agree at4
and198 respectively; matching IDs do not clear the elementwise gate.
[Full comparison](../artifacts/qwen35-context256-2026-11-09/fidelity-final.json),
[decision](../artifacts/qwen35-context256-2026-11-09/decision.json).

The two reference tolerance intervals themselves are disjoint at473checkpoint
coordinates at128 and2,659at256: 3,132total checkpoint coordinates, maximum
positive interval gap0.0008465593. No scalar output can satisfy both frozen
inequalities at those coordinates. This diagnoses an incompatible acceptance
requirement on this workload; it does not establish an alternative precision
policy or excuse a failed gate. No threshold, profile requirement or model
adoption criterion is changed.
[Closed-reference envelope analysis](../artifacts/qwen35-context256-2026-11-09/reference-envelope-diagnostic.json)
includes four formula controls and depends only on the two independent profiles.

Native replay/recovery checks pass. All256 complete-logit rows compare bitwise
under chunk partitions32/17/1, including the irregular17-input checkpoint split.
Observer failure, cancellation and a late GPU head failure preserve the128-input
prefix; explicit repair/retry, stale-future NaN masking, reset, invalid/capacity
admission, completed-work counters and zero warmed numerical Rust allocations
pass. Capacity512/chunk32 and the6GiB ceiling remain fixed. Native physical peak
is2,764,966,624B; the closed independent reference peaks at809,584,128B.
[Native controls](../artifacts/qwen35-context256-2026-11-09/native-controls-final.json),
[reference closure](../artifacts/qwen35-context256-2026-11-09/reference-2/checks.json).

The reference streams461,893,632bytes under its512MiB capture allowance. Native
checkpoint rows use153,964,544bytes; the complete-logit replay file uses
254,279,680bytes. Both are streamed to bounded files; no full-logit history is
retained in RAM. Native diagnostic ownership reserves256MiB before numerical
allocation; the reference has its own bounded ledger and six-hour deadline.
Target intermediate values never enter the independent producer.

Required checks pass: formatting, strict CPU/Metal all-target Clippy,
644CPU/778Metal tests (41/319ignored),26producer/reader unit tests,
24actual resealed native and30actual independent-source rejection controls.
Existing native58 package/three46-export/C ABI evidence is reused after auditing
that the changed/new Rust is test-only. Production library, shaders, dependencies,
ABI, Swift and packaging inputs remain unchanged;3,940Rust inputs and retained
outputs are hash-checked. Ordinary CPU tests require no model or Apple GPU.
[Required checks](../artifacts/qwen35-context256-2026-11-09/required-checks-4/checks.json),
[native evidence controls](../artifacts/qwen35-context256-2026-11-09/native-reader-controls-3.json),
[independent evidence controls](../artifacts/qwen35-context256-2026-11-09/reference-reader-controls.json),
[package reuse](../artifacts/qwen35-context256-2026-11-09/native-package-reuse.json).

A separate scalarCPU128 localization capture peaks at2,634,811,768B. All18.18M
CPU/Metal values satisfy their pairwise tolerance, max ratio0.78954, while
6,910,567FP32 bit patterns differ. Its ideal-prefix comparison also fails.
This localizes the discrepancy beyond a GPU-only difference and does not clear
the independent gate or identify a cause.
[Scalar comparison](../artifacts/qwen35-context256-2026-11-09/cpu128-comparison.json),
[pairwise tolerance](../artifacts/qwen35-context256-2026-11-09/cpu-metal128-tolerance.json).
The [pinned norm audit](../artifacts/qwen35-context256-2026-11-09/norm-sanitization-primary-audit.json)
confirms the investigated stored-norm convention; it is not upstream numerical parity.

The separate stagedFP32 recurrence prototype matches169,728 tiny/control values,
then1,225,024full tiny-graph values in both profiles and16,333,568actual-source
17-input values. Its closed128/256 trial now matches every bit of all38,491,136
original scalarFP32 values/2,102arrays. Source/runtime checks,8reader unit controls
and32actual resealed capture controls pass; physical peak722,060,728B. The original
collector remains unchanged. Isolated3-warmup/10-pair recurrence cohorts reduce
645/619ms medians to22.4/22.6ms; these are reference-step measurements, not model
or whole-graph speedups. The full collections run different profile sets alongside
other diagnostic work and provide no matched timing comparison.
[Isolated experiment](../artifacts/qwen35-context256-2026-11-09/recurrent-vector-experiment-summary.json),
[full byte parity](../artifacts/qwen35-context256-2026-11-09/reference-graph-vector-experiment/full256-parity.json),
[variant evidence controls](../artifacts/qwen35-context256-2026-11-09/reference-graph-vector-experiment/full256-reader-controls.json).

Initial macro/import compilation failures, storage-exhausted first captures and
the prototype's unrounded-control rejection remain recorded. Cache recovery
preserves source, raw evidence, archives and link outputs; fresh native2,
required4 and reference2 complete. Failed attempts remain rejected/incomplete.
[Storage failure/recovery](../artifacts/qwen35-context256-2026-11-09/storage-failure-audit.json),
[prototype correction](../artifacts/qwen35-context256-2026-11-09/recurrent-vector-first-correction.json),
[preservation audit](../artifacts/qwen35-context256-2026-11-09/iteration-preservation-8.json).

The decision retains rejection evidence and bounded diagnostic/recovery tools
with scoped Astra approval; [checks/review/commit](../artifacts/qwen35-context256-2026-11-09/publication-checks.json). Earlier one/17-input results retain their original
scope. Full requests, TTFT/overhead,512/1024-input numerical execution, task
quality, managed adoption, whole-GPU execution and iPhone runtime are unqualified.
The next selected model is9B, with separate source/head/fit/codec/numerical gates;
[model plan](model-observability-plan.md),
[frozen contract](../artifacts/qwen35-context256-2026-11-09/acceptance-contract.json).

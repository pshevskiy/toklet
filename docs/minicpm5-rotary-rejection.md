# MiniCPM5 rotary precision experiment: rejected

The model-owned rotary coefficient experiment fails **six original short-suite
numerical comparisons**. Retain the investigation and rejection record, with
the candidate implementation archived outside the engine commit. The existing
ordinary loader, 512/chunk32 policy, separate 1,025-position diagnostic and all
numerical, memory, quality and adoption gates remain unchanged. Astra approves
this documentation-only retention, with no actionable findings.

## What the investigation established

The prior deeper diagnostic still rejects 82 values in 1.773B comparisons. Eight
selected failing projection coordinates reproduce exact rational dots of their
own native/FP32/FP64 inputs and independently reconstructed affine weight rows.
That local evidence points upstream; it does not qualify every projection.
Seven traced token rows cover 50 selected stages. Five first diverge from FP64 at
layer 0 rotary; two differ earlier at input normalization. No universal cause is
inferred from this sparse trace.

- [Exact scalar replay](../artifacts/minicpm5-projection-isolation-2026-11-10/isolation-2.json)
  and [owned resource check](../artifacts/minicpm5-projection-isolation-2026-11-10/isolation-checks-2.json).
- [Stage trace](../artifacts/minicpm5-projection-isolation-2026-11-10/trace-3.json)
  and [resource check](../artifacts/minicpm5-projection-isolation-2026-11-10/trace-checks-3.json).

The experiment precomputes binary64 phases, splits sine/cosine into FP32 high/low
words and uses compensated complex products before Half storage. An explicit
model constructor owns one immutable table/pipeline; sessions borrow it. The
table is 1,049,668 bytes at 1,025 positions or 524,356 bytes at 512, with a separate 64 MiB
pipeline allowance. Ordinary factories are unchanged in the candidate as well.
The fixed table has scalar 4-byte loads and seven guards; no stronger alignment
or assumed GPU execution width is required.

The independent original binary64 CPU routine is frozen before primitive GPU
outcomes. All 7,084,800 same-input Half Q/K values across 1,025 positions and
layers 0/1/41 match numerically; signed zeros compare equal. The ordinary primitive
differs at 82,027 values on those same inputs. This establishes local fidelity in
that domain, without claiming universal correctly rounded transcendentals or
an end-to-end improvement.

[Model experiment contract](../artifacts/minicpm5-projection-isolation-2026-11-10/prospective-contract-2.json),
[primitive contract](../artifacts/minicpm5-projection-isolation-2026-11-10/prospective-contract-1.json),
[independent reference](../artifacts/minicpm5-projection-isolation-2026-11-10/rotary-reference-1/manifest.json),
[optimized primitive result](../artifacts/minicpm5-projection-isolation-2026-11-10/primitive-1/report.json),
[composed-source GPU checks](../artifacts/minicpm5-projection-isolation-2026-11-10/composed-gpu-checks-1.json).

## Why the model variant is rejected

The original six cases, chunks 1/3/32, both original CPU profiles and all original
gates remain unchanged. All 2,614 native arrays complete; 52,896,768 values are
compared. At `tile17`, row 4/column 106, FP64 comparisons fail at two layer 41 stages
under every chunk schedule:

| Stage | Native | Original FP64 | Absolute error | Original bound |
| --- | ---: | ---: | ---: | ---: |
| Attention projection | -3.46875 | -3.4296875 | 0.0390625 | 0.037609375 |
| Attention residual | 377.5 | 371.5 | 6 | 4.3715 |

A fresh full offline replay reproduces the byte-identical rejection report.
These six failures are binding. Full/greedy readback selection, exact causal
prefixes, the original teacher-forced histories and 512 boundary/reset controls
pass. State correctness does not override the numerical rejection.

[Native owner](../artifacts/minicpm5-projection-isolation-2026-11-10/short-checks-1.json),
[complete short comparison](../artifacts/minicpm5-projection-isolation-2026-11-10/short-comparison-1.json),
[fresh replay receipt](../artifacts/minicpm5-projection-isolation-2026-11-10/short-replay-checks-2.json),
[independent raw witnesses/negative controls](../artifacts/minicpm5-projection-isolation-2026-11-10/short-audit-1.json).

The deeper run had started when the short result became available. Its verified
owned process was deliberately terminated because a mandatory prerequisite had
failed. Exit code -15, no final capture manifest and `passed=false` remain authoritative;
the complete journal prefix and raw data are preserved. There is no completed
variant 1,025 comparison, full-context memory qualification, quality score or
performance/overhead measurement. No speed or 1% telemetry claim follows from
diagnostic clocks. Performance cohorts are unwarranted after this numerical
rejection.

[Stop decision](../artifacts/minicpm5-projection-isolation-2026-11-10/rejection-stop-decision-1.json),
[closed failed owner](../artifacts/minicpm5-projection-isolation-2026-11-10/native-checks-1.json),
[partial-capture interpretation](../artifacts/minicpm5-projection-isolation-2026-11-10/stopped-native-interpretation-1.json).

## Candidate validation and retention boundary

Candidate checks pass: fmt, both strict Clippy configurations, 683 CPU/831 Metal
tests, nine CLI example tests, two explicit GPU tests and 36 portable tests.
The real-model GPU control observes 84 actual table encoders, shared ownership,
tight-budget rejection and reset/replay. Native69 compiles three arm64 slices
with 46 C exports each; CPU and Metal C smokes pass. The short capture's final RSS
is 2,209,497,088 bytes and sampled lifetime physical peak 2,792,458,712 bytes, under
the unchanged 6 GiB guard. Final physical peak remains unavailable.

[Required checks/counts](../artifacts/minicpm5-projection-isolation-2026-11-10/required-checks-1/counts.json),
[required phase receipt](../artifacts/minicpm5-projection-isolation-2026-11-10/required-checks-1/checks.json),
[portable log](../artifacts/minicpm5-projection-isolation-2026-11-10/portable-all-1.log),
[native package](../artifacts/minicpm5-projection-isolation-2026-11-10/native69-publication.json),
[C smokes](../artifacts/minicpm5-projection-isolation-2026-11-10/native-smoke-69/checks.json).

Six failing coordinates replay independently using stdlib Half/F32 decoding and
scalar binary64 bounds; 12 resealed actual metadata/owner copies reject. The
original eight-source CPU producer and 14 selected primitive-reference sources
remain byte-identical. The uncompleted deeper consumer's full numerical path
is not qualified by its portable checks.

[Reference protection](../artifacts/minicpm5-projection-isolation-2026-11-10/reference-protection-1.json).
All 4,038 historical primitive build inputs also reconstruct exactly from
unchanged files, three saved pre-composition files and two immutable base files.
[Source reconstruction](../artifacts/minicpm5-projection-isolation-2026-11-10/primitive-source-reconstruction-1.json).

The complete 14-file rejected code/fixture patch and source copies are retained
as experimental evidence. This commit proposes only five research/planning
documents; no candidate kernel, model loader, CLI mode or reader is installed.
The isolated prototype checkout remains frozen for its owner/source provenance.
The retained engine matches all 4,034 prior Cargo inputs, 421 native inputs and
ten package outputs; their prior checks apply to those unchanged bytes.
System timing, clone-index and audit-schema preparation failures are preserved
with their corrections; no contract or gate was changed after outcomes.

[Rejected patch](../artifacts/minicpm5-projection-isolation-2026-11-10/rejected-implementation-1.diff),
[source manifest](../artifacts/minicpm5-projection-isolation-2026-11-10/rejected-source-1/manifest.json),
[timing failure](../artifacts/minicpm5-projection-isolation-2026-11-10/resource-attempt-1-interpretation.json),
[preflight corrections](../artifacts/minicpm5-projection-isolation-2026-11-10/reference-protection-1.json).

Next, audit whether every intersection of the unchanged original CPU bounds
contains a representable native Half/F32 value. Continuous interval overlap by
itself does not prove representability. This is a pending question, without an
infeasibility finding or permission to alter precision, tolerances or quality
policy. Both original 10k-target/100-task comparisons, model adoption and the
whole roadmap remain open. The 6 GiB limit, 2 GiB disk floor and deferred screen/
iPhone scope remain intact.

[Review/commit and scope](../artifacts/minicpm5-projection-isolation-2026-11-10/publication-checks.json).

[Astra review](../artifacts/minicpm5-projection-isolation-2026-11-10/astra-review-result-1.json).

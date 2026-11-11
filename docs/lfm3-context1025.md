# LFM3 BF16 parity through 1,025 positions

The authenticated 24-layer BF16 graph runs a fixed 14-chunk schedule
through 1,025 positions. Its first nine chunks reproduce the previously
qualified 513-position schedule. The added chunks cover full attention
at longer depths and further rollover of the 256-token sliding windows.
The probe compares layer outputs, history, K/V proposals, final norms,
last-row tied logits, and actual attention views. It checks failure before
publication and retries at offsets 257 and 1,024.
[Schedule and limits](../tests/fixtures/lfm3-context1025-v1/contract.json).

## Independent reference

Two initial reference attempts exceeded the 512 MiB RSS ceiling. The
accepted reference runs separate guarded phases for the three layer
groups and the tied head. Each phase checks the pinned runtime and
checkpoint. Consecutive phases exchange hashed BF16 hidden rows.
Full attention with more than 513 keys calls the same pinned eager
operation per KV group. Four independent grouped/full-head controls
match 788,480 BF16 words exactly.

The guarded phases peak at 485,376,000 bytes of conservative aggregate
RSS. Their combined payload matches all 549 observations from the prior
513-position reference. It also matches all 854 observations and
68,054,016 BF16 words from the full-head capture. The original failed
guard and its cross-check role are recorded.
[Reference reconciliation](../artifacts/lfm3-context1025-2026-11-11/source-reconciliation.json),
[first guard failure](../artifacts/lfm3-context1025-2026-11-11/attested-context1025-checks.json),
[second guard failure](../artifacts/lfm3-context1025-2026-11-11/attested-context1025-bounded-checks.json).

## Native comparison and release timing

All 854 native observations and 68,054,016 BF16 words match exactly.
The 549 overlapping native stage digests match the earlier run. 30
attention views contain two wrapped old spans. Both late aborts leave the
published prefix and distribution intact, and retry reproduces candidate
logits. The native run's combined final RSS upper bounds total
3,528,048,640 bytes under the 6 GiB engine ceiling.
[Comparison](../artifacts/lfm3-context1025-2026-11-11/report.json),
[memory guard](../artifacts/lfm3-context1025-2026-11-11/verify-checks.json).

The release probe on M1 Pro records the first ordinary one-token forward,
three warmups and ten measured commits at offsets 1,025 through 1,038.
The measured median is 1.285 seconds, with 7.65 ms median
absolute deviation. AC power, Low Power Mode and thermal checks pass.
The timed commits follow the exact comparison. The numerical qualification
ends at 1,025 positions, and observer overhead remains unqualified.
[Raw samples](../benchmarks/results/2026-11-11-lfm3-context1025-v1.json).

Formatting, both strict Clippy configurations, 733 CPU tests, 879 Metal
tests and nine targeted example tests pass. Native package and C smoke
inputs match the previous verified build, including 444 source pins,
ten package outputs and both Mac C smokes. The engine implementation
and ABI are unchanged.
[Native reuse](../artifacts/lfm3-context1025-2026-11-11/native-reuse.json),
[Checks](../artifacts/lfm3-context1025-2026-11-11/checks.json).

Astra approved the diagnostic, source reconciliation and release evidence
with no actionable findings.
[Review](../artifacts/lfm3-context1025-2026-11-11/astra-review.json),
[commit](../artifacts/lfm3-context1025-2026-11-11/publication-checks.json).

## Original quality inputs

The original corpus produces 294,504 identical IDs with both pinned
tokenizers. The first 10,001 define all 10,000 loss targets in ten blocks.
The 100 original grading cases and output caps are preserved. All 30 prose
and 30 continuation inputs match. The stock template trims a trailing
newline before the end-of-message marker in each of the 40 code prompts.
The modified-parent and candidate text assets are byte-identical.
The complete input suite requires a capacity of 1,025.
[Input reconciliation](../artifacts/lfm3-context1025-2026-11-11/quality-inputs/reconciliation.json).

A separate source pilot matches all 16,384,000 BF16 tied-head words for 128
real hidden rows, including the previously captured endpoint logits.
Its bounded FP64 reduction matches the direct complete-vocabulary NLL on
three rows. It uses synthetic targets. Full held-out loss, task scoring
and managed adoption remain open.
[Scoring pilot](../artifacts/lfm3-context1025-2026-11-11/quality-inputs/head-pilot-result.json).

# LFM3 original quality campaign

The [frozen input reconciliation](../artifacts/lfm3-context1025-2026-11-11/quality-inputs/reconciliation.json)
defines 10,000 teacher-forced loss targets and all 100 tasks in their original
order. The loss inputs match across profiles. The stock template trims a
trailing newline in the 40 code prompts. Both quality comparisons retain the
original mean NLL allowance of 0.05 nats per token and two percentage points
of task pass-rate loss.

## Collection

The original stock and modified BF16 references use the pinned eager
arithmetic. Source task generation streams a layer bank across independent
task caches. Each guarded round records its input tokens, cache receipts,
and complete tied-head decision digests. A round advances only after all
layer phases and the head pass. Successful rounds discard expired cache
files and retain the evidence needed to reconstruct the trajectory.

Toklet collects tasks through its existing JSON console. Each case uses
the prepared IDs, a 128-token prefill chunk and the original output cap.
The collector records every token event, hidden stop, logical admission
ledger and reset. The offline auditor replays each completed case from
the captured console events.

The `lfm3_quality_loss` Cargo example scores each frozen target from the
authenticated native graph. It verifies the checkpoint and the tied-head
tensor, checks the projected endpoint against the published logits, and
uses the shared stable FP64 NLL reduction. The diagnostic admits its
additional BF16 head bank and work buffers within the 6 GiB limit. The
guarded loss runner seals full blocks and validates every source and target
ID. Native task and loss jobs run serially on this Mac.
[Native loss queue](../artifacts/lfm3-context1025-2026-11-11/quality-inputs/native-loss-queue.json).

## Evidence and decisions

The modified-reference pilot completes one original continuation case.
An independent layer-streamed replay reproduces all 16 greedy decisions
and digests of 2,048,000 BF16 logits. The source collection stays within
512 MiB. The guarded decoder verifies the same decision chain and decodes
the visible IDs with the pinned tokenizer. The pilot misses the literal
expected word.
[Trajectory](../artifacts/lfm3-context1025-2026-11-11/quality-inputs/source-task-70-trajectory-replay/result.json),
[decoder audit](../artifacts/lfm3-context1025-2026-11-11/quality-inputs/source-task-70-decoded-v2/result.json).

Both references finish all 10,000 matched loss targets. The stock mean
NLL is 3.27640 and the modified mean is 3.28421 nats per token. The increase
of 0.00781 passes the original 0.05 allowance. Both reference runs stay
within 512 MiB.
[Complete reference loss](../artifacts/lfm3-context1025-2026-11-11/quality-inputs/loss-reference-complete.json),
[reference guard summary](../artifacts/lfm3-context1025-2026-11-11/quality-inputs/loss-reference-guard-summary.json).

The first 1,024 Toklet loss targets match all modified-reference greedy
IDs. The largest absolute NLL delta is 3.55e-15. The release pilot takes
793.7 seconds with a conservative aggregate peak of 3,769,417,728 RSS
bytes. It uses the prototype collector with the same graph and tied-head
arithmetic. The standard example uses the shared FP64 reduction.
[First-block comparison](../artifacts/lfm3-context1025-2026-11-11/quality-inputs/native-loss-pilot/comparison.json),
[memory receipt](../artifacts/lfm3-context1025-2026-11-11/quality-inputs/native-loss-pilot/guard.json).

The original literal grader receives complete independently decoded
collections and runs HumanEval in the qualified isolated guest.
`run_lfm3_quality_grading.py` owns the grading process group through exit
and independently verifies the guest returns to its stopped state. It
preserves the qualified timezone and offers the existing ordinary-first
CPU witness for ambiguous exit 137 under the original four-second limit.
[Stopped-guest check](../artifacts/lfm3-context1025-2026-11-11/quality-inputs/grading-window-live-smoke/result.json).
Astra approves the grading-window integration with no findings.
[Window review and commit](../artifacts/lfm3-context1025-2026-11-11/quality-inputs/grading-window-publication-checks.json).

The final decision checks both 10,000-target pairs and both 100-task pairs
before publishing a quality outcome. Both comparisons use the same
authenticated modified-reference task collection and decoded outputs.
Both complete comparisons now pass. On all 10,000 targets, the native mean NLL
matches the modified reference's 3.28421 nats per token, every greedy ID agrees,
and the maximum absolute NLL delta is 3.66e-15. The equal-weight literal task
scores are 35/100 for stock and 37/100 for both modified and native. All 5,094
native sampled decisions, stops and decoded rows match the modified collection.
Both isolated grading windows independently confirm the guest stopped.
[Final loss](../artifacts/lfm3-context1025-2026-11-11/quality-inputs/quality-final-loss.json),
[complete decision](../artifacts/lfm3-context1025-2026-11-11/quality-inputs/original-quality-decision.json),
[complete task comparison](../artifacts/lfm3-context1025-2026-11-11/quality-inputs/native-complete-source-comparison-1.json).

A turn interruption ended the first remaining-loss supervisor. Its authenticated
orphan was stopped, and that partial output remains unqualified. A fresh run of
the remaining nine blocks under the unchanged guard and frozen binary completes
8,976 targets in 6,486 seconds with a conservative aggregate RSS peak of
4,085,530,624 bytes, within 6 GiB. The recovered finalizer reuses the authenticated
complete decodes and preserves the original thresholds and isolated grader.
The decision above applies to the original native binary. The fresh
finite-widening candidate has its own complete results below. Astra approves
scoped CPU qualification and retention; managed adoption remains open.
[Retry guard](../artifacts/lfm3-context1025-2026-11-11/quality-inputs/native-loss-remaining-retry-2/guard.json),
[recovery completion](../artifacts/lfm3-context1025-2026-11-11/quality-inputs/finish-quality-result.json).
[Grader preflight](../artifacts/lfm3-context1025-2026-11-11/quality-inputs/grader-preflight.json).

The finite-widening candidate compiles a fresh loss example and collects all
10,000 original targets and 100 original tasks, preserving output caps, normal
EOS, the isolated four-second CPU grader and unchanged quality gates. Both loss
pairs pass. All native greedy loss IDs and 5,094 task decisions/stops/decoded
rows match the modified reference; maximum NLL delta is 3.66e-15. The original
loss executable is not reused as candidate evidence.
[Fresh loss-build binding](../artifacts/lfm3-finite-guard-2026-11-11/fresh-quality-1/candidate-loss-binding.json),
[complete task identity](../artifacts/lfm3-finite-guard-2026-11-11/fresh-quality-1/native-complete-source-comparison.json).

The first isolated grading attempt stops on process inspection of owned setuid
`ps`; its failed receipt remains immutable. Native process-group discovery
replaces that child, with unreadable live processes still failing closed.
Concurrent-working-set accounting preserves retrospective lifetime peaks without
summing disjoint serial clients as simultaneous allocations. The 6 GiB guard
also reserves the entire 2 GiB guest. SDK/live checks, 14 original isolation
modes, 28 authored literal programs and a 260-command inert sequence pass.
The fresh retry completes all 80 isolated program runs, retains 37/100 passes
for both modified reference and candidate, has no unresolved cases, and
independently verifies guest shutdown. Both complete quality comparisons pass.
[Retained grading failure](../artifacts/lfm3-finite-guard-2026-11-11/fresh-quality-grading-failure-audit-1.json),
[recovery checks](../artifacts/lfm3-finite-guard-2026-11-11/grading-recovery-checks-1.json),
[fresh complete quality decision](../artifacts/lfm3-finite-guard-2026-11-11/fresh-quality-recovery-4/original-quality-decision.json).

Astra approved the tooling after the shared modified-reference identity
check was added. The six portable tests cover differing completions with
equal grades and absent or malformed provenance.
[Review and commit](../artifacts/lfm3-context1025-2026-11-11/quality-inputs/quality-publication-checks.json).

The primary and full 156-request compatibility/sustained gates pass. All 62
capacity/state checks and exact event/memory replay then pass, within 6 GiB.
All 441 portable tests across 60 local modules pass in both Python modes under
512 MiB. Earlier runner/preflight failures remain retained; no numerical,
quality, capacity or memory gate is reduced. The finite-widening change and
host-helper update pass combined Astra review with no actionable findings;
CPU retention is approved. Managed adoption remains open.
[Full state replay](../artifacts/lfm3-finite-guard-2026-11-11/state-analysis-4.json),
[candidate scope and measurements](lfm3-finite-widening.md).
[Review/commit](../artifacts/lfm3-finite-guard-2026-11-11/finite-publication-checks.json).

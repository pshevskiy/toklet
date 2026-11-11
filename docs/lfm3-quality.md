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

The first 1,024 Toklet loss targets match all modified-reference greedy
IDs. The largest absolute NLL delta is 3.55e-15. The release pilot takes
793.7 seconds with a conservative aggregate peak of 3,769,417,728 RSS
bytes. It uses the prototype collector with the same graph and tied-head
arithmetic. The standard example uses the shared FP64 reduction.
[First-block comparison](../artifacts/lfm3-context1025-2026-11-11/quality-inputs/native-loss-pilot/comparison.json),
[memory receipt](../artifacts/lfm3-context1025-2026-11-11/quality-inputs/native-loss-pilot/guard.json).

The original literal grader receives complete independently decoded
collections and runs HumanEval in the qualified isolated guest. The
final decision checks both 10,000-target pairs and both 100-task pairs
before publishing a quality outcome. Both comparisons use the same
authenticated modified-reference task collection and decoded outputs.
Full results and managed adoption remain open.
[Grader preflight](../artifacts/lfm3-context1025-2026-11-11/quality-inputs/grader-preflight.json).

Astra approved the tooling after the shared modified-reference identity
check was added. The six portable tests cover differing completions with
equal grades and absent or malformed provenance.
[Review and commit](../artifacts/lfm3-context1025-2026-11-11/quality-inputs/quality-publication-checks.json).

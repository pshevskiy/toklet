# Parallel engine and CLI delivery

Shorten the path from a measured bottleneck to a checked, reviewed change.
Keep the original [acceptance policy](../PLAN.md#validation-and-performance-policy),
model priorities and numerical decisions. Engine/CLI leads; interactive screen
capture and iPhone runtime remain deferred. No delivery speedup is measured yet.

## Current assignments

The Qwen3.5-2B bounded numerical search is closed. The LFM3 two-token experiment
is also closed after one dispatch correction: large operator gains did not close
the original one-token compatibility gate. Astra approved evidence-only closure;
all candidate code was reverted. [Experiment decision](lfm3-two-token-projection.md).

| Lane | Exclusive ownership | Ready handoff |
| --- | --- | --- |
| Implementation | One mechanism's numerical source and shared signatures | Small frozen diff, layout/ownership/numerical invariants, targeted controls |
| Independent validation | Oracle fixtures and artifact collection/replay | Exact input/output pins, smallest controls, reproducible commands and unresolved gates |
| Evidence/documentation | One report or next-command preparation | Concise actual pass/reject scope, receipts and limitations |
| Integrator | TODO/improvements, shared API, heavy queue and publication | Earliest integrated check, one combined Astra review, retained/reverted decision, commit/push and exact-head CI |

Independent static preparation may overlap compilation. Before performance work,
the integrator requires explicit **paused** acknowledgements from every active
lane and confirmation that local preparation processes have finished. Record
this barrier before launch. All preparation stays paused through measurement.
The first corrected LFM3 screen started too early and was excluded; its complete
unchanged matrix was rerun after pause. Do not repeat that scheduling mistake.
[Retained exclusion](../artifacts/lfm3-tokens2-2026-11-13/tokens2-screen-exclusion-2.json).

This ownership and queue policy avoids file collisions and measurement overlap;
delivery speedup remains unmeasured. Reassign a lane to Astra when the combined
packet is ready; a finite failed experiment ends with an explicit closure.

## Keep four slots useful

Use one integrator and up to three independent owners. Give each owner an
exclusive file set, concrete output, acceptance command and stopping rule.
Agree on shared signatures before source edits. Each handoff contains paths,
input hashes, the next executable command and unresolved dependencies; link logs
instead of copying them. Preparation is not successful execution.

Reassign a finished lane to Astra when the combined diff and check results are
ready. During review, other owners may prepare independent fixtures, commands or
documentation. Explicitly dispatch follow-ups to completed agents and require a
running acknowledgement; a message alone does not restart them. Integrate useful
ready slices without waiting for unrelated preparation to finish.

Keep one implementation experiment active. Read-only preparation for the next
task can overlap a settled pass/reject decision and its evidence closure.
Do not open a second model campaign to fill a slot.

## Use short decision checkpoints

1. Confirm the bottleneck using current source and a representative workload.
   Freeze one hypothesis, primary metric, finite candidate list and stop rule.
2. Implement one mechanism with an independent oracle. Run cheap boundary,
   arithmetic, error, allocation and state controls before full-model work.
3. Run the first integrated compile as soon as the small interface is ready.
   Fix integration failures before collecting expensive captures.
4. Advance to applicable full graph/state/quality and resource gates only after
   local gates pass. A failure stops dependent performance/adoption cohorts.
   Further experiments require new evidence and a new declared hypothesis.
5. Measure a promising candidate in release mode, including complete requests,
   confirmation and sustained workloads required by PLAN. Retain or revert.
6. Submit one frozen diff, invariants and combined results to Astra. Resolve
   findings in that review, then commit/push and verify exact-head CI.

The [completed 2B checkpoint](../artifacts/qwen35-2b-attention-2026-11-13/two-f32-qk-check-1/result.json)
ends its finite search. It does not establish that every possible algorithm fails.
Original numerical, quality and observer-overhead gates remain unchanged.

## Protect the heavy queue and reuse evidence

Only the integrator starts compilers, conversions, numerical runtimes, model
loads and quality jobs. Performance cohorts run alone with recorded power and
thermal conditions; even cheap background preparation pauses during measurement.
Keep the 6 GiB engine, 512 MiB setup including 32 MiB helper, 3 GiB compiler-only
and 2 GiB free-disk limits. Reuse the clean worktree and Cargo cache.

Use existing probes and guards. Check interpreter, arguments, input/runtime pins,
output paths and phase disk admission before entering the queue. Add a helper
only when an existing tool cannot express a necessary measurement or control.

Reuse successful checks only when their relevant source, dependency, feature,
compiler and runtime inputs are unchanged. Record each invalidation and rerun
reason. Documentation-only changes need content/link/whitespace checks. Kernel
changes retain required fmt, strict CPU/Metal Clippy, CPU/Metal tests and affected
model/session/device checks. New numerical inputs invalidate their downstream
captures; unchanged model bytes do not waive a changed kernel's validation.

## Measure the workflow

Use the existing milestone receipt as one event ledger: dispatch/acknowledgement,
handoff ready, first integrated check, input freeze, queue start/end, checks
complete, Astra request/approval, push and green CI. Preserve actual timestamps;
leave missing historical clocks unset.

Report elapsed time to reviewed green CI, heavy-queue wait/execution, review wait,
first-check time, rework and repeated-check reasons. Estimate the next checkpoint
from the longest dependency chain and observed jobs, with numerical uncertainty
separate. Do not sum parallel estimates or predict a workflow speedup without
comparable completed milestones.

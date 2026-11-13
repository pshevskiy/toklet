# Parallel engine and CLI delivery

The current track is Qwen3.5-2B real-source conversion and engine/CLI
diagnostics, followed by numerical investigation. The typed-source/tiny-graph slice is published as `025bcd1e`;
Ministral's `8b6220e9` numerical rejection remains explicit. Keep acceptance in
[PLAN](../PLAN.md#validation-and-performance-policy) and preserve model order,
existing source eligibility and earlier rejection decisions.

## Assign independent work

Use three implementation lanes and one integrator, with exclusive file ownership.
Keep one model milestone in flight. Parallelize its independent mechanisms and
preparation; close its pass/reject decision before opening another model campaign.
The [previous dispatch receipt](../artifacts/qwen35-2b-source-planning-2026-11-13/parallel-work-plan-1.json)
records the completed typed-source slice. Current source/conversion evidence is
in [TODO](../TODO.md#current-and-next-task); keep the current phase's owners and
next executable commands separate from completed history.

| Owner | Current work | Handoff |
| --- | --- | --- |
| Source lane | Parent lineage, bounded Q4 conversion and saved-byte audit | Frozen source/child manifests and complete tensor coverage; prepare the next bounded source task after handoff |
| Graph lane | Explicit token-ID CLI, native capture and transaction/repair controls | Exact build/test/CLI commands with environment and output paths; existing numerical loops unchanged |
| Reference lane | Independent reader, full-coordinate reference and comparison | Frozen protocol, runtime admission and executable capture/check commands, independent of native output |
| Integrator | Shared declarations, manifest parser, serial check queue, status, Astra and publication | One bound set of source inputs, combined results/review and exact-head CI |

Agree on the small API before editing shared files. Hand off a concrete function
signature, ownership/accounting rule and acceptance command; identify any unknown
that blocks actual qualification. Mark a lane ready for integration only when its
files and generation commands are available. Do not wait for a lane's entire
full-model campaign before integrating its first useful change.
Prefer one executable handoff plus a short summary of changed files, inputs,
outputs and blockers. Link raw logs and manifests instead of copying them into
each handoff or status update.

Reuse the clean worktree and Cargo cache. Each lane may run cheap static or
stdlib-only controls over its owned files. The integrator owns shared edits,
compilation and model/runtime execution. A completed lane frees its slot for
Astra or independent next-task preparation; at most four agents run at once.
Explicitly dispatch and acknowledge a follow-up when an owner has completed;
sending a message alone does not restart an idle agent. Confirm a running owner
before waiting on a handoff that blocks the heavy-job queue.

## Follow the dependency chain

For the current real-source phase, source authentication, parent comparison,
conversion/audit, Metal/GPU/state/CLI controls and the short graph have passed.
Context17 rejects three of 680,841,024 comparisons with identical native traces
and completed offline audits. Conditional fixed255 is not launched. Astra
approves diagnostic retention; local-input investigation and the original deeper
graph/codec/quality/fit → performance/observer-overhead gates remain open. A
numerical rejection stops dependent cohorts and becomes an explicit reviewed
decision; it does not qualify the remaining gates. Model qualification can span
several reviewable code milestones.

1. In parallel, implement typed source loading, graph binding and independent
   fixture/reference work. The integrator resolves source lineage and storage
   requirements and wires the agreed interface.
2. Run the first integrated compile as soon as those interfaces are present.
   Prove the tiny ratio-one graph, exact F32 small values, session recovery and
   memory accounting before spending time on full payloads or captures.
3. Authenticate source values and conversion semantics before admitting a real
   child. Reserve the complete selected phase, including sources, child, captures,
   temporary workspace and disk floor, before downloading its payloads. Admit
   deeper graph/stock-quality phases separately once their prerequisites pass;
   their original quality and memory gates remain required before performance.
4. Freeze the combined inputs. Run applicable fmt, strict Clippy, CPU, Metal and
   explicit session/GPU checks. Reuse results only with verified unchanged
   relevant source/dependency inputs. Record why a repeated check was necessary.
5. A failing numerical, state, codec or fit gate stops its dependent quality and
   performance cohorts. Preserve the failure and record a scoped diagnostic or
   rejection decision. Independent portable work can continue.
6. Send Astra the frozen diff, invariants and combined results in one request.
   Resolve findings there; commit/push the reviewed milestone once and verify
   exact-head CI. Do not create separate routine helper/publication reviews.

The [current critical-path analysis](../artifacts/qwen35-2b-payload-2026-11-13/velocity-critical-path-1.json)
records five concrete changes: one active model, cheap preflight before heavy
work, reuse across matching input hashes, independent preparation during the
serial queue, and short graph pass/reject checkpoints before dependent campaigns.
The [graph/CLI handoff](../artifacts/qwen35-2b-payload-2026-11-13/graph-execution-handoff-1.json)
and [reference handoff](../artifacts/qwen35-2b-payload-2026-11-13/reference-execution-handoff-1.json)
contain prospective commands; preparation does not count as successful execution.

## Serialize scarce resources

One queue owns builds and memory-intensive conversion, numerical, model and
quality jobs. Performance cohorts run alone, with the required power/thermal
conditions and no competing preparation. Keep the 6 GiB engine, 512 MiB
setup/audit including its 32 MiB helper reservation, 3 GiB compiler-only and
2 GiB disk-floor policies. Parallel agents do not grant parallel heavy jobs.

Prepare bounded codec metadata, control inputs, documentation or a future diff
while a heavy job runs. Watch remote CI or prepare the next source contract while
Astra reviews. Avoid importing a numerical runtime or starting another model to
fill an idle slot. Engine/CLI work requires no screen-sharing picker.

## Measure delivery velocity

Use the milestone receipt as the event ledger: dispatch, first integrated check,
source freeze, required checks complete, review request/approval, push and green
CI. Preserve actual timestamps; record missing clocks as missing. Track total
elapsed time, heavy-queue execution/wait, review wait, rework and repeated-check
reasons. Keep raw logs in artifacts and summarize only counts and evidence links
in TODO/improvements.

Estimate remaining time from the longest dependency chain and observed job
lengths, with unresolved source/numerical issues stated separately. Do not sum
parallel lane estimates or promise a workflow speedup before comparable completed
milestones demonstrate one. The earlier
[conversion timings](../artifacts/ministral3-conversion-2026-11-13/velocity-observations-1.json)
are descriptive observations, not a measured effect of parallel delivery.

# Parallel engine and CLI delivery

Use one end-to-end milestone with three implementation lanes and one integrator.
The next milestone is Ministral's real packed Metal graph and token-ID CLI route.
Acceptance remains in [PLAN](../PLAN.md#validation-and-performance-policy); the
[graph interface inspection](../artifacts/ministral3-graph-planning-2026-11-13/integration-plan-1.json)
records the concrete integration points. Existing model order and rejected
candidate decisions stay in force.

## Work ownership and dependencies

| Owner | Exclusive work | Handoff |
| --- | --- | --- |
| Source lane | New child adapter and source/alias tests in `src/mlx/`; coordinate existing `qwen.rs` and `qwen/logical.rs` edits before dispatch | Strict nested geometry, physical-to-logical aliases, mixed-affine inventory and positional profile |
| Metal lane | Resident positional ownership, planning and graph integration in `src/metal/` | Shared immutable tables, query/key-specific transforms, independent projection widths and tied output |
| Reference lane | Independent tiny/full graph fixtures, checker and state schedules | Freeze expected arrays, gates and identity bindings before observing native output |
| Integrator | Shared declarations/API contracts, raw CLI, status records, heavy-check queue, review and publication | Integrate early, resolve shared-file changes, run qualification and publish |

Before dispatch, name the exact files each lane may edit and agree on the small
source-to-Metal interface. Allow at most four active agents including the
integrator. Reuse the existing clean worktree and Cargo cache; avoid duplicate
checkouts/build directories. Keep each lane's commit boundary subordinate to the
combined reviewed milestone.

Source, Metal and reference work can begin together. CLI wiring can use the
agreed interface while those lanes work. Integrate small compiling changes early
so incompatible APIs fail before full-model capture. First prove a tiny
distinct-width, mixed-affine, tied-head graph; then run the actual converted
model. Include a query-gain composition control: admitted real positions alone
do not expose a query/key transform mix-up.

## Keep the critical path short

- Run cheap targeted controls during implementation. Freeze the combined inputs,
  then run the applicable fmt, strict Clippy, CPU, Metal and session/GPU checks.
  Reuse prior checks only when all relevant inputs are verified unchanged.
- Use one queue for builds and memory-intensive numerical/model/quality runs.
  Keep performance cohorts isolated from competing jobs. Preserve the 6 GiB
  engine, 512 MiB setup/audit, 3 GiB compiler and 2 GiB disk-floor policies.
- Prepare bounded codec metadata/fixtures or observe remote CI while another
  lane uses the heavy queue. Do not import a numerical runtime or start another
  model just to fill an idle slot. Full quality/performance runs depend on the
  preceding graph and codec gates passing.
- Send Astra one frozen diff, invariants and combined check results. Follow up
  findings against that request. Reassign a finished lane's slot to Astra;
  overlap only independent next-task preparation with review.
- Commit and push the reviewed milestone once. Keep logs in existing evidence
  directories and link one concise status receipt; avoid separate review and
  publication cycles for each helper. Engine/CLI work requires no sharing picker.

## Measure delivery velocity

Record dispatch, first integrated check, source freeze, required-check completion,
review request/approval and pushed green CI timestamps in the milestone receipt.
Report total elapsed time, heavy-queue time, review wait and rework/repeated-check
reasons. Estimate remaining time from the dependency chain and observed durations,
not the sum of concurrently running lane estimates.

The [last milestone's descriptive timings](../artifacts/ministral3-conversion-2026-11-13/velocity-observations-1.json)
show 2,089.8 seconds for source acquisition, 95.0 seconds for conversion and
22.8 seconds for saved-byte auditing. Three source lanes, one review and verified
reuse of 4,641 unchanged Rust inputs were already used. The next graph milestone
can use the authenticated local child. These observations do not establish a
workflow speedup or engine-performance improvement.

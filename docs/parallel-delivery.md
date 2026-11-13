# Parallel engine and CLI delivery

The active milestone is the Qwen3.5-2B typed source and existing hybrid graph
route. Ministral's graph/CLI milestone is published as `8b6220e9`; its numerical
rejection remains explicit. Keep acceptance in
[PLAN](../PLAN.md#validation-and-performance-policy) and preserve model order,
existing source eligibility and earlier rejection decisions.

## Assign independent work

Use three implementation lanes and one integrator, with exclusive file ownership.
The [current dispatch receipt](../artifacts/qwen35-2b-source-planning-2026-11-13/parallel-work-plan-1.json)
records the assigned agents, paths, interface, queue limits and pending gates.

| Owner | Current work | Handoff |
| --- | --- | --- |
| Source lane | New `mlx::qwen35::two_b` plan and CPU/Metal stores; opt-in F32 small-vector loading and accounting | Exact source pins, 24-layer recipe, preserved F32 `A_log`, checked learned bytes; existing 4B/9B eligibility unchanged |
| Graph lane | `model::qwen35` source binding, Metal factory and ratio-one/state controls | Tied-head binding using existing numerical loops; no new kernel without evidence |
| Reference lane | New typed reader, bounded conversion/audit tools and independent tiny fixture/checker | Freeze geometry, values, expected arrays, gates and identity bindings before observing native output |
| Integrator | Shared declarations, CLI, source lineage and disk reservation, check queue, status, Astra and publication | First integrated compile, one frozen diff, one combined review and exact-head CI |

Agree on the small API before editing shared files. Hand off a concrete function
signature, ownership/accounting rule and acceptance command; identify any unknown
that blocks actual qualification. Mark a lane ready for integration only when its
files and generation commands are available. Do not wait for a lane's entire
full-model campaign before integrating its first useful change.

Reuse the clean worktree and Cargo cache. Each lane may run cheap static or
stdlib-only controls over its owned files. The integrator owns shared edits,
compilation and model/runtime execution. A completed lane frees its slot for
Astra or independent next-task preparation; at most four agents run at once.

## Follow the dependency chain

1. In parallel, implement typed source loading, graph binding and independent
   fixture/reference work. The integrator resolves source lineage and storage
   requirements and wires the agreed interface.
2. Run the first integrated compile as soon as those interfaces are present.
   Prove the tiny ratio-one graph, exact F32 small values, session recovery and
   memory accounting before spending time on full payloads or captures.
3. Authenticate source values and conversion semantics before admitting a real
   child. Reserve the complete campaign, including sources, child, captures,
   temporary workspace and disk floor, before downloading payloads.
4. Freeze the combined inputs. Run applicable fmt, strict Clippy, CPU, Metal and
   explicit session/GPU checks. Reuse results only with verified unchanged
   relevant source/dependency inputs. Record why a repeated check was necessary.
5. A failing numerical, state, codec or fit gate stops its dependent quality and
   performance cohorts. Preserve the failure and record a scoped diagnostic or
   rejection decision. Independent portable work can continue.
6. Send Astra the frozen diff, invariants and combined results in one request.
   Resolve findings there; commit/push the reviewed milestone once and verify
   exact-head CI. Do not create separate routine helper/publication reviews.

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

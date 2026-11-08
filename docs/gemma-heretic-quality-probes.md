# Heretic Gemma quality-only probes

G1.Cb execution, grading and combined Astra review are complete. The
[selected-context fidelity prerequisite](gemma-heretic-fidelity.md) is committed
as `0e2cbaf9`. The full numerical and quantized adoption decisions below reject.

The modified BF16 parent is separately pinned to
`p-e-w/gemma-4-E2B-it-heretic-ara` at
`c9a1d4c031981f14d86eeb0c7d87de7fafd34513`, with complete payload SHA256
`15605c16e479a8dc53928dd4b521a4ead9fe66222dc14d1ce392384e999a544b`.
Its test-only inspector accepts its exact config and the unchanged original
header. Only the known disabled `moe_intermediate_size:null` alias is removed
for shared graph validation. Ordinary official inspectors, payload pins,
codec/catalog identity and managed routes remain strict.

The resident BF16 delivery keeps a private expected-source pin in its transient
preparation object. It still authenticates the whole source before publishing
weights, retains bounded authenticated PLE row access, and uses the existing
resident projection and session ownership. No full PLE/FP32 copy is introduced.

The quantized route gains a test-only model constructor bounded to 1,025
positions. Public diagnostics remain capped at 512 with chunks at most 32.
The 6 GiB process policy includes a separately reserved 128 MiB diagnostic
allowance. Planned/admitted bytes, owned session bytes and observed lifetime
physical peaks are reported separately. These probes do not claim a new speedup
or qualify observability overhead.

Both sources reuse the frozen wrap/overflow/reset, 10,000-target loss,
100-task generation and fresh raw replay helpers. Wrap capture covers two
chunk schedules, eight selected complete distributions, all retained K/V at
four eviction milestones, stable allocations and exact admission/rejection.
Independent row-backed FP64/staged FP32 math uses absolute producer positions,
with no native session or KV-ring implementation. Every compared value retains
the original `1e-5 + 1e-4 * abs(reference)` limit; near-tied greedy differences
require investigation. Owned source/runtime receipts and full capture hashes
remain part of qualification.

The complete frozen quality protocol needs 1,025 loss positions and 567 task
positions. Mean loss increase must be at most 0.05 nats/token and task pass-rate
loss at most two percentage points relative to the unquantized reference.
Original, modified BF16 and quantized identities stay separate. The 100 tasks
are collected as raw token journals; generated code runs only in the existing
isolated grader. There is no prompt truncation or gate relaxation.

Fresh611 CPU/745 Metal tests,27 Python controls, formatting and strict
CPU/Metal all-target Clippy pass. Nine explicit GPU regressions, native52
three arm64 slices/46 exports and both Mac C smokes pass. The public512limit
and6GiB ceiling stay unchanged; iPhone runtime remains deferred.

Both owned independent reference collections complete with authenticated source
and runtime endpoints. Their measured physical peaks are1.177/1.187GB. Each
source compares48,246,784 selected-logit/retained-state observations through
1,025positions under both original profiles. All16selected greedy IDs match,
but both numerical gates fail:

| Source | FP64 failed observations / max gate ratio | Staged FP32 failed observations / max gate ratio |
| --- | --- | --- |
| Modified BF16 |629 /7.899 |0 /0.00137 |
| Quantized MLX |6 /2.011 |2 /1.860 |

A reference-only compatibility diagnostic finds60outward-disjoint intervals
and61values with no common FP32 result for the parent, across11,543,552unique
reference values. The quantized references have no such conflict. This diagnostic
never consults native values or changes either acceptance interval. Native
numerical rejection remains authoritative; previous512selected-context evidence
has its earlier scope.

The unchanged original-source quality baseline scores52/100tasks. Each source
uses10,000held-out loss targets and all100fixed tasks, without truncation:

| Source | Mean NLL / change from original | Task passes / loss from original | Quality metric gates |
| --- | --- | --- | --- |
| Original source |9.995776 /0 |52/100 /0pp | Baseline |
| Modified BF16 |9.711180 /−0.284596 |50/100 /2pp | Pass at task limit |
| Quantized MLX |9.228555 /−0.767222 |43/100 /9pp | Reject task quality |

Quantization also loses7percentage points against the modified BF16 parent,
exceeding the same2point limit despite lower mean loss. The task journals match
fresh raw replay for4,244/4,378decisions with zero mismatches. Both isolated
graders finish cleanly and verify stopped-VM restoration. Generated code never
runs on the host. The complete paired case outcomes and receipt hashes are in
[quality summary](../artifacts/gemma-heretic-quality-probes-2026-11-08/quality-summary.json).

**Reject full1,025-position numerical qualification for both sources; reject
this quantized checkpoint for adoption on task quality too.** Passing loss,
state or sampled-token checks does not waive either failure. No managed route,
codec or application identity changes.

The quantized task pilot exposed aggregate budget errors converted into an
untyped backend error. Its factory now shares construction with fault injection
and returns the absolute typed `MemoryBudget` preflight. Thresholds and
allocations remain the same; exact-boundary/zero/overflow controls pass.
Earlier build2 loss/wrap and parent-task receipts retain their original pins.
Their only subsequent Rust input delta is this cold mixed-generation factory
and its unit test; numerical operators, layouts and parent generation are
unchanged. Affected quantized generation and raw replay use final build4.
Astra approves this relevant-input reuse, the complete diff and negative
qualification decisions with no findings. Approval covers diagnostic machinery
and the budget fix; it grants no adoption or gate waiver.

Initial compile/fixture failures, Clippy module placement, attempt1 sandbox
memory-measurement failures and an overly strict repeated-grader telemetry
comparison remain in the [iteration record](../artifacts/gemma-heretic-quality-probes-2026-11-08/iteration-notes.json).
All original numerical and quality gates remain unchanged.

[Contract](../artifacts/gemma-heretic-quality-probes-2026-11-08/acceptance-contract.json),
[verification](../artifacts/gemma-heretic-quality-probes-2026-11-08/verification.json),
[review/commit](../artifacts/gemma-heretic-quality-probes-2026-11-08/publication-checks.json).

# Same-model session replacement in the native worker

`replaceWithEmptySession` is an internal worker transaction. A request and
selection UUID bind the exact current model and session IDs, capacity, chunk,
whole session budget and coexistence reserves. It neither reloads weights nor
replays old tokens. This task adds no Settings Apply action or preference
persistence.

Candidate preparation and typed native report/context validation run on the
owning thread before acceptance. A dedicated queue path preserves ordinary
snapshot fields on rejected, cancelled or stale attempts. The final condition
lock rechecks identity, cancellation, closing, queued cleanup and quarantine,
then swaps the prepared session, settings and context together. Model, tokenizer
and loaded upper bound remain retained. Success starts an empty native cache
and clears cache/generation-derived fields. Transcript and draft are owned by
the caller and are untouched by this worker path.

The [native creator](native-empty-session-candidate.md) rechecks individual and
aggregate storage after allocation. The typed reader independently checks the
[transition ledger](native-session-transition-report.md), flags, modes and
request identity. CPU constructor accounting includes its shared model table;
Metal constructor accounting excludes that host table while aggregate
accounting includes it once. Both metadata/driver and bridge allowances remain
in coexistence accounting. Text/application reserves are caller declarations;
logical Swift buffer extents and historical context admission are not measured
allocator capacities or OS memory peaks. The last-load runtime plan remains
historical evidence, not a new-session allocation report.

Retirement occurs after acceptance. A retirement error reports committed plus
restart required, never rollback. A BUSY/WRONG_THREAD result retains at most one
known-live owner for a shutdown release attempt; ambiguous PANIC destruction is
never retried. Diagnostic PANIC simulation reports uncertainty only after a
real successful release. Cancellation after acceptance may clear the transient
result but cannot undo the new owner. A future Friendly Apply consumer needs
durable committed configuration facts; a vanished result does not mean rollback.

## Verification

Independent saved review approved the bounded worker evidence. All 14 fresh V2
cold/CPU/app commands passed: strict Swift 6 typed-consumer
checking; three standalone control compiles using the existing Swift 5 worker
concurrency baseline; the CPU worker run; and builds, architecture checks and
load-command checks for Mac, iOS and iOS Simulator apps. The apps were not
executed. Exact unchanged Rust/header identity inherits 469 default tests
(20 ignored), 577 Metal tests (186 ignored) and 32 exports. Native slices were
not rebuilt for this Swift task.

Both separately approved Mac jobs passed against the newly selected
[32-export bridge](apple-session-transition-package.md): the worker transition
controls and the unchanged preview/generation regression. CPU and Metal each
passed 37 typed/domain/accounting controls, 15 deterministic lifecycle/staleness
barriers, two retirement cases, empty adoption and subsequent usability, and a
postacceptance cancellation witness. The legacy Metal regression passed its
21 barriers and tiny generation witness. All 16 successful command groups
closed with no endpoint errors; raw logs and actual products are retained.

Seven raw records per backend retain the used current state, budget rejection,
smaller empty adoption, late-cancel pair and retirement outcomes. On Metal,
the current witness has six history tokens, five committed and one pending;
the budget-rejected snapshot preserves that prefix. Smaller adoption changes
session 2/capacity 32 to session 4/capacity 3 on model 1, with chunk 2 and zero
history, committed and pending tokens. Loaded upper remains 32. Its aggregate
is 136,538,396 bytes: 28,364 shared + 17,472 current + 5,392 planned candidate +
136,487,168 reserves. The actual context admission changes from 67,170,700 to
67,158,620 bytes; context admission and aggregate/current-owned accounting have
different scopes. The corresponding CPU aggregate is 136,590,052 bytes.

The Metal late-cancel pair retains model 40/session 42/capacity 16 and admission
67,165,004 after the transient result disappears. Retirement records retain
committed adoption with restart required and status 4/6. Full ordinary snapshot
preservation, stale-error suppression, continuation and ownership cleanup are
completed assertions. The seven records are not full logits/weights exports or
independent reconstructions of every internal assertion. No performance or
complete process-peak claim follows from sparse process observations.

## Preserved first attempt

V1's 14 cold/CPU/app checks passed, but its first Metal control failed at an
acceptance-barrier deadline; the second regression was not run. That process
group closed without endpoint errors. Stdout was empty and stderr contained
only the fatal deadline message, so the actual decoder rejection was not
captured. Static producer/consumer inspection identified the decoder applying
the CPU constructor equation to Metal, incorrectly adding a 4,480-byte host
model table. V2 corrects that exact equation without changing the native
report, budgets, fit inequalities or aggregate sum. Four additional controls
check both backends' table inclusion/omission. The test now retains a bounded
actual snapshot/message on unexpected rejection or deadline. V1 remains failed;
the current 14-command phase was a fresh run after the correction.

This qualifies the bounded worker boundary and linked tiny CPU/Mac controls,
not rendered UI, physical-phone execution, a real text-profile adoption,
Settings Apply, persistence, larger context eligibility or whole-device fit.
The next UI task must reconcile durable accepted settings and preserve the
conversation/draft while explicitly announcing cache reset.

[Machine-readable evidence](../benchmarks/results/2026-10-30-native-worker-session-transition-v1.json)
binds the four source files, receipts, raw observations, preserved failure,
compiled products and independent saved review.

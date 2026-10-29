# Typed Swift planning consumers

`NativeContextLimits` and `NativeCapacityPlan` copy and validate the native JSON
reports on their creating OS thread, then return immutable Sendable values.
Their synchronous wrappers always release the existing bounded JSON handle.
They do not create a worker, change UI/settings, recreate a session or establish
artifact identity. Project membership follows with worker integration.

The limits consumer checks schema/report kind, expected ordinary/audited-Smol
profile, accepted/rejected/unknown states, nullable enforced bounds, backend
build availability and required unevaluated fields. It validates the reported
minimum of enforced bounds without reproducing model-shape tables. An
unavailable or rejected report has no `planningUpper`. Metadata acceptance is
not payload authentication, device fit or load permission.

The capacity consumer takes explicit context, upper, chunk, offset, modes and
separate budgets. It checks request arithmetic and every echoed value, shared
resource accounting, range/limits, point fit flags and adjacent witnesses.
Full, partial, empty and single-point intervals remain distinct. The selected
point must fit, but its fit does not require the original upper to fit.
`selectedFitsNumericalBudgets` additionally requires both loading flags; it
cannot grant actual loading or whole-app fit.

`resources` describes the requested upper, while a lower selected point has its
own footprint. The consumer does not relabel upper-capacity KV as selected KV.
The capacity API has no audited-Smol selector, so a separate limits report is
not cross-profile authentication. Equal directory strings or equal JSON cannot
prove equal payloads; actual loading still revalidates. Asynchronous request
identity/stale-result handling is the next worker task.

The old `NativeResourcePlan.decode` signature, size handling, fixed expectations
and inspector remain unchanged. Its field/accounting guard is extracted into
an internal validator. The old call passes exactly its former context, offset,
mode, budget and attention expectations; new request restrictions are applied
only by the capacity consumer.

Verification passes 75 new synthetic rejection controls and actual C-producer
checks against default and Metal libraries. The latter cover full fit, selected
17/adjacent18 under a lower budget, zero-budget no-fit, false loading flags
despite session fit, old-resource equality and immutable metadata snapshots after
a temporary declaration change. Default mode keeps unavailable planning explicit.
No model forward or token generation is involved.

The unchanged old resource decoder/worker regression passes 25 wire mutations
in each mode. Metal additionally checks actual tiny load/session/suspend/unload,
three queued cleanup paths and actual source revalidation after a plan. It
loads tiny owners but generates no tokens. These are separate legacy regression
results, not extra cases in the new 75-control count.

New consumers/controls compile under Swift 6 with complete strict concurrency
and warnings as errors; the unchanged legacy regression retains its qualified
Swift 5 strict mode. Nine build/compile/CPU jobs and two explicit Mac jobs pass.
The existing Metal archive and Rust checks (default462/Metal569,20/184 ignored)
are reused after exact 333-input engine/header identity verification; only the
default archive is rebuilt. Both actual library variants and all five Swift
executables are retained. Rust tests were not rerun for Swift-only changes.

Astra verifies current Swift, inheritance, products, aliases, runtime, raw outputs
and process closure. [Exact evidence](../benchmarks/results/2026-10-30-swift-planning-consumers-v1.json)
retains compiled assertion scope; full wire payloads are not retained for an
additional independent saved-byte decoder. No worker/UI/persistence or project
integration, source identity, enlarged-context, real-model, phone, whole-app or
performance qualification is claimed.

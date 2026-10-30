# Worker planning preview

`NativeCommand.preview` uses the existing dedicated worker thread and single
queue. It is accepted only while idle for the current selection. Preview
selection and request UUIDs correlate results; neither UUIDs nor directory
strings authenticate model bytes. The result is separate from the last actual
load's `runtimePlan`.

Preview enqueue bypasses the ordinary request-clearing path. Its early worker
branch also bypasses `Engine.perform`, context refresh and the ordinary state
publisher. A completed preview changes only preview metadata, busy/phase and
polling revision. It restores the previous phase and preserves the active
model/session, prefix, pending token, chat result, score, timings, generation
status, context facts, admission and load plan. Preview errors belong to the
preview result rather than replacing an earlier generation/load error.

Selection changes, accepted ordinary commands, cancellation and lifecycle
cleanup immediately invalidate preview results and their active request key.
The worker checks the complete request and selection before capacity inspection
and before publishing success or error. Synchronous inspection cannot itself be
interrupted; invalidation suppresses subsequent work and publication. Native
JSON handles are copied and released on their creating worker thread.

The requested absolute context is bounded by the inspected ordinary factory
upper and an optional caller ceiling. Checked subtraction of offset gives the
remaining relative capacity. The requested upper is clamped to that remainder;
chunk, modes and separate budgets are preserved or rejected. `resources`
continues to describe the requested upper, while the selected lower point has
its own footprint. A selected fit does not account for coexistence with an
already loaded model or grant whole-app/device admission.

Audited-Smol results remain metadata only: the capacity API has no audited
profile selector. Unsupported or unavailable ordinary metadata similarly does
not authorize capacity inspection. Actual loading still reopens and validates
the source, device and session; a preview cannot bypass those checks. One
bounded result and its separate application allowance are retained, with no
preview history or automatic retry queue.

The Xcode project includes the C1 limits/capacity consumers and the new preview
types. This source membership is not binary readiness. The currently selected
XCFramework is not assumed to export the new APIs: app linkage still requires
a coherent current 30-export bridge and a separate app build. This task adds
no UI, settings application or persistence.

The first verification attempt stopped at a missing closing brace in a test
switch after three successful type/SDK jobs. No runtime test ran. V2 corrected
only that brace. Its nine compile/type/SDK jobs passed, then the first CPU
preview runtime failed because the test selected a numerical-array fixture
instead of a loadable model directory. All failed receipts and their closed
process groups remain retained. V3 corrected fixture routing and included the
native status/message in the load-failure diagnostic. It also routed the
unchanged legacy Metal worker test to its original Qwen fixture with the
existing token-47 oracle; the new Metal preview test retains its separate
Qwen3 fixture. Worker/core/project semantics and control counts did not change.

The fresh V3 verification passed all 12 compile/CPU jobs and all three bounded
Mac Metal jobs, with 15 closed process groups and no endpoint errors. New
planning types passed Swift 6 strict checking. Worker integration and both
platform variants compiled under the existing Swift 5 strict baseline;
iOS worker source type checking and project membership checks also passed.
The immutable default/Metal libraries were reused after exact engine/header
identity checks; this Swift task did not rerun unchanged Cargo suites.

Default and Metal preview tests respectively passed 16 and 21 barrier cases.
Compiled assertions also checked request derivation, metadata-only outcomes,
stale success/error suppression, active-engine and generation-state
preservation, and tiny generation witnesses. These assertions are not reported
as additional barrier cases. Separate unchanged resource-plan tests rejected
25 malformed reports per mode; the Metal variant also passed three cleanup
cases and source revalidation. Separate legacy worker tests passed their
original token-47, lifecycle, replay, cancellation and release assertions in
CPU and Metal modes. Timing fields are retained as ordering evidence, not a
performance result.

[Evidence](../benchmarks/results/2026-10-30-worker-planning-preview-v1.json)
binds source, products, successful checks, both failed attempts and independent
saved review. This qualifies the bounded worker preview integration. It does
not qualify app linkage, a real model, a phone, a context range or performance.
The coherent XCFramework and actual app-build gate remains separate.

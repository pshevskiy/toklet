# Apply settings to the current chat

The source, controller and app-build checks passed: 22 cold/CPU/app commands
and four separately approved Mac jobs. An independent saved audit authenticated
the source, products, raw observations and closed process groups. Rendered UI
and physical-phone checks remain separate.

The Settings planning section offers one explicit Apply action for the selected
estimate. It retains the conversation, unsent draft and reply preference while
replacing only the loaded model's native session with an empty cache. The next
Send re-encodes the actual conversation and checks the rendered prompt plus
requested reply against the new shared window. These settings are not persisted
for the next launch.

Apply binds the actual worker, model/session instances, conversation and model
key, exact preview request/result UUID, displayed selected capacity, chunk and
whole memory budget. It requires a fitting selected point and rejects estimates
above the loaded model limit instead of clamping or reloading. Loading-fit flags
are separate from replacement of an already resident model. A reply preference
which does not fit is rejected without changing it. Native model/session
identity, current limits and D1/D2 actual individual/coexistence admission remain
final; a directory or preview is not payload authentication.

The worker's new Friendly memory policy keeps the accepted profile process
allowance as the total coexistence cap, reserves the separate text allowance and
272 MiB for application work including 16 MiB staging, and reads fresh process
headroom after the actual model-bound transition report. Incremental headroom
must cover candidate-owned memory, its 64 MiB metadata/driver allowance, one
bridge, the JSON report allowance and 16 MiB staging. This is an admission policy,
not measured heap usage or free system RAM. Mac headroom remains the configured
min(RAM/2, 6 GiB) process allowance less TASK_VM_INFO footprint; iOS uses its
existing process-available-memory policy. Diagnostic replacement requests keep
their previous behavior and do not query Friendly headroom.

Durable accepted configuration is reconciled before stale preview selection is
cleared, even when Settings is hidden or late cancellation erased the transient
result. New field edits clear the previous Applied notice while retaining the
accepted session configuration. Selection invalidation does not cancel unrelated
generation. A committed cleanup error retains the new configuration and requires
restart; Send is disabled while quarantined. Current capacity/chunk/budget come
from the durable descriptor; the initial KV/load report is labelled historical.

## Verification

All 22 fresh cold/CPU/app-link commands passed: strict Swift 6 typed checks,
eight standalone control builds under the existing Swift 5 worker concurrency
baseline, four CPU runs, and nine Mac/iOS/iOS Simulator app build and platform
inspections. The project has 32 iOS entries and the Mac app 34 source files.
Apps were not launched. Exact identity of 338 unchanged native/header inputs
inherits the prior 469 default tests, 577 Metal tests and 32 exports; there was
no Cargo or native-slice rebuild. Ten actual products were retained for the
separate Mac phase. All 26 command groups closed without endpoint errors.

CPU controls passed 57 existing planning and 28 new Apply cases, ten headroom
records, and the unchanged 37 transition plus 11 configuration controls with
15 lifecycle/staleness barriers, two retirement cases and twelve raw records.
The headroom controls rejected zero and one byte below the observed requirement
84,047,968, then accepted the exact requirement. They also checked diagnostic
bypass, aggregate rejection before querying, three invalid/query-error cases,
two stale-query barriers and old-state preservation. These controlled values do
not measure available device memory. Raw records and completed executable
assertions have distinct scopes; no full binary state reconstruction is claimed.

The four Mac jobs passed the same ten headroom records and transaction/configuration
controls, the unchanged 21-barrier tiny generation regression, and a loaded
Friendly controller using the pinned cached Qwen3 artifact. The tiny Metal
headroom boundary was 84,049,664 bytes; exact passed and one byte under failed.
These are admission controls, not memory-performance measurements.

The loaded run used one ordinary model load, four previews and three actual
empty-cache replacements. Model instance 1 remained loaded; new session instances
3, 4 and 5 each used capacity 256, chunk 16 and a whole session budget of
1,610,612,736 bytes. Each acceptance had a distinct request UUID and fresh process
headroom check. Required incremental headroom was 125,957,028 bytes; observed
policy headroom was 5,585,959,712, 5,627,067,144 and 5,627,050,784 bytes. The required
sum was 41,919,140 candidate-owned bytes + 67,108,864 metadata allowance + 20,480
bridge + 131,328 JSON allowance + 16,777,216 staging. This does not qualify
whole-device memory fit.

The first replacement survived cancellation and Settings dismissal before UI
polling: its durable configuration reconciled despite the missing transient
result. Each of the three accepted notices was followed by a different planning
field edit; the old notice cleared without losing the accepted configuration.
Workspace and file hashes remained identical, and persistence calls stayed at
two throughout these Applies and edits. Reply 128, conversation and unsent draft
were preserved. A separate preview of capacity 64 rejected that reply preference
without changing it or the edited draft.

With the final capacity still 256, the next Send used 140 fixed repetitions of
`hello ` and reply 128. The actual worker rejected the rendered request with
“This request needs 281 tokens including its reply; the current shared window is
256.” The command did not start generation or emit tokens, its native history
remained empty, messages were restored and a newer draft entered while the command
was queued survived. This later Send/draft-edit scenario has normal persistence
activity; the two-write equality above applies to the earlier Apply/edit sequence.
No successful real-model generation was performed. The separate tiny regression
did exercise generation.

The private source error-oracle and stale-notice findings, final blank-line
correction and owner V1 parser-order finding were corrected before execution.
They are preserved as static review history, not failed runtime/model outcomes
or changed admission policies. The loaded controller retained 17 raw records;
structured IDs, accounting, hashes and the error are observed data, while the
remaining full-state rollback claims also rely on the pinned executable's
completed assertions. No full binary-state reconstruction is claimed.

The parent’s current UI availability check reported that the Mac was locked;
no app was launched or controlled in that check. This is an outstanding visual
observation, not a failed controller or model run.

Rendered Mac UI and physical-phone observation remain separate from source,
controller and app-build evidence. No settings persistence, artifact-bound
preference restoration, expanded loaded context, whole-device fit, quality or
performance qualification follows. These checks do not authorize new application
context limits or a larger loaded model.

[Machine-readable evidence](../benchmarks/results/2026-10-30-friendly-chat-apply-v1.json)
binds the current 12 source files, receipts, raw observations, retained products
and independent saved review. The existing [durable session configuration](native-session-configuration.md)
and [worker transition](native-worker-session-transition.md) contracts remain separate.

# Context facts and shared-window admission in the app

Chat and Expert now consume the fixed native context and request records from
[L3.1](context-admission.md). This connects the existing runtime contract to the
app; it does not raise the current engine, codec, text or output limits.

`NativeContext.swift` validates ABI/size, reserved and known-field bits, explicit
unknown declarations, backend/status values, nonzero instance IDs and checked
position/history arithmetic. The retained model value contains configuration
facts only. A separate live-session value contains session identity, committed
KV positions, sampled history and pending tokens. Suspend drops live facts while
retaining the model/configuration; explicit Chat Send creates a fresh session.
Expert Append/Continue still require explicit replay. Cleanup immediately clears
published facts, and a finishing command cannot restore them after queued unload
or close. Same-folder model reload is distinguished by a new model instance.

Generation queries fresh facts on the native owner thread, calls the same native
preflight used by both modes, verifies its identity and arithmetic, then invokes
generation, which validates again. Preflight is a snapshot, not a reservation.
Chat counts the entire rendered conversation, including retained turns and control
tokens, plus the requested reply. Reset starts at zero; earlier native history is
not counted again. Expert Append/Continue include pending sampled history.
Over-budget rejection preserves the previous token prefix and restores the draft
without overwriting newer edits or an intentionally cleared draft.

The Chat summary shows the model declaration separately from selected runtime
positions and explains partial support. Settings also explains the current
codec/event output cap. Expert shows loaded/global/session limits, offset,
instance IDs, active remaining capacity and the last request budget. Retained
configuration after suspend is not presented as a live-session capacity claim.
Unknown declaration remains unknown. Qualification and whole-app/device fit are
not granted by these records.

Mode policy preserves current constraints: Chat input512 rendered tokens,
output256 tokens, conversation/decoded text64KiB and draft16KiB; Expert diagnostic
output128 tokens and token-ID text256KiB. Limits are intersected with selected
capacity and any known separate model input/output caps. Saved unsupported reply
preferences remain visible and preserved offline. Reply choices do not shrink
because of an earlier conversation's native history; actual rendered admission
remains authoritative for every Send.

## Verification scope

Strict Swift6 complete-concurrency/warnings-as-errors checks pass for both Mac
modes and both iOS SDK source sets. Portable policy tests reject27 invalid context
records and12 invalid request records, and replay five frozen native arithmetic
cases. The legacy CPU witness completes exact32 positions, rejects maximum+1,
and checks suspend/replay/cancellation, new identities, unchanged prefixes and
two queued-cleanup publication races. Existing mobile-interface, scoring and
pinned Smol real-model recovery regressions also pass.

The real Mac Qwen3 workload requests385 rendered input plus127 output tokens,
and rejects385 plus128 with selected capacity512 in both modes. It cancels after
one emitted token; this checks admission and state preservation, not completed512
execution or full model-window quality. The model declaration is40960. All three
draft restoration/edit/clear cases pass; offline unsupported preferences remain
preserved. Recorded process peaks are1,001,964,984 physical/1,027,850,240 resident
bytes for this functional workload, without a performance or device-fit claim.
The actual context-summary/error component is rendered and visually inspected;
this is not a complete interactive or phone visual audit.

A fresh three-slice native package exposes all28 C entrypoints, including context
query/preflight. Both iOS app builds pass: device unsigned, simulator ad-hoc signed
and verified. The existing generated library symlink now selects this package,
with its previous target/build identity retained. No phone is contacted. All32
new/failed/nested check groups are absent, and package ownership was independently
checked. The first frozen attempt failed because the controller resolved the
Swift driver symlink to its frontend name; the tool-only retry preserves the
driver invocation. Failure logs remain. The publisher verifies endpoint-only
compiler alias/target/hash identity; it does not claim continuous alias checking.

See the [public receipt](../benchmarks/results/2026-10-27-context-app-integration-v2.json).
Progressive numerical window qualification, larger codec/event/text limits and
physical iPhone qualification remain separate tasks in the
[active plan](long-reply-engine-plan.md).

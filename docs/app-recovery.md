# Recoverable Chat and model selection

23 October 2026. The iPhone button audit exposed shared controller problems;
these fixes also apply to the optional Mac window. This report covers controller
correctness, separately from the ongoing physical UI audit.

A turn rejected before numerical generation now restores its editable prompt and
removes the tentative user/assistant pair. Forbidden control literals and context
rejection therefore do not poison later turns. Draft edit identity is tracked
outside persisted history, including intentional edits back to empty. A new edit
is never overwritten by rollback. Native validation and context limits remain
unchanged. Existing saved rejected turns are not silently deleted.

Lifecycle cancellation before command dequeue is explicitly marked unstarted,
keeps cancellation reason 3, and uses the same recovery. Once generation begins,
ordinary cancellation preserves its terminal/partial reply. Empty output alone
never decides whether a turn is rolled back.

New chat uses the loaded text model even when another model’s history is selected.
Deleting the last conversation retains the loaded-model association in its empty
replacement. Opening an unrelated folder checks its bounded pinned configuration
before changing history or unloading a working model; full tensor/tokenizer and
memory validation still runs at load time. A stale worker/UI mismatch refuses the
selection and asks for a retry. Delayed downloaded-model verification is scoped
to the requesting conversation, so New/select/delete during verification cannot
redirect the later Open completion.

## Verification

The [receipt](../benchmarks/results/2026-10-23-app-recovery-v1.json) records the
initial failing real-model reproduction, exact source copies, final strict Swift
checks and both iOS SDK app builds. The first baseline compile was blocked by the
host sandbox’s SwiftUI macro service. The second compiled cleanly and failed the
intended rejected-control-literal recovery assertion. An intermediate combined
runner omitted WorkerChecks.swift and failed compilation; its corrected runner
is retained alongside it.

The final local SmolLM2 check covers rejected literal → corrected successful
send, context rejection, newer nonempty and intentionally empty drafts, delete
last → successful Send without reloading, New chat after other-model history,
unrelated folder preservation, and admitted-config/stale-worker rejection.
Six deterministic dequeue-gated checks cover suspend/unload/close, each with an
untouched draft and an intentionally cleared newer draft; all permit a later
corrected real reply. A separate cancellation after at least three emitted IDs
preserves a partial generated reply. Existing Chat, delayed Hub verification and
Mac close/reopen/quit regression suites pass. The mode-switch fixture explicitly
admits the two supported outcomes: rolled-back unstarted turn, or terminal turn
whose generation started; its memory-warning check uses a completed real turn.

Astra reviewed the implementation and regression changes. These are functional
checks; no quality, performance, new architecture or completed physical UI audit
is claimed by this controller report. The installed phone initially had an older
Smol-only app. The user requested every function/button be checked; physical
verification of the current replacement app remains part of that ongoing audit.

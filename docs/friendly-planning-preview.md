# Friendly Settings planning preview

Settings now contains a nonpersistent **Preview only** draft for the requested
window, chunk size and engine memory budget. Each prefilled input retains a
visible label. Preview uses the actual loaded conversation model and its
ordinary profile; it does not select a different model, apply settings, resize
a session, load a second model, or save the planning draft. Chat's existing
512-position window, chunk 32 and loading budgets remain unchanged.

The display separates the model's declared window, the ordinary engine planning
limit, the requested upper, the effective search upper and the selected estimate.
The effective upper follows the existing factory policy, with offset zero and
no extra 512-position ceiling. This allows capacity discovery without presenting
the hypothetical range as the active Chat range. The selected lower point has
its own memory footprint; the requested-upper KV estimate is not relabeled as
that lower footprint. Current Chat capacity and fixed KV accounting are shown
separately. Loading-fit flags, detailed budget accounting and text/preview
allowances are in disclosures.

The whole engine budget is converted once through the existing numerical-budget
helper, reserving the bridge allowance. Text and preview application allowances
remain separate. Estimates do not include app/tokenizer memory or coexistence
with the currently loaded model. Opening a model still reopens its files and
checks actual loading and session memory; neither a preview nor its request ID
authenticates payload bytes or changes admission.

A target is promoted only from an accepted load's exact directory and settings
when that load completes with a nonzero model instance. It is scoped to the
owning worker, model instance, conversation UUID and model key. A valid local
URL need not have a retained security-scoped URL; the accepted directory is
stored separately. Existing security access remains tied to the loaded worker's
lifetime and is not released on Settings dismissal or draft invalidation.

Draft edits, Settings dismissal, conversation/model changes, ordinary actions,
worker replacement and inactive/expert transitions invalidate request and result.
Invalidation uses `selectPreview(nil)` rather than cancelling ordinary engine
work. The complete request and target are checked again when observing the
actual worker result. An absent result while a request is pending is distinct
from an invalidated completed result. Previous active model/session, chat,
generation metrics and load report remain separate from preview metadata.

The 57 authored synthetic controls cover parsing and byte-bounded text retention,
request derivation, accepted-load promotion, stale successes and errors, target
replacement, actual unloaded-store wiring, and unchanged workspace/persistence.
A separate explicit Mac integration harness opens the cached pinned real Qwen3
model through normal Chat admission, flushes pending draft autosave, then
previews 512 and 256. It captures the actual worker result and checks unchanged
model/session IDs, active 512/chunk-32 settings, workspace/file bytes and
persistence-call counts before hiding the preview and closing the worker. This
integration performs no generation. The unchanged C2 tiny Metal regression
retains its separate generation witness against the current C3 Mac archive.

The fresh V3 verification passed all 13 compile/CPU/app-link commands and both
separately approved Mac jobs, with 15 closed process groups and no endpoint
errors. The 57 synthetic controls passed. Mac, iOS simulator and iOS device apps
compiled and linked against the selected coherent 30-export C3 bundle; their
arm64 platform/load metadata and bundled Tiny resources passed inspection.
Shared app/worker code used the existing Swift 5 complete-concurrency and
warnings-as-errors baseline. Unchanged Rust/header checks and the retained
libraries were reused after exact identity checks; no Rust suite was repeated.
No app was launched or explicitly signed by this verification. An implicit
linker ad-hoc signature is not excluded.

The real cached Qwen3 integration completed both 512- and 256-position previews
through ordinary admission, with matching requested/effective/selected values.
Model/session IDs stayed 1/2, active capacity stayed 512 and chunk stayed 32.
The persisted workspace and unsent draft bytes stayed unchanged; the initial
save established two persistence calls, and neither preview nor dismissal
added a call. The worker closed cleanly. The separate unchanged tiny Metal
regression passed all 21 barrier cases and its existing generation witness
against the current C3 Mac archive. The real loaded integration did not generate
text. This verifies the store/worker wiring, not rendered UI behavior.

Two failed attempts remain evidence. V1 stopped in its first compile when the
SwiftUI macro server could not create its sandbox; no test ran. The unchanged
retry used the app-build execution context already used by C3. V2 passed three
compiles, then its synthetic control incorrectly compared polling revision as
part of unchanged engine state. The test-only V5 correction retains all fields
except revision in that comparison, requires both preview fields to remain nil,
and rejects an injected functional-field mutation. Fresh raw before/after
snapshots showed revision 0 to 3 and otherwise identical reflected state. All
57 controls and the production/view/project/loaded-harness bytes were preserved
apart from that test oracle correction; all five failed-attempt process groups
closed without endpoint errors.

[Evidence](../benchmarks/results/2026-10-30-friendly-planning-preview-v1.json)
binds all seven source files, both failed attempts, successful products and
checks, actual workspace witnesses and independent saved review. Sparse direct
process memory observations are retained without a whole-device or compiler-tree
peak claim; no elapsed-time performance result is reported.

App compilation and loaded-store integration remain distinct from rendered UI
or physical-phone checks. A later Mac UI inventory attempt reported the Mac was
locked; no app was controlled or launched, so rendered Mac inspection remains
pending physical access. The pinned parent transcription records that limitation;
it is not counted as a passing check. There is no Apply action, planning-draft persistence,
new usable Chat context range, text-budget expansion, whole-device-fit or
performance qualification. Transactional session replacement, model-bound
transition accounting and any later reconciliation of Chat text limits remain
separate tasks.

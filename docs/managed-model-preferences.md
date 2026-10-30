# Managed model settings

Friendly Chat can save context capacity, prefill chunk size, whole-session
budget and reply limit for the authenticated managed Qwen3 profile. Choose the
model through Models → Use; after the audited model and tokenizer open, Toklet
checks saved settings against the current loaded limit and fresh memory facts.
Startup restores conversation history without automatically loading a model.
Other managed profiles, external folders and Expert remain separate persistence
work; this change does not grant arbitrary Hugging Face model support.

Each local record is at most 4 KiB. Its key hashes the preference schema and
accepted artifact/backend/load/text/planning namespace. Paths, process/session
IDs and measured headroom are excluded. The whole-session budget includes the
bridge once. The record contains user choices, never a stored permission to
allocate memory. Unknown, corrupt or oversized records remain untouched.

An explicit accepted Apply or valid reply edit can save settings. Ordinary
loading, preview, automatic restore and fallback do not rewrite the record.
Restoration uses the existing same-model empty-session transition and current
coexistence, text and application allowances; both the planned and actual
candidate allocations must pass fresh admission. Send stays unavailable while
restoration is pending. Rejection keeps the current admitted session and saved
record, with a nonmodal explanation in Settings.

The reply setter invalidates a pending transition before reading actual worker
state. A global edit generation prevents an older restore from replacing newer
choices. A native commit remains committed after late cancellation, but saved
settings and restored replies wait until old-session cleanup finishes. Cleanup
quarantine requires a restart and leaves the prior record intact. Failed file
writes report an error while retaining the actual accepted native settings.
Same-snapshot completion handling prevents a busy-to-idle race from dropping a
deferred save. Newer planning drafts remain separate from accepted values.

Reply limits must satisfy the existing allowlist and current Chat output limit.
Reply 32 with capacity 32 is rejected; reply 32 with capacity 33 is only a mode
check. Send separately tokenizes the actual rendered conversation and requires
prompt plus reply to fit. Its rejection guidance now includes increasing context
and explains that formatting contributes tokens. No template length is assumed.

The local store uses atomic replacement and the existing iOS file-protection
policy. This is not a power-loss durability or hostile-filesystem guarantee.
No migration, pruning, cloud storage or account operation is introduced.

All 23 build/CPU jobs and both Mac jobs pass in the
[verification record](../benchmarks/results/2026-10-30-managed-model-preferences-v1.json):
60 preference controls, 28 Apply, 57 planning and 37 routing controls, current
worker/configuration regressions, tiny CPU preview and all three artifact-local
app builds. The tiny Metal preview/generation regression passes 21 barriers.

One cached audited Qwen3 load restores capacity 256/chunk 16/reply 64 without
rewriting its record. Precommit reply editing cancels the candidate and saves
the original configuration after cleanup. Postcommit edits retain the actual
128/chunk 8 session and newer draft/reply, with no write while cleanup is busy.
At capacity 33/chunk 1/reply 32, the actual minimal rendered request needs 45
tokens: it rejects before generation, preserves the empty native prefix and
keeps the newer draft. That count is observed, not hardcoded in the test.

The final injected cleanup failure commits capacity 64/chunk 8, requires restart
and leaves the healthy capacity-33 preference record intact. Five prepared
candidates yield four commits, including this unsavable one, and five explicit
preference writes; automatic restore writes zero. Owning close succeeds. The
controller generates no real-model tokens and uses isolated test directories.
Astra independently authenticates all 25 closed process groups, current source,
retained products and output hashes, including all ten preference-record hashes.

Synthetic terminal-state and failed-save controls do not qualify an actual
low-headroom restore or process restart. Rendered UI, iPhone runtime, successful
real-model generation after resizing, broader model persistence, quality and
performance remain separate work. Native Rust and the selected 35-export
package are unchanged.

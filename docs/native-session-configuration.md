# Durable accepted-session configuration

`NativeSnapshot.sessionConfiguration` records one accepted session’s model and
session instance IDs, capacity, chunk, whole session budget and optional
replacement-request UUID. It is configuration metadata, not live token-history
authority, artifact authentication or permission to ignore `restartRequired`.
Numerical operations still need fresh context and ordinary worker admission.

Ordinary load and create/replay populate a descriptor with no accepted UUID
from validated native context and exact settings. Same-session generation,
reset, score and preview preserve it, including while queued work suppresses
mutable `liveContext`. The
[empty-session replacement](native-worker-session-transition.md) stages the new
descriptor before acceptance and swaps it with the owner, settings and context
under the existing condition lock. Late cancellation can clear the transient
transition result without erasing the committed configuration. Retained
`Engine.settings` remain a separate resume policy.

Load/unload/suspend/interrupt/close clear visible configuration immediately.
Actual release clears Engine configuration before a potentially failing native
call; missing or invalid native context also clears it. Replayed sessions get a
new session ID and no accepted UUID while keeping resumed capacity, chunk and
budget. Committed cleanup failure keeps the candidate descriptor alongside
`restartRequired` until shutdown; this cannot authorize further work.

The descriptor checks positive distinct model/session IDs, capacity and loaded
bounds, offset zero, a positive chunk no greater than capacity, whole budget
greater than the bridge allowance and no greater than 6 GiB, and native context
admission within that budget. It does not narrow existing constructors: an
accepted ordinary CPU chunk of 2,048 remains representable. Preview and
replacement-request bounds remain unchanged. No native API, kernel, project
membership, UI Apply or persistence code changes in this task.

## Verification

Independent saved review approved the bounded evidence. All 14 fresh cold/CPU
commands passed: strict Swift 6 typed-consumer checking; three standalone
control compiles using the existing Swift 5 worker concurrency baseline; the
CPU run; and Mac, iOS and iOS Simulator app builds, architecture checks and
load-command checks. Apps were not launched. Exact unchanged Rust/header
identity inherits 469 default tests (20 ignored), 577 Metal tests (186 ignored)
and 32 exports; native slices were not rebuilt.

Both separately approved Mac jobs passed against the selected
[32-export archive](apple-session-transition-package.md). CPU and tiny Metal
each passed the 37 existing transition controls plus 11 new configuration
controls, 15 lifecycle/staleness barriers and two retirement cases. The separate
preview/generation regression passed its 21 barriers and tiny generation
witness. All 16 successful command groups closed without endpoint errors.

Twelve raw records per backend include configuration after cancellation before
the first observation, evaluated-token generation, replay, cancelled replay
and close, alongside the prior transition witnesses. On Metal, model 43/session
45 retains capacity 16, chunk 3, whole budget 134,217,728 and accepted request
`435D7A9B-0C3E-476A-93C0-38FBA35101F5` after late cancellation and subsequent
two-token evaluation. Replay changes to session 46 with no accepted UUID and
the same resumed options. Cancelled replay has no live session or descriptor
but retains saved prefix `[1,7]`; closed state has neither model nor session or
descriptor. CPU records show the same state relationships with their own UUID.
These process-local IDs and UUIDs are test observations, not persistent identity.

The raw cleanup records keep the adopted descriptor and restart-required
state. Score/reset/preview preservation, in-flight configuration alongside nil
live context, rejected-request equality, lifecycle clearing and subsequent
usability also rely on completed assertions. The records are not full state,
logit or model-weight exports and do not independently reconstruct every
assertion. Prior D3 failed-attempt evidence remains retained; this task does not
relabel it as passing or change its accounting correction.

This establishes the descriptor and bounded worker lifecycle behavior. It does
not qualify rendered UI, a physical phone, real text-profile replacement,
Settings Apply, preference persistence, a larger context window, whole-device
fit or performance. The next UI consumer must still bind the selected
conversation/model/session and explicitly explain native cache reset.

[Machine-readable evidence](../benchmarks/results/2026-10-30-native-session-configuration-v1.json)
binds the three source files, actual products and logs, CPU/Metal records,
completed receipts and independent saved review.

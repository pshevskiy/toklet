# Phone state boundaries and actual OS lifecycle

Both selected models pass the bounded state and actual scene interruption/replay
checks on the wired iPhone 14 Pro (`iPhone15,2`), iOS27.0.1. This is a qualification
of these mechanisms on this Pro; the plan's base iPhone14 and overall M7.3 remain
open. The [qualification and exact evidence](../../benchmarks/results/2026-10-21-m7-phone-state-lifecycle-qualification-v1.json)
retain the raw flags (`physicalDeviceQualified=false`, `m7Complete=false`).

| Model | Check | Completed witness | Lifetime process peak, bytes | Thermal states |
|---|---|---|---:|---|
| small | state | 512 prompt; 128 committed / 384 pending; full capacity 1152 | 1,114,311,096 | [0, 1] |
| small | lifecycle | 19 events; 10 interrupted IDs; 522 replayed prefix | 1,114,360,224 | [0, 1] |
| larger | state | 512 prompt; 128 committed / 384 pending; full capacity 1152 | 1,928,170,440 | [0, 1, 2] |
| larger | lifecycle | 19 events; 3 interrupted IDs; 515 replayed prefix | 1,929,808,864 | [2] |

The state jobs use a512-token prompt, chunk128 and capacity1152. They require an
actual partial committed prefill with cancellation reason3 and zero emitted IDs;
the50ms timer is only a trigger. Completing that retained prompt and decoding three
IDs matches a fresh prompt. Fully cached prompts preserve the scalar score;
pending sampled IDs are committed before append, and a changed first token forces
the exact fresh reference. A fully committed1152-position prefix rejects continue,
append and a1153-token replacement with zero callbacks and identical saved tokens,
counts and before/after scalar diagnostics. These scalar checks do not compare an
entire distribution. Reset/fresh decode succeeds afterward; close zeros state and
releases ownership.

Each lifecycle job reaches active-generation READY, then the controller activates
Settings and returns to Toklet. Actual scenePhase callbacks record inactive,
background and active transitions, with original observation clocks. Fresh process
queries and activation metadata bind the same original PID throughout. The exact
sampled prefix survives cancellation, suspension drops KV, replay rebuilds that
prefix, and continuation matches uninterrupted reference IDs. The subsequent
synthetic memory-warning trigger exercises the shared release handler, replay
rejection after unload and a fresh usable load. It does **not** establish delivery
of a real UIKit memory warning, jetsam recovery, cleanup while OS-suspended, or a
safe process budget under every system pressure condition.

The signed app is prepared-v16, built from the reviewed state-app source and native
package-v5. The host-only activation argv fix follows that build; app source and
all seven signed-file hashes are identical throughout these accepted checks.
Collection freezes111files per attempt, including109matching Git HEAD files and
the owned driver/checks. Only the two historical host helper files map to their
exact older source copies. There are13accepted stages across both models, including
current-app tiny CPU/Metal integration; the13clean signed preparation commands are
separate from the accepted collection command count.

Two failures remain intact: the first lifecycle attempt placed device options after
the app bundle (exit64; fixed and reviewed in
[the argv report](2026-10-21-m7-lifecycle-cli-fix.md)); the next larger-state attempt
received an empty installed-app query and correctly refused launch before inference.
A fresh read-only exact-identity recheck and the separately guarded retry passed.
The accepted small lifecycle storage proof includes one exact-owned pre-copy reset,
and the later larger-state retry includes one exact-owned prelaunch reset. Both
interventions precede model execution. Failed attempts are excluded from
successful totals. No quality gates or tolerances changed, and no performance or
energy claim follows from these diagnostic jobs.

Six original-driver, six retry-driver, four tail-driver and12analysis regressions
pass. Astra accepted source and completed evidence with no findings after a byte-identical
independent replay. The independent replay checks exact manifests and driver prerequisite
closures, receipt order/hashes, installed identity, device/profile endpoints,
storage proof/copy freshness, private/public commands, exact raw console/launch
results, natural absences, both owned resets and the same-PID OS round trip.

Accepted collection: 191 clean owned commands, 26 natural absence observations and 2 exact-owned preflight resets. No system memory-warning, jetsam or overall device qualification is claimed.

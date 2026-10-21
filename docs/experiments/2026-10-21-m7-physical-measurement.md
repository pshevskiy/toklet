# M7 physical measurement controller

The physical controller now admits exact local model/job inputs and a verified
signed installation, prepares transfers without inference, and reuses authenticated
transfer receipts for a launch-only retry. Actual iPhone14Pro CPU and Metal measurements now pass after fixing filesystem
alias containment and startup activation ordering. The corrected Swift runner
also passes actual Mac CPU/Metal checks and both iOS SDK builds. These are tiny
measurement mechanism checks; real-model phone qualification and M7.3 remain
open. Astra accepted this code task and its evidence with no remaining findings.

The [controller](../../tools/experiments/m7_physical_measurement.py) uses frozen
local tiny fixtures, the M6 closure's selected model artifacts and benchmark
prompts, and the independently qualified M7 short raw-ID references. It creates
immutable host hard links rather than another full model copy. Exact remote file
inventory and size checks precede a config-SHA-authenticated app launch. Cached
signing observations are hash-bound and re-evaluated for current expiry, team,
device and application; current physical identity and installation checks remain
fresh. Preparation has a distinct receipt format, no inference output and two
independent process-absence checks. Reuse cannot treat a preparation receipt as
a measurement pass, and cannot guarantee launch before the phone locks.

The output checker requires ordered signed-file proofs, complete raw samples,
exact IDs/state, sequential request clocks, bounded finite times, valid unsigned
integer ranges, monotonic process lifetime peaks, foreground conditions and
close/release. Warmups are separate from measured repetitions. Sustained mode
stops at the first eligible measured request reaching the declared accumulated
native last-event duration. Source endpoint failures cannot prevent cleanup:
cleanup targets only the freshly admitted exact installation and re-queries its
executable before every signal. No other app or remembered PID is targeted.

The first two tiny CPU measurement attempts were denied because iOS reported the
phone locked; neither ran inference. Transfer-only preparation then allowed the
third CPU attempt to launch. It emitted matching installed-file proofs but failed
before model loading with an artifact-inventory bound error, zero samples and
clean process absence. That attempt remains failed and unqualified. A separate
Mac probe reproduces Foundation enumerating `/private/tmp` when the supplied
root uses `/tmp`. The old lexical component comparison rejected that alias.

The fourth CPU attempt confirms matching model artifact proofs and successful
loading with the alias fix, then rejects the still-inactive app before its first
sample. Close, release and independent process absence pass. This remains failed.
The diagnostic had begun in App.init before UIKit became active. A startup-only,
off-main polling gate now waits for active before config/proof/load work. A
monotonic30s checked deadline is verified before and after the main-queue state
read; the outer controller remains the process watchdog. Deterministic immediate,
delayed, exact-deadline and late-state cases ensure timeout prevents continuation.
Once the runner begins, interruption observers and active-state checks precede
artifact proof and worker creation; all per-request foreground checks remain.
This startup wait is untimed and does not qualify background recovery.

The alias fix canonicalizes the trusted root and enumerated entries before comparing
containment and deriving logical inventory names. Each child symlink is rejected
**before** canonicalizing that child. Entry-count and containment errors are now
separate. Actual CPU requests through a trusted root alias and Metal requests
through the canonical root preserve identical admitted inventories and tiny
results. File and directory symlink rejection through the alias remains checked;
extra files and changed bytes remain rejected. Fifteen Mac functional cases
retain26 timed samples, all44 related portable checks pass, and strict Release
builds for both device and simulator SDKs pass. These are mechanism checks;
concurrent SDK compilation makes the Mac timings unsuitable for performance
claims. The inherited Xcode orientation/AppIntents warnings remain.

Astra cleared the controller, transfer reuse, alias fix and startup gate source.
The final signed Release app runs on the wired physical iPhone14Pro (`iPhone15,2`,
iOS27.0.1), built with Xcode27.1/SDK27.1 and the reviewed native package.
Each backend completes three warmups and ten measured requests, retaining26 raw
samples overall. Prompt `[1,7,13,3]`, capacity32/chunk3 and maximum2 preserve exact
`[47,47]`, history6/committed5/pending1, reason1/status0. Both signed-file proofs,
model inventories, monotonic native/caller clocks, zero interruptions,
unload/close/weak-owner release and two independent process-absence checks pass.

| Backend | Requests | Maximum observed lifetime physical peak (bytes) | Native logical admission (bytes) |
| --- | --- | --- | --- |
| CPU | 3 warmup +10 measured | 21,546,656 | 67,210,976 |
| Metal | 3 warmup +10 measured | 23,037,600 | 67,164,260 |

Both use provisional128MiB native/256MiB process caps. Logical admission and
physical charge are different measures; neither establishes real-model phone
limits. `TASK_VM_INFO` is the named own-process memory instrument. All samples
report active application state, nominal thermal state, low power off, full
battery and plugged/full battery status. These brief tiny requests cannot
establish sustained thermal behavior or representative model throughput. Raw
native event times and submission/callback/completion clocks remain available;
no speedup, tail percentile, energy or sustained claim follows.

The fifth CPU attempt, before the final unlock, was denied by iOS Locked and
ran no inference. Earlier failed portable edits, the temporary deadline finding
and refused build preparation remain recorded. The
[qualification receipt](../../benchmarks/results/2026-10-21-m7-physical-measurement-qualification-v1.json)
binds current inputs, private evidence hashes, exact safe published copies and
historical executed source mappings. Raw signing/device inventories stay private.
No Rust numerical code, quality gate or default changes occur; no unrelated
Cargo/performance suite is attributed to this host/Swift controller task.

After this task, independently observe available phone filesystem capacity before
multi-GiB real-model transfers. Trial native/process caps are provisional
admission policies, not supported phone limits or hard allocator ceilings.
Real-model numerical/state/context checks, full repetitions, at least600s of
completed sustained requests, confirmation, thermal observations and actual OS
lifecycle/pressure evidence remain. Energy stays unset without instrumentation.
Results on this iPhone14Pro cannot establish the separate base iPhone14 budget.

Astra accepted source and publication after verifying815 distinct current,
historical and evidence paths,35 historical mappings,106 exact published/staged
copies, both console replays, privacy, links and staged whitespace. Review
performed only offline checks, with no concurrent native or phone workload.

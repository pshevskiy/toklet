# M7 authenticated measurement runner

The shared Swift diagnostic runner can now admit local model jobs, measure
completed raw-token requests, observe whole-process physical footprint and
lifetime peak, and preserve artifact endpoint errors after failed trials.
Actual tiny CPU and Metal checks on this Mac validate the mechanism. No new
phone execution, mobile limits, performance, sustained or lifecycle qualification
is accepted here; M7.3 remains open.

## Identity, memory and ownership

`--measure RELATIVE_CONFIG SHA256` selects a diagnostic app launch. The config
is bounded to512KiB, read and SHA-checked twice, decoded with exact top-level
fields, and validated before loading. Model/packed paths are relative to the
app's Documents root; traversal, symlink components, special files and unexpected
inventory entries fail. At most256 files/512 enumerated entries per directory
are admitted. Each file is SHA256 streamed in256KiB chunks with a4GiB per-file
bound. The exact before/after inventory includes every model and packed input.
The after proof runs after close even on request/output failure; separate endpoint
errors preserve the original failure. A combined mutation plus throwing-output
regression checks both errors and owner release.

Immutable weights and session/scratch still belong to the existing dedicated
native worker. Native admission uses the explicit numerical cap. A separate
trial process budget is chosen in the config, capped by the diagnostic policy
at4GiB; it is not inferred from device RAM or the Mac6GiB qualification. This is
**post-operation observed peak rejection**, not an enforceable allocation ceiling.
A native allocation may fail or the OS may terminate the app before a checkpoint;
such incomplete trials cannot pass. File hashing, Swift storage, frameworks,
loading/session creation, requests and release are covered by the process's
lifetime physical high-water mark. System services outside this process are
outside that measurement.

`TASK_VM_INFO` reads only this process. A localized, initialized C record is
rebound for the bounded call with no overlapping Swift access. The returned
status/count must cover the SDK-derived end of `ledger_phys_footprint_peak` and
fit the supplied capacity; negative signed peaks fail. The implementation uses
the actual C field offset because sizeof-based SDK revision macros are not
imported into Swift. `phys_footprint` is the current physical charge and the ledger
peak is the process lifetime peak, not an isolated operator or per-request peak.
No mobile peak is inferred from this Mac test.

## Timing, repetition and interruption

Each request resets the session, validates empty committed state, then uses an
explicit raw prompt, output count and optional expected output prefix. Stops are
empty; tokenization and formatting remain outside this harness. The runner
requires complete IDs, exact pending/committed counts, native/callback timing
records and the explicit budget after completed work. Samples stream immediately
to bounded-record JSON output instead of accumulating a long sustained transcript
in Swift memory. Three warmups are separate from at least ten measured requests.
An optional600–3600s target sums measured native **last-event elapsed seconds**;
this excludes warmups and is distinct from wall duration or complete native
return time. The2048-request ceiling fails an incomplete sustained sequence.
This component has exercised the zero-duration functional mode only.

Native event timestamps, callback timestamps after lock acquisition, and caller
completion after wakeup retain the boundaries in the
[timing component](2026-10-20-m7-worker-timing.md). Loading plus session creation
has a separate caller bracket; native conversion and compilation are not yet
separated by this ABI. Thermal state and low-power state are sampled; iOS also
reads battery and app state on the main thread. Energy remains unavailable.

The iOS diagnostic disables the idle timer and enables battery monitoring for
that process. Inactive requests cancellation; background queues session release;
a memory-warning notification queues full unload. Any observed interruption
invalidates foreground measurement, even if the app resumes before the next
sample. These are handler/invalidation mechanics, not accepted background
recovery or real OS-pressure evidence. Close waits for native completion and
weak-owner release; cleanup errors and incomplete app output remain failures.

## Verification and exclusions

The reproducible
[Mac checker](../../tools/experiments/m7_native_measurement_check.py) admits the
reviewed package and complete source/header/archive/helper map before **every**
owned child, including toolchain probes. It records the actual allowlisted
environment, Xcode/Swift versions and SDK path, then checks the compiled executable
before launch. Ordinary admission tests prove changed headers, archives and
helpers produce zero child launches; a changed binary also fails its recheck.

Twelve functional cases pass, including exact tiny CPU/Metal IDs and state,
config SHA/path/field/size rejection, checked self-memory reads, artifact mutation,
extra files, symlinks, trial budget rejection, wrong oracle, output failure,
combined mutation/output failure and final endpoint mutation. Each actual backend
runs three warmups plus ten measured two-token requests;26 raw samples retain
IDs, clocks, state, memory and environment. Both close/release acknowledgements
pass. Twenty-seven related Python checks and both strict Release SDK builds
pass. The inherited Xcode iPad orientation and AppIntents warnings remain;
the two raw Xcode stdout paths and raw Swift-version stderr have narrow
whitespace exceptions so their original tool-emitted bytes remain intact. Source/documentation whitespace checks remain active.
No numerical Rust/ABI/default changes or new Cargo qualification are attributed
to this task.

The first Mac checks pass but are excluded because their failure paths omit the
final artifact proof, as Astra found. The second checker preserves a nil-optional
JSON parsing failure; the third passes but precedes the fresh-checkout test setup
addition. Final v4 passes with current sources. The earlier SDK v10 build is
preserved with its exact executed source snapshot; final v11 matches current app
sources. Astra's checker provenance finding is fixed by full prelaunch map checks,
helper pins, toolchain/environment records and zero-launch regressions. No
numerical gate is relaxed.

The [qualification receipt](../../benchmarks/results/2026-10-20-m7-measurement-runner-qualification-v1.json)
binds current code, native package, build/runtime inputs, historical source
mappings, raw samples and exact published copies. Next is the physical controller:
transfer exact frozen model/job inputs to the verified app data container, run
short real-model oracles under an independent trial budget, then qualify phone
memory/context, repetition/sustained and actual lifecycle behavior. The
[device acceptance protocol](../../apple/DEVICE_QUALIFICATION.md) remains in force.

Astra accepted the code and publication with no remaining findings after
verifying164 evidence pins, six code pins, five historical mappings,50 copies,
current build/runtime maps,27 related checks and364 local link targets. Phone,
sustained and actual OS lifecycle qualification remain open.

The final staged check also found a trailing space in Swift's raw version line.
That evidence-only path now preserves the original byte with a narrow Git
attribute; source and documentation whitespace checks remain active.

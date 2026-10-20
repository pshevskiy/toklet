# M7 signed physical tiny integration

The signed Release harness passes its independent tiny CPU and Metal integration
checks on a wired **iPhone 14 Pro (`iPhone15,2`), iOS27.0.1**. This establishes
physical native integration; M7.3 and overall M7 remain open. No real-model
phone memory budget, model/context limit, throughput, thermal, energy or actual
OS lifecycle qualification follows. Pro evidence does not qualify a base
iPhone14, ProMax, iPad or another Mac.

The [qualification receipt](../../benchmarks/results/2026-10-20-m7-physical-tiny-qualification-v1.json)
binds the three changed code files,168 evidence inputs,27 historical source
snapshots and69 exact published copies. Raw device inventories, provisioning,
account/team identifiers and signing output stay in ignored artifacts; their
hashes are retained. Published copies are scanned against the private selection
and available private inventory/signing identifiers. The engine's numerical
code, defaults, tolerances and protected CPU binaries are unchanged.

## Mechanism and ownership

The [controller](../../tools/experiments/m7_physical_integration.py) admits an
exact private device/team configuration and the reviewed native-package receipt.
It verifies current physical identity, wired connection, Developer Mode and
developer services; isolated signed Release build; arm64/iOS18 deployment;
strict signature; development entitlements; and a valid profile admitting the
selected device/team/app. It checks the literal and resolved native-library link
and all source/package/fixture endpoints. Private command output is hash-bound;
public receipts redact nested errors as well as commands.

The app streams SHA256 hashes for seven fixed admitted files before and after
the test: executable, Info.plist, PkgInfo, tiny config/weights, CodeResources and
development profile. The host requires an exact local seven-file inventory and
matching phone proofs. These proofs establish correspondence for those files;
they do not enumerate every possible file in the installed bundle. The app's
self-test disables the foreground idle timer for its short execution. Normal UI
behavior and numerical workers remain unchanged. CryptoKit and Combine imports
are explicit.

A preparation-only receipt supports signing/installing once, without inference.
Fast reuse requires its exact SHA, original signed files, still-current app/native
inputs, historical controller/test snapshots, signature/profile validity, and
fresh current installation metadata before **each** launch. A changed or missing
installation prevents launch. An existing manual harness causes refusal; the
separate preflight stop records target only the freshly verified installation.
Cleanup re-queries that exact executable before bounded TERM/KILL and verifies
absence independently of the host launcher. Query/signal failures remain errors,
even if a later absence check succeeds. No unrelated app is targeted.

## Observed result and checks

The [passing runtime receipt](../../benchmarks/results/2026-10-20-m7-physical-tiny/m7-physical-tiny-v7/integration.json)
contains75 source/package inputs,37 private evidence hashes and17 successful
commands. Both actual backends emit initial prompt-plus-continuation IDs
`[47,47,47,47]` and the same ten exact states:

| State | History | Evaluated | Pending | Saved | Reason |
|---|---:|---:|---:|---:|---:|
| load | 0 | 0 | 0 | 0 | 0 |
| prompt | 6 | 5 | 1 | 6 | 1 |
| continue | 8 | 7 | 1 | 8 | 1 |
| suspend | 0 | 0 | 0 | 8 | 0 |
| replay-cancelled | 0 | 0 | 0 | 8 | 3 |
| replay | 8 | 8 | 0 | 8 | 1 |
| after-replay | 9 | 8 | 1 | 9 | 1 |
| reset | 0 | 0 | 0 | 0 | 0 |
| cancel | 5 | 4 | 1 | 5 | 3 |
| resume | 6 | 5 | 1 | 6 | 1 |

Subsequent outputs have separate assertions in the shared worker checks. Full
unload, close acknowledgement and weak-owner release pass. Both launch receipts
match the executable and report normal termination; console error markers are
also rejected, because an earlier failed app diagnostic demonstrated that
CoreDevice's reported zero exit alone is insufficient. Before/after proofs match
all seven signed source files. All three independent final remote-process checks
find the app absent, with no cleanup or endpoint error. The published
[CPU output](../../benchmarks/results/2026-10-20-m7-physical-tiny/m7-physical-tiny-v7/private-logs/cpu-self-test.stdout)
and [Metal output](../../benchmarks/results/2026-10-20-m7-physical-tiny/m7-physical-tiny-v7/private-logs/metal-self-test.stdout)
are exact copies; both stderr files are empty. Qualification flags stay false.

Sixteen targeted checks cover admission, profile/installation/launch metadata,
console/state/file mismatches, cleanup failures, nested redaction, reuse input
mutation and preparation-only orchestration (success, cleanup failure and
endpoint mutation all require zero launches). Six related suites run in isolated
processes and pass all52 checks with unchanged source endpoints. Both simulator
and device SDK Release builds pass strict Swift concurrency/warnings-as-errors;
Xcode still emits the inherited iPad orientation validation warning and expected
AppIntents extraction warning. They are not Swift compiler warnings or claims
of a warning-free entire Xcode log. Rust numerical code is unchanged, so no new
engine benchmark or Cargo suite is attributed to this integration task.

## Preserved failures and limits

All attempts are immutable. Physical v1 stops on the installation JSON schema;
v2 receives an iOS security denial; v3 is locked; v4 refuses an already running
manual harness; v5 launches but fails its bundle-proof guard before inference;
v6 again refuses a manual harness. Corrected v7 passes. The v5 path guard failure
motivates fixed literal file names; its precise root cause was not independently
proven. The schema and console checks follow the observed Xcode27 fields.
Prepared-v1 refuses the existing harness; prepared-v2 passes with zero inference
launches. Separate preflight stop receipts verify the exact admitted executable
and absence. Each physical/preparation attempt retains the exact three executed
code snapshots; late incorrect prepared-v1 copies are excluded and preserved
privately, while published replacements match the original recorded hashes.

Earlier SDK builds and a test syntax failure remain excluded historical evidence.
Several combined related-test runs encounter an intermittent host permission
error in the existing timeout-probe cleanup. Original standalone, traced and
uninstrumented reruns pass; final52-check qualification uses separate processes
per module. No timeout helper fix or tolerance relaxation is claimed. The failed
combined logs remain published alongside the passing isolated record.

Next work is real-model numerical/state admission, whole-process loading/peak
memory, measured phone limits, actual inactive/background/recovery and
memory-warning handling, repeated performance samples, at least600 completed
seconds of sustained execution and confirmation, with thermal/power observations
and energy only where actual instrumentation supplies it. The
[device protocol](../../apple/DEVICE_QUALIFICATION.md) retains all acceptance gates.

Astra accepted the code task and publication packet with no remaining findings,
after verifying198 current/historical pins,69 exact copies, executed-source
mappings, privacy,363 local link targets and whitespace. M7.3 remains open.

# Toklet native harness

This is a raw-token diagnostic app over the same Rust engine and local model
format. It does not duplicate tokenization or chat formatting in Swift. The
bundled independent tiny fixture makes integration checks reproducible without
model downloads. The current package supports arm64 Mac15+, iOS18+ and arm64
simulator build targets. Physical model/context limits are not yet qualified.

## Build

The verified toolchain is Xcode27.1, Rust1.90 and Python3.12. Install the required
Rust targets and SDKs first; build commands use locked offline dependencies and
do not install toolchains or change global Xcode selection.

From the repository root, create a new native package with
`python3 tools/experiments/m7_apple_package.py artifacts/apple-package-local-v1`.
Review its `build.json` and record its SHA256. Then run:

```sh
python3 tools/experiments/m7_swift_harness_build.py artifacts/apple-package-local-v1 REVIEWED_PACKAGE_SHA256 artifacts/apple-harness-local-v1
```

Replace `REVIEWED_PACKAGE_SHA256` with the reviewed build receipt's digest.
Destinations must be new. The builder verifies exact archive/source/header pins,
creates `apple/Binaries/Toklet.xcframework` and `apple/Generated/Tiny`, then builds
release simulator and unsigned device apps. It refuses foreign or differently
linked generated inputs. In this checkout the generated library link already
selects the qualified `artifacts/m7-apple-package-v4` package; reuse that package
when rebuilding the app. Its receipt digest is
`1185314043649692b7e41aff9b90e37355201eef71852965c20ef69513e6f13c`.

Open `apple/TokletHarness.xcodeproj` after generated inputs exist. The shared
scheme selects arm64, iOS18 minimum, complete Swift concurrency checks and
warnings as errors. Simulator builds use an ad-hoc signature; the device build
is deliberately unsigned until a signing team is configured.

## Use

Start with Tiny fixture, CPU, capacity32, chunk3 and numerical cap128MiB.
Load, prompt IDs`1,7,13,3`, generate2 IDs, then Continue. The initial four outputs
are`47,47,47,47`. Settings stay locked while a model is loaded. Unload before
changing the artifact/backend/settings. Copy real local artifacts into the app's
Documents/Model folder using device file sharing; packed Q4 v2 artifacts need
Metal and their separate local packed-directory path. CPU keeps its existing
2,048-token cap; Metal limits follow the actual model and explicit admission.
No mobile budget is inferred from the Mac's6GiB qualification.

The UI exposes Prompt, Append, Continue, Reset, Cancel, Suspend, Replay, Unload
and Close. It polls bounded raw-ID events every100ms; this polling interval is
not an engine timing measurement. The dedicated worker serializes native calls.
Non-UI diagnostics can use `waitUntilIdle(seconds:)` and bounded `tokenTimings`
in the returned snapshot. Native elapsed time and callback uptime have distinct
boundaries; see the [timing component](../docs/experiments/2026-10-20-m7-worker-timing.md).
Inactive requests cancellation. Background requests session release, retaining
immutable weights and exact history for explicit replay. Memory warnings request
full unload; saved replay state is then cleared. Actual completion around OS
suspension needs physical-device testing. Close is terminal for that worker.

For an automated tiny check, launch with`--self-test` (CPU) or
`--self-test --metal`. It prints one`TOKLET_SELFTEST_JSON=` record and exits.
This checks native ownership/state/replay/cancellation/release, not physical
performance or supported limits. `physicalDeviceQualified` stays false.
The simulator runner `tools/experiments/m7_swift_harness_run.py` additionally
admits a reviewed app build SHA, verifies the installed copy, captures the idle
UI and verifies Shutdown of the explicitly owned lab simulator. It is scoped
to the frozen Inference Lab iPhone14 admission records, not arbitrary devices.

## Physical setup and qualification

The [20 October physical tiny integration](../docs/experiments/2026-10-20-m7-physical-tiny.md)
passes CPU and Metal on the wired iPhone14Pro (`iPhone15,2`, iOS27.0.1).
Development signing, profile trust, installed-file correspondence, exact native
states and process absence are verified. The earlier
[setup observation](../benchmarks/results/2026-10-19-m7-physical-setup-v1.json)
remains a historical pre-signing record. The subsequent
[Pro device report](../docs/experiments/2026-10-21-m7-iphone14pro.md) accepts the
selected models' full quality, repeated context/sustained, memory/thermal and
actual scene lifecycle evidence within explicit trial limits. M7 is complete
after the user's [explicit device-target amendment](../docs/experiments/2026-10-21-m7-release.md).
Energy remains unavailable; real warning delivery/jetsam stay unqualified.
Pro results cannot establish base iPhone14 or ProMax limits.

Connect the phone with a cable and keep it unlocked. In Xcode27, open
Xcode > Open Developer Tool > Device Hub, select the physical phone and follow
pairing instructions; accept Trust This Computer on the phone if prompted.
Earlier Xcode versions use Window > Devices and Simulators. Developer Mode may
stay hidden in Settings until pairing starts. Then enable Developer Mode under
Settings > Privacy & Security, restart and confirm it on-device using
[Apple's instructions](https://developer.apple.com/documentation/xcode/enabling-developer-mode-on-a-device).
In Xcode > Settings > Apple Accounts, sign in with your Apple Account. Open
`apple/TokletHarness.xcodeproj`, select TARGETS > TokletHarness > Signing &
Capabilities, enable Automatically manage signing and choose your team.
A Personal Team supports local device testing without paid membership. Use
[Apple's team setup instructions](https://help.apple.com/xcode/mac/current/en.lproj/dev23aab79b4.html).
Account authentication and the phone's confirmation require its owner. Keep
credentials and private device identifiers out of committed files. A successful
sign-in alone does not prove that a valid development certificate/profile exists;
verify the signed app and physical installation before accepting test results.
If macOS requests access to the development signing key, its owner must approve
the Keychain prompt. If iOS rejects the installed developer app, its owner must
trust that developer under Settings > General > VPN & Device Management.
An unlocked phone on the Home Screen is needed for launch.

The physical controller takes the reviewed package path/SHA, a private selection
JSON path/SHA, and a new output directory. Private selection fields are exactly
`device_identifier`, `device_udid`, `development_team`, `product_type` and
`os_version`; never commit their identifying values. This controller deliberately
admits only the current iPhone15,2/iOS27.0.1 target. For example:

```sh
python3 tools/experiments/m7_physical_integration.py artifacts/m7-apple-package-v4 REVIEWED_PACKAGE_SHA256 PRIVATE_SELECTION.json PRIVATE_SELECTION_SHA256 artifacts/physical-prepared-local-v1 --prepare-only
python3 tools/experiments/m7_physical_integration.py artifacts/m7-apple-package-v4 REVIEWED_PACKAGE_SHA256 PRIVATE_SELECTION.json PRIVATE_SELECTION_SHA256 artifacts/physical-run-local-v1 artifacts/physical-prepared-local-v1 REVIEWED_PREPARATION_SHA256
```

Preparation verifies a signed installation with zero inference launches. Reuse
checks the original app/native inputs, signature/profile and current installed
location before launching CPU and Metal. Save the three code files under the
preparation's `source-snapshot/` with their repository-relative paths when
retaining a preparation for later controller revisions; their hashes must match
that receipt. Existing manual harness processes cause refusal. Do not bypass
that guard or launch against a stale installation. Private output stays ignored;
publish only reviewed redacted receipts and safe exact console copies.
See [the device qualification protocol](DEVICE_QUALIFICATION.md) before accepting
any mobile model/context limit or closing M7.

## Explicit measurement jobs

The diagnostic `--measure RELATIVE_CONFIG SHA256` launch reads a bounded,
SHA-authenticated JSON job relative to Documents. It does not run a job entered
through UI text. A job specifies model/packed folders, exact relative file hashes,
backend, capacity/chunk, numerical and separate process trial budgets, raw prompt,
output count, expected output prefix, three warmups, ten initial repetitions and
optional minimum native seconds. See the
[runner report](../docs/experiments/2026-10-20-m7-measurement-runner.md) for the
exact timing, memory and interruption boundaries. Model/packed inventory is
checked before loading and after cleanup, including failed requests. JSON samples
stream as `TOKLET_MEASUREMENT_SAMPLE_JSON=` and the final summary as
`TOKLET_MEASUREMENT_JSON=`; errors, missing output or false qualification flags
must not be interpreted as a supported mobile limit. Actual phone qualification
still requires a separately reviewed physical run.

Repeat the Mac mechanism checks with a new output directory:

```sh
python3 tools/experiments/m7_native_measurement_check.py artifacts/m7-apple-package-v4 REVIEWED_PACKAGE_SHA256 artifacts/measurement-check-local-v1
```

This compiles the Swift checker with the reviewed Mac archive and runs the tiny
fixture; it does not install a phone app or qualify phone memory/performance.

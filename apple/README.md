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

## Physical setup and remaining qualification

The latest setup check,19 October2026, finds a wired physical iPhone14Pro
(`iPhone15,2`) on iOS27.0.1 with Developer Mode enabled and developer services
available. The earlier wireless iPhone14ProMax (`iPhone15,3`, iOS26.6.1) still
reports Developer Mode disabled. These are separate devices. The wired Pro is
the next qualification target; its results cannot establish base iPhone14 or
ProMax limits. The first observation found no signed-in Xcode account. After
owner sign-in, Xcode shows a Personal Team, but the Mac still has zero valid
signing identities; development signing is not yet configured for the harness.
No physical app was installed or run.

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
The [latest setup observation](../benchmarks/results/2026-10-19-m7-physical-setup-v1.json)
records readiness only, not physical execution or qualification.
See [the device qualification protocol](DEVICE_QUALIFICATION.md) before accepting
any mobile model/context limit or closing M7.

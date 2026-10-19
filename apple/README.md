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

The paired physical device is an iPhone14ProMax (`iPhone15,3`) on iOS26.6.1.
A fresh check reports Developer Mode disabled, DDI services unavailable and zero
valid local signing identities. No physical app was installed or run.

Enable Developer Mode on the phone, restart and confirm it on-device using
[Apple's instructions](https://developer.apple.com/documentation/xcode/enabling-developer-mode-on-a-device).
Assign the app target to a signing team with automatic signing using
[Apple's team setup instructions](https://help.apple.com/xcode/mac/current/en.lproj/dev23aab79b4.html).
Account authentication and the phone's confirmation require its owner.
See [the device qualification protocol](DEVICE_QUALIFICATION.md) before accepting
any mobile model/context limit or closing M7.

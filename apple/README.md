# Toklet native harness

Toklet Harness is one app over the Rust engine. It opens a friendly local Chat
screen with model selection, saved conversations and settings. The original
raw-token diagnostic form is available under **Settings → Expert mode**.
See the [interface guide](../docs/app-interface.md) for supported text models,
loading and current limitations. Tokenization/chat formatting remain in Rust,
outside numerical kernels. The bundled independent tiny fixture keeps Expert
checks reproducible without downloads. Targets are arm64 Mac15+, iOS18+ and
arm64 simulator; device qualification remains specific to each recorded trial.

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
linked generated inputs. The current generated library link selects
`artifacts/chunk-pools-native41-2026-11-06/Toklet.xcframework`, with all 41 C
exports and complete Mac/iOS/simulator app compile/link evidence. Its native
build receipt SHA256 is
`68edb4d767518d20099f6fef5a9dfafb583c389ee4f9c977bb6f24f896b27666`.
The previous native41 and native40 packages are preserved.
[Package refresh evidence](../benchmarks/results/2026-11-06-chunk-pools-native41-v1.json).
[Qwen2.5-1.5B Mac controller and package evidence](../benchmarks/results/2026-11-06-qwen15-mac-app-v1.json).
The [shared Mistral integration](../docs/mistral-managed-preferences.md),
[Qwen2.5 integration](../docs/qwen2-managed-preferences.md) and earlier
[Smol/Qwen3 integration](../docs/smol-managed-preferences.md) record strict decoding,
explicit audited loading, actual Mac settings checks and guarded selection. [Managed settings](../docs/managed-model-preferences.md) save accepted
context/chunk/budget/reply choices and restore them after fresh model and memory
admission. Smol [bound advisory preview/Apply](../docs/smol-advisory-preview.md)
also uses actual loaded-owner admission; all existing
[session Apply](../docs/friendly-chat-apply.md) routes retain their loaded bounds.
Other profiles and external folders remain ephemeral for these engine settings.
The previous37 [Qwen2 package](../docs/qwen2-managed-preferences.md),
36 [shared Smol package](../docs/smol-managed-preferences.md),
35 [identity package](../docs/swift-accepted-identity.md),
32 [session-transition package](../docs/apple-session-transition-package.md),
30-export [planning package](../docs/apple-planning-package.md) and older 28 package
are preserved. Package selection does not install or launch an app or increase
context limits.

The [text profile](../docs/app-chat-profile.md),
[interface guide](../docs/app-interface.md) and
[physical model-switching follow-up](../docs/mobile-model-switching.md) retain
their recorded application/device scopes. Earlier packages and installed-app
results are not retroactively upgraded by a new compile/link result.

Open `apple/TokletHarness.xcodeproj` after generated inputs exist. The shared
scheme selects arm64, iOS18 minimum, complete Swift concurrency checks and
warnings as errors. The legacy builder explicitly ad-hoc signs simulator apps and leaves the device
app without a development signing stage. The reviewed planning and session-transition package gates omit
all explicit codesign/account/provisioning operations; its Mac/simulator link
outputs can still contain LC_CODE_SIGNATURE. See their reports before making any
signing or installation claim.

## Use

For ordinary text, follow the [Chat guide](../docs/app-interface.md). Models
supports [Hugging Face search and verified downloads](../docs/model-hub.md) for
the selected SmolLM2 and Qwen eight-bit profiles and the pinned Mistral four-bit
profile, with offline opening. The [Mistral Mac report](../docs/mlx-mistral-app.md)
separates functional Chat from actual network-transfer qualification. For diagnostics open the conversations
icon → Settings → Expert mode.

Start with Tiny fixture, CPU, capacity32, chunk3 and numerical cap128MiB.
Load, prompt IDs`1,7,13,3`, generate2 IDs, then Continue. The initial four outputs
are`47,47,47,47`. Settings stay locked while a model is loaded. Unload before
changing the artifact/backend/settings. Copy real local artifacts into the app's
Documents/Model folder using device file sharing; packed Q4 v2 artifacts need
Metal and their separate local packed-directory path. For a supported local
MLX artifact, enable Metal and **MLX-format model**, with the separate
Q4 folder empty. The bundled Tiny button clears MLX because that fixture is
floating-point. CPU keeps its existing
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
For the separately copied packed tiny fixture in Documents/MLXTiny, launch
`--self-test --mlx`; the [worker report](../docs/mlx-apple-worker.md) specifies its
scope. Do not point this fixed-ID self-test at a real checkpoint.
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

Chat input follows the admitted active session window and model input cap,
with a32768-token codec ceiling. Prompt plus reply must fit together; reply
output remains capped at256. Managed Qwen2 offers an explicit512/1024 next-load
limit and starts with512-token KV before saved settings/Preview/Apply.
[Mac boundary evidence](../benchmarks/results/2026-11-02-session-window-chat-v1.json)
covers successful longer Chat and overflow preservation, not full-window
quality, rendered UI or physical-phone qualification.

Advanced window planning supports an optional KV ceiling in MiB. Blank means
unlimited; zero is explicit. Preview counts physical cache buffers, and Apply
checks both resident planned and actual candidate KV before committing. The
ceiling itself is not saved; accepted window settings use the existing record.
[KV-ceiling evidence](../benchmarks/results/2026-11-03-kv-preview-apply-v1.json).

The default Mac catalog includes the pinned Llama3.2 1B eight-bit profile at512
positions; iOS retains four profiles pending physical qualification. The original
[quality gates](../docs/llama3-quality-qualification.md) and fresh
[ranged-download/offline journey](../docs/llama3-download-offline.md) pass.
Open Llama through Models to use its audited loader and text policy.

All five adopted Mac profiles have [fresh download, Stop/recovery and offline evidence](../docs/mac-hub-journeys.md).
An optimized app from unmodified production sources is available at
`artifacts/mac-hub-journeys-2026-11-04/production/Toklet.app`; it was built without
launching or replacing a user session. iPhone qualification remains separate.

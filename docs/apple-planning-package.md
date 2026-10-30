# Apple planning bridge and app build/link verification

The current bridge and full app sources compile and link for arm64 Mac,
iOS device and iOS simulator. All three fresh native slices expose exactly
the current 30 C functions, including capacity planning and context limits.
The previous selected package exposed 28; standalone Swift checks could not
establish app linkage against that older bundle.

The bounded owner reused the existing M7 slice/package algorithm with the
reviewed pipe supervision and file-identity helpers. All 30 child jobs passed:
21 toolchain/native-package jobs, six device/simulator app-build inspections
and three Mac app-build inspections. Every owned process group closed and
endpoint checks passed. These are compile/link results, not app runtime tests.

The build used installed Rust 1.90.0 (`1159e78c4`), Xcode 27.1 build 27A9275,
macOS 27.0 SDK and iOS/simulator 27.1 SDKs. Cargo used locked offline
release builds with Metal and panic=unwind in a task-local target directory.
The three target/minimum pairs are arm64 macOS 15, arm64 iOS 18 and arm64
iOS Simulator 18. No toolchain or dependency was downloaded.

Each slice's exported C symbol set exactly equals the current header, and
identical header/modulemap bytes accompany every archive. C smoke executables
link without running; they reference the new context-limits API. The new
capacity API is covered by exact symbol equality and the typed Swift consumers
in the app link. All three linked executables and apps have the expected
architecture and LC_BUILD_VERSION platform/minimum.

App sources and resources were copied to an exclusive artifact-local snapshot.
The iOS project resolves all 28 source entries, including the C1 limits/capacity
consumers, C2 planning-preview types and existing WorkerChecks dependency.
The Mac compile consumes 30 sources: all Harness Swift files except the iOS
entrypoint, plus the Mac sources. The actual current worker and project remain
in Swift 5 mode with complete strict concurrency and warnings as errors.
Separate C1 typed-consumer Swift 6 results remain separate from this whole-app
build configuration. The packaged Tiny resource remains byte-identical.

No explicit codesign command, signing identity, account, provisioning action,
installation, simulator boot or app launch occurred. The owner field
`signed:false` means there was no signing stage. Mac and simulator load metadata
nevertheless contain LC_CODE_SIGNATURE; device load metadata does not. This
report does not claim that the Mac/simulator binaries are cryptographically
unsigned or verify a development/distribution signing identity.

The job limit was 1,800 seconds, with 16 MiB retained per output stream, a
6 GiB observed direct-process charge guard and 15-second TERM plus 15-second
KILL cleanup. A 16 GiB free-disk reserve was checked before launch. Sampled
parent-process memory is not an aggregate compiler/child-process peak. Actual
wall duration was not retained and is not inferred from sample counts. SDK
metadata and selected tools were pinned; this is not a full SDK-file-tree audit.

Independent saved-result review passed and authorized guarded adoption.
[The machine-readable evidence](../benchmarks/results/2026-10-30-apple-planning-package-v1.json)
records that audit and the subsequent exact old-target-to-new-target link
replacement. The workspace link remained unchanged throughout the build and
now selects this verified candidate. The previous package is preserved.
Selection changes build inputs; it does not install or launch an app.

This gate adds package and app compile/link evidence. It does not validate UI
behavior, a planning Settings interface, model loading or generation in these
apps, new context or factory admission, phone operation, signing/distribution,
performance or power. Earlier installed-app and device results retain their
original packages and scopes. The preview-only UI task remains separate.

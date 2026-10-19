# M7 Swift iOS harness integration

The minimal SwiftUI app consumes the same raw-token native worker and local
artifact format. It exposes load, prompt/append, bounded token-ID streaming,
continuation, reset, cancellation, suspension, same-owner replay, unload and
close. This is a numerical integration harness; text tokenization/chat formatting
stay outside the kernels and are not reimplemented in Swift. Persistent labels
identify filled fields, and model settings are locked while a model is loaded.
Input fields reject oversized values; the worker checks path bytes before
allocating UTF8 input arrays.

The app polls a bounded snapshot every100ms without per-token UI work queues.
Native model/session calls remain on the dedicated thread; callbacks retain
explicit synchronous lifetimes. Inactive requests cancellation, background
requests session release, and memory warnings request full unload after active
work returns. These are handler requests, not claims about iOS background
completion. Same-owner replay preserves pending history transactionally. Full
unload clears saved state; close is terminal. Physical lifecycle completion
still needs actual-device evidence.

The Xcode project selects arm64, iOS18 minimum, complete Swift concurrency
checking and warnings as errors. It consumes separate device/simulator slices
from the admitted XCFramework. The controlled builder verifies the exact package
SHA, all archive/source/header pins, the generated library link's admitted,
literal and resolved targets before each build and at completion, bundled tiny
resources, app architecture/platform/deployment and simulator ad-hoc signature.
Device apps are unsigned. Tool commands have bounded deadlines and owned-group
cleanup; failures and source endpoint changes are preserved.

Eight builder tests cover foreign generated inputs, fixture mutation/directory
redirection, link replacement before binding and after admission, binary plist
identity/escape checks, bounded metadata and zero-launch rejected metadata.
Five runtime-controller tests cover exact app inventory/executable bindings,
conflicting pins, installed-copy mutation, rejected-build zero launch and owned
cleanup preserving the original failure. The runtime controller admits a
reviewed app receipt, checks its actual installed bundle inside the owned
simulator, runs the shared tiny self-test, captures idle UI and requires verified
simulator Shutdown. Host launcher termination alone is not simulator cleanup.

The first build fails on duplicate CToklet module maps and inherited x86_64
selection; explicit arm64 and the XCFramework's processed headers fix both.
The next build passes but is excluded from final qualification because link
admission still had a controller gap found by Astra. Exact executed source
snapshots and raw failures remain preserved. Later builds pass with the tightened
controller; visual inspection finds and fixes placeholder-only field labels.
The final v5 source has persistent labels and locked loaded-model settings.
No numerical precision, tolerance, kernel or engine default changes occur.

The final native package has three passing compile/link/export/deployment slices
and unchanged endpoints. The final simulator and unsigned iPhone app builds pass
all nine commands, with70 source/package bindings and11 bundle files.
The actual simulator app self-test passes all ten exact state transitions:
load; prompt6/5/1; continuation8/7/1; suspend with8 saved IDs; precancelled replay
with8 saved IDs/no session; full replay8/8/0; continuation9/8/1; reset;
cancellation5/4/1; and resume6/5/1. The initial prompt-plus-continuation list is
[47,47,47,47]; subsequent emitted IDs have separate oracle assertions. Full
unload, close acknowledgement and weak-owner release pass. Installed copies,
source and bundle endpoints match, and simulator cleanup exits cleanly with
Shutdown verified. A captured screenshot verifies the final idle form labels.

This establishes M7.2 software integration on the named simulator. No simulator
Metal, real-model phone, supported mobile budget/context, throughput, energy,
thermal or physical lifecycle qualification follows. A fresh physical inventory
check finds iPhone14ProMax/iPhone15,3/iOS26.6.1 with Developer Mode disabled and
DDI services unavailable; an elevated signing query finds zero valid identities.
Only the redacted status is published. The paired phone's account/device setup
requires its owner. Base iPhone14 and other device limits remain separate.
The [harness guide](../../apple/README.md) and
[physical acceptance protocol](../../apple/DEVICE_QUALIFICATION.md) record the
remaining work; M7.3 and overall M7 stay pending.

The [component receipt](../../benchmarks/results/2026-10-19-m7-swift-harness-qualification-v1.json)
binds11 changed sources,133 evidence files and58 exact published copies,
including the final image, private-identifier-free availability summary,
protected CPU binary hashes and exact excluded source snapshots. Final runtime
binds86 source/package/app inputs and verifies all6 installed bundle files;
all7 execution commands and2 cleanup commands exit successfully, with no
endpoint or cleanup error. Physical acceptance remains false.

Astra accepted task6/M7.2 with no remaining findings after checking current and
historical bindings,58copies, installed files, ten states, clean shutdown,
helper checks, final UI/lifecycle wording and redacted device status.
M7.3 and overall M7 remain pending physical qualification.

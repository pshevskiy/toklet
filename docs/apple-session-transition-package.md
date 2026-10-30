# Apple session-transition bridge package

Three fresh native slices and the current app sources compile and link against
all 32 C exports at D2 commit `c83625ceda4e97fa0fe9df15a847f40ef5dbecd6`.
The added APIs provide [same-model transition accounting](native-session-transition-report.md)
and [empty candidate preparation](native-empty-session-candidate.md). This
package makes those native symbols available to future Swift consumers; the
current app does not yet adopt candidate sessions or apply new Chat settings.

All 30 bounded build/link jobs passed, with closed process groups and no endpoint
mismatches: 21 native/toolchain/package jobs and nine app-build/inspection jobs.
Every native slice exports exactly the current header's 32 functions, with
matching bundled headers and module maps. C smoke executables reference empty
candidate preparation, cancellation and context APIs and were linked but not
run. The three apps were compiled and inspected but not launched.

The targets are arm64 macOS 15, arm64 iOS 18 and arm64 iOS Simulator 18. Installed
Rust 1.90, Xcode 27.1 and the existing SDKs were used, with locked offline Cargo
release builds, Metal and panic=unwind. No downloads or toolchain changes were
performed. The artifact-local snapshot includes all 30 current iOS project
source entries and 32 Mac compile sources, including the preview-only Settings
files added in C4. The app retains Swift 5 with complete concurrency checking
and warnings as errors. This does not replace separate Swift 6 consumer checks.
Bundled Tiny resource bytes match the authenticated inputs.

The independent audit authenticated 2,428 input bindings, 77 output identities
and 42 copied source files. It checked each executable's architecture and
LC_BUILD_VERSION platform/minimum, all three archive/header/module-map sets,
and the old selected framework's preservation. The workspace framework link
was unchanged during all build jobs. The guarded adoption recorded in the
[machine-readable result](../benchmarks/results/2026-10-30-apple-session-transition-package-v1.json)
subsequently selected only this candidate. The preceding 30-export C3 bundle
remains available; its earlier runtime checks retain their original scope.

No explicit codesign, signing identity, account, provisioning, installation,
simulator boot, app launch or model/GPU execution occurred. The receipt's
`signed:false` means no explicit signing stage. Mac and simulator C/app binaries
contain LC_CODE_SIGNATURE; iOS-device binaries do not. It does not mean the
Mac/simulator outputs lack implicit ad-hoc signatures.

The owner retained the qualified limits of 1,800 seconds per child, 16 MiB per
output stream, a 6 GiB sampled direct-process guard and 15-second TERM plus
15-second KILL cleanup. The prospective free-disk reserve was 8 GiB, based on
the prior complete 1,519,861,760-byte build tree. Actual free space was
13,269,934,080 bytes before and 11,923,083,264 after. Those endpoint observations
are not a peak-disk measurement. Sparse direct-child memory samples are not an
aggregate compiler/process-tree peak; no elapsed or performance claim is made.
Tools and SDK metadata are pinned, not the complete SDK file tree.

An initial metadata invocation used a relative saved-review path where exact
absolute-path identity was required. It stopped before writing the package or
spawning children. That failed attempt is retained; the retry used the same
reviewed scripts and policies with the correct absolute arguments.

This closes coherent package and app compile/link coverage for the 32-export
header. It adds no worker adoption, UI Apply, persistence, new context admission,
rendered UI, physical-phone, numerical or performance qualification. Native
D1/D2 verification remains separate. The next task is a small model/session-bound
worker transaction that validates a candidate before atomic acceptance.

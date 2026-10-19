# M7 Apple library packaging

The Metal library and native boundary now compile for arm64 macOS and iOS.
Production target gates include iOS consistently through dependencies, Half
weight loading, the Metal module and native dispatch. The runtime availability
check requires macOS 15 or iOS 18 before selecting precise Metal floating-point
APIs. Unified-memory, MPS and pipeline capability checks remain required; GPU
profiling counters remain optional. Numerical kernels, precision policy and
engine defaults do not change.

The package builder creates separate arm64 macOS, iPhone and simulator archives
and combines them into an XCFramework. Device and simulator code are separate
slices despite sharing the same instruction architecture. Each archive must
export exactly the C entrypoints declared in the header. A real C client links
against every slice with warnings treated as errors; its Mach-O build command
must identify the exact platform and minimum deployment version. Final metadata,
archives and headers are checked and hashed after XCFramework creation.

The build uses installed Rust 1.90 and Xcode SDKs, offline locked dependencies,
explicit compiler paths, unwind panic behavior and isolated output directories.
An allowlisted environment excludes inherited Cargo settings. Applicable local
and global Cargo configuration files must be absent at both endpoints. All
commands have deadlines and bounded process-group termination/reaping, including
build descendants that outlive their direct parent. Ten portable tests cover
those cleanup cases, environment/configuration rejection, bounded metadata and
symbols, platform/deployment matching and package corruption/path traversal.

The first build attempt compiled and linked all three slices, but remains
unqualified because Astra identified environment admission and descendant cleanup
gaps in its controller. The next attempt fixed both gaps but failed during Mac
C linkage because its restricted environment omitted a writable temporary
directory. Both attempts and the exact executed controller sources are retained.
The corrected build sets TMPDIR explicitly inside its exclusive output folder.
No tolerance, numerical assertion or deployment minimum was relaxed.

Build qualification is distinct from runtime qualification. Simulator execution
can establish integration behavior only. Physical iPhone quality, supported
context limits, independent memory budget, lifecycle, sustained thermal behavior
and energy remain separate M7 acceptance requirements.

The corrected v3 build passes all three compile/link/export/deployment checks
and exact XCFramework copies: macOS 15 arm64, iOS 18 arm64 and iOS 18 arm64
simulator. Xcode 27.1 uses the MacOSX27.0 and iPhoneOS/iPhoneSimulator27.1 SDKs;
Rust is 1.90.0. All 21 tool commands exit successfully, with no cleanup or source
endpoint errors. The build receipt binds 38 sources and all 10 package files.
CPU107/Metal-feature119 tests, formatting and strict all-target Metal Clippy
pass; 8/59 hardware/model checks respectively remain explicitly ignored.
Protected M6 CPU executables remain unchanged.

The final integration controller admits a separately reviewed build SHA and
exact three-slice inventory before launching any executable. It rejects
conflicting bindings, missing/remapped executables and changed frozen fixture
payloads. The dedicated simulator's prior Shutdown state and owned boot evidence
are admitted explicitly. Once ownership is admitted, outer cleanup shuts down
that simulator even if an earlier Mac worker fails, then requires an independent
simulator inventory response to verify Shutdown. Host launcher process-group
cleanup alone is insufficient evidence of simulator cleanup. Seven focused
integration tests cover zero-launch provenance failures, primary/endpoint error
preservation and simulator cleanup/state failures.

A deliberately TERM-ignoring, bounded C witness reaches the launcher deadline.
It is already absent after host process-group cleanup; simulator Shutdown and
PID absence are separately verified. This does not claim a child survived host
cleanup. Its exact executed controller source is preserved because final
integration uses distinct cleanup-log names to avoid overwriting witness logs.

Final integration passes Mac CPU, Mac Metal and iPhone simulator CPU with exact
[47,47,47,47] outputs from the frozen independent tiny artifact. The compiled
client also asserts retained ownership, pending-token continuation, callback
Busy, cancellation, reset and orderly release. All 60 admitted integration
bindings remain unchanged. Three numerical clients and both simulator cleanup
commands exit successfully; Shutdown is verified with no primary, cleanup or
endpoint errors. This is a tiny integration test, not a real-model phone quality
or performance result. No simulator Metal runtime qualification is claimed.

The [component receipt](../../benchmarks/results/2026-10-19-m7-apple-package-qualification-v1.json)
binds 41 sources, 115 evidence files and 94 exact published copies. It preserves
excluded attempts, the precise historical witness source and protected CPU
binary identities. M7.1 boundary and build packaging are complete in this scope;
M7.2 Swift integration and M7.3 physical-device qualification remain open.

Astra accepted final source, evidence and publication after independently
verifying all 41 source pins, 115 evidence pins, 94 copies, protected binaries,
historical source mapping, slice identities, results and shutdown checks.

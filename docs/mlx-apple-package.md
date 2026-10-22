# MLX-enabled native package and linked Apple apps

The current generated app library alias selects `artifacts/m8-apple-package-v1`.
Its reviewed build receipt SHA256 is
`00d3701912b3371a70485bf4cff7bd4d8217db0d99e53206274c17d47095bee5`.
The preceding `m7-apple-package-v5` directory and historical qualification remain
immutable. A separate driver records the exact known alias switch before invoking
the existing app builder; it does not replace an unknown checkout or library.

The [package builder](../tools/experiments/m7_apple_package.py) now binds every
Rust source and embedded `.metal` file, alongside Cargo/header/test inputs. The
old fixed shader inventory covered `kernels.metal` and would have omitted the
new `mlx.metal` input. A focused regression requires both libraries and an
additional nested shader, while excluding unrelated binary data. All 11 focused
portable methods pass, including existing owned-process cleanup and metadata/
platform/bundle checks. No numerical code, compiler math policy or ABI layout
changes in this task.

All 21 native package commands pass. The three separate arm64 slices target
macOS15, iOS18 and iOS18 simulator; each links its C smoke executable and exposes
exactly the 15 header-declared C entry points, including native MLX loading. The
XCFramework's ten-file inventory, archive/header copies, platform/minimum metadata
and source endpoints are checked. Build outputs stay separate from the protected
historical CPU binaries.

After Astra admitted that exact native receipt, all nine linked-app build commands
pass: release simulator and unsigned device app compilation/linking, architecture
and deployment inspection, plus simulator ad-hoc signing/verification. The linked
app build receipt SHA256 is
`51f3ce3c804085525d8902e85b72de8388f5abe76f16237b10b10ea325cb900d`.
All current worker/UI sources are included, so MLX's C entry point is resolved
against the new native library. This is distinct from device development signing,
installation and physical execution.

The freshly packaged Mac C executable also passes all three explicit local modes:
CPU and Half Metal with the existing floating fixture, and native MLX Metal with
the frozen packed fixture. Each produces exact `[47,47,47,47]` IDs and completes
its existing ABI/score/state/callback/retained-owner/cancellation/reset checks.
These are functional tests rather than performance measurements.

The [task receipt](../benchmarks/results/2026-10-22-native-mlx-apple-package-v1.json)
joins the focused checks, native and app builds, generated alias binding, raw logs,
Mac checks and unchanged source/protected-binary endpoints. All runtime Rust and
shader source bytes are unchanged in this packaging task. Ordinary Rust suites
are not represented as fresh runs; the package performs fresh release builds and
link checks on all three targets.

The current apps are ready for explicit M8 signing/installation and sealed
real-model job integration. This report does not accept real checkpoint quality,
process peaks, performance, simulator execution or physical iPhone MLX execution.
Those remain open in the [M8 plan](mlx-implementation-plan.md).

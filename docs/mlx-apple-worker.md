# Explicit MLX selection in the Apple worker and UI

The harness has an explicit **MLX-format Qwen model** toggle alongside Metal.
It selects the reviewed native Rust `toklet_model_load_mlx` entry point through
`NativeSettings.mlx`. Metal must be enabled and the separate Toklet Q4 folder
must be empty. The worker enforces these constraints before loading, including
for callers that bypass the UI. CPU plus a separate Q4 folder also rejects.
Existing CPU/Half/Q4 choices keep their defaults and loading routes.

The bundled Tiny fixture button clears MLX selection because that fixture is
floating-point. A supported real MLX artifact uses its own local model directory;
the app does not convert or download it. Settings remain locked while loaded.
Prompt/append/continue, cancellation, reset, suspend/replay and unload continue
through the existing dedicated worker and bounded native handles.

`--self-test --mlx` explicitly runs the frozen packed tiny fixture placed in
Documents/MLXTiny. It uses the same worker ownership/state oracle as the bundled
CPU/Half self-test; it is not a real-model or performance test. Its backend label
is `metal-mlx`. Existing `--self-test` and `--self-test --metal` retain their
original backend labels and bundled directory. The Mac checker accepts a local
fixture path and the optional `mlx` argument.

The Mac worker passes all three configurations: CPU, compensated Half Metal and
native MLX Metal. Each checks exact four sampled IDs and ten lifecycle/state
records, including pending continuation, cancelled replay, successful replay,
callback cancellation, reset, unload, close and final worker release. The MLX
run additionally rejects three conflicting backend settings and a missing local
artifact without retaining native owners, then successfully loads and runs the
valid fixture. Native timing events remain within each completed request.
All harness sources and the shared worker checker typecheck for arm64 iOS18+
with complete concurrency checks and warnings as errors. The Mac checker uses
the same strict Swift5 settings as the Xcode project.

The first passing run is retained with its original UI source. Before the final
run the Tiny fixture button was refined to clear MLX selection; no numerical or
worker behavior changed. Eight final owned commands and exact source/archive/
fixture/protected-binary endpoints pass in the
[task receipt](../benchmarks/results/2026-10-22-native-mlx-apple-worker-v1.json).
No Rust or shader source changed in this task. The current native archive is
rebuilt and linked, while ordinary Rust suites remain the preceding bridge
commit's evidence rather than fresh tests claimed here.

This verifies worker execution on Mac and iOS source compatibility. New native
packages, linked app builds, sealed measurement/quality job selection and actual
physical iPhone execution still require their own checks. Existing M7 device
qualification is unchanged. Real checkpoint quality, latency, process peaks and
physical-device qualification remain open in the [M8 plan](mlx-implementation-plan.md).

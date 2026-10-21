# Resident native MLX Qwen graph and sessions

`metal::Model::load_mlx` loads the validated local [Qwen source](mlx-qwen-source.md)
into Toklet's existing Rust/Metal graph. Callers must explicitly choose
`ProjectionMode::MlxAffine`; default sessions reject MLX models, and ordinary
Half/Q4 models reject this mode. No MLX runtime dependency is added. Current
architecture/configuration and geometry restrictions remain enforced.

Packed weights retain their original unsigned words and separate affine metadata.
The loader streams directly into final GPU buffers; it retains neither a complete
FP32 expansion nor a second packed model copy. Separate metadata buffers round up
to four-byte lengths. Floating exclusions, norms and biases convert directly to
Half, with rejection of nonfinite or overflowing conversion. Tied embedding/output
storage is owned once. Upload enforces the [qualified affine GPU value
domain](mlx-metal-affine.md); broader scalar-readable artifacts can still reject
at this GPU boundary.

The graph uses Half activations, KV and floating weights. Packed embedding gathers
round to Half; packed projections reconstruct in FP32 and use compensated reduction.
The final tied packed output projection uses the original packed parameters and
returns FP32 logits. Floating projection exclusions use the existing compensated
Half operators. The selected numerical tests use compensated normalization. The
existing reference and online attention selections remain explicit.

The shared shader ABI now includes input/output row strides and input offset in
scalar units. The encoder checks shapes, alignment, buffer extents and integer
conversions before dispatch. This handles padded intermediate rows and the final
row's nonzero MPS input offset without transposing or copying the model. Pipeline
selection occurs during loading. Model-owned pipelines and buffers survive through
the session's checked command completion; dependent operators are encoded into the
existing chunk command without a new host wait per operation. Diagnostic and token
readback use existing session publication rules.

Logical weight accounting includes actual padded Half buffers, original packed
bytes and padded metadata; the tiny fixture uses 18,152 bytes. Loader admission
adds the existing 64 MiB library/metadata estimate and a 64 KiB stream buffer.
Source-file admission also applies independently. Session admission adds actual KV,
scratch and readback allocations through the existing machinery. This is bounded
logical accounting, not measured process peak or a mobile budget. The loader's
`conversion_seconds` currently covers payload reading, validation, upload and
conversion together; `compilation_seconds` is separate, while `load_seconds` is
total elapsed loading time. No pure conversion-time or loading-speed claim follows.

## Checks on Apple M1 Pro, 22 October 2026

The [frozen independent Half oracle](mlx-half-oracle.md) predates this target graph
execution. Its manifest, all data/source identities and numerical gates remain
unchanged. Five cases at three chunk sizes compare exactly 525 unique diagnostic
snapshots (35 stages per case/chunk) with checked shapes and 74,784 values. Cases
include rotary offset, changed future token, one token and capacity 32. Reference
attention's internal score/probability arrays are not exposed in these snapshots;
the independent fixture retains them. Ordinary and diagnostic final-row logits,
fully cached no-op behavior and the 38-event trace also pass.

State tests cover four teacher-forced greedy steps with both attention choices and
both full-logits/token-only readback modes (16 selected IDs), pending samples,
exact and changed prefixes, reset and actual three-token partial prefill. Loader
checks reject insufficient source budget, float-to-Half overflow and unsupported
affine values, then reload and run valid data. Both model/mode mismatch directions
reject. A finite, domain-admitted packed embedding row that overflows Half gather
fails without advancing the four-token committed prefix or publishing a new result;
restoring the cached prefix recovers the original distribution.

Two private GPU tests additionally verify nonzero input offset, padded input/output
strides, Half/FP32 projection and gather guards, and session transaction outcomes
for both readback choices. The transaction test completes real staged GPU work,
then injects Nonfinite and Completion errors. Completion is fault injection, not an
observed Metal command failure; it requires reset/replay. Completed nonfinite work
preserves the valid prior prefix under the existing contract.

All four prior affine GPU tests pass after the shader ABI change: 54 cases,
864 completed calls and 390,096 exact values. The existing compensated Half
regression also passes 2,520 stage arrays and its warmed allocation/state checks.
These are correctness regressions; no timing repetitions or performance gate is
claimed. Full repository formatting, strict CPU/Metal-feature Clippy, ordinary
CPU/Metal-feature Rust tests, all three offline MLX fixture checks and the seven
portable Half-oracle tests are recorded in the
[task receipt](../benchmarks/results/2026-10-22-native-mlx-metal-model-v1.json).
Protected historical CPU binaries are unchanged.

This qualifies the selected tiny graph/session configuration on M1 Pro. Real
checkpoint quality, loading/steady-state process peaks, performance, CLI/app
selection, iPhone execution, wider architectures and discovery/downloads remain
open in the [M8 plan](mlx-implementation-plan.md).

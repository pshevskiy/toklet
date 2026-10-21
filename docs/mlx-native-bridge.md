# Native MLX model loading through the C bridge

The additive ABI-v1 function `toklet_model_load_mlx` accepts the existing bounded
UTF-8 directory path, `TokletOptions`, model-handle output and error record. It
loads a compatible local MLX Qwen model using the resident Rust/Metal loader. The
header is [apple/include/toklet.h](../apple/include/toklet.h); existing C layouts,
ABI version and CPU/Half/Q4 entry points remain unchanged.

Sessions created from this explicit MLX handle select MlxAffine projections,
compensated normalization, online attention, Half KV and full-logits readback.
Selection uses the loaded model's explicit origin, once at session creation.
Packed representation is not inferred from an absent Toklet Q4 format flag.
Unsupported builds/platforms return UNSUPPORTED without a CPU fallback. Supported
builds preserve the source/geometry/payload/domain and memory-admission checks of
[the resident model loader](mlx-metal-model.md).

The new loader uses the existing panic/error boundary, clears a valid output slot
before options/path/loading checks, and publishes a handle only after complete
loading succeeds. Handles retain the same thread-affinity, synchronous callback,
non-reentrant session and model/session lifetime requirements as other native
functions. Existing session generation, state/history, cancellation, score, reset
and release calls work with the MLX model; no MLX-specific numerical code is added
to the C boundary.

The explicit Mac Rust test compares a committed native score with a direct session
using the same immutable model and independently reduces loss in FP64. It then
releases the public model handle, checks that the live session retains ownership,
generates four fixed IDs, verifies exact pending state and cancellation/input
rejection preservation, resets, and confirms final owner release. The C smoke
program's `mlx` argument invokes the new symbol through the C header and preserves
its existing ABI layout, scoring, callback reentry, history, retained-model,
cancellation and reset checks. Both use the frozen local tiny packed artifact.

Ordinary tests reject empty/NUL/invalid UTF-8 paths, a bad ABI version and null
output without accessing a model/GPU; unavailable builds also verify explicit
UNSUPPORTED and a cleared output handle. Formatting, strict CPU/Metal-feature
Clippy, ordinary Rust tests, the explicit Mac GPU Rust/C runs and an iOS Rust
compilation check are retained in the
[task receipt](../benchmarks/results/2026-10-22-native-mlx-native-bridge-v1.json).
Historical CPU binaries remain unchanged.

This adds the native engine entry point for Apple callers. The harness still needs
an explicit UI/worker selection and a rebuilt package before app-level MLX trials.
iOS compilation is not physical iPhone execution or device qualification. Real
checkpoint quality/performance/process peaks and broader architecture/download
support remain open in the [M8 plan](mlx-implementation-plan.md).

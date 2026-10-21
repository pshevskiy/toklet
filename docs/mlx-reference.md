# Full-model scalar reference import for MLX Qwen

The explicit [reference importer](../src/mlx/reference.rs) loads a supported local
MLX-format Qwen artifact into Toklet's existing FP32 scalar model. It provides a
complete-graph correctness reference for later resident GPU integration.
Production performance paths do not select this diagnostic expansion.

## Ownership and admission

The importer first opens the validated [artifact source](mlx-qwen-source.md),
checks the total expanded FP32 weights, and admits that total plus the largest
packed-tensor temporary, a 64 KiB stream buffer and a 64 MiB metadata estimate.
It expands one affine tensor at a time using the established scalar FP64 affine
multiply/add with FP32 weight storage. Temporary original packed buffers are
released after each tensor. Floating payloads stream directly into their final
FP32 arrays. It retains no second source model and no redundant tied output table.

All size arithmetic is checked, admission precedes expansion, and malformed or
nonfinite payloads return an error before an executable model is published.
The final model uses the unchanged scalar graph and can run the existing bounded
cached sessions. Sessions, diagnostic traces, tokenizer/application state and any
physical process peak require separate admission/measurement. This path does
expand the full model once and is not the device-fit or performance backend.

## Independent tiny model fixture

The [exporter](../tools/reference/mlx_qwen.py) starts from the immutable tiny-v1
FP32 weight arrays. Actual pinned MLX quantization creates packed embeddings and
projections, with a floating excluded O projection and a per-module 8-bit/group-64
down-projection override among 4-bit/group-32 matrices. Independent NumPy unsigned
word extraction and FP64 affine arithmetic reconstruct FP32 reference weights;
the existing NumPy graph generates all 39 intermediate arrays for five cases,
including changed future tokens, a rotary offset, one token and full capacity.

The fixture also saves actual mlx-lm Qwen2 FP32 logits on both CPU and GPU for the
four zero-offset cases. These separately validate the canonical reconstruction
and graph under the unchanged tiny-v1 `1e-5 + 1e-4 * abs(reference)` gates. Offset
references come from the independent NumPy graph. No native-MLX offset result,
complete-model Half GPU parity, real-checkpoint quality or latency is claimed.

The [fixture](../tests/fixtures/mlx-qwen-v1/manifest.json) contains 195 NumPy trace
arrays and eight native MLX logits arrays in a 320,128-byte payload, plus the local
safetensors/config snapshot. Fixture hashing is stdlib-only; ordinary Rust tests
require neither Python/MLX nor a model download or GPU.

The initial exporter recorded source/producer hashes after numerical work. Astra
identified that as a provenance gap. That original export and source are retained
as historical evidence rather than silently receiving corrected identities.
The corrected export admits one bounded manifest byte buffer against the accepted
tiny-v1 SHA before parsing, captures selected producer hashes before numerical
imports/execution, records the consumed array identities and checks all of those
paths again before publishing. The selected files do not claim a complete Python
or shared-library dependency closure. Regeneration preserves identical numerical
array and safetensors bytes; its manifest records the corrected provenance.

## Verification scope

The [portable tests](../tests/mlx_reference.rs) compare every trace array and all
saved native logits, exercise cached chunking/offset/exact-prefix/reset execution,
check expansion-budget thresholds, and reject nonfinite floating payloads and
affine reconstruction overflow before publishing a model.

All four focused tests, formatting, strict CPU/Metal-feature Clippy, 130 CPU tests
(8 ignored), 144 Metal-feature tests (63 ignored) and both offline MLX fixture
checks pass. Both actual oracle exports completed; final evidence uses the
corrected producer/source pinning. Protected historical CPU binaries are unchanged.
The [task receipt](../benchmarks/results/2026-10-22-native-mlx-qwen-reference-v1.json)
binds the final sources, fixture, checks and retained original export.

Complete resident GPU graph execution and state/quality/memory qualification,
CLI/native-app integration, additional architectures, discovery/downloads and
Mac/iPhone checkpoint trials remain open in the [M8 plan](mlx-implementation-plan.md).

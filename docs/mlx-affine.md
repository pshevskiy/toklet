# Native MLX affine tensor reader and scalar reference

This is the first native Rust primitive for importing MLX-format models. It does
not link MLX or Swift into Toklet. It does not yet load a complete MLX checkpoint,
run a new model architecture or qualify GPU model execution. The wider
[implementation plan](mlx-implementation-plan.md) remains open.

## Storage profile

The compatibility profile is `mlx-affine-u32-le-v1`, based on pinned MLX 0.32.3
oracle execution and the upstream storage implementation. This is Toklet's reader
profile label, not an on-disk MLX magic number or a universal future-format claim.
The [Rust reader](../src/mlx/affine.rs) borrows three contiguous little-endian byte
slices; it imposes no pointer-alignment requirement and allocates no expanded
matrix. Public `Layout` construction checks dimensions and byte-size arithmetic.

| Tensor | Dtype | Rank-two shape | Layout |
| --- | --- | --- | --- |
| `weight` | U32 packed codes | `[rows, columns × bits / 32]` | Unsigned codes in feature order, low bits first; rows contiguous |
| `scales` | F16, BF16 or F32 | `[rows, columns / group_size]` | One scale per contiguous row-local group |
| `biases` | Same dtype/shape as scales | `[rows, columns / group_size]` | One affine bias per group |

Supported bit widths are **2, 3, 4, 5, 6 and 8**; supported affine groups are
**32, 64 and 128**. Columns must be a positive multiple of the group size;
there are no padded partial groups or Toklet Q4 headers/row padding. Codes for
3/5/6 bits continue across byte and U32-word boundaries. Tensor byte lengths
must match exactly, with no trailing bytes. Safetensors dtype/rank/name/index
validation and global/per-layer quantization configuration belong to the later
model loader, not this raw-slice matrix API.

Each weight represents `scale × unsigned_code + bias`. Negative scales are valid;
zero scales permit arbitrary codes, with reconstruction governed by ordinary
floating-point zero/sign behavior. Metadata must be finite. These rules differ
from Toklet Q4's positive-scale and canonical constant/tail conventions.
Group-wise **`biases`** are distinct from an optional linear-layer **`bias`**.
Quantized embedding row lookup and a tied output projection must preserve the
same shared packed storage when they are implemented.

Floating quantization modes such as MXFP/NVFP, per-tensor mixed schemes, higher
ranks and unsupported dtypes/parameters are outside this first primitive.
They must not be interpreted as affine data by a future loader.

## Explicit numerical reference

Storage decoding and execution rounding are separate contracts. This reader's
scalar reference converts metadata to FP32, computes the affine expression in
FP64, then rounds the reconstructed weight to FP32:

```
w_ref = FP32(FP64(FP32(scale)) * FP64(code) + FP64(FP32(bias)))
y_ref[row] = FP32(ordered_FP64_sum(FP64(w_ref[row, col]) * FP64(input[col])))
```

This is an independent storage/matvec oracle. It does not reproduce every MLX
CPU/GPU reduced-precision dequantization or fused matrix-multiplication rounding.
In particular, an F16 output may overflow where the FP32 reference stays finite,
and FP32 multiply-then-add can differ from the FP64 reference after cancellation.
The regression suite preserves these distinctions explicitly. Future native
kernels require their own accumulation/output policy and numerical/quality gates;
this task does not relax any existing tolerance.

`value()` checks indices and rejects nonfinite reconstructed FP32 results.
`matvec()` checks shapes and finite input before work, validates all row results
before publishing any caller output, and rejects reconstruction/result overflow.
Its two-pass scalar implementation is a correctness reference, not an optimized
kernel. Bit width and metadata type select statically dispatched operations
outside matvec's scalar loops; group metadata is loaded once per row-local group.

## Independent fixtures and checks

The [frozen fixture](../tests/fixtures/mlx-affine-v1/manifest.json) contains 54
cases: all six bit widths × three groups × three metadata dtypes, each with two
256-column rows. Literal packed patterns enumerate every unsigned code in order;
the second row complements the packed bits. No production packing/unpacking
helper creates the reference bytes.

The [exporter](../tools/reference/mlx_affine.py) uses pinned MLX 0.32.3 on CPU and
Metal GPU. For chosen exact-product scales, both backends' dequantization with
metadata promoted to FP32 agrees bit for bit and is independently checked against
the literal code semantics. The fixture separately retains native-dtype CPU/GPU
outputs as rounding witnesses, without treating them as the Rust FP32 contract.
Native-dtype CPU/GPU arrays differ in 31 of 54 cases; all promoted-FP32
reference arrays agree. These differences remain explicit, with no tolerance
waiver or claim of reduced-precision equivalence. Ordered Python binary64
accumulation supplies the matvec reference. The manifest
records versions, backend scope, exporter/oracle-binary identity and exact array
hashes; ordinary Rust tests need neither Python, MLX nor an Apple GPU.

[Seven Rust tests](../tests/mlx_affine.rs) cover every code, byte/word/group/row
boundaries, all saved reconstruction/matvec bits, independent metadata groups,
negative/zero/signed-zero/subnormal/extreme values, unaligned slices, malformed
layouts/lengths, NaN/Inf, checked overflow and preservation of every output element
on failure. The first targeted run exposed an expected-value bug: computing the
BF16 minimum subnormal via FP32 `powi(-133)` yielded zero. The fixed expectation
uses its exact FP32 bit pattern; parser output and numerical gates are unchanged.
The original failure/source remain retained.

The first oracle export and final provenance export have identical 332,592-byte
array payloads. The later export identifies the exporter and MLX extension module in the manifest.
No model was downloaded or evaluated, and no latency, quality, mobile-memory or
whole-model architecture-support claim follows from these primitive checks.

## Upstream references

Storage/specification reference: MLX core commit
`1f8e74e3f12f31365464a6867c6579f0e9b29d85`, inspected through the pinned MLX Swift
0.32.3 source closure. This source reference is distinct from the identified
Python oracle binary; no identical-build assumption is made.

- [Quantization/dequantization operations](https://github.com/ml-explore/mlx/blob/1f8e74e3f12f31365464a6867c6579f0e9b29d85/mlx/ops.cpp)
- [CPU unsigned bit extraction](https://github.com/ml-explore/mlx/blob/1f8e74e3f12f31365464a6867c6579f0e9b29d85/mlx/backend/cpu/quantized.cpp)
- [Metal dequantization and typed arithmetic](https://github.com/ml-explore/mlx/blob/1f8e74e3f12f31365464a6867c6579f0e9b29d85/mlx/backend/metal/kernels/quantized.h)
- [Quantized embeddings and linear layers](https://github.com/ml-explore/mlx-swift/blob/19601207e9a0de51e03ee6ec0c3c5f3784275075/Source/MLXNN/Quantized.swift)

## Task validation

Formatting, strict all-target Clippy and full CPU/Metal-feature tests pass with
isolated build products: **118 CPU tests (8 ignored), 130 Metal-feature tests
(59 ignored)**, plus all seven focused reader tests and offline fixture hashing.
The ignored hardware/model suites remain separate. Both protected historical CPU
release executables retain their hashes. The
[task receipt](../benchmarks/results/2026-10-21-native-mlx-affine-v1.json) binds
source/check/raw identity, retained attempts, producer identity and final review.

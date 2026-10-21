# Native MLX Qwen artifact source

The [Rust source](../src/mlx/qwen.rs) validates a local MLX affine Qwen2
safetensors snapshot and streams its original tensors into caller-owned buffers.
This is the loading boundary for M8.2; it does not yet build an executable model,
run a new GPU operation, qualify a checkpoint, or expose search/download UI.

## Graph and storage contract

The importer reuses Toklet's strict existing Qwen2 graph validation: tied output,
SiLU, QKV bias, ordinary rotary positions, full causal attention and supported
configuration fields. It bounds the logical name inventory before constructing
it. Config JSON is limited to 1 MiB and duplicate keys are rejected recursively,
including nested quantization instructions. Local single-file and indexed shards
reuse the existing validated safetensors parser, plain shard filenames, exact
index-to-shard mapping, duplicate-tensor rejection, complete-file offsets and
16 MiB aggregate header ceiling.

`quantization` and `quantization_config` are accepted aliases; if both appear,
their JSON values must agree exactly. The supported declaration contains global
integer `bits`/`group_size`, optional `mode: "affine"`, and optional exact module
paths such as `model.layers.0.mlp.up_proj`. A module instruction is `false` to
exclude it or an object with its own complete bits/group size and optional mode.
Unknown module paths/options and other modes reject. Equivalent declarations
with different JSON structure are deliberately outside this first source API.

Following the pinned upstream loader's `.scales` presence guard, global defaults
and per-layer settings describe matrices with stored scales. Floating matrices
without scales remain floating, even under a global quantization declaration.
Packed tensors under an explicit exclusion reject. Required `.weight`, `.scales`
and `.biases` names/shapes/dtypes are checked together; partial or stray payloads
reject. A projection's ordinary `.bias` remains a separate floating vector.

Logical matrices are row-major `[rows, columns]`; packed weights are U32
`[rows, columns * bits / 32]` and scales/biases share floating dtype and shape
`[rows, columns / group_size]`. The [affine storage contract](mlx-affine.md)
defines supported widths/groups/metadata and reconstruction precision. Floating
tensors retain their F16/BF16/F32 source precision. No model-wide FP32 or Half
copy is made by this source.

## Ownership, admission and failure

`Source` owns validated metadata and open local shard handles. Its immutable
`TensorSpec` inventory resolves storage choice at load time; numerical execution
should retain concrete buffers instead of looking up names per operation.
`visit_float` streams finite FP32-decoded values through the shared 64 KiB buffer.
`read_affine` reads into three exact caller-provided allocations, then validates
finite scales/biases. It checks destination sizes before writing. On read,
metadata or consumer failure, destination buffers can be partial and must be
discarded before publishing a model. Opening alone validates headers/config,
not every floating payload; successful reads perform payload validation.

An optional redundant `lm_head` must have the exact same storage layout and byte
contents as the embedding. Its weight and, where packed, scales/biases are hashed
through bounded streaming. Different floating dtypes, signed-zero encodings or
quantized representations reject even if a semantic equivalence might exist.
This conservative check is not a general numerical-equivalence test. The public
ownership plan omits the redundant head: embedding and tied output share one
future resident tensor. Payload identity is not a model-hub provenance claim;
immutable downloaded-snapshot identities are a later boundary.

Admission counts the original planned tensor payloads once, plus a 64 MiB
metadata estimate and 64 KiB stream buffer. It checks overflow and a caller limit
within the initial 6 GiB ceiling. This is not a measured physical peak or a phone
budget. GPU alignment/padding, any conversion, sessions/KV/scratch, tokenizer and
application allocations need their own admission before execution. Metadata is
allocated during opening, before payload admission; payload weight buffers are
never allocated by `Source::open`.

## Verification and remaining work

The [portable tests](../tests/mlx_qwen.rs) exercise mixed floating/packed weights,
single/indexed shards, every supported bit width and metadata dtype, per-layer
overrides/exclusions, aliases, exact tied-head ownership, streaming beyond
64 KiB, finite checks, truncation, wrong destination sizes, malformed graph/config,
duplicate instructions, metadata shapes/dtypes, bad index paths/mappings and
budget boundaries. Literal Q4 patterns independently check preserved packed
bytes/reconstruction; the separate affine suite retains the pinned MLX oracle.

The first focused run had five passes and two failures in synthetic fixture
construction: repacking an already packed shape and a malformed view rejected by
the safetensors serializer. Test setup was corrected; loader behavior and
numerical gates were unchanged. The failure logs/source remain retained.

All eight focused tests pass, together with formatting, strict CPU/Metal-feature
all-target Clippy, 126 CPU tests (8 ignored), 138 Metal-feature tests (59 ignored)
and offline affine-fixture hashes. Both protected historical CPU binaries remain
unchanged. The [task receipt](../benchmarks/results/2026-10-21-native-mlx-qwen-source-v1.json)
binds the completed checks, exact sources, raw logs and review.

M8.2 remains open for resident packed projection/embedding kernels, complete model
assembly, numerical/quality/state/memory checks and explicit CLI/native-app
selection. Wider architecture families, device-aware discovery/downloads and Mac/
iPhone qualification remain open in the [M8 plan](mlx-implementation-plan.md).

Upstream configuration/presence-guard references: mlx-swift-lm commit
`bb0d5b048f02e14dbf8f2d7bb1a20911a295470a`,
[BaseConfiguration.swift](https://github.com/ml-explore/mlx-swift-lm/blob/bb0d5b048f02e14dbf8f2d7bb1a20911a295470a/Libraries/MLXLMCommon/BaseConfiguration.swift)
and [Load.swift](https://github.com/ml-explore/mlx-swift-lm/blob/bb0d5b048f02e14dbf8f2d7bb1a20911a295470a/Libraries/MLXLMCommon/Load.swift).
Support is deliberately narrower than every format accepted by that runtime.

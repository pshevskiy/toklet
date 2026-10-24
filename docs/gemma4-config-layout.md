# Gemma4 E2B configuration and tensor-layout adapter

The named `mlx::gemma4` module now validates the inspected E2B text graph and its
MLX storage metadata. It is separate from `dense::Architecture` and the existing
loader/compatibility registry. Its returned plan describes structure; it does
not make Gemma4 runnable or eligible for search, download, Chat or Expert.
The [receipt](../benchmarks/results/2026-10-25-gemma4-config-layout-v1.json)
binds the code, original metadata and checks.

## Exact supported metadata branch

The input is the converted wrapper/config, single-shard index and raw JSON
header already authenticated by the [G1 inspection](gemma4-e2b-inspection.md).
The copied [fixture manifest](../tests/fixtures/gemma4-e2b-metadata-v1/manifest.json)
binds these unchanged source bytes at repository revision
`238767527555cb75a05732a84dff5d6ba0dd6809`. Neither the fixture nor this API
contains/opens the3.55GB tensor payload. Its full integrity remains unverified.

Callers must explicitly select `TextOnly`. The wrapper must name the known
conditional-generation architecture and recognized audio/vision configurations;
the text graph must match the audited E2B branch. Unknown/missing text options,
alternate depth/head/PLE/shared-KV geometry, MoE, K-equals-V, bidirectional/bias
semantics, different activation/norm/rotary/cap and conflicting quantization
declarations reject. The exact branch uses hidden1536,35 layers, Q8/KV1,
local D256/window512 and global D512/128 rotated coordinates. Producers0..14
own K/V, later local/full layers reuse13/14, and their MLP widths double from
6144 to12288. Per-layer input width is256 and the output table is tied.

All three in-memory JSON buffers have a1MiB bound and reject duplicate fields.
The shared cold JSON helper retains existing dense-loader behavior. Counts and
dimensions require JSON integers; floating graph options compare exact finite
FP64 values, so equivalent exponent spellings do not create different graphs.
No numerical tolerance is introduced by that comparison.

## Layout, names and exclusions

The adapter resolves540 logical text tensors into1096 physical descriptors:
278 affine matrices and262 BF16 vectors/scalars. Every matrix uses the existing
[MLX affine layout](mlx-affine.md), four bits/group64 with U32 words
`[out,in/8]` and BF16 scales/biases `[out,in/64]`. Learned norm vectors retain
their direct convention. Consumer layers have no private K/V projection or K
norm tensors; a redundant tied output table is not silently accepted. Named
tables, layer plans and physical offsets are cold metadata for later loading;
forward execution need not resolve tensor names in numerical loops.

Every text name, dtype, shape, byte span and index-to-shard entry must match.
The single shard is explicitly `model.safetensors`. All header spans, including
excluded components, must cover the advertised payload exactly with no gaps or
overlap. Integer/product/offset arithmetic is checked before building the plan.
Known audio/vision tower/projector roots are explicitly separated; unknown roots
reject. Their internal modality graph options and values are not qualified or
executed by the text-only plan. Root generation/special-token metadata likewise
grants no tokenizer, stop or chat semantics.

The derived text payload is2,603,981,382 bytes; excluded component payload is
946,370,944 bytes. The large per-layer embedding accounts for1,321,205,760 stored
bytes. These are authenticated-header layout counts, not payload-value checks,
resident allocation plans, process peaks or fit estimates. The declared131072
positions remain configuration metadata, with no qualified real-engine capacity
and no131072-output-plus-prompt capability.

## Checks and remaining work

Seven offline methods bind the exact fixture and check every layer's producer/
rotary/MLP plan, physical inventory and byte totals. Mutations cover every missing
text field, representative changed graph options, explicit modality/quantization
policy, unexpected text/private-KV names, index mismatches/path variants, wrong
dtypes/shapes, gaps/overlap, overflow, truncation, duplicate fields, buffer limits
and equivalent finite float spellings. They allocate metadata only.

The first preliminary run is retained with two failed methods. A direct JSON
number comparison incorrectly distinguished exponent spellings under
`arbitrary_precision`; one negative case also named a nonexistent excluded
tensor. Finite numeric field semantics and the actual captured tensor name
correct those harness/parser mistakes. The original metadata and all numerical
gates stay unchanged; failed source snapshots/logs remain unaccepted.

Formatting, strict default/Metal Clippy,279 CPU/301 Metal-feature tests and both
iOS compile targets pass. Existing dense families retain their loader/registry
behavior. Astra reviews exact source/evidence before commit. No GPU numerical
trial or Gemma payload download is represented as executed here.

The actual artifact/payload reader, finite value/reconstruction admission,
general scalar or resident Metal graph, real checkpoint quality/generation,
tokenizer/chat/stop profile, allocation/device fit and app integration remain
open. The [bounded synthetic scalar graph](gemma-text-graph-scalar.md) is a
separate32-position reference; its geometry cannot qualify E2B by inspection.
Both original short-GPU and long-rotary discrepancies remain unwaived.

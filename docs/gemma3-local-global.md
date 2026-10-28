# Gemma3 local/global composition

B3 selects the public [Gemma3 270M four-bit text export](https://huggingface.co/mlx-community/gemma-3-270m-it-4bit/tree/ff1143e3a10547c9f2129e94ca37059b096b23f4),
revision `ff1143e3a10547c9f2129e94ca37059b096b23f4`. The first task adds a
strict cold metadata inspector. It does not load weights, execute a graph, enable
Chat or qualify a device. The full payload and tokenizer files have not been
downloaded; only metadata and an exact safetensors header range were inspected.

## Concrete plan

| Source fact | Selected value |
| --- | --- |
| Hidden / query / KV widths |640 / 1024 / 256|
| Layers / heads / head width |18 / 4 query,1 KV / 256|
| MLP / vocabulary |2048 / 262144|
| Attention |Five local layers then one global, repeated three times|
| Local window / declared positions |512 / 32768; neither is an execution admission|
| Rotary |Ordinary split-half; local theta10000, global theta1000000|
| Normalization |Four hidden and two Q/K unit-offset gamma vectors per layer; no V norm|
| Other graph facts |Tanh-GELU, query scale1/16, BF16-rounded embedding multiplier25.25|
| Storage |127 affine4/group64 matrices with BF16 coefficients;109 raw BF16 norms|
| Tied output |One physical262144×640 table serves embedding and output|

The header has490 physical tensors and236 logical tensors. Its53,490 bytes put
the payload at byte53,498, unaligned to4 bytes. All150,885,632 advertised payload
bytes have contiguous, shape-derived extents. The full150,939,130-byte file's
advertised SHA256 is
`bf8b442cff5dd7c9ac6b5695310d5b66a4d0a4c3780f5c774fb90cd0c352bd9d`;
this is not a locally verified payload checksum.

`mlx::gemma3::inspect` accepts bounded, duplicate-free config/index/header
buffers, validates the exact selected graph and resolves immutable layer and
physical/logical tensor plans. Changed or unknown graph fields, scaling,
softcaps, modalities, tensor shapes, shards, aliases or extents reject. Norm
coefficients stay raw BF16; inspection performs no `1+gamma` conversion. The
single tied table is counted once; no physical head alias is introduced.

A small private `gemma_metadata` module shares concrete metadata types and
validators with the existing Gemma4 inspector. Gemma4 publicly reexports its
previous types; its helper bodies and payload behavior remain unchanged.
Family-specific graph options and inventories remain separate. The shared
mapper is deliberately restricted to internally constructed Q4/group64/BF16
plans. It is not a general architecture registry or numerical dispatcher.

The common model shape already permits hidden640 with query width1024. A
regression witness preserves that capability, avoiding a needless shape-system
redesign. No executable family or native Chat admission changes in this task.

## Numerical decisions belong to the next task

Gemma3 is not the existing two-pre-norm SiLU dense graph or Gemma4's direct-norm,
PLE/shared-KV graph. Each attention/MLP branch is pre-norm, operation, post-norm,
then residual. Q/K norm precedes RoPE. Full and local attention have distinct
rotary bases and visibility; every layer owns its own cache.

The [Transformers reference](https://github.com/huggingface/transformers/blob/c27795950091cdaf1b0e32c4aed58618fb41ffa0/src/transformers/models/gemma3/modeling_gemma3.py)
widens gamma before adding one. [Pinned MLX](https://github.com/ml-explore/mlx-lm/blob/1f7057ca530f8a594d4aab50bac41a2ed5e5c5c7/mlx_lm/models/gemma3_text.py)
adds in the weight dtype and clips residual sums for FP16 activations. Those
profiles are not bitwise interchangeable. Later fixtures must freeze coefficient
rounding, activation stores and overflow behavior before target outcomes. Never
silently pre-fold gamma into Half. The conversion card's older sanitizer also
does not establish exact quantized runtime ownership or successful old-runtime
loading; the physical header and tied configuration determine this plan.

At local query512, key0 must be excluded while key1 and key512 remain visible;
global attention retains key0. Existing CPU window views support immutable
old-ring/new-chunk unions. Native append-only caches do not acquire rolling
semantics from those CPU tests. Resident window state needs its own transaction,
staging and lifetime checks.

## Remaining tasks

1. Freeze independent tiny CPU graph/operator fixtures with unequal widths,
   unit-offset norms, both rotary branches and local/global visibility. Exercise
   chunks larger than a tiny window, failure-before-commit and reset/replay.
2. Compose resident packed projections, explicit norm precision, postnorm scratch
   and local/full cache ownership. Preserve existing families and Reference
   arithmetic; prove the resident rolling cache independently.
3. Freeze one bounded real acceptance protocol. The prospective544-token scope
   crosses window512 with selected rows511/512/513/543 and two chunk partitions.
   It still needs implementation, reference and controller review before runs.
4. Acquire and verify actual payload/tokenizer bytes; check raw generation,
   resource admission and whole-process6GiB fit. Keep production app/codec,
   phone, full declared window and adoption quality in B5/M8.

One544×262144 FP32 logits owner alone costs570,425,344 bytes. Source/native
collection must count it and retain only prospectively selected witness rows
or streaming hashes where specified. Static payload/KV estimates never establish
whole-process fit or speed.

Tokenizer metadata's `<image_soft_token>` ID262144 lies beyond this numerical
head; future text-only input must reject it before mutating model state. No
tokenizer coverage, exactly-once BOS or decode fidelity is qualified here.

## Cold checks

Formatting and strict default/Metal all-target Clippy pass, along with415 default
and293 Metal-library tests, five new Gemma3 controls and14 Gemma4 regression
tests; suites overlap. Seven owned groups close normally. The first full run
caught an omitted genuine index field: this export includes exact
`total_parameters:268098176` as well as total_size. The corrected schema validates
both and rejects missing/changed counts; failed v1 is retained. Source metadata
and numerical policies are unchanged. Astra authenticates3,914 current bindings,
retained output identities and closed groups, approving cold-plan retention and commit.

See [compact evidence](../benchmarks/results/2026-10-28-gemma3-cold-plan-v1.json),
[TODO](../TODO.md) and [improvement record](improvements.md).

# Q4 v2 real model artifacts and independent references

Both previously selected Qwen checkpoints now have source-bound group-8 Q4 v2
artifacts and independent reconstructed-model references frozen before GPU parity
or held-out evaluation. The fixed candidate, source checkpoints and quality gates
remain as specified in [Q4 v2](../q4-v2.md); v1 rejection evidence is preserved.
This qualifies reference production, not real GPU behavior, quality or speed.

| Checkpoint | Packed matrices / bytes | Largest staging, bytes | Predicted resident weights, bytes |
| --- | ---: | ---: | ---: |
| 0.5B |168 /357,837,312|4,358,208|630,251,436|
| 1.5B |196 /1,310,208,256|13,762,624|1,777,247,428|

Packing uses the committed row-streaming packer at parent`b18ce0f`. Both complete
source/config identities match the pinned checkpoints; helper endpoints remain
unchanged. Packed sizes exactly reproduce the frozen layout predictions. The
complete packing process records lifetime native charges14,598,792bytes after
small and17,253,048bytes after larger. These are source conversion charges, not
inference memory or packing-performance claims. Logical resident-weight values
remain predictions until the separate GPU admission check.

## Independent reconstruction and frozen cases

[The independent reader](../../tools/reference/qwen_storage_v2.py) decodes
8-byte records in128-row stripes, with finite metadata, canonical nibble/tail/
padding and version checks. It hashes consumed bounded bytes before interpretation,
checks exact tensor/source/config inventories and rechecks retained source descriptor,
path and packed endpoints. The single tied Half table is cached; other tensors
use bounded stripes/current matrices. There is no full live FP32/FP64 model.
Original v1 oracle helpers and gates remain unchanged.

[The exporter](../../tools/reference/qwen_storage_v2_oracle.py) validates the four
original source cases and their complete228-array reference before execution.
It adds one fixed authored17-token pattern,`([9707,11,358,2776]*5)[:17]`, at absolute
offsets0and5. These two shape cases activate tiled prefill; offset5 itself is
absolute-position evidence, not nonempty-prefix reuse. GPU irregular-chunk checks
will separately exercise cached prefixes. Inputs are frozen before GPU/held-out
results and do not use calibration or held-out parameter search.

Each model freezes six cases/342 full arrays at embedding, layers0/1/last, final
normalization and all-row logits. The original four retain unquantized FP32 source
identities/argmax; the new two have null source-array SHA and only reconstructed
FP32/FP64 argmax. No unavailable unquantized reference is invented. The numerical
gate remains `max(1e-4,2*pooled_FP32_FP64_error,2*equivalent_prefix_error) +
1e-3*abs(reference)`, evaluated independently before GPU use.

| Model | Independent logical admission, bytes | Completed native lifetime charge, bytes | Full reference manifest SHA256 |
| --- | ---: | ---: | --- |
| Small |906,264,640|433,210,280|`6413875965b4bf364a376988f66bf114519e80ef7e33fcd515bfb79cfbaf8527`|
| Larger |1,211,203,648|724,534,280|`79ce188cbc130dd2c97f64df4a83f27b5e4055ad4b93dd94ee5ddf08c3ae3d1c`|

Both fit6GiB. Complete manifest serialization and native memory observation
precede ordinary publication; a rejected/unavailable observation leaves only a
pending manifest and failed receipt. Native lifetime charge complements explicit
allocation accounting, without establishing complete shared physical memory.
The [probe selector](../../tools/reference/qwen_storage_v2_probes.py) preserves
all full-array hashes/gates and selects64coordinates plus all available argmax
positions. Both committed probe sets contain342arrays, with48,156/48,092payload
bytes and separately pinned manifests. Full references remain local artifacts.

## Checks and review

Astra found a decode-stripe lifetime overlap that undercounted temporary ownership.
Explicit deletion of expanded/codes/scale/bias/views after copying each stripe
prevents the previous expanded array from surviving the next allocation. A weakref
regression checks this before the next allocation and after return; the declared
9-byte-per-padded-value stripe allowance is conservative. This changes memory
ownership, preserving separate FP32 reconstruction arithmetic and every gate.

All51reference Python checks pass, including seven new reader/memory/provenance
checks and five new probe checks. Eight manifest mutation variants reject version,
quantizer, exclusions, source, duplicates/missing entries, remapping and shapes;
authenticated v1 matrix bytes under a v2 manifest reject. Source descriptor/path
replacement, exact-byte hashing, canonical storage, publication failure and
unavailable source argmax are checked. One ordinary Rust integrity test validates
both pinned probe sets, every payload/shape/index/gate and reference availability
without GPU/downloads. Formatting, Metal-feature all-target Clippy and whitespace
pass. Engine/source/qualified executables are unchanged by this tool/data task.

Packing, reference numerical work, builds/tests and large hashes run sequentially;
no new real-model GPU or held-out work runs before freezing references. The
[receipt](../../benchmarks/results/2026-10-14-q4-v2-real-oracle-qualification-v1.json)
binds exact artifacts, source/helpers, independent references/probes, completed
native memory and check logs. Astra accepted final code, all1,788bindings,684full/probe relationships,
gate/reference availability, native/check evidence and283local links with no findings. Next qualify both GPU geometries,
full load/session budgets and unchanged loss/task/performance gates.

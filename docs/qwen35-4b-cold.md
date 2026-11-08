# Qwen3.5-4B pinned cold text recipe

Q1.A is implemented and Astra-approved in its cold metadata scope. This adds metadata and packing validation for
`avan-ag/Qwen3.5-4B-Uncensored-MLX-4bit` at
`fb98fbc59c71a567f3e23435386c9e93951261db`. It introduces no numerical backend,
qualified context, model/session memory admission or ordinary catalog/Chat route.
Gemma remains the first active quality task; the next model stays Qwen3.5-4B,
then Qwen3.5-9B and the selected smaller-model track.

The [publisher card](https://huggingface.co/avan-ag/Qwen3.5-4B-Uncensored-MLX-4bit/blob/fb98fbc59c71a567f3e23435386c9e93951261db/README.md)
identifies a conversion from DavidAU's modified/fine-tuned source using MLX-LM
0.31.1. This lineage description is not a reproduced transformation or a quality
result. Original, modified and quantized identities need separate later gates.

The recipe requires exact config/header hashes and the declared single-shard
length, plus an explicit text-only choice. All924physical tensors must match the
fixed graph and representation; missing, unknown, duplicate, overlapping,
truncated, mistyped or changed-shape descriptors fail. The dense Qwen3 admission
path remains unchanged and rejects this configuration.

| Serialized owner | Count | Original payload bytes |
| --- | ---: | ---: |
| Affine4 codes, group64 |249 |2,102,394,880 |
| BF16 scales and affine biases |498 |262,799,360 |
| BF16 vectors, including A_log/dt_bias |153 |350,208 |
| BF16 depthwise convolution `[8192,4,1]` |24 |1,572,864 |
| Total |924 |2,367,117,312 |

There are426logical tensor owners. The embedding and tied output head alias one
physical matrix. Reconstructed logical parameter count is4,205,751,296, including
unquantized vectors/convolutions and excluding quantizer metadata. This exceeds
the strict under-4B tier; Qwen3.5-4B remains the separate selected second model.

The layer recipe distinguishes24gated-delta/recurrent owners and24convolution
histories from8attention-KV owners. Declared delta geometry has16key heads and
32value heads, each dimension128. Full attention declares16query heads,4KV
heads, head dimension256 and64rotated dimensions, plus a gated query projection.
The original Q projection has8192output rows, including its output gate. MRoPE
sections `[11,11,10]`, theta10,000,000 and the declared FP32 recurrent-state dtype
are preserved as metadata. No recurrence, convolution, gate, rotary or norm
arithmetic is implemented by this recipe. The existing three-tap Metal
convolution is not a four-tap executor.

The source header is119,829bytes; payload begins at byte119,837, which is
unaligned even for16-bit reads. Ranges stay byte offsets. A later loader must
use validated little-endian reads into suitably aligned owned buffers; this
plan provides no typed mapping/alignment guarantee. The full file is declared
as2,367,237,149bytes with SHA256
`0dfd6217f56cadcc373c90283d58d5e673833650fed1d6c299313fb536999043`.
Only metadata was read, so that complete payload digest is not authenticated.
The 6 GiB engine ceiling remains unchanged; tensor byte totals do not establish
actual load/session/scratch/framework memory or device fit.

Eight focused CPU controls, fresh618CPU/750Metal tests (34/296ignored),
formatting and strict CPU/Metal all-target Clippy pass. The first compile hit
JSON macro recursion; splitting the fixed specification resolved it. The check
script accidentally repeated CPU Clippy under its Metal label; a separate
explicit `--features metal` run passes, with both original and corrected logs
retained. The release probe binary and3,908source/fixture/Cargo pins are unchanged
across checks and metadata collections.

Two fresh-process metadata cohorts each use3warmups and10observations. Median
local read/hash/parse/drop times are2.527ms and2.537ms, with physical high-water
5,439,896B and5,325,184B. Each observation reads122,853metadata bytes and zero
tensor payload bytes. Other Gemma jobs were running, so these are metadata
observations, not an engine benchmark or a device/model fit qualification.
Astra approves with no findings; [verification](../artifacts/qwen35-4b-cold-2026-11-08/verification.json)
and [raw observations](../artifacts/qwen35-4b-cold-2026-11-08/metadata-observations-1/checks.json)
retain the exact scope and commands.
[Checks/review/commit](../artifacts/qwen35-4b-cold-2026-11-08/publication-checks.json)
records publication separately.

Payload loading, independent tiny/full hybrid graph/state checks,
tokenizer/template/stops, capacity/failure/cancellation, source quality, request
metrics and performance qualification are separate implementation work.
iPhone runtime remains deferred.

[Metadata provenance](../artifacts/qwen35-4b-preparation-2026-11-08/metadata-1/checks.json),
[header inventory](../artifacts/qwen35-4b-preparation-2026-11-08/header-inventory.json),
[contract](../artifacts/qwen35-4b-cold-2026-11-08/acceptance-contract.json),
[iteration](../artifacts/qwen35-4b-cold-2026-11-08/iteration-notes.json).

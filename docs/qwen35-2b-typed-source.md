# Qwen3.5-2B typed-source diagnostic

This is an explicit structural diagnostic for the selected 2B model. It adds a
source plan, authenticated typed storage and binding to the existing hybrid graph.
It does not qualify the real model's values, contexts, tokenizer, quality,
performance or managed application route. No ordinary loader or CLI admission is
added by this change.

## Source and ownership contract

`mlx::qwen35::two_b` admits exactly 24 layers: eighteen recurrent layers and six
full-attention layers, hidden width 2,048, intermediate width 6,144 and tied
248,320-row embeddings. Recurrent key/value head counts are both sixteen. The
child recipe is affine Q4/group64 with BF16 scales/biases, 97 BF16 vectors,
eighteen original F32 `A_log[16]` vectors and eighteen BF16 convolutions. It has
320 logical owners, 694 physical tensors and a 1,059,315,904-byte payload.

The caller supplies exact config/header/container lengths and SHA256 identities
through `ArtifactPins`. These identities bind bytes; they are not proof of source
lineage or a conversion certificate. Metadata inspection checks bounded sizes,
complete tensor coverage, exact shapes and config semantics. Loading consumes
and hashes the complete stream, rejects trailing data and publishes no store on
failure. Unaligned source bytes are decoded explicitly.

Original F32 values remain F32, including bits that BF16 cannot represent. Only
BF16 small tensors widen to F32. Existing 4B/9B descriptor eligibility and public
`PayloadBytes` remain unchanged. F32 metadata handling is a separate opt-in;
legacy Gemma/LFM recipes retain their BF16 restrictions.

The runtime learned-tensor count is 1,060,412,672 requested bytes. Allocation
capacity, load/upload overlap, framework/context resources, session state and
application owners are additional. This count is not a full-model fit result.
Consuming Metal upload retains packed GPU matrices and small CPU parameters.
Recurrent, convolution and attention state remain CPU FP32; this is a hybrid
executor. No numerical hot loop or new kernel is introduced.

## Independent checks and remaining gates

A separate tiny fixture uses recurrent key/value ratio one, tied output and
original F32 decay values. It binds all stages and state to independent FP64 and
staged FP32 references under the existing `atol=1e-5`, `rtol=1e-4` gates. Checks
cover chunking, offsets, cancellation, failure/retry, reset and memory admission.
The producer imports no target-engine output. Tiny checks do not qualify the
complete 2B source or all-workload memory use.

Formatting, strict CPU/Metal Clippy, 781 CPU and 940 Metal-feature tests,
twelve portable tests per Python mode and the explicit tiny GPU storage check
pass. The first Clippy attempt found a missing legacy-example enum arm; its
failure is retained and the corrected check passes. All 4,126 Rust/fixture inputs
remain frozen. Astra approves scoped diagnostic retention with no findings.
[Review/publication](../artifacts/qwen35-2b-source-planning-2026-11-13/publication-checks.json).
[Checks and scope](../artifacts/qwen35-2b-source-planning-2026-11-13/qualification-1.json),
[dispatch and queue](../artifacts/qwen35-2b-source-planning-2026-11-13/parallel-work-plan-1.json).

The selected TheCluster BF16 source is pinned at `78efacde`. Its declared Hauhau
parent is pinned at `2bcf35c1`, whose inspected inventory exposes GGUF artifacts.
Those metadata declarations do not prove tensor-value lineage or the publisher's
normalization/layout/dtype transformations. Authenticate those semantics and
reserve the complete campaign before real conversion or payload qualification.
The original numerical, quality, 6 GiB memory and request-performance gates remain
in force. The next dependent milestone is real source/conversion qualification,
followed by graph/CLI qualification if its prerequisite gates pass.

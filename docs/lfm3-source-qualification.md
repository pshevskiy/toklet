# LFM3 BF16 source qualification

Status: both complete objects authenticate and all tensor comparisons pass.
Astra approves scoped source-audit retention. Numerical execution,
quality, engine memory admission, performance and managed adoption remain open.
This work adds standalone source-audit commands, without changing engine kernels,
loaders, inference CLI policy or the application.

The candidate is `mchen04/LFM3-1.7B-Heretic-Abliterated-MLX-bf16` at
`88636d45d403e7c8ab6a40e85c9d0fa90cc5a253`. Its declared modified parent is
`Vortecks/LFM3-1.7B-Heretic-Abliterated` at
`dc37c60c02a35603456c85c2aecd7bc22b23b89c`. Both objects contain
3,457,635,448 bytes, including 286 BF16 tensors and 1,728,801,536 parameters.
The authenticated child SHA256 is
`b6ff17ac993dddf0f6f51bd872b12ae1437f4453a54a0b4878cedc7864fbc07e`;
the authenticated parent SHA256 is
`2e44203266e14abedc1d56a0d2379840abf5456729e30ad878117f015e503e28`.

The [prospective contract](../artifacts/lfm3-source-2026-11-10/prospective-contract-1.json)
was frozen before complete payload outcomes. Fresh API, source assets and range
headers authenticate 25 Git/LFS asset identities. The web reader failed; bounded
HTTPS fetches with exact range/size checks succeeded. This is recorded in the
[metadata receipt](../artifacts/lfm3-source-2026-11-10/metadata-verified-1.json).
No remote Python code is imported or executed.

## Exact conversion and ownership

The 13 depthwise convolution declarations change from parent `[2048, 1, 7]` to
MLX `[2048, 7, 1]`. Their flat BF16 byte order is identical: both address
`channel * 7 + tap`, because the exchanged axis has extent one. All remaining
matrix and norm shapes must match exactly; no tensor may be missing, extra,
overlapping or separated by an undeclared payload gap. This includes every one
of the 119 learned norm tensors.

The child config differs only by removing the parent's `auto_map` and adding
`model_file: "lfm3.py"`. The generation config, tokenizer config, tokenizer and
chat template are byte-identical between parent and child. Stock LiquidAI MLX
metadata is pinned separately; stock payloads and stock-to-modified quality are
not authenticated or scored by this comparison.

The [producer](../tools/reference/lfm3_source.py) admits a fixed schema and bounded
metadata, verifies the entire retained child before and after transfer, compares
parent bytes to child tensors by name in at most 4 MiB blocks, and requires a
complete parent hash, EOF, successful TLS/HTTP receipt and closed transfer.
Intermediate tensor journal entries remain unauthenticated until the complete
object passes. Only one candidate payload is retained; the parent is discarded.

The independent [offline verifier](../tools/reference/check_lfm3_source.py)
reconstructs coverage, validates the raw transport receipt and owned process/
clock/resource ledger, and rehashes all child blocks, tensors and the whole child
file. It seals the retained witness bytes. It cannot independently rehash the
discarded parent; parent identity relies on the checked full-stream producer.
Recorded block hashes are not Merkle membership proofs.

The audit has a separate 512 MiB aggregate producer/curl/controller guard and
2 GiB disk floor. This is a CPU source audit, not model execution or engine
memory admission. The engine ceiling stays 6 GiB, with existing codec/application
allowances unchanged. Audit elapsed time includes network transfer; it is not an
inference performance result. Engine telemetry overhead remains unqualified.

## Graph work that follows

The pinned code requires 24 layers: 13 convolution, six full-attention and five
sliding-attention layers. Hidden width is 2,048, MLP width 7,680, vocabulary
128,000, and attention uses 16 query/four KV heads of width 128. The following
source semantics need independent numerical fixtures and implementation:

- Seven-tap causal depthwise convolution over `B * x`, with six history rows;
  an independent `C` gate precedes the output projection.
- Sandwich RMS norms around operator and MLP residual branches, plus per-head
  Q/K RMS norms and one sigmoid output gate per query head.
- Full attention uses theta 1,000,000 and a half-rotary zero-frequency tail;
  sliding attention uses theta 10,000 and a 256-token rotating window. Positions
  remain absolute across cache rotation and chunk boundaries.
- Source BF16 rounding boundaries, FP32 normalization/softmax/nonlinear work,
  and the source's split-half rotary convention must be qualified explicitly.
- `model.embedding_norm` normalizes the final hidden state after all layers,
  despite its name. The output projection uses the tied embedding table.

Existing three-layer LFM2 fixtures and the specialized three-tap Metal kernel
do not qualify LFM3. The scalar leaf already accepts variable kernel sizes;
reuse it and qualify seven-tap fixtures plus six-row transactional history. The
[read-only inspection](../artifacts/lfm3-source-2026-11-10/next-convolution-inspection-1.json)
records existing kernel/fixture coverage. Start with independent history/chunk fixtures;
qualify one mechanism before composing the graph or optimizing Metal dispatch.
Keep exact prefix, pending-token, cancellation/failure and capacity semantics.
Before adoption, require the full original 10,000 scored targets/100 tasks/caps
for stock-to-modified and modified-to-native comparisons. Byte equality alone
provides no numerical, quality or runtime qualification.

## Checks and review

Nineteen portable tests pass across producer/schema/stream and offline
witness/resource validation, including actual corrupt/truncated bytes, a second
4 MiB block with a short tail, tensor reorder, singleton-axis equivalence,
duplicate metadata, coverage holes/overlaps, clock/PID/resource controls and
qualification rejection. The complete offline replay passes all 1,014 block and
286 tensor digests, including the complete child hash. Twenty-eight actual CLI
controls reject coherent altered identity/layout/transport/resource/clock claims
and an actual changed byte in an independent APFS clone. The corrupted clone was
removed after checking; its exact reconstruction recipe remains. Original bytes,
source inputs and canonical capture evidence stay unchanged. Astra approves
scoped retention with its RSS finding resolved.

Astra found that coherently understated final RSS values could pass the original
consumer despite higher sampled RSS. The corrected verifier reconciles each
final peak with the same PID's sampled resident-size maximum; it keeps physical
and RSS scopes separate. Portable controls cover each PID and all three together;
a coherent altered actual campaign also rejects. A fresh complete offline replay
and all 28 CLI controls pass, with both ordinary and optimized Python passing all
19 portable tests. Producer, primary captures and contracts are unchanged.
[Counterexample](../artifacts/lfm3-source-2026-11-10/astra-resource-counterexample-1.json)
and [historical source snapshot](../artifacts/lfm3-source-2026-11-10/source-epoch-1/manifest.json)
retain the initial finding and exact earlier inputs.

[Parent owner/resources](../artifacts/lfm3-source-2026-11-10/parent-checks-1.json),
[offline replay](../artifacts/lfm3-source-2026-11-10/offline-checks-2.json),
[witness seal](../artifacts/lfm3-source-2026-11-10/witness-seal-2.json),
[negative controls](../artifacts/lfm3-source-2026-11-10/negative-checks-2.json),
[temporary-clone recipe](../artifacts/lfm3-source-2026-11-10/negative-controls-2/temporary-clone-removal.json).

The complete parent campaign closes with a conservative sum of final process
RSS peaks of 247,611,392 bytes and a sampled aggregate RSS peak of 244,580,352
bytes (producer, curl and controller). Corrected offline replay closes at 117,194,752 bytes
conservative final RSS sum. Both remain below the separate 512 MiB audit guard;
final physical peaks are unavailable. The canonical candidate remains the sole
retained complete model object for this audit.

Fresh formatting passes. Prior strict Clippy, 681 CPU/827 Metal/nine example
checks and native68 export/C smoke checks are reused only after exact matches of
4,034 Cargo inputs, 421 native inputs and ten package output hashes.
[Reuse evidence](../artifacts/lfm3-source-2026-11-10/reused-engine-checks-1.json).

[Astra review](../artifacts/lfm3-source-2026-11-10/astra-review-result-2.json),
[commit/evidence](../artifacts/lfm3-source-2026-11-10/publication-checks.json).

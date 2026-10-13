# Streamed larger Half and Q4 references

The bounded tensor reader reproduces all 228 existing small-model independent
Half payload hashes and numerical gates exactly. It then freezes 228 complete
larger Half arrays and 228 reconstructed-Q4 arrays at layers0/1/27, embedding,
final normalization and all-position logits, before any larger GPU comparison.
These are independent CPU references, not GPU geometry or quality acceptance.

Only the tied Half embedding table is cached. Other Half tensors stream from
the pinned BF16 source through bounded 1024-row stripes; each Q4 matrix is
independently reconstructed into one temporary FP32 matrix through 128-row
decode stripes. No source mapping or complete FP32/FP64 weight model is retained.
The existing `qwen_fp16.forward` implementation, storage boundaries and Half
gate derivation are unchanged. Q4 reconstruction checks explicit version,
little-endian metadata, nibble order, separate FP32 multiply/add, canonical
constant/tail/padding rules and finite metadata. Hand-computed 65-column fixtures
exercise codes, tails, padding and malformed metadata independently of packing.

Accepted source full-manifest hashes, exact four cases, three layers and 228-array
inventory are checked before forward execution. Source/config checkpoints match
the earlier preselection. The reader hashes the exact bytes it parses or decodes;
it rechecks retained source descriptor, pathname, config, inventory and packed
payloads. All numerical and memory-observation helper identities are bound and
rechecked. An exclusive receipt and disjoint fresh output prevent collisions.
Manifest serialization finishes in an exclusive pending file. Completed native
memory admission precedes publication of the normal manifest; rejected/unavailable
observation leaves only the pending file and failed receipt.

Half limits retain the frozen pooled independent Half-versus-source FP32/FP64
envelope and equivalent-prefix policy, with relative limit1e-4. Reconstructed-Q4
limits use independent BLAS FP32/FP64 arithmetic and matching stored boundaries:
`max(1e-4, 2*pooled_precision_error, 2*equivalent_prefix_error) + 1e-3*abs(reference)`.
They are selected before GPU evaluation. This numerical Q4 oracle does not measure
lossy quality; the unchanged 10000-target and 100-task gates remain separate.
It does not resolve the separately recorded tiny offset17 decoder discrepancy.

Larger Half admission is 1,159,593,984 bytes, Q4 admission1,205,469,248 bytes.
Each includes the 466,747,392-byte tied Half table, one current matrix, bounded
BF16/FP64/decode stripes and a fixed 512 MiB application/reference allowance.
Completed native lifetime peaks are 724,665,376 and 724,878,416 bytes. The small
parity export peaks at432,309,040 bytes. All are below6,442,450,944. Native charge
complements explicit allocation admission; it does not establish performance.

The final v3 exports include completed memory observation before publication.
Earlier v1/v2 receipts and exact helper snapshots are retained for audit; their
payloads/gates agree, but they are not the final publication implementation.
Bounded probe export preserves complete-array identities and gates while selecting
64 coordinates plus all source/FP32/FP64 argmax coordinates. The larger Half/Q4
probe manifests and payloads are committed for ordinary integrity checks.

Fourteen new tests cover exact-byte/provenance races, source descriptor replacement,
config endpoints, stripes/tied ownership, reconstruction/padding, output collisions,
frozen reference admission, pooled/equivalent-prefix gates, publication failure and
argmax insertion outside the64 base probes. All34 reference Python tests pass.
Rust engine bytes are unchanged from the preceding checked commit. The
[qualification binding](../../benchmarks/results/2026-10-13-streamed-storage-oracle-qualification-v1.json)
retains complete/probe hashes, receipts, source snapshots and checks.

Astra reviews source, test and actual export evidence before commit. Larger GPU
geometry, short native parity, matched loss/task quality and M5 release acceptance
remain pending. No held-out feedback changes Q4 v1 or any frozen acceptance gate.

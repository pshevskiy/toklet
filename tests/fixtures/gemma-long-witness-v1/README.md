# Full-sequence tiny long oracle

This compact fixture was frozen before incremental reference execution or a
native context-ceiling increase. The full-sequence producer preserves the
historical batch algorithm, with WITNESS-only1025-position admission and
projection/gather batches no larger than32. It uses no native or incremental KV
implementation. See [the reference record](../../../docs/gemma-long-witness.md).

`manifest.json` is346994 bytes, SHA256
`bee93a03eb8f3ff6fe1e46919621d8702c1f4eb1a7ada850edb6a987fc750465`.
`arrays.bin` is6974448 bytes, SHA256
`2940891fd6a32d22c7fa574aad633d8adc3681a1281c23722a97e1111e938ec3`.
Each descriptor specifies its original tensor shape, dtype, selected absolute
positions, byte offset/extent and SHA. Data retains head-major ordering where
the source has that layout; position selection preserves every other dimension.

Both original/future complete logits and original producer K/V arrays are kept
at all1025 positions. Other original intermediates keep positions
0/1/2/3/4/7/8/16/17/31/32/510/511/512/513/1023/1024. Other future intermediates
keep the final position; its preceding complete1024-position causal prefix is
checked separately. The tiny local window remains4, not E2B's512.

Two cold complete captures have identical files. Each sequence file is109051800
bytes; original SHA256 is
`0318d2bec8719300097886ddd25d0a92d85e9e57a4244a6ed0b189ab4d721256`,
future SHA256 is
`defd719079f57e44146d0056aad3223891313169f28e1143e1e505316bc1572d`.
Full files and per-state manifests remain in
`artifacts/m8-gemma-long-witness-checks-v1/graph-a` and `graph-b`.
The checks receipt is800933 bytes, SHA256
`17f7a8a2f7eb779204d0527b74b226ed3541c209cdef1af574038b9a21ed5297`.

Independent replay checks360 complete states,360 frozen32-prefix comparisons,
180 future-prefix comparisons and48 selected scalar operator boundaries per
capture. Eight coherent corruption controls pass. This is mathematical tiny
reference preparation; native context remains32 and real E2B numerical/quality,
phone fit and full-model-window qualification remain separate.
Selected rotary/attention checks use captured q_norm/QKV, and softcap uses the
captured head; embedding/PLE/scaling alone derive directly from frozen weights.

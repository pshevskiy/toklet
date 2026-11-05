# Granite3.1-1B-A400M on Mac Metal

5 November 2026. Apple M1 Pro. The reviewed experiment uses the original
218-tensor BF16 checkpoint at revision
`0da7a48b0276d500ce5922fd2b33944091fc6c09`. The payload and tokenizer
identities were verified before execution. Toklet streams conversion into
resident Half and uses compensated projections and normalization with Reference
attention. The stock MLX worker retains BF16.
[Source and arithmetic](granite-moe-next.md).

The frozen 27-token prompt produced token 38 (the answer `4`) and stop token
0 on both runtimes. Both final-layer selected expert sets matched. Independent
ranking and selected-softmax replay passed on the native score witnesses.
Stock retains its own route slot order.

| Decision prefix | Stock/native ID | Mean absolute logit difference | Maximum |
| ---: | ---: | ---: | ---: |
| 27 | 38 / 38 | 0.46235 | 2.01336 |
| 28 | 0 / 0 | 0.17879 | 1.16677 |

All 49,155 logits per decision are in the evidence. The native session matched
bitwise across chunks 1 and 32 for both decisions and a synthetic 512-token
prefix. The capacity rejection preserved history, position and distribution.
Reset reproduced the original distribution.

| Memory on Mac | Bytes |
| --- | ---: |
| Resident Half weights | 2,669,259,756 |
| Allocated F16 KV at 512 | 25,166,496 |
| Native admission, chunk 32 | 2,906,501,564 |
| Native admission, chunk 1 | 2,896,620,872 |
| Native peak RSS | 2,532,081,664 |
| Native sampled peak physical footprint | 2,719,173,152 |
| Stock sampled peak physical footprint | 2,990,754,792 |

The stock and native workers ran in separate processes under a 6 GiB cap.
This covers the pinned checkpoint, the selected two decisions and native
state at capacity 512. Broader fidelity, held-out quality, larger context,
performance and Chat/device adoption remain separate gates.
[Evidence](../benchmarks/results/2026-11-05-granite-moe-real-v1.json).

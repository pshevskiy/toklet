# Original Llama3 reference and short alignment

A public BF16 mirror at `alpindale/Llama-3.2-1B-Instruct`, revision
`f92201d8185818a9d079b3b52efdab4b68bdd17f`, matches Meta revision
`9213176726f574b556790deb65791e0c5aa438b6` for the weight payload and four text/config
Git blobs. The acquired 2,471,645,608-byte checkpoint verifies SHA-256
`1ff795ff6a07e6a68085d206fb84417da2f083f68391c2843cd2b8ac6df8538f`.
Both revision-pinned Hub replies are archived beside the acquisition receipt.
[Meta inventory](https://huggingface.co/api/models/meta-llama/Llama-3.2-1B-Instruct/revision/9213176726f574b556790deb65791e0c5aa438b6?blobs=true),
[mirror inventory](https://huggingface.co/api/models/alpindale/Llama-3.2-1B-Instruct/revision/f92201d8185818a9d079b3b52efdab4b68bdd17f?blobs=true).

Acquisition keeps TLS verification enabled, refuses existing destinations and
checks remaining payload plus 1 GiB free space before and during the large stream.
It publishes one verified inode without a second full-file copy. This reference
asset is not a new production BF16 model backend.

The independent NumPy graph validates 146 original BF16 tensors, widens exactly to
FP32, gathers only needed embedding rows, retains one widened projection and
uses vocabulary tiles for the tied output. RMSNorm, scaled split-half RoPE,
causal GQA, SwiGLU and residuals keep FP32 intermediates. It does not reuse the
Half-intermediate graph as an original-FP32 baseline. Numerical provider files
and source inputs are pinned; reference and native processes run sequentially.

The frozen first 18 raw corpus IDs define prefixes 1/4/8/17 with next target
`ids[n]` at logit row `n-1`, no BOS insertion or chat formatting. Before outcomes,
Astra froze every-logit batch/prefix tolerance `1e-3 + 1e-4*abs(prefix)` and target
NLL difference ≤ 1e-4. Saved FP32 rows independently rescore in a stdlib-only
checker. All four comparisons pass: maximum logit difference 5.8174e-5 and
maximum target-NLL difference 2.4153e-5.

The audited native MLX8/Half profile scores the same targets. Its four-case mean
NLL difference is −0.01427516 nats, descriptive only. This does not decide the
original 10,000-target or 100-task adoption gates. Original peak RSS is
2,670,002,176 bytes; native lifetime physical peak is 1,520,928,520 bytes. Each
process remains below 6 GiB. No throughput measurement is claimed.

All 100 unchanged task budgets were re-tokenized with the pinned plain Llama
policy. `HumanEval/32` needs 310 prompt + 256 reply = 566 positions; the other 99 fit 512.
The frozen 1024-target loss blocks need 1025 positions. A bounded quality execution
range must be implemented and checked before full collection; no cases or output
reserves are dropped to fit the current 512 route. The original 0.05-nat/2-percentage-
point thresholds remain unchanged.

[Evidence](../benchmarks/results/2026-11-03-llama3-original-short-v1.json),
[frozen short inputs](../benchmarks/llama3-original-short-v1/manifest.json).
The physical iPhone 14 Pro is unavailable; the available Pro Max is not substituted.
Phone Settings/session qualification remains open while Mac work proceeds.

## Long-row alignment

The explicit quality reference projects selected hidden rows through the tiled
tied head, capped at 1,025 input positions. Its ordinary entry point stays capped
at 512. Corpus preparation freezes 10,001 IDs once, checks the previous 1,025 IDs
byte-for-byte and separately hashes the first 1,026 IDs. The last ID supplies the
held-out target after a 1,025-token input. Ten unchanged blocks cover targets
1–10,000 exactly once with position resets and one-token overlaps.

At prefixes 512, 513, 1,024 and 1,025, all eight saved original rows pass independent
stdlib replay. Maximum logit discrepancy is 3.9101e-5 and maximum target-NLL
discrepancy is 1.9024e-5, within the previously frozen gates. Peak RSS is
3,181,379,584 bytes. The native report digest was pinned before inference; its
four-case mean NLL difference of 0.02133 is descriptive. Fifteen focused controls,
prior short-row regression and Astra review pass. The initial checker argument
failure is retained. Commit: `7ac121e`.
[Evidence](../benchmarks/results/2026-11-03-llama3-original-long-v1.json).

Full loss and task adoption outcomes remain separate requirements. No app or
physical-phone qualification follows from these alignment checks.

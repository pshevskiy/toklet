# Gemma E2B8-bit feasibility decision

5 November 2026. **Do not download weights or implement an8-bit backend for the
current Mac owner layout.** The header-based text lower bound plus existing
512-position/chunk32 numerical owners, codec and friendly-app policies exceeds
6GiB by102,686,350bytes. This is a planning rejection, not a measured Q8 peak or
a claim that a different memory design could never fit.

The [immutable Hub artifact](https://huggingface.co/mlx-community/gemma-4-e2b-it-8bit/tree/03dcf209f3f549b4075e7191e77cf69b3d48e1b2)
is affine8/group64 with BF16 scales/biases. Its model card identifies original
`70af34e20bd4b7a91f0de6b22675850c43922a03`; that is a publisher lineage assertion,
not an independently reproduced conversion. Text config exactly matches the
pinned Q4 artifact, and tokenizer SHA256 is the same
`cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f`.

Only metadata, tokenizer and HTTP-range tensor headers were fetched. Immutable
Git blob identities, complete header offsets/shapes/dtypes, index ownership and
container lengths validate. The two containers total5,866,581,158bytes; selected
text comprises1096tensors and4,918,123,078bytes. Tied main/PLE tables are counted
once;1415 modality or unused tensors are excluded. Header verification does not
authenticate weight payload bytes that were never downloaded.

| Current-layout term | Bytes |
| --- | ---: |
| Selected raw text lower bound | 4,918,123,078 |
| FP32 KV and staging | 20,054,016 |
| Host/backend scratch excluding logits | 75,925,032 |
| Working/published logits | 67,108,864 |
| Two token histories | 4,096 |
| Model/session metadata and allowances | 100,773,408 |
| Codec and response stream | 1,077,936,128 |
| Friendly app and transition staging reserves | 285,212,672 |
| Total | 6,545,137,294 |
| Existing policy | 6,442,450,944 |

Numerical terms derive from the checked567-position generation receipt, reduced
to512 for global KV, attention scratch and histories; local512 capacity and
chunk32 stay fixed. Q8-specific guards, descriptor differences and BF16-vector
widening are not counted, making the text term optimistic. No current Q8 native
owner exists; these are explicit design estimates rather than allocator facts.

The [existing loader](gemma-resident-weights.md) does not require a second full
CPU model: it streams code/scale/bias chunks, then converts vectors. A hypothetical
Q8 chunk at16384rows and8960columns requires155,975,680bytes of source staging;
load admission also has separate64MiB context,64MiB cold metadata and128KiB hash
stack policies. Those transient terms cannot rescue the rejected steady state.
The ledger assumes no old model remains live; replacement overlap adds costs.

Initial metadata attempts failed on Python CA setup, a wrong one-shard
assumption, a TLS timeout and a redirect body larger than an8-byte curl ceiling.
Preserved failures precede the successful verified HTTP1 range reads; no TLS
verification was disabled and no model weight payload was fetched.

Astra approves the header-only planning rejection. No Rust inputs changed, so documentation/content,
header inventory and arithmetic replay are the relevant checks. The
[Q4 rejection](gemma-full-tasks.md), historical numerical failures and absent
ordinary Gemma app/phone qualification remain. [Machine evidence](../benchmarks/results/2026-11-05-gemma8-feasibility-v1.json).

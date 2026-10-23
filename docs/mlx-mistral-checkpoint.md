# Immutable Mistral full-attention checkpoint

The selected public candidate is `mlx-community/Mistral-7B-Instruct-v0.3-4bit`
at revision `a4b8f870474b0eb527f466a03fbc187830d271f5`.
The complete snapshot is now acquired and admitted as data. This establishes
checkpoint identity, not native Mistral execution or answer quality.

The 4,077,480,215-byte safetensors file has SHA256
`d7182d886f93a9efa73d6d556ee33ffe839a4e09e15965eeef46f295ab5b503d`.
Its 84,239-byte header describes 743 tensors: 226 packed U32 matrices, 452 F16
scale/bias arrays and 65 F16 norms. Payload size is 4,077,395,968 bytes. Embedding
and untied output each have logical shape `[32768,4096]`; the source remains
packed. The 32-layer full-attention configuration has hidden 4096, FFN 14336,
32 query heads/eight KV heads of dimension 128, rotary theta 1e6 and epsilon 1e-5.
`sliding_window` is null. Sliding-window and other Mistral variants remain
unsupported until separately implemented and qualified.

| File | Bytes | SHA256 |
| --- | ---: | --- |
| config.json | 721 | `efc145d2f53d6d6ef2639be0556b4ecf6faac6cf8a7199ad2b7bb5ebcf443bdd` |
| model.safetensors.index.json | 52,381 | `525940738b16d210fdfd692a24752d050c15a50cbe208dd82b06204d640d687b` |
| tokenizer.json | 1,961,548 | `e553af6fff7d7ad76e830608b218c5c0b0822998d5a1a96099a74cd3c1cb1a49` |
| tokenizer_config.json | 138,039 | `e8fbcfef34d855dba2c7f1d91a2c5470cf493d00ecfd06488ce21e45325bc8ae` |
| special_tokens_map.json | 414 | `6fa06efa2785e450051989a6f8fb4416b10149ded485ddd3f127a40734f5cfd0` |
| tokenizer.model | 587,404 | `37f00374dea48658ee8f5d0f21895b9bc55cb0103939607c8185bfd1c6ca1f89` |
| README.md | 612 | `22112370a61fd1cb3b22946979100f3b74fd3d45f6c445393ebd7a8d744f70d6` |

The data-only retry uses 244 disjoint HTTP ranges, four bounded workers and
16 MiB maximum range bodies. Exact 206 status, Content-Range, total length,
identity encoding and captured command diagnostics are checked. Ordered assembly
rehashes each consumed chunk and the complete file before exclusive publication.
Small files also match their immutable Git blob or LFS identities. All 252 owned
commands complete cleanly; all 252 process groups are subsequently observed
absent. Seven portable transfer checks and a 1 MiB actual range pilot pass.

Astra independently audited the source, eight published files, 252 command rows,
756 diagnostics and exact ordered reconstruction before admitting receipt SHA
`88fde696338c7c58f8a84ee1d744a27ced5e8a36c8417ca5cc55f3b78d738bf9`
for the [CPU reference export](mlx-mistral-oracle.md). The
[checkpoint receipt](../benchmarks/results/2026-10-23-mistral-checkpoint-v1.json)
retains the source copies, complete identities and owned-group absence result.
An earlier serial transfer failed with curl18 and published no model; its logs
and subsequent absence checks remain retained. The mixed earlier ignored
candidate inspection remains invalidated, with the collision recorded.

This acquisition does not qualify the app's URLSession path. A separate Mac
public search/download test is running against the unchanged catalog and a
private pinned test inventory; it executes no numerical inference and grants
no Mistral fit or compatibility label. Numerical reference, target execution,
app Chat integration, latency and held-out quality remain separate gates.
The phone remains paused at the user's request.

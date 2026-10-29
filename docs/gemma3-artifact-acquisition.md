# Verified Gemma3 270M artifact acquisition

29 October 2026. B3d1 acquires the exact public
[Gemma3 270M four-bit export](https://huggingface.co/mlx-community/gemma-3-270m-it-4bit/tree/ff1143e3a10547c9f2129e94ca37059b096b23f4)
at revision `ff1143e3a10547c9f2129e94ca37059b096b23f4`. The new local directory
is `models/MLX-Gemma-3-270m-it-4bit-ff1143e`. Acquisition authenticates data;
it does not enable numerical execution or Chat in the application.

Eleven files total190208344 bytes: eight previously pinned metadata files plus
the original model, tokenizer JSON and SentencePiece model. The model container
is150939130 bytes with SHA256
`bf8b442cff5dd7c9ac6b5695310d5b66a4d0a4c3780f5c774fb90cd0c352bd9d`.
All data fits the200MiB transport cap. Payload sizes/SHA256 and canonical Git
LFS pointer identities are checked separately; payload bytes are not hashed
as pointer files.

Three credential-free bounded HTTPS transfers follow immutable revision URLs.
Each completes with HTTP200 after its redirect, empty stderr and a closed owned
process group. Kernel file-size limits and bounded raw logs prevent oversized
unadvertised output. Exact hashes, all eleven files and the original header/index
must pass before a single exclusive atomic directory publication. An existing
destination, including an empty directory or symlink, is not replaced. Retained
successful partials and published payloads share the same device/inode and
linkcount2; those paths do not duplicate physical payload storage.

Astra independently rehashes the current input/runtime identities, all published
files and nine raw outputs before and after its readback. The exact53490-byte
header matches the frozen metadata fixture. Its490 indexed tensors cover
150885632 contiguous payload bytes starting at file offset53498:127 original
affine triples and109 BF16 norm vectors, with no physical output-head duplicate.
All55936 norm words are finite; range hashes are retained. Finiteness and layout
do not establish normalization arithmetic or quantized execution correctness.

The [conversion card](https://huggingface.co/mlx-community/gemma-3-270m-it-4bit/blob/ff1143e3a10547c9f2129e94ca37059b096b23f4/README.md)
records MLX-LM0.26.3. Its pinned
[model/sanitize code](https://github.com/ml-explore/mlx-lm/blob/6c876ca5d18fee91bda8d45fd0c42852241a4d5a/mlx_lm/models/gemma3_text.py),
[conversion](https://github.com/ml-explore/mlx-lm/blob/6c876ca5d18fee91bda8d45fd0c42852241a4d5a/mlx_lm/convert.py)
and [loading code](https://github.com/ml-explore/mlx-lm/blob/6c876ca5d18fee91bda8d45fd0c42852241a4d5a/mlx_lm/utils.py)
do not fold gamma. That positive source chain supports an explicit raw BF16 gamma
interpretation for the next selected-artifact reference. The original Google
checkpoint revision remains unknown, separately from this quantized artifact's
authenticated identity. Numerical/text qualification and original unquantized
adoption quality still require their own evidence.

No model, tokenizer, downloaded code or numerical graph runs in this acquisition
task. Declared context32768 and local window512 remain source metadata. The next
source/native protocol must freeze IDs, arithmetic profiles and selected rows
before the bounded544-token Mac journey. This acquisition grants no real-model,
application, iPhone or performance admission. It changes no Rust code and needs
content/link/hash/whitespace checks rather than another engine benchmark run.

See [machine-readable evidence](../benchmarks/results/2026-10-29-gemma3-artifact-acquisition-v1.json),
[completed tiny graph](gemma3-resident-graph.md) and [TODO](../TODO.md).

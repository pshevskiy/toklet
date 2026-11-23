# Nanbeige T128 Metal kernel

This packaged 13,472-byte library contains one affine Q4/G64 BF16
matrix multiplication specialization. The explicit Nanbeige Metal opt-in
uses it for Q, O and Down on eligible 128-token prefill chunks. The
validated host was Apple M1 Pro on macOS 26.6.2. The loader keeps the
existing dense or packed execution path available if optional admission
fails.

Build with `cargo build --locked --offline --release --features metal`.
From the repository root, use the pinned local Nanbeige checkpoint and
a file with 512 little-endian u32 prompt IDs.

```sh
target/release/toklet generate \
  --model /path/to/pinned-nanbeige-checkpoint \
  --model-profile nanbeige4.2-3b-mlx-q4 --backend metal \
  --token-file /path/to/prompt-ids.u32le --context-capacity 544 \
  --prefill-chunk 128 --max-new-tokens 32 --no-eos --dense-prefill \
  --qmm-prefill-metallib assets/nanbeige-qmm/qmm.metallib --json
```

The `qmm_prefill` statistics give admission, the committed QMM chunk and
row counts, the reserved bytes, and the hash, load and pipeline times.
The loader authenticates the exact file before passing the same owned
bytes to Metal. SHA-256 is
`78496343fbbcb95ac8949cb4c384baeb8c0edf4f8f87a37e506b844b02d88353`.

For the experimental prepacked Steel route on M1 Pro with macOS 26, add
`--steel-prefill-metallib /path/to/mlx.metallib` to the command above. Use
the pinned MLX 0.32.4 wheel image with SHA-256
`c6d7a3facb83568f6e4cb2cf4a8c736ee29302571a48343cb9a3a4d1cd3e3893`.
Admission requires exactly 512 prompt IDs, a 32-token output limit,
capacity 544 and chunk size 128. An unsupported configuration returns
an error. The terminal JSON reports `steel_prefill: true` for this route.

On that fixed Steel route, Q, O and Down use the embedded 64-row
specialization `qmm_bm64.metallib`, SHA-256
`941137a5e36e1dbe293f3c1656b6e15be47e91f841489b4c01d5122bcb5f87f2`.
It shares the pinned MLX source and Apple Metal toolchain described below.
The supplied 32-row image is still required for admission and remains the
kernel for the generic QMM route.

The two independent warm confirmation orders improved completion by
8.63% and 8.45% against Toklet’s dense option, at 5.17 GiB peak. Decode
intervals crossed zero. A single cold CLI sample completed in 13.738 s
with QMM and 12.570 s with dense. [Performance record](../../docs/improvements.md)

The Metal translation unit instantiates the kernel from MLX v0.32.4
at commit `36969c094354086e15d2a327a9c50feb20b9cf11`. The packaged image
was built with Xcode 27.1 build `27A9275`, macOS SDK 27.0 and Apple Metal
Toolchain `27.1.266.1`, targeting macOS 26.0. Keep Apple’s MIT notice
with the translation unit and compiled image.
[MLX source](https://github.com/ml-explore/mlx/tree/36969c094354086e15d2a327a9c50feb20b9cf11) ·
[MIT notice](vendor/LICENSE)

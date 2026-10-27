# Original quantized Qwen3 source capture

[manifest.json](manifest.json) is the exact canonical source manifest, SHA-256
`0183168d5f61b8fb103af547cd89e74ae7b053764146eb94c565dbf88b0c788a`.
It freezes six calibration cases, the original quantized checkpoint/config,
704 parameters,197 quantized modules, output/tile geometry and complete hashes.
The second cold capture has identical source bits and widened arrays.

Each set contains204201984 BF16 logits and their bit-exact FP32 widening,
1225211904 bytes. Raw files remain outside Git under
`artifacts/m8-qwen3-quantized-source-capture-v1/capture-a` and `capture-b`.
Both full readbacks verify every value and descriptor/path endpoints.
No validation inputs or native model execution enter this capture.

This is an original-source precision experiment, not a quality baseline or
adopted numerical contract. Source BF16 storage does not establish every
internal accumulation dtype. See [the report](../../../docs/qwen3-quantized-source.md)
and [public evidence](../../../benchmarks/results/2026-10-27-qwen3-quantized-source-v1.json).

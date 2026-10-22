# Independent tiny Llama decoder oracle

The [exporter](../tools/reference/mlx_llama.py) specifies a bias-free dense Llama
graph before native target implementation: SiLU, RMSNorm, full causal GQA,
split-half ordinary RoPE, and tied or independently owned untied output weights.
The tiny geometry is two layers, hidden32, intermediate64, four query/two KV
heads of width8, vocabulary64 and capacity32. It does not qualify scaled RoPE,
sliding attention, projection biases or other Llama graph variants.

The [frozen fixture](../tests/fixtures/mlx-llama-v1/manifest.json) has SHA256
`824d2ee2ae9dd5b623c5aa9db10c13b2e4dee32647ea5fa9d78274fb3529a1f5`:
809 arrays in 1,026,048 bytes. Both tied/untied artifacts mix affine four-bit
matrices, an eight-bit override and a floating projection exclusion. Original
U32 low-bit-first words and source-precision affine metadata reconstruct FP32
parameters independently. Embedding gathers round to Half in the mixed graph;
packed output dots use the original reconstructed weights. No Qwen projection
bias is included. The untied head has different rows/scales, changes selected
IDs and has a saved wrong-tied-head substitution witness.

Ordered FP32 and FP64 reductions independently evaluate all39 stages across five
cases: prompt, changed future token, rotary offset5, single token and capacity32.
The FP32 reference gate is `1e-5 + 1e-4 * abs(reference)`. Mixed-Half gates are
`max(1e-4, 2 * FP32_FP64_envelope) + 1e-3 * abs(reference)`, frozen before native
target evaluation. Four greedy steps use the mixed graph's own history and
lowest-ID tie breaking: tied `[47,47,47,47]`, untied `[3,3,3,3]`.

Actual upstream MLX0.32.3/MLX-LM0.32.0 runs on CPU and GPU agree with the FP32
reference for all zero-offset cases and actual two-plus-two-token cached
execution. Standalone offset5 is an independent scalar witness; it is not
presented as an upstream cache experiment. These installed numerical producers
are distinct from the newer static inventory in [architecture coverage](mlx-architectures.md).
Selected package source/native files, the complete non-cache MLX runtime tree
including Metal libraries, consumed input bytes and exporter are pinned before
imports and rechecked at publication.

All [12 portable methods](../tools/reference/test_mlx_llama.py) pass, including
mandatory finite/non-Boolean gates, overflow rejection, exact shapes/dtypes,
coverage, versions, producer closure, greedy/witness replay, causal prefixes,
exclusive export and stdlib-only checking with an external manifest SHA.
The [receipt](../benchmarks/results/2026-10-22-llama-oracle-v1.json) binds owned
export/check logs and unchanged Rust/shader/Swift/protected-binary endpoints.
Prior Rust checks are inherited; this task changes no numerical target code.
Astra reviewed the source before export and the completed fixture afterward.

This closes only the independent tiny Llama oracle task. Native execution,
real-checkpoint compatibility, quality, performance and phone qualification
remain open in the [M8 plan](mlx-implementation-plan.md).

# Real Phi3 independent reference

The pinned Phi-3-mini-4k four-bit checkpoint now has an independently generated
CPU reference and plain-text contract for the next native comparison. This
completes B2c1; real Metal execution remains B2c2. The existing tiny composition
checks are described [separately](phi3-dense-composition.md).

## Graph and ownership

The lazy reader authenticates the complete original file, header/index and all
455 physical entries, including the unaligned byte51879 payload start. It maps
original fused QKV/gate-up row ranges independently of the Rust loader. Embedding
lookup reconstructs only requested rows; each projection owns one temporary FP32
logical matrix. Norms and stored activations/KV use Half; scores, probabilities
and final logits use FP32. The untied packed output head remains distinct.
No full15,284,318,208-byte FP32 model is allocated, and no remote Python executes.

Main NumPy2.3.2 FP32 BLAS and wide FP64 profiles use the same Half storage
boundaries, ordinary head96 split-half RoPE and explicit offsets. These are
independent native arithmetic references, not exact canonical Torch/MLX runtime
parity or original unquantized quality evidence. The six fixed probes cover one,
four and17 tokens, changed future tail, offset5 and offset509 with three local
keys. Selected source layers are0,1,31. Position511 is not a full512-key or4096
context accuracy test.

All349 main/wide pairs,8,197,056 values and19,025,664 bytes per profile pass
saved-byte identity, finite-value, shape and extent verification. Each exact
case/stage envelope is frozen before target execution. The only fatal native
comparison will be
`abs(native-main) <= max(1e-4,2*maxabs(main-wide)) + 1e-3*abs(main)`.
Wide comparisons remain diagnostic; no cross-case pooling or prefix widening.
Main and wide greedy IDs are both `[3,3,3,3]` on the fixed raw prompt, a diagnostic
observation rather than a text-generation quality claim.

## Text boundary

Sixteen rendered cases cover eight conversations with both generation flags.
The pinned Jinja template and manual renderer agree; tokenizer replay produces
identical full43,382-byte fixtures. There is no automatic BOS or hidden truncation.
ConfigEOS32000 and tokenizerEOS32007 are the stop union; IDs1 and2 are not stops.
All53 unmapped numerical head slots32011..32063 reject before decode. Numerical
logits retain all32064 slots. Added-token rstrip, metaspace and byte-fallback
behavior is explicit: rendered bytes need not equal decoded bytes.

These fixtures specify a future native text adapter. They do not implement or
qualify production Chat, native decode-error rollback, phone behavior or natural
EOS generation.

## Checks and measured fit

Ten synthetic original-row/ownership/corruption tests and five portable text
controls pass independently of downloads and GPU availability. Source graph
export, saved-bit verification and two text exports all finish in five owned,
closed process groups with authenticated endpoint identities. Runtime is
Python3.12.8, NumPy2.3.2 with Apple Accelerate, macOS26.6.2/25G83 on MacBookPro18,3.

The conservative source admission is4,489,855,655 bytes, including the full
mapped file, largest temporary matrix, graph/BLAS and framework allowances.
Sampled resident maximum is2,842,066,944 bytes; the OS lifetime RSS peak is
2,846,588,928 bytes. Observed lifetime physical footprint is522,881,880 bytes.
All stay below6GiB; clean mapped pages explain why physical footprint is smaller
than RSS. This is process-fit evidence, not latency or throughput qualification.

Astra independently decodes every saved array and recomputes its envelope,
hashes and greedy selection; it verifies full text replay and process closure.
A pre-export whitespace failure was corrected by removing only an EOF blank
line from the new graph; old frozen producers and all numerical gates remain
unchanged. See [compact evidence](../benchmarks/results/2026-10-28-phi3-real-reference-v1.json),
[TODO](../TODO.md) and [improvement record](improvements.md).

Next: freeze the reviewed native capture/checker, run fixed short numerical and
raw session controls, and measure native process fit. Full declared context,
production app/phone integration, adoption quality and performance remain open.

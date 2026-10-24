# Independent causal and sliding-window GQA reference

This G2 operator witness freezes local attention mathematics before its native
target. Inputs are FP32 Q `[query_heads,query_tokens,D]` and separate K/V
`[kv_heads,key_tokens,D]`. Query heads map contiguously to KV heads using
`query_head/(query_heads/kv_heads)`; no repeated KV heads are materialized.

The query sequence ends at the last available key. With key origin b, query i
has position `b+key_tokens-query_tokens+i` and key j has position `b+j`.
Causality requires key≤query. A positive window W additionally requires
`query-key<W`, including exactly W positions when enough keys exist. A missing
window means full causal attention. Score scale is explicit FP32; Gemma4 uses1,
which the hand witness distinguishes from an implicit `1/sqrt(D)`.

Independent NumPy FP64 dot products, max-subtracted softmax over included keys,
and context accumulation produce scores, probabilities and output. Masked
probabilities are exactly zero. Inputs remain immutable. Empty sequences make
no numerical target calls. The bounded format admits only its eleven named
cases, exact field/shape/dtype inventories, finite exactly representable FP32
inputs and an external SHA256. Boolean masks require actual Boolean values.

The [fixture](../tests/fixtures/gemma-local-attention-v1/math.json) contains77
arrays and14640 output scalars. Cases cover single/prefill/cached-prefix input,
translated trimmed origins, window1, wide/global attention, several GQA ratios,
actual8-query/1-KV-head geometry at D256 and D512, and an explicit512-window
boundary. That last case uses D2 and queries at512–514 over515 keys: each query
includes exactly512 keys while excluding expired and future positions. It tests
the full window boundary separately from the actual-width cases.

Seven portable methods check hand scale/head mapping, future and expired-key
independence, masked zero probabilities and row sums, exact chunks/origin
translation, invalid geometry/shapes/dtypes/finiteness, strict external identity
and source selection, and retention of a completed raw mask when context fails.
Three ownership and three cold-runtime-admission tests also pass.

The explicit adapter authenticates `base.py` at MLX-LM revision
`0d4d3d04f8ab7ce4a51e7209e8615755a48ba94d`, SHA256
`63d8e58a4b6f8c0902719499c693647fbf486bbd5d22b09868a07d37e08c83d6`.
Only captured `create_causal_mask` and `scaled_dot_product_attention` definitions
execute, with cache/sinks absent and no quantized KV path. Cold admitted
NumPy2.3.2/MLX0.32.3 CPU/GPU processes each retain20 raw arrays. Masks match
exactly; context passes the unchanged `1e-5 + 1e-4*abs(expected)` policy, with
maximum error/policy ratios about0.00810 CPU and0.00487 GPU. These are numerical
errors, not latency measurements. Raw arrays are saved before later failures.

The first passing capture covered small windows. It is retained as preliminary,
without being credited for the512 boundary. Astra reviewed the added bounded
case before a fresh complete capture. Independent stdlib replay validates all
saved math arrays and source comparisons. The
[receipt](../benchmarks/results/2026-10-24-gemma-local-attention-reference-v1.json)
binds the exact final source/runtime, tests, captures and replay. Existing engine
sources and protected binaries remain unchanged. Astra reviews final evidence
before commit.

This is not physical ring-order, cache mutation/commit/rollback, shared-KV
ownership, normalized/projected full graph, quantized KV, quality, performance
or phone qualification. It does not resolve the separate upstream rotary
precision gap. The [standalone scalar target](window-attention-scalar.md) now
passes its frozen cases and required checks; cache ownership witnesses are
next in the [Gemma4 plan](gemma4-implementation-plan.md).

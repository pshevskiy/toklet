# Independent Qwen FP16 references, 10 October 2026

The [oracle](../../tools/reference/qwen_fp16.py) freezes real Qwen FP16 numerical
references before Metal decoder evaluation. All 24 layers execute over the four
unchanged [qwen-v2 cases](../../tests/fixtures/qwen-v2/manifest.json), capturing
layers 0, 1 and 23, final normalization and all-position logits. The separate
[FP16 manifest](../../tests/fixtures/qwen-fp16-v1/manifest.json) contains 228
selected arrays, 31,616 payload bytes, full-array identity anchors, and 228 gates
against the original FP32 arrays. Full reconstructed arrays total 9,359,136 bytes
and remain local in `artifacts/qwen-fp16-full-v2`.

Safetensors/Torch only read/convert the pinned BF16 tensors directly into
988,065,536 bytes of owned FP16 weights. NumPy executes the decoder, independently
of Transformers forward, Rust or Apple APIs. Row-major linear weights are
`[out,in]`, activations `[tokens,hidden]`, and rotary Q/K `[tokens,heads,dimension]`.
GQA maps query heads to shared KV heads without expanding K/V. Every named
activation rounds to FP16, including normalization, projections, rotary, SiLU,
gating and residuals; scores/probabilities/logits store FP32. Bias is added after
the complete dot and before its single storage cast. Separate FP32 and FP64
calculations preserve the same storage-rounding boundaries. BLAS association is
not claimed to match the ordered scalar tiny oracle. Weight conversions use
at most 1,024 rows; the largest FP64 stripe is 34,865,152 bytes. No complete
FP32/FP64 weight copy is constructed.

Gates pool both reconstructions' FP16-versus-source-FP32 errors across each named
operation/layer and all cases. Exact common prefixes at matching absolute
offsets contribute a shape-variation envelope. The fixed policy is
`atol=max(1e-4,2*pooled_error,2*prefix_error)`, `rtol=1e-4`. No target Metal output
was inspected. Existing FP32, tiny and Core ML gates remain unchanged. Future
target comparisons use the original FP32 arrays with these new gates; the
saved FP16 reconstructions supply additional diagnostic references.

Both reconstructions preserve all twelve source greedy IDs. Maximum main FP16
logit error is 0.394946575; including FP64 arithmetic gives an envelope of
0.464270592 and pooled absolute logit gate 0.928541183. The largest intermediate
discrepancy is 2.123859406 at offset-case final normalization, whose pooled gate
is 4.247718811. These measured differences do not establish held-out quality.
Persistent caching, long context, target GPU numerical checks, the 10,000-token
loss gate and 100-task quality gate remain pending.

The [functional record](../../benchmarks/results/2026-10-10-qwen-fp16-export.json)
binds generator/manifests, repository state, versions and power boundaries.
NumPy 2.3.2, Torch 2.9.1 and safetensors 0.8.0 are pinned; the BLAS thread
environment is one. Generator SHA is
`c16994767d257ebc4d6bee38d5b1a20d9fcfd9e134b598eead54e25b8553b789`.
Own-process physical charge peak is 1,166,165,720 bytes, which can omit clean
mapped source pages; total system memory/energy are unavailable. The whole
source mapping is another 988,097,824 logical bytes during loading. The
6.416-second functional export includes source validation/loading, both
calculations, writing and verification; it is not an inference benchmark.
AC power was attached, AC low-power mode disabled and no thermal warning recorded.

Exports refuse existing/overlapping directories and nonfinite arithmetic. A
ten-minute watchdog bounds only the explicit export process. Offline verification
checks source/generator identity, dtypes, hashes, selected coverage and stored
gate formula without weights or Torch. Exact full/probe regeneration additionally
checks every envelope component and byte; this explicit real-model check passes.
Six offline tests cover manual bias/storage rounding, stripe tails, FP16-range
normalization, causal GQA/rotary, all tiny cases against the existing scalar/gates,
prefix pooling, overwrite admission, real probes and corruption. Formatting,
Clippy with warnings denied, 74 ordinary Rust tests and 39 offline Python checks
pass. Rust execution and release binaries remain unchanged.
Astra accepted the completed task without findings after independently checking
all full/selected anchors, identities, pooled/prefix envelopes and greedy IDs.
Its review ran four lightweight arithmetic/integrity tests and no heavy jobs.

The scalar transport driver was paused before sample two finished. Its numerical
worker completed the timed request normally (748.973 seconds), then waited
between requests while export and checks ran. The driver resumed at 00:23:55 UTC;
no timed request overlapped verification, and frozen binary hashes still match.
Exploratory v1 output remains ignored; final v2 export and exact regeneration
supply the accepted evidence. This preparation closes no milestone.

```sh
.venv/bin/python tools/reference/qwen_fp16.py --output artifacts/qwen-fp16-candidate --probes-output artifacts/qwen-fp16-probe-candidate --record benchmarks/raw/qwen-fp16-export-candidate.json
.venv/bin/python tools/reference/qwen_fp16.py --check tests/fixtures/qwen-fp16-v1
.venv/bin/python tools/reference/qwen_fp16.py --output artifacts/qwen-fp16-full-v2 --probes-output tests/fixtures/qwen-fp16-v1 --regenerate-check
.venv/bin/python -m unittest discover -s tools/reference -p 'test_qwen_fp16.py'
```

Export/regeneration require the pinned weights/config and original full qwen-v2
arrays. Recreate them in new local directories using the
[reference recipe](../../tools/reference/README.md), verify committed anchors,
then install the ignored full source arrays. Offline checks need neither model
downloads nor Apple devices. Run export and heavy checks outside performance jobs.

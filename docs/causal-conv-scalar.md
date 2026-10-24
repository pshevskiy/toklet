# Scalar causal depthwise convolution

`cpu::causal_depthwise_conv_step` implements one raw token from the
[frozen independent convolution reference](causal-conv-reference.md). It remains
a standalone E4/L4 operator; no model loader/session or Metal graph uses it yet.

Input/output are FP32 `[channels]`, weights `[channels,kernel]`, and immutable
history plus separate candidate history `[kernel-1,channels]`. History is oldest
to newest and each channel's weight taps run oldest to current input. Products
and increasing-tap sums use FP64, then one cast stores FP32 output. Finite FP32
products and any admitted usize-bounded sum fit FP64; the final cast can overflow
and rejects. This is a correctness reference, with no optimization claim.

Checked nonzero dimensions, overflow-safe element counts, exact slice lengths
and finite inputs precede all writes. Invalid admission preserves both outputs.
Arithmetic failure can leave partial convolution output; discard it and the
candidate. Previous history remains intact on every result. Only after every
channel succeeds does the function shift the bounded history and append raw
input. Kernel one has zero history. Empty sequences make no step calls. SiLU
stays a separate existing CPU operation. The numerical function allocates no
buffers, performs no session commit and needs no extra alignment beyond FP32.

Four offline tests authenticate the accepted reference receipt's literal hash
before reading fixture identities. They compare frozen convolution output and
composed SiLU at the unchanged `1e-5 + 1e-4*abs(expected)` policy. Copied raw
history and final history compare exactly. Independent hand cases distinguish
channel/tap order and raw input from filtered output, cover kernel one, and
preserve FP64 cancellation when separate products exceed FP32 range. Every
input class's NaN/infinities, all five slice lengths, zero/overflow dimensions
and output overflow exercise preservation and replay.

The [receipt](../benchmarks/results/2026-10-24-causal-conv-scalar-v1.json) binds
source, frozen reference, raw logs, protected binaries and check results.
Formatting, strict default/Metal Clippy, 207 CPU and 229 Metal-feature tests
(15/107 ignored respectively), and both iOS cross-checks pass. Compiler cache reuse happens
in a separate copy-on-write target, without changing the previous accepted
target's files or qualifying build/runtime performance.

Gated output normalization, hybrid config/tensor mapping, model/session state
commit and replay, resident Metal execution, real artifacts, chat, long-context,
quality/performance and iPhone qualification remain open. This does not broaden
model eligibility or native/app execution limits.

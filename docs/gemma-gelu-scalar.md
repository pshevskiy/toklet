# Scalar tanh-GELU and GeGLU

`cpu::gelu_tanh` and `cpu::gated_gelu_tanh` implement the first native G2
operators from the [independent Gemma4 witness](gemma-gelu-reference.md).
They are standalone functions; no model graph calls them yet.

Inputs and output are contiguous FP32 slices with equal nonzero length. GELU
uses FP64 for `0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x³)))`, then rounds once
to FP32. GeGLU multiplies that rounded activation by the FP32 up input in FP32,
preserving the separate upstream activation/product rounding. The shared factor
is calculated outside the scalar loop. Finite FP32 cubic intermediates fit FP64;
GELU remains finite even at FP32 extrema. This is tanh-GELU, not erf-GELU or SiLU.

Nonzero width, exact lengths and finite inputs are checked before output writes.
GeGLU rejects a non-finite product; it can leave partial output, which the caller
must discard. Inputs remain immutable and a subsequent valid call can replay.
The functions allocate no buffers, own no session state, require no extra
alignment beyond FP32 and use static direct calls. An empty sequence makes no
operation calls. Existing kernels and graph eligibility remain unchanged.

Four offline tests authenticate the frozen reference receipt's literal hash
before parsing the fixture identity. All nonempty fixture activation/product
arrays pass the unchanged `1e-5 + 1e-4*abs(expected)` policy. Composed FP32
activation-times-up results match GeGLU bit for bit. Literal hand values
distinguish the activation variant and gate order; relative tiny-input checks
reject accidental zeroing. FP32 extrema are mathematical boundary tests, not
additional upstream qualification. Empty/short/long/NaN/infinity admission
checks preserve output sentinels; gated overflow checks partial output, input
preservation and replay.

The [receipt](../benchmarks/results/2026-10-24-gemma-gelu-scalar-v1.json) binds
exact source, fixtures, raw logs, protected binaries and seven passing commands:
formatting, strict default/Metal Clippy, 211 CPU tests and 233 Metal-feature tests
(15/107 ignored respectively), plus iOS device/simulator cross-checks. Builds use
a separate copy-on-write compiler cache; no latency or throughput is qualified.
Astra reviews code and final evidence before commit.

Proportional rotary, local/shared KV, value normalization, per-layer paths,
decoder ordering, logit capping, artifact/session/Metal graph integration and
real-model/tokenizer/quality/device qualification remain open in the
[Gemma4 plan](gemma4-implementation-plan.md). No output/context/byte capability
was raised and this is not an install-ready phone package.

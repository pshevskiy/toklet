# Causal depthwise convolution reference

This independent reference precedes native convolution implementation for the
[hybrid decoder plan](mlx-qwen35-design.md). It fixes tap order, channel layout,
post-convolution SiLU and raw-input history separately from recurrent attention.
No native model/session consumes this operator yet.

Inputs are FP32 `[tokens,channels]`, weights are FP32 `[channels,kernel]` and
initial history is FP32 `[kernel-1,channels]`. History runs oldest to newest;
weights run oldest history tap to current input. Each channel is independent.
The producer widens inputs to FP64, constructs causal windows and uses a NumPy
contraction rather than a Toklet kernel. It computes stable FP64 SiLU and saves
all outputs, every prefix history and final history. Five tiny cases cover empty
input, kernel one, asymmetric taps/channels, odd channel count and 128 channels.
Hand arithmetic, chunking, causality, negative SiLU, rejection and format tests
supply additional witnesses.

A bounded stdlib reader requires external manifest and binary hashes before
consumption. It validates exact fields, geometry, dtype, finite values, aligned
accounted spans, padding and per-array identities. A neutral cold runtime helper
admits the interpreter, selected producer sources and complete pinned NumPy/MLX
package/native/Metal/metadata files before numerical imports. Producer and
upstream invocations use separate exact runtime manifests and fresh bytecode
cache directories. Frozen gated-delta v1 tooling is unchanged.

The explicit upstream comparison authenticates the captured Qwen3.5 source at
MLX-LM revision `0d4d3d04f8ab7ce4a51e7209e8615755a48ba94d`. Only its Conv1d
constructor, concatenation, unmasked history update and SiLU assignments execute;
no model module or repository code is imported. The actual admitted MLX
`nn.Conv1d` receives weights `[channels,kernel,1]`, groups equal to channels,
no bias and zero padding. CPU and GPU are separate completed invocations on an
Apple M1 Pro with NumPy 2.3.2 and MLX 0.32.3. Each retains 154 raw arrays and
passes 154 comparisons at the unchanged `1e-5 + 1e-4*abs(expected)` policy.
The largest error/policy ratios are approximately 0.001020 (CPU) and 0.000735
(GPU). These are numerical comparisons, not timing or throughput measurements.

Full calls, repeated prefixes and chunk sizes one/three check output and carried
raw-input history. Empty input is an explicit mathematical no-op; no upstream
empty call is claimed. Kernel-one convolution is compared, but its upstream
model cache branch is excluded: the source's `-0:` slice retains all input,
whereas the mathematical operator has zero history. Masks, length-dependent
caches and full model/cache behavior remain outside this reference.

The owned capture retains exact commands, logs, source snapshots, input/runtime
pins, output bytes and endpoint guards, including failures if a helper raises.
Cooperative interruption publishes partial evidence after suppressing repeated
signals; a hard kill can still lose a partial child report. Three orchestration,
nine math/format/source-selection and three runtime-admission tests pass.
The [receipt](../benchmarks/results/2026-10-24-causal-conv-reference-v1.json)
binds the final source and checked evidence. The first capture is retained as
superseded: Astra required immediate failure-preserving observation capture,
numerical prefix checks and source-buffer substitution protection. Targeted
regressions and a fresh complete capture pass after those corrections. The final
runtime helper differs from its executed snapshot only by removal of one extra
EOF blank line; the receipt binds both exact versions without a numerical rerun. Frozen arrays are at
[the fixture manifest](../tests/fixtures/causal-conv-v1/manifest.json).

The subsequent [scalar convolution](causal-conv-scalar.md) is checked separately.
Gated output normalization, hybrid weight/config mapping, state/session commit,
resident Metal execution, real checkpoints, chat,
long-context, quality and iPhone qualification remain open. Current app/native
model eligibility and execution limits are unchanged.

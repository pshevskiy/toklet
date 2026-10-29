# Gemma3 clipped Half residual

29 October 2026. Gemma3's resident Half graph now uses a separately named
residual operation at both post-attention and post-FFN additions. For finite
Half operands, it adds in FP32, clips to ±65,504, then stores Half. It preserves
the signed-zero result and ordinary Half rounding. Invalid inputs still set the
sticky numerical status; clipping does not make NaN or infinity admissible.

The first real Gemma3 source export stopped at the layer8 BOS FFN residual:
64,864 +14,520 =79,384, beyond finite Half. The failed run,398 partial groups
and669,568 original output bytes remain recorded. Pinned MLX-LM0.26.3 explicitly
applies this FP16 adaptation at both residual sites. This changes the declared
Half arithmetic profile; it changes no gamma interpretation, other operation,
finite check or numerical tolerance. Ordinary BF16/FP32 residual semantics are
separate. The original generic Metal residual kernel remains unchanged.

## Implementation and ownership

`residual_half_clamped` borrows the existing contiguous Half input/output and
read-only branch buffers. One checked scalar operation handles each active
value; guard/tail bytes are untouched. It uses the existing status owner and
command lifecycle. The new pipeline has its own queried execution width and
preserves existing kernel indices and legacy width selection. No new model,
session allocation, host readback or per-layer host wait is introduced.

## Checks and qualification

Formatting, strict Metal Clippy,327 Metal library tests and seven targeted
controls in both debug and release pass. The release primitive checks20,962
active values,16 literal bit-pattern cases, signed zeros/subnormals/ties, five
invalid inputs, sticky status/reset recovery and nonzero guards. A64-value
same-command addition/copy/addition control observes the actual saturated store
boundary. Legacy overflow still produces its original nonfinite/status result.
Eight warmed primitive encodes allocate zero Rust objects; command creation and
driver allocations are outside that primitive measurement.

Three additional GPU jobs rerun the complete tiny decoder:428,832 main-profile
values across2,400 stage snapshots, cache publication/poison/reset controls and
11,520 rotary comparisons. All unchanged gates pass; the maximum main graph
error/gate fraction remains0.4126969243. Twelve warmed full forward calls allocate
zero Rust objects, including command creation/completion, excluding the driver.
Sparse memory observations are not continuous peaks or performance measurements.

Independent Astra saved-output review and final publication review are recorded
in the [evidence](../benchmarks/results/2026-10-29-gemma3-half-residual-v1.json).
This qualifies the primitive and complete tiny graph regression. The corrected
real reference export, actual real native window/text journey, app admission and
iPhone qualification remain separate tasks in [TODO](../TODO.md).

See the [original tiny resident graph](gemma3-resident-graph.md) and
[verified artifact](gemma3-artifact-acquisition.md). The large real checkpoint
and diagnostic captures remain local artifacts; they are not committed weights.

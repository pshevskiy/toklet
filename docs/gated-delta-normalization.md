# Scalar recurrent Q/K normalization

This adds the raw-input normalization required by the
[scalar recurrence](gated-delta-scalar.md), using the independently frozen
[gated-delta witnesses](gated-delta-oracle-preparation.md). It remains an operator
foundation for E4/L4, with no hybrid model/session admission.

`cpu::gated_delta_normalize` takes Q/K `[heads,key_dim]` and separate output
slices in the same contiguous layout. For each head it computes
`Q / sqrt(sum(Q²)+epsilon) / sqrt(key_dim)` and
`K / sqrt(sum(K²)+epsilon)`. Epsilon is added to the sum, not the mean. There
are no norm weights. This is distinct from a transformer's per-head RMSNorm.
The Q readout scale is folded exactly once and is not reapplied by the step.

Input storage is FP32. Squares, increasing-coordinate sums and normalization
use FP64, followed by one FP32 output rounding. Finite FP32 squares and sums fit
FP64 for the checked slice dimensions; positive finite epsilon gives nonzero
finite denominators. Normalized coordinates stay bounded, including zero heads,
epsilon-dominated heads and maximum finite FP32 inputs. No FP32 squared-norm
overflow or reciprocal-infinity special case is needed.

Positive dimensions, checked element products, exact slice lengths, finite Q/K
and finite positive FP64 epsilon are validated before any write. Admission
failure preserves both output buffers. The operator allocates no storage;
outputs add two caller-owned `[heads,key_dim]` FP32 buffers to any later graph's
memory plan. It changes no recurrent state, positions or session ownership.

Three new ordinary offline regressions authenticate the accepted reference's
literal receipt SHA before using its manifest/binary identities. They compare
all raw normalization cases and the composed recurrent states/outputs to the
independent FP64 fixture at the unchanged source policy. Hand calculations
separate sum from mean epsilon, verify the single Q scale, zero heads,
epsilon-dominated output comfortably above the absolute tolerance, and maximum
finite FP32 input. Invalid epsilon, nonfinite Q/K, shape/size overflow and output
length mismatches verify pre-write preservation.

The [receipt](../benchmarks/results/2026-10-24-gated-delta-normalization-v1.json)
records formatting, strict default/Metal Clippy, 200 CPU tests, 222 Metal-feature
tests and both iOS cross-checks (15/107 ignored respectively). No new GPU operator
or physical-device check follows. Compiler-cache files were seeded into a
separate copy-on-write target directory; the accepted previous target and
protected release executables remain independent. No build-time or runtime
performance claim follows from that cache reuse.

Gate construction and convolution still need native operators. Hybrid layer
mapping, state/session commit, resident Metal execution, model admission,
long-context quality and Mac/iPhone app qualification remain open. Existing
native/app output and context limits are unchanged.

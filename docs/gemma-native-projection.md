# Independent Gemma native projection replay

The bounded CPU primitive independently implements original Q4/group64/BF16
unpacking and the projection's specified arithmetic order. It accepts at most
128 output rows, 32 input rows and width12,288. Explicit FP32 FMA reconstructs
weights; each lane accumulates the rounded product and its FMA residual with
two-component compensation. Increasing-hop merges preserve the lane order.
Buffers have typed alignment, separate ownership and checked byte lengths.
Nonfinite input/metadata, arithmetic failure and a non-nearest rounding mode
reject execution. It is reference-only and loads an explicitly supplied library.

Apple clang21 builds the C module with `-O2 -ffp-contract=off -fno-fast-math`
and strict warnings. Optimized arm64 assembly retains separate multiply and
explicit fused instruction sites. Four new controls use exact rational RNE at
every operation across lane widths1/2/8/16/32/64. They cover literal low nibbles,
group/row/token boundaries, cancellation requiring the product residual, signed
zero, subnormal and midpoint cases, shape/byte/finite admission and overflow.
Together with twelve existing affine controls, all16 pass. Astra approves this
CPU primitive scope; existing required Cargo results are reused because their
source/dependency inputs are unchanged.

A feasibility measurement authenticates the original checkpoint and replays its
first128 V rows at owner13/position433 from the archived native input. At assumed
width32 all128 archived output words match exactly. One warmup and three raw
samples take1.425–1.431ms for one input and32.588–32.838ms for32 repeated copies
of that same input; peakRSS30,244,864bytes on the M1 Pro. Source inference was
running concurrently. These are diagnostic CPU tile timings. The first runner
attempt failed while repeating a memoryview before projection; its script,
plan and logs are preserved, and the corrected bytes conversion passes.

Actual Metal pipeline width and in-process subnormal/flush behavior must still
be measured. `fegetround()` establishes the rounding mode only. Fresh GPU
fixtures, the independent host-provider graph and fatal coupled comparisons
remain required by the [Mac conformance design](gemma4-native-conformance.md).
All historical numerical rejections and app/phone limitations remain unchanged.
[Evidence](../benchmarks/results/2026-11-04-gemma-native-projection-v1.json).

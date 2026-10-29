# Independent tiny LFM2 Half reference

B4d4b fixes a numerical target for the resident mixed graph before native
outcomes. It reads the authenticated original tiny Q4/G64/BF16 container from
the scalar fixture, rather than importing that graph or its numerical outputs.
The geometry is H256/I256/V64/Q4/KV2/D64/C12, with Conv→Attention→Conv. The
18 affine parents, nine direct norms, two Conv3 tensors and tied vocabulary
table retain their original ownership and bytes.

Both profiles compute affine sums exactly with integer dyadics over original
FP32 reconstructed weights and stored Half inputs. They round directly to Half,
or directly to FP32 for the tied output head. A common exact power of two reduces
integer sizes without discarding bits. Controls distinguish Half/FP32 midpoint
double rounding, cancellation, subnormals, parity and overflow. This mathematical
center is fixed uniformly; it does not emulate a native reduction lane order.

MAIN reuses qualified direct-gain epsilon1e-5 Neumaier RMS and the explicit FP32
three-tap convolution. Every B*x, Conv, C*z, affine, SiLU, gating and ordinary
unclamped residual has its declared Half store. Attention preserves serial
Reference FP32 score/probability/context stores and grouped-head mapping. WIDE
uses the declared diagnostic leaf arithmetic and ordered FP64 softmax mass,
while retaining the same score/probability FP32 stores and exact affine center
on its own Half inputs. No embedding lookup normalization or gain+1 is added.

RoPE MAIN uses the explicit FP32 exponent/pow/angle/trig/product stores. WIDE
constructs frequencies using Decimal ln/exp at80 digits and checks their FP64
stability at120 digits, then uses FP64 angle/trig/products. That is a stability
check of the same method, not certified transcendental rounding or an emulation
of the device intrinsic. Components cover offsets0/1/5/11 and9..11.

An analytic combined witness makes the three convolution stores observable:
B=C=x is Half0x3c01, the current tap is1.25 and the output coefficient is1.5.
The u/Conv/Cz/output words are3c02/3d02/3d03/3f84. Omitting the u, Conv or Cz
store instead changes a later word to3d03,3d04 or3f85. No target output supplies
these expectations.

Four cases retain all63 groups in each profile:504 descriptors/2,212,480 bytes.
Each case/stage freezes MAIN's floor1e-4, twice its MAIN/WIDE envelope and
rtol1e-3 before any target. MAIN is fatal and WIDE diagnostic. Raw future scores
remain observable; earlier-prefix invariance checks only actually visible score
columns, while masked probabilities are exact positive zero. Histories are raw
u suffixes. Source tokenwise snapshots do not invent within-chunk native captures;
native checks must use observed chunk boundaries and active KV prefixes. Source
inactive positive zeros do not require a native reset to erase dirty KV tails.

The twelve portable controls and bounded four-file export/replay pass. The first
export had stopped before any graph arrays because its diagnostic counterexample
assumed Python builtin float sum meant ordered FP64 addition. A separately reviewed
successor uses explicit ordered additions and records0 alongside exact dyadic1.
The failed run/source/logs remain preserved. Graph arithmetic, cases, limits and
gates did not change. The successful workers stay below the600s/512MiB limits.

Independent saved-array qualification now passes: Astra reconstructs every one
of the1,097,888 MAIN/WIDE values bit-exactly and all252 envelopes/gates exactly.
It independently decodes1,327,104 original FP32 affine weights and verifies
290,304 exact dot results/74,317,824 scalar products, both raw history families,
active KV snapshots,88,576 inactive snapshot positive zeros,1,080 masked positive
zeros, causal visible prefixes and every component witness. Source workers peak
below173MB; these are CPU reference-worker observations, not native model fit.
Rust engine bytes remain unchanged; all six repository checks also pass,
including formatting, strict default/Metal Clippy,450 default/541 Metal tests
and optimized test compilation. Source validation uses its twelve controls,
exact replay and complete saved-array audit.

[Machine evidence](../benchmarks/results/2026-10-29-lfm2-half-reference-v1.json)
records the protocol, source correction, exact exports and independent audit.
No resident native graph, real LFM2 model, text/app behavior, process fit, speed,
full context or mobile qualification follows. The [LFM2 plan](lfm2-hybrid-plan.md)
next composes and checks the concrete resident mixed-state graph.

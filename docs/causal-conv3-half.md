# Borrowed raw Half causal Conv3 on Mac

B4d2 qualifies a reusable standalone three-tap causal depthwise convolution on
Apple M1 Pro. The input is already stored Halfu; coefficients are original BF16
widened exactly to FP32. Old history `[2,H]` remains immutable. The operation
writes output `[T,H]`, a separate raw-u candidate `[2,H]` and sticky status.
It does not perform B*x, C*z, activation, clipping or state publication.

The profile admits H5/256/1024 and T1..32 with checked input/output strides.
Each lane computes three separate FP32 products and `(p0+p1)+p2`, then one Half
output store. Contraction is disabled. Explicit operand, product, partial-sum
and final-store checks set shared sticky status; invalid output/candidate must
be discarded. A candidate is the last two rows of old+rawu, copied as UInt16 bits.
Only the first H global lane IDs write both candidate rows, covering T1 without
races. Coefficients are `[H,3]`, oldest/middle/current. Buffer extents, UInt32
index arithmetic, aliases, device identity and retained resources are validated
before encoding. Borrowed encode allocates no operands or host wait.

The independent source precedes native outcomes:12 cases contain46,571 MAIN
and46,571 WIDE output words plus7,238 raw candidate words. MAIN has explicit
FP32 stores; WIDE uses ordered FP64 arithmetic on the same operands and admitted
finite-intermediate domain. MAIN remains fatal at the unchanged per-case gate
`max(1e-4,2*maxabs(main-wide))+1e-3*abs(main)`; WIDE is diagnostic. Exact tap,
premature-Half-store, coefficient-narrowing, subnormal and signed-zero witnesses
supplement the tolerance. Eight portable controls, exact four-file replay and
Astra's full source/partition reconstruction pass.

The release GPU journey has274 submissions and286 retained raw captures:
12 paired strided/tight cases,35 partition schedules/233 actual boundaries,
two causality controls and27 invalid/sticky/reset phases across nine sites.
Astra independently reconstructs312,670 valid outputs exactly. Fixed source
MAIN/WIDE comparisons cover312,530 normal/partition plus90 recovery values;
the remaining50 are the causal pair. Exact candidates cover200,750 normal/
partition,20 causal and180 recovery words. Every old/input/coefficient hash,
physical guard, candidate suffix, active store witness and status transition
passes. Twenty-four release encode measurements report zero Rust allocations;
driver allocation is outside that counter.

The partial-overflow control also overflows the Half WIDE store, so status alone
does not isolate the intermediate check. This limitation was recorded before
hardware outcomes; the explicit finite checks are additionally reviewed in the
shader. Invalid NaN/overflow raw outputs remain evidence, not an equality oracle.

Full checks pass formatting, strict default/Metal all-target Clippy,450 default
and536 Metal-feature tests and optimized release test compilation. The shader
and wrapper are new; existing CPU convolution, RMS and gating remain unchanged.
The first unrun controller draft omitted its log-cap constant. Astra caught it
before launch; the reviewed successor restores16MiB per-stream caps and checks
both global bounds as well as the reused supervision functions. No numerical
source, gate or executable changes accompany that correction.

Both GPU jobs close cleanly under the qualified pipe-only controller. Actual
raw logs total3,347,784 bytes, below the4,521,573-byte pre-run bound and16MiB
per-stream cap. Usage samples are16 and1 early observations, not complete process
peaks or model-fit evidence. The exact executed release bytes are retained after
success for later Cargo rebuilds; the receipt distinguishes the original executed
path from that identical retained copy.

[Machine evidence](../benchmarks/results/2026-10-29-causal-conv3-half-v1.json)
records source/host/native reviews, raw audit, executable identity and the unrun
controller hold. This closes only the raw native leaf. Resident model publication,
strided binary products, multi-KV staging, real-model/app/text, full quality,
performance and iPhone remain separate [LFM2 tasks](lfm2-hybrid-plan.md).

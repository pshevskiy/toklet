# Resident FP32 packed projection for Gemma

The standalone Metal projection now passes its frozen mathematical/scalar
contracts on Apple M1 Pro. It consumes original packed weights with FP32
activations and outputs. The older operator's Half activation path is unchanged.
This is a G4 prerequisite, not complete Gemma execution or an app/device grant.
The current admission also has the independently frozen per-token L1 fallback
described below; the original max proof remains the first, unchanged fast path.

## Fixed storage and ownership

The format is four-bit unsigned codes, low nibble first, with group 64 original
BF16 scales and affine biases. Weights are row-major `[outputs, columns/8]` U32;
metadata is `[outputs, columns/64]` BF16. Inputs are contiguous FP32
`[tokens, columns]`, outputs `[tokens, outputs]`; there is no linear bias. Columns
are multiples of 64 through 12288, outputs 1..16384 and batch capacity 1..32.
These bounds cover projection widths, not an entire embedding vocabulary.

A standalone Context caches the existing Gpu pipelines and one Safe/Precise
projection pipeline, with contraction disabled. The 64 MiB context allowance is
policy, not measured residency. An owner uploads original codes/coefficients
once and reuses fixed input/output/status buffers for batches 1..capacity.
Smaller batches retain sentinels across inactive capacity; later full batches
reuse the same weights. Six buffers include seven trailing guard words each.
Admission counts code/metadata padding, scratch and the Rust owner before
numerical allocations, and checks device buffer limits. There is no expanded
FP32 weight matrix. Maximum admitted geometry is about 108 MiB packed tensors
plus 3.5 MiB numerical scratch; it is not exhaustive GPU qualification of all shapes.
The largest owner exercised here is 168,640 bytes, including a 240-byte structure.
Caller storage and simultaneous owners require separate admission.

One queried SIMD group handles each token/output row. Adjacent lanes traverse
adjacent columns. Original BF16 coefficients promote to FP32; explicit fused
multiply-add reconstructs a weight once. Product and explicit FMA product
residual feed compensated pairs, then increasing-offset cross-lane merges and
FP32 output. Shape/format selection stays outside scalar loops. Observed SIMD
width is 32, queried rather than assumed.

The existing Float32 MPS path was inspected as an alternative. It expects dense
matrices; expanding the full model is outside the 6 GiB policy. Bounded dequantized
tiles plus MPS remain a viable prefill comparison. This task introduces a small
correctness matvec, without a tuned GEMM or speed claim. A substantial optimized
matrix path requires representative completed measurements first.

## Conservative numerical admission

The primary reference reconstructs original BF16 coefficients in FP64, stores
FP32 weights, then keeps ordered FP64 and exact rational dot products distinct.
Metadata and every possible reconstructed code 0..15 must be zero or FP32-normal;
weights must also have magnitude at most 2^32. Unused codes are checked too.
This prevents flushed coefficient/weight amplification. With columns at most
12288 and weights at most 2^32, lost original subnormal-input contributions sum
to less than 2^-80, far below the unchanged absolute gate. No bitwise underflow
or universal FP64-accuracy equivalence is claimed.

Before staging, the original fast proof checks both bounds, computed with
upward-rounded positive FP64 products and exact-zero handling:

- `S >= columns * max_abs_weight * max_abs_input`, with `S <= F32_MAX/4`.
- `U >= 2 * columns * 2^-53 * S`, with `U <= 1e-5/16`.

If this proof rejects, the current engine tries the tighter L1 proof below
under the same S/U limits. Both failing returns the original preflight error.

Products of FP32 values are exact finite FP64 values here. For the bounded column
count, `gamma_K = K*u/(1-K*u) <= 2*K*u`, where `u=2^-53`. Thus the second bound
keeps ordered-FP64 versus exact-dot disagreement below one sixteenth of the
absolute gate. It is a conservative execution subset, not a theorem bounding
every GPU error. Actual Gemma activation-domain fit remains open.

Astra found the reason to distinguish the bounds before target code existed:
for 64 unit weights and input `[2^100, 1, -2^100, 0, ...]`, ordered FP64 returns 0,
while the exact dot is 1. The overflow bound alone admits it. The original design
and both oracle values remain retained; the reviewed summation bound rejects it
before dispatch. A separate finite-reference case with naive FP32 product
overflow rejects for its absolute-product bound. Adjacent FP32 inputs immediately
below/above the summation cutoff freeze the boundary before implementation.
Zero-weight owners admit even maximum finite inputs with exact-zero bounds.
No numerical tolerance is changed.

## Output transaction and diagnostics

Constructor/domain/budget errors publish no owner. Input preflight errors leave
the caller's output unchanged and the existing owner reusable. Once staging starts, any failure permanently
poisons the owner. Commands retain resources, CPU shared-buffer access requires
checked completion, and output publishes only after guards, inactive capacity,
status and finite-result checks. No legitimate completed overflow is expected
inside the admitted sum envelope; driver/in-flight failure injection is not
qualified. This diagnostic uses explicit host copies/readback; whole-graph
resident scheduling and shared contexts remain later work.

The pre-target [producer](../tools/reference/gemma_affine_math.py) uses Fraction
accumulation and stores exact rational sums separately from ordered FP64 values.
Two owned captures match byte for byte. The
[independent Decimal400 replay](../tools/reference/check_gemma_affine_math.py)
and Astra's independent exact replay verify all 27 cases / 306 outputs, codebooks,
packing and admission. Twenty-one cases admit, three reject input admission and
three reject constructor domains. They cover actual widths, non-Half FP32 inputs,
odd output rows, mixed scales/signs, repeated rows, varied lane/group cancellation,
and deep 12288-column cancellation retaining residual `2^-14`. Twelve portable
reference methods pass. Fixture hashes are bound in the
[manifest](../tests/fixtures/gemma-metal-affine-v1/manifest.json).

Four unchanged [previous cases](gemma-packed-reference.md) are also exercised.
Their historical ordered outputs are already FP32, promoted for comparison;
their mathematical contract is affine-ideal FP64. They are not relabeled as new
exact-rational or unrounded ordered-FP64 references. Their reconstructed weights
happen to be exact FP32. The shared test helper and raw main-case records retain
these distinct contract labels.

The portable scalar test checks 330 ordered outputs and 324 admitted mathematical
comparisons. The conflicting scalar/exact result remains an explicitly rejected
diagnostic. All 346 CPU / 372 Metal-feature tests, formatting, both strict checks,
iOS device/simulator compilation and one exact optimized admission method pass.
Sixteen CPU and 114 Metal-feature ignored tests remain separate hardware/real-model
suites. The empty
default MLX CLI harness is independently listed as empty, without execution credit.
Thirty-one owned verifier regressions exercise complete bare/prefixed transcripts
and corruption of numerical output, packed tensors, odd-BF16 padding, active/
inactive capacity, status, ownership, reference labels and rejection/recovery.

Completed optimized GPU execution passes 25 cases twice: 648 output values and
1296 comparisons across the two correctly labeled contracts. Fifty same-owner
single-token partitions match full-batch bits; 25 returns to full capacity retain
those bits. All immutable buffers, padding, guards, inactive capacity and status
match. Three constructor rejects, three input rejects and subsequent valid
recovery pass, as do six malformed-input/capacity preflights and maximum-input
zero-weight recovery. Maximum absolute error is 5.3641443287233415e-8; maximum gate
fraction is 0.00040189941001414194 under unchanged
`1e-5 + 1e-4 * abs(expected)` gates.

Hardware is Apple M1 Pro, MacBookPro18,3, 16 GiB, macOS 26.6.2 (25G83). Completed
process peak RSS is 34,537,472 bytes and physical footprint 22,479,424 bytes,
within the separate 1 GiB standalone policy. These are not whole-model memory or
inference-latency measurements. The
[receipt](../benchmarks/results/2026-10-25-gemma-metal-affine-v1.json) binds source,
fixtures, owned commands, exact release binary and first captured buffers.

## Decision and remaining work

Retain the operator as an FP32 packed-projection correctness prerequisite.
Existing shaders, scalar kernels, old families, ABI and application limits remain
unchanged. Whole-model upload without redundant copies, embedding/tied-head
chunking, FP32 attention/KV/pointwise graph composition and actual activation
admission need their own evidence. No app or phone profile is enabled.

G2/G3/G4 remain open, as do 57 cross-precision failures in five arrays, 27 native
FP64-ideal logit failures per raw schedule, 15 dense and 15 packed source-GPU
intermediate failures and eight long-rotary source failures. Original-BF16,
broader quality, representative performance, Mac/phone fit and 131072 output
tokens plus a nonempty prompt remain open. No prior rejection is waived.

## Per-token L1 fallback qualification

The [independently frozen L1 witnesses](gemma-affine-l1.md) now drive an
allocation-free fallback in input preflight. When the old proof rejects, sum
FP32 magnitudes separately per token in FP64: skip zeros, copy the first
positive term exactly, then take the nearest sum followed by one `next_up`
step for each subsequent positive term. Take the maximum row sum A and compute
upward `S >= maxW*A`, then `U >= 2*K*2^-53*S`. Limits remain F32_MAX/4 and
1e-5/16; shader, codebook/shape limits, buffers and ownership are unchanged.
Both proofs failing preserves the original error and pre-staging transaction.

The portable scalar check compares all 47 new outputs with frozen ordered
references and 42 admitted outputs with the independent mathematical contract.
The Metal-feature portable method matches every per-row sum, S/U and original
max-bound bit pattern; test-only counting confirms old passes never traverse
the fallback. All 347 CPU / 374 Metal-feature tests, both strict checks,
formatting, iOS device/simulator compilation and one exact optimized L1 method
pass. Sixteen CPU / 115 Metal ignored cases remain explicit separate suites.

Completed M1 Pro execution passes 15 new cases twice: 84 output values and
168 dual-contract comparisons, including twelve cases newly admitted by L1.
Sixteen same-owner single-token partitions and 15 returns to full capacity
preserve output bits and original weights. Four rejected cases preserve caller
output and reusable state, then recover with valid zeros. A warmed batch whose
first row itself needs/passes L1 and whose second row rejects leaves all six
device buffers unchanged and recovers. Owner size remains 240 bytes; the
largest new owner exercised is 119,476 bytes. Maximum new absolute error is
1.401298464324817e-45, the admitted tiny-input loss, under unchanged gates.
This does not grant bitwise subnormal equivalence.

The unchanged previous GPU cases also pass again: 648 outputs, 1296 contract
comparisons, 50 partitions, domain/input rejects and recovery. Ten owned
capture-checker regressions reject changed original weights, guard/status
damage, wrong token rows, repeat differences, lost records and late-failure
buffer mutation. Coherently changing both repeats, partitions and resume to
one finite wrong value fails the numerical gate itself.

On MacBookPro18,3 / Apple M1 Pro / 16 GiB / macOS 26.6.2 (25G83), the new GPU
process peaks at RSS 26,460,160 / physical 15,385,176 bytes under separate
1 GiB limits. The old-case process uses 34,521,088 / 22,463,088 bytes. The
capture regression process uses 72,204,288 / 31,736,384 bytes under separate
512 MiB limits. These qualify these processes, not model residency or speed.
The [fallback receipt](../benchmarks/results/2026-10-25-gemma-metal-affine-l1-v1.json)
binds source/build identity, exact optimized binary, complete raw captures,
resources and process closure.

Retain the standalone fallback. The real frozen trace has sufficient reference
bounds for 276 calls plus a full-head geometry rejection, but this task does
not execute those real projections or establish native activation parity.
Table/head chunking, resident full graph, original source/precision failures,
quality/performance, app/phone admission and 131072 output plus prompt remain
open. No device capability or earlier failed numerical result changes.

## Complete real-checkpoint projection qualification

The [explicit real-projection run](gemma-real-projections.md) now passes all
276 fitting calls with complete frozen inputs, original packed/BF16 tensors,
two full repeats, partitions/resume and independent device-buffer replay.
All repeated GPU outputs match both rounded FP32 references for this trace.
Keep the 206 captured-state inputs distinct from 70 mathematical GELU products.
The full head's exact geometry rejection is retained. This closes the standalone
short-trace projection gate; native activation parity, table/head chunking,
resident graph, source/precision failures and app/device/long-output gates remain.

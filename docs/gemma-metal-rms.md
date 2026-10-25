# Standalone resident FP32 Gemma RMS

The fixed-shape weighted/unweighted Metal operator now passes independent
mathematical and scalar checks on Apple M1 Pro. It provides FP32 normalization
needed by the Gemma4 graph. Whole-model GPU execution, source parity, performance,
app admission and iPhone qualification remain open.

## Numerical contract and implementation

Inputs/outputs are contiguous row-major `[rows, width]` FP32; optional immutable
weights are direct learned `[width]` scales, with no implicit `+1`. Unlearned
normalization uses unit scales. Accepted geometry is rows 1..256, width 1..1536,
and exactly `1e-6_f32` epsilon. Other epsilon values reject explicitly. Actual
reference shapes include local head 256, global head 512, hidden 1536 and PLE width 256,
plus width 1, odd tails, zeros, extreme finite inputs and negative/zero scales.

One queried SIMD group processes each row. Adjacent lanes visit adjacent columns.
The maximum input magnitude or once-rounded host square root of epsilon supplies
a scale, avoiding overflow in squared values. Each lane accumulates scaled
squares with FP32 compensation; increasing-offset cross-lane merges combine
both sums and corrections. The denominator is dimensionless. The mathematical
oracle retains the original exact FP32 epsilon; the cached square root is an
implementation approximation, not a replacement oracle epsilon.

A weighted result can be normal even when `input / scale` would be subnormal.
The weighted kernel therefore decodes original stored input, weight and scale
bits into normal mantissas and integer exponents, including subnormal inputs.
It combines mantissas through the denominator and applies the combined exponent
once with `ldexp`. Explicit zero/sign handling and unchanged numerical gates
cover the amplified result. This avoids flushing a meaningful intermediate.
It does not promise bitwise active-underflow behavior or universal source parity.
Both kernels use Safe/Precise math with contraction disabled; their selection
occurs at owner creation, outside scalar loops. The scalar CPU implementation
remains unchanged.

## Ownership and failure behavior

A standalone Context caches the existing Gpu pipelines and two RMS pipelines.
Its 64 MiB allowance is policy, not measured residency. Fixed owners borrow that
context and retain disjoint input/output/weight/status UInt32 buffers with seven
trailing guard words each. Weights upload once; unlearned mode has one dummy
weight word. Admission counts actual buffer bytes and the Rust owner before any
numerical allocation, and checks queried device buffer limits. The maximum
admitted geometry has 393,216 values; geometry bounds are not exhaustive GPU
qualification of every shape. The largest owner exercised by the frozen GPU
cases is 100,604 bytes; its Rust structure is 136 bytes.

Bad input shape/nonfinite input rejects before submission, preserves caller
output and leaves the owner reusable. Every failure after staging begins
permanently poisons it. Commands retain resources, and CPU shared-buffer access
requires checked completion. Caller output publishes only after completion,
guards, status and finite-result validation. The completed overflow probe retains
raw GPU output/status/guards before checking unchanged caller sentinels, sticky
poison, rejected retry and successful recreation of the same weighted variant.
Driver/in-flight failures were not injected.

This standalone correctness API copies inputs and reads outputs. Resident
whole-graph composition, shared context ownership across operators and efficient
command scheduling remain later work.

## Independent evidence and checks

The [maintained producer](../tools/reference/gemma_rms_math.py) froze 16 cases
and 57,685 outputs before target creation. Two owned captures match byte for byte.
An [independent Decimal80 consumer](../tools/reference/check_gemma_rms_math.py)
replayed every value and exactly matched all once-rounded FP32 bits; Astra's
separate Decimal90 replay confirmed them. Nine portable reference methods pass.
The fixture's SHA256 is
`935e1ecac3d8418cd4f668b5200e3e58730c0798391561bebe1b945391e859bb`.
The complete finite-input overflow witness is checked separately.

The [portable Rust integration tests](../tests/gemma_metal_rms.rs) compare the
unchanged scalar RMS with all 57,685 mathematical outputs and reject the expected
overflow. The full 345 CPU / 370 Metal-feature tests, both strict checks, formatting,
iOS device/simulator compilation and one exact optimized geometry test pass.
Sixteen CPU and 113 Metal-feature ignored tests remain explicit hardware/real-model suites, rather
than automatic qualification. The empty default MLX CLI harness is independently
listed as empty; its missing tests receive no execution credit.

Twenty-nine owned synthetic verifier methods validate correct bare/prefixed
transcripts and reject malformed/missing/duplicate records, input/output/weight/
guard/status corruption, wrong accounting, altered overflow/poison/recreation,
boolean/integer substitutions and a finite off-gate output altered consistently
in caller and device captures. This tests the result verifier, not driver failure.

The optimized completed GPU probe checks 16 cases twice: 115,370 numerical
comparisons under unchanged `1e-5 + 1e-4 * abs(expected)` gates. All 31 row
partitions match the full-row output bits; repeats are exact. Weighted-ones and
unlearned results agree numerically. Observed execution widths are 32 for both
pipelines, queried rather than hardcoded. Maximum gate fraction is
0.0011703413751736325. Maximum absolute error is 1.1552709867686105e31 at extreme
FP32 scales; it remains inside the corresponding relative gate. No tolerance is
relaxed for amplified tiny inputs.

Hardware is Apple M1 Pro, MacBookPro18,3, 16 GiB, macOS 26.6.2 (25G83). The separate
1 GiB diagnostic policy passes with 46,055,424 bytes peak RSS and 31,326,808 bytes
peak physical footprint. These are completed-process observations for the
standalone probe, not full-model residency or inference latency measurements.

For reproduction use the fixed fixture and the ignored test
`metal::gemma_rms::tests::real_frozen_math_guards_partitions_and_failure_state`
in an optimized Metal-feature build with a separate target directory. The
[receipt](../benchmarks/results/2026-10-25-gemma-metal-rms-v1.json) records source,
controller, fixture, executable, first-buffer and command identities.

## Decision and remaining gates

Retain this correctness operator as a concrete Gemma GPU prerequisite. There is
no speed claim or default kernel replacement. Existing shaders and Half RMS remain unchanged; the new shader is separate.
Model/source/codec ABI and application limits are unchanged.

G2/G3/G4 remain open. The 57 cross-precision failures in five Gemma arrays,
27 FP64-ideal native logit failures per raw schedule, 15 dense and 15 packed
source-GPU intermediate failures and eight long-rotary source failures remain
unwaived. This operator neither identifies nor repairs their graph-level cause.
Original-BF16/quality, actual resident model/KV/scheduling, device fit and real
Mac/iPhone generation remain separate. The 131072-output-plus-nonempty-prompt
requirement also remains open; no context/output limit changes here.

# Gemma4 proportional rotary precision diagnostic

The next G2 investigation compares the pinned MLX-LM `ProportionalRoPE` class
against independent FP64 mathematics before choosing native precision semantics.
This is a completed diagnostic with numerical disagreements, not an accepted
rotary fixture or implementation. The [Gemma4 plan](gemma4-implementation-plan.md)
and long-context quality gates remain open.

The input is one FP32 head/token of width512, with rotated dimensions128,
base1000000 and factor1. For pair index i<64, frequency is
`base^(2*i/512)`, angle is `position/frequency`, and coordinates i and i+256
rotate together. Remaining coordinates copy unchanged. This explicitly tests
the full-width denominator and half-head pair distance; it is not rotation of
the first128 contiguous coordinates. Deterministic values are exact multiples
of0.125. Independent FP64 sine/cosine and pair arithmetic produce the ideal.

The exact source class comes from `rope_utils.py` at MLX-LM revision
`0d4d3d04f8ab7ce4a51e7209e8615755a48ba94d`, source SHA256
`ccba2f0c7b246c008a1188dc3760c162bdd906b15a1c490461dee4be64bc62df`.
Only that captured class executes. The cold admitted NumPy2.3.2/MLX0.32.3
runtime includes interpreter/package/native/Metal file identities. CPU and GPU
run in separate owned processes on Apple M1 Pro, MacBookPro18,3, macOS26.6.2
(25G83), arm64. Timing and memory performance are not measured.

The unchanged policy is `abs(actual-ideal) <= 1e-5 + 1e-4*abs(ideal)`.
Maximum error divided by that allowance is shown below; values above1 fail.

| Position | CPU ratio | GPU ratio | Both within policy |
| --- | ---: | ---: | --- |
| 0 | 0 | 0 | Yes |
| 1 | 0.00651 | 0.00430 | Yes |
| 7 | 0.02389 | 0.06284 | Yes |
| 31 | 0.18823 | 0.10253 | Yes |
| 127 | 0.70353 | 0.34157 | Yes |
| 511 | 1.69195 | 4.90384 | No |
| 1023 | 4.47092 | 3.10987 | No |
| 8191 | 38.02952 | 59.36941 | No |
| 131071 | 642.25824 | 314.81909 | No |

At131071, maximum absolute differences are approximately0.0130611 CPU and
0.00890017 GPU. Passing isolated positions through127 does not establish a
context limit, whole-graph agreement or quality. No tolerance was changed and
no failing observation was removed from acceptance.

All input/output/frequency/ideal arrays are retained before validation, including
inactive positive-infinity frequencies in raw binary. Independent stdlib replay
authenticates the captures and all eighteen comparisons. Saved CPU/GPU active
frequencies differ. As a diagnostic only, reconstructing from the captured CPU
frequencies with a rounded FP32 reciprocal and FP32 position multiplication
reduces the long-position residual to about1.78e-7. Equivalent GPU reconstruction
still leaves about0.01157 error, so it does not establish the complete GPU cause.
Captured upstream frequencies cannot admit an independent production oracle.

The published MLX0.32.3
[Metal source](https://github.com/ml-explore/mlx/blob/v0.32.3/mlx/backend/metal/kernels/rope.metal)
uses FP32 phase arithmetic and fast sine/cosine for this operation. That source
corroborates mechanisms to investigate; it does not prove how the captured
installed binary was compiled or fully explain the GPU residual.

Astra required stronger interruption, restoration, raw-capture and collision
handling before the first launch. Unexecuted proposals are retained separately.
Both final child commands exit successfully, with clean endpoint/capture guards;
their success means observations completed, not numerical acceptance. The
[receipt](../benchmarks/results/2026-10-24-gemma-rotary-precision-v1.json) binds
exact diagnostic scripts, runtime/source, raw arrays, replay and environment.
Protected CPU binaries and all tracked engine code remain unchanged.

Next: resolve the precision contract with independently generated frequencies,
explicit phase arithmetic and broader shape/chunk/offset fixtures, retaining the
FP64 oracle and these disagreements. Then qualify a native scalar operation and
its Metal counterpart without silently adopting one backend's errors. No Gemma
model eligibility, output/context capability, phone package or131072-output
qualification follows from this diagnostic.

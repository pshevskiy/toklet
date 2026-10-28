# Gemma3 Half GELU and embedding scaling

29 October 2026. B3c2 adds two internal optional Metal operations: tanh-GELU
and scalar scaling. The [preceding RMS task](gemma3-half-rms.md) supplies the
other missing precision primitive. These leaves remain test-only until the
resident graph composes them; they do not enable model loading or Chat.

## Arithmetic and reuse

The GELU operation consumes stored Half input and evaluates the prescribed
FP32 sequence: cubic, scaled cubic/addition, `sqrt(2/pi)` multiplication,
precise tanh, addition and final product. Contraction is disabled. The result
is stored in a physical Half activation buffer. The existing `Kernel::Gating`
then multiplies that buffer by the stored Half up projection and writes Half.
No new product kernel or operator framework is introduced.

The scalar operation consumes Half input, multiplies by FP32 factor 8 or 25.25
and stores Half. For the selected Gemma3 checkpoint, 25.25 is the BF16-rounded
embedding factor, not an unrounded `sqrt(640)`. Future graph composition must
first store the packed gather result as Half, then scale that stored result.
The scalar leaf cannot independently certify the actual gather implementation.

The checked plan accepts rows 1–32, GELU widths 128/2048 or scale widths 64/640,
and separate source/destination row strides. Kernel and factor selection occurs
once at the operation boundary. Only active columns are touched. UInt32 indices,
rounded dispatches, operand lengths, disjoint buffers, device identity and
retained command references are validated before creating the encoder.

One owner compiles both safe/precise pipelines once. The caller owns guarded
Half input/output, status, command completion and the additional tight Half
activation/up/product buffers when composing gating. Encoding uploads no
operands, resets no status and waits for no work. Release Rust allocations are
measured separately from Objective-C/Metal encoder allocations. Compiler/context,
caller and diagnostic allocations are separate from checked numerical buffers;
the constructor checks one 64 MiB compiler reserve plus its Rust owner, not a
reserve per layer or a measured driver-residency figure.

## Frozen source checks

The independent stdlib-only producer exports four cases with 152,512 output
values per profile across six case/stage groups. Main rounds prescribed steps
to FP32; wide evaluates in FP64. Both preserve the Half activation before the
product. Export and replay match byte-for-byte. Astra independently recomputes
all saved main/wide words, operands, padding, factors and gates.

Each stage uses the prospectively frozen MAIN gate
`atol=max(1e-4,2*envelope), rtol=1e-3`; wide remains diagnostic. Exact controls
add narrower checks that the tolerance could miss:

- GELU input .25 and up 5 require activation `0x30ca` and product `0x39fc`.
  Delaying the intermediate Half store produces the rejected `0x39fd`.
- Signed subnormal inputs `0x0002/0x8002` require GELU `0x0001/0x8001`,
  then existing gating must produce `0x0005/0x8005`.
- Scaling input 1 by 25.25 requires `0x4e50`. An analytical source witness
  distinguishes scaling before the Half gather store, which gives `0x4e51`.
- Smallest signed Half subnormals scaled by 8/25.25 require signed words
  8/25 respectively. Seven invalid controls require sticky error status;
  invalid output bit patterns are retained but are not a gate.

## Native consumer and qualification

The consumer checks a strided GELU result and, separately, a tight physical
Half activation consumed by existing gating in the same command. It compares
the tight and strided active words exactly. Scale is checked on its own strided
buffers. This produces 218,432 source comparisons, including repeated activation
coverage; it is not 218,432 distinct source values.

All raw buffers, command status/error, status guards and immutable operand hashes
are retained before acceptance assertions. Three short-buffer checks and a true
equal-extent alias control exercise preflight rejection. Seven invalid controls
run error, sticky-valid and explicitly-reset-valid phases. Successful command
completion requires completed status and no Metal error. The consumer measures
eight normal release encoding calls, including reused gating; driver allocations
remain outside the Rust measurement.

Formatting, strict default/Metal Clippy, 420 default tests, 302 Metal-library
tests, two targeted debug and two release cold controls pass. Counts overlap.
Both explicit release GPU jobs pass on Apple M1 Pro. Astra independently verifies
all 218,432 MAIN comparisons bit-for-bit, with maximum error/gate fraction zero.
WIDE has 21,051 differing words, maximum absolute difference
`4.76837158203125e-7` and zero diagnostic gate exceedances; it is not bit-exact.
All eight measured release encodes have zero calling-thread Rust allocations.
Every raw buffer, exact store/subnormal control, guard, immutable input/hash,
status transition and command status/error passes. The 21 recovery phases
include 5,888 checked valid output words.

The first cold run stopped on Clippy's redundant `width: width` spelling before
target tests; the corrected shorthand preserves numerical code and gates, with
the failed receipt retained. Both owned GPU process groups close normally.
Direct-process memory samples, especially the recovery job's single sample,
cannot establish an execution peak or whole-model fit. No graph, real-model,
app, iPhone, latency, throughput or quality qualification is implied.
See [machine-readable evidence](../benchmarks/results/2026-10-29-gemma3-half-pointwise-v1.json).
The [checklist](../TODO.md) continues with rolling KV state and full resident
graph composition.

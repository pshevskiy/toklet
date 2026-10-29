# Gemma3 native-store reference contract

29 October 2026. The independent Gemma3 reference now defines the intended
mixed precision explicitly: accurate projection stores, FP32 normalization and
serial FP32 reference attention. Eighteen portable controls, exact source export/
replay and independent saved-byte review pass. Native qualification under this
reference remains a separate step.

## Why the contract changed

The previous complete native capture passed state, schedule, codec and resource
checks, but its numerical reader rejected92 of18,537,288 MAIN comparisons.
All failures were intermediates; no final or teacher logits failed. That result
remains rejected with its raw arrays and old reference intact.

Independent reconstruction found that the inspected native RMS and projection
were correct for their own inputs. An earlier rounding boundary explained why
the two CPU profiles did not enclose the native implementation. At layer0,
France token row3, FP32 normalization reaches the exact Half midpoint−84.90625
and rounds to−84.875. FP64 normalization reaches−84.9062508376195 and rounds to
−84.9375. Accurate projection of those different inputs changes158 V values.
A separate one-request six-token GPU trace directly confirms the frozen FP32
normalization and V predictions. Astra independently verifies all13 snapshots,
42,528 finite values and11,520 own-input RMS words.

The former MAIN used FP32 BLAS projections; WIDE used FP64 throughout. Native
execution combines accurate compensated projection stores with FP32 RMS and
serial FP32 attention. Two empirical CPU profiles do not form an interval for
that combination. A compensated-attention candidate was held: its local score
accuracy improvement changed only one context word and did not explain the
larger profile divergence. No numerical engine change follows from this
investigation.

## Uniform reference arithmetic

MAIN uses `NativeHalf_certified_projection_serial_attention_v3`. It applies the
same rule to every projection, including the tied FP32 head. Finite Half inputs
and reconstructed FP32 weights have exact binary64 products. At most1,024 output
rows are converted per stripe. An outward-rounded summation-error bound
certifies a destination only when both interval ends round to the same finite
Half/FP32 word. Ambiguous midpoint, cancellation or underflow cases use an exact
signed integer dot in units2^-173 and direct nearest-even destination rounding.
Plain binary64 BLAS alone is not claimed to be correctly rounded. Certification
assumes the pinned BLAS evaluates ordinary binary64 dot products; this is not
a qualification of unexamined backends.

FP32 Neumaier RMS, effective FP32 `1+raw BF16 gamma`, rotary conventions, explicit
Half stores, clipped residuals, causal/local masking and stop semantics retain
their declared meanings. MAIN attention follows serial FP32 reference order;
vectorization spans independent outputs. No GPU code or output table is imported
into the producer. WIDE's arithmetic and all3,232 saved array hashes remain
identical to the earlier reference, as do teacher input histories.

Every old per-group atol/rtol dictionary is copied literally from the frozen
reference. New MAIN/WIDE envelopes are diagnostic and cannot increase a limit.
The MAIN center changes, so `rtol*abs(MAIN)` and the resulting acceptance set
also change. This is a separately versioned reference-contract correction;
it is not an unchanged baseline or a retroactive pass for the old rejection.

## Validation and limits

Both exports contain3,232 groups/6,464 arrays and13,473,568 finite values per
profile. The97,110,272 raw bytes replay exactly. Independent review verifies
all legacy coefficients, WIDE hashes, teacher chains, masks, norms, residual,
gating and embedding stores. The projection ledger accounts for143,984,128
stores:143,857,582 interval-certified and126,546 exact fallbacks. All32 exported
fallback words are independently checked for nearest-even rounding and original
weight-row identity;31 exact dots are reconstructed from retained inputs. One
input was not retained, and its exact sum cannot be reconstructed independently
from a hash alone.

The4,931,096,570-byte prospective source ledger remains below6GiB. Actual maximum
weight conversion is10MiB per stripe. The two worker runs take137.51/139.16
seconds, with OS RSS peaks2,577,956,864 /2,582,560,768 bytes. These describe the
CPU reference jobs, not engine throughput. Both owned groups close before any
new native model run.

See [evidence](../benchmarks/results/2026-10-29-gemma3-native-store-reference-v1.json),
[previous reference](gemma3-real-reference.md),
[original-byte loader](gemma3-original-loader.md) and [TODO](../TODO.md).
Application, iPhone and original-unquantized adoption quality remain separate.

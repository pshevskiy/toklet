# Gemma frozen-reference precision localization

The real E2B FP64-versus-staged-FP32 discrepancy is now localized at all five
failing saved states, with same-input endpoint replay and original source-backed
coefficients. This is counterfactual analysis of existing independent references,
not a new native/GPU trial, a corrected model, a product precision selection or
closure of the other G2.1 discrepancies. Original gates remain unchanged.

See [public evidence](../benchmarks/results/2026-10-26-gemma-precision-localization-v1.json),
[original complete reference](gemma4-text-math.md), [real hybrid execution](gemma-hybrid-real.md)
and the [active Gemma plan](gemma4-implementation-plan.md).

## Contracts and first divergence

The original mathematical profile uses FP64 activations over FP32-reconstructed
packed weights. The staged profile rounds explicit operator/residual/product
boundaries and selected constants to FP32, with FP64 reductions. This is neither
an upstream original-BF16 implementation nor a description of GPU accumulation.
Both use the same original checkpoint and four IDs `[2,100,257,100]`.

All713 states are compared in the actual frozen execution order. The canonical
shape dictionary lists the head near the beginning; that order must not be used
to identify the first failing operation. No state or failing coordinate is dropped.

The first numerical inequality is scaled embedding, index0. Its signed difference
is−9.750119889240239e−8: rounded `sqrt(1536)` contributes−5.745432818393681e−8,
and output storage contributes−4.0046870708465576e−8. Its gate fraction is only
0.0005807885708314598. Ordinary FP32 inequality is not itself a defect, nor does
this prove that this one rounding causes every downstream failure.

The first tolerance failure in execution order is layer14 post-FF normalization,
index5014 (token position3, feature406), with fraction1.3071418445078735.
The original `1e-5 + 1e-4*abs(ideal)` policy still rejects57 values:

| State | Failed values |
| --- | ---: |
| Layer14 post-FF normalization | 1 |
| Layer15 post-FF normalization | 1 |
| Layer26 Q projection | 1 |
| Tied head | 27 |
| Corresponding capped logits | 27 |

The head and logit failures are related observations, not54 independent causes.
These are reference cross-precision failures. The separate native hybrid record
has its own captured outputs and unchanged comparison results.

## Ordered counterfactuals

Every analysis first reconstructs a saved endpoint from its complete saved input
and original learned coefficient. FP64 reconstruction must satisfy the existing
`1e-13 + 1e-13*abs(expected)` reference-replay bound; staged endpoints must match
FP32 bits exactly. All31 cases meet these prerequisites. The FP64 projection
residual includes differences between the original NumPy/BLAS reduction and
`math.fsum`; it is not production GPU error.

RMS uses the complete1536-wide input row, direct learned scale and the original
FP32-representable epsilon. The path first changes the selected numerator while
holding the ideal denominator, then changes the denominator using the full
staged input, then rounds the output. Nonlinear contributions depend on this
order; this is an explicit path rather than unique causal attribution.

| Endpoint | Inherited numerator/input contribution | Denominator contribution | Final storage contribution |
| --- | ---: | ---: | ---: |
| Layer14 norm | −1.6171505842012962e−5 | −1.5148471435577626e−9 | +5.922084003739947e−11 |
| Layer15 norm | +1.6475210199204567e−5 | −7.043926414374013e−10 | +2.1306193788428462e−11 |
| Layer26 Q | −1.324626282679306e−5 | Not an RMS operation | +5.362608135328628e−10 |

The Q projection reproduces the original endpoint from its staged input and
original packed row. Its signed reconstruction residual is3.3202607330196088e−15.
For all27 head failures, inherited-input contributions range from+1.003012316419012e−5
to+1.3918883419897915e−5; the largest local storage contribution in magnitude is
7.307860983019054e−10. Maximum reconstruction residual is3.403256409323374e−15.

The head→softcap path is also replayed: inherited head change, rounded division
by30, rounded `tanh`, then rounded multiplication by30. The respective maximum
local rounding magnitudes are4.0745357435256757e−10,8.399522084812094e−10 and
4.656612873077393e−10; signed residuals are zero. The inherited head effect is
as large as1.3918539707936245e−5. These local rounding steps do not dominate the
selected logit failures. Substituting an ideal input is a counterfactual, not an
available one-operator production fix.

## Authentication, capture and verification

The analysis authenticates both complete original reference files and the
independent packed-storage inventory. It consumes complete source components
while retaining only selected excerpts: two learned norm vectors, one Q row and
27 embedding/head rows,30336 original bytes in total. Packed reconstruction
retains the established4-bit/group64/BF16 coefficient conventions and FP32
reconstruction boundary; no entire dense model is expanded.

The producer hashes the whole model before/after; the controller independently
binds that full digest around every owned command. The reader separately
re-extracts and hashes the eight consumed original components and header. These
are distinct authentication scopes. File descriptors, paths and bounded regular
file sizes are checked. Source excerpts and attempted counterfactuals are
flushed before their later gates; incomplete attempts cannot pass replay.

Six mechanism and eight capture/corruption controls pass, including original
packing, FP32 boundaries, same-input rejection, source tampering/short reads,
reordered states, altered contributions, softcap phases and
partial journals. All four owned process groups close. Producer physical peak
is136151784 bytes below its512MiB diagnostic policy. All engine/shader files and
protected executables remain unchanged; no unrelated engine benchmark or Cargo
suite is needed for this reference-only task. Link and whitespace checks pass.

## Decision and next acceptance

Retain the existing FP32 diagnostic and unchanged original gates. This evidence
supports **deferring an isolated head/norm rounding change**: inherited input
differences dominate at these selected endpoints. It does not locate every
upstream contribution or justify changing one operation, default or tolerance.

Freeze held-out/source-quality evaluation next to inform a product precision
choice. Keep a named candidate's numerical contract distinct from original
BF16 behavior and document the actual checkpoint/tokenizer/runtime. The separate
source-BF16, prior tiny dense/packed GPU and long-rotary discrepancies still need
their own resolution. G2.1/G5 are not closed by this localization. Resident
scheduling, progressive32/512/model-window qualification, app integration,
phone fit and the wider architecture/Hugging Face plan remain open.

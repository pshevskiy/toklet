# Gemma3 bounded native conformance policy

Status, 29 October 2026: **prospective acceptance policy; not qualification**.
This corrects the B3d comparison method before checker implementation or a new
target collection. It does not change the engine or the frozen
[native-store reference](gemma3-native-store-reference.md). The human authorized
Astra to advise, investigate and correct the plan autonomously. The earlier
[Qwen3 native-profile decision](qwen3-native-profile.md) distinguishes native
implementation conformance from strict CPU-trajectory fidelity; its model results
are not inherited by Gemma3.

## Why the comparison changes

The original coupled-trajectory screen rejects 92 MAIN values. Its corrected
native-store-reference rescore still rejects 84 of 18,537,288 values, with 98
WIDE diagnostic exceedances. Both results remain **rejected**. The 84 failures
are 42 distinct case/row/stage/channel values repeated in exact native schedules.
Saved-operand investigation reproduces all 42 failing scalars independently;
33 complete own-input operator groups contain 24,984 values with zero failures
under their existing local numeric gates. This includes eight exact-dot
projection certificates; a separate ninth projection witness is outside those
42 scalars. It does not establish correctness of all unexamined operators.

All final and teacher readouts already pass their strict coupled bounds:
12,582,912 values, with zero WIDE output exceedances. Independent witnesses also
show that small same-input GELU and rotary differences fit the published Metal
intrinsic accuracy contracts. Those are compatibility proofs, not observations
of hidden GPU intrinsic results. NumPy bit identity is not the provider contract.

The new acceptance set is explicit: validate local operator semantics on actual
native operands, while retaining strict coupled attention and output checks.
Local conformance alone cannot bound accumulated error. No coefficient fitting,
new reference center, coordinate whitelist or automatic waiver follows from the
saved failures.

## Frozen identity and domain

The artifact remains `mlx-community/gemma-3-270m-it-4bit` revision
`ff1143e3a10547c9f2129e94ca37059b096b23f4`, original Q4/group64/BF16 affine
payload and raw BF16 norms. Source3 MAIN and WIDE, all 3,232 literal legacy gate
dictionaries and teacher input chains remain fixed. Source3 raw arrays SHA256 is
`355683397397c40cc744e6ba32e69b42f9fa920f0a4376b3a90f7240bd5f9657`;
the legacy gate inventory SHA256 is
`f40fa2275a9c9f193287f32f2764a3dbf9e4d8836a56706583df1d5555151bdd`.

This policy covers the selected 18-layer geometry H640/I2048/Q4/KV1/D256,
capacities 1–544, absolute positions 0–543, local window512, theta10000 or1000000,
and the existing selected rows, partitions, teacher chains and generation/state
protocol. Declared32768 is distinct. Preserve existing MLX normal-metadata and
finite reconstructed nonzero-weight magnitude ≥2^-79 admission. Stored Half
operands must be exactly finite Half values; intermediate FP32 arithmetic uses
round-to-nearest-even and the declared operation order. There is no new generic
flush-to-zero or alternative rounding allowance.

## Every selected value retains a fatal check

| Route | Groups | Values | Acceptance |
| --- | ---: | ---: | --- |
| Coupled attention scores | Included below | 307,620 | Source3 MAIN and existing coefficient gate |
| Coupled attention context | Included below | 294,912 | Source3 MAIN and existing coefficient gate |
| Coupled final and teacher logits | Included below | 12,582,912 | Source3 MAIN and existing coefficient gate |
| **Coupled subtotal** | **624** | **13,185,444** | All remain fatal |
| Actual-input operators | **5,808** | **5,351,844** | Both semantic rules below and existing local numeric gates |
| **Total** | **6,432** | **18,537,288** | No skipped value |

The rule applies universally by stage kind. Every teacher record and every
stage named logits, attention_scores or attention_context takes the coupled
route. Full native KV operands are not retained, so attention scores/context
cannot receive an own-input exemption. Every other selected stage takes the
actual-input route. The separate 64 native-free distributions retain their
existing codec, greedy and replay checks; they are not new source comparisons.

Actual-input numeric centers come from independent operations on saved native
operands and original weights, never target outputs. The original per-group
`atol + rtol * abs(reference)` coefficients are unchanged, but these local
centers change their acceptance sets explicitly. Original coupled intermediate
drift continues to be reported diagnostically, including its rejected values.

## Actual-input requirements

- **Lookup, scale, product and clipped residual:** reconstruct original embedding
  words, apply the exact embedding factor25.25 and explicit FP32 operations/Half
  stores. Require exact declared results. Both Gemma3 residuals add in FP32,
  clip to ±65504 and store Half; the generic residual operation is unchanged.
- **RMS:** ascending FP32 Neumaier accumulation of separately rounded squares,
  then separately rounded total, mean, epsilon addition, square root and
  reciprocal. Epsilon is FP32(1e-6); effective gamma is once-rounded
  FP32(1 + rawBF16). Apply input × inverse × gamma with separate FP32 stores,
  then Half RNE. Require exact declared results; no FMA or reassociation.
- **Half projection:** use the exact mathematical dot of reconstructed original
  FP32 weights and actual Half inputs. Require the native word to be one of the
  two adjacent Half values bracketing that exact result, or the single value
  when exactly representable. This is faithful rounding, not a one-step radius
  around nearest-even. Certified FP64 intervals may prove membership; ambiguous
  cases require exact dyadic fallback. Also require the unchanged local numeric
  gate against independent nearest-even projection. Overflow is failure.
- **GELU and rotary:** derive intrinsic arguments through the declared FP32
  operations, evaluate independent true-result bounds, propagate the published
  admissible intrinsic values through every remaining FP32 operation and Half
  store, and require membership. Also require the unchanged local numeric gate.
  No output epsilon may be inferred from observed discrepancies.
- **Softmax:** use the actual full materialized score row and exact absolute
  causal/local mask. Compute the visible maximum and exact FP32(score − max)
  arguments, which must be nonpositive. Propagate admissible exp values through
  increasing-key FP32 mass accumulation and division without unproved
  independence/correlation shortcuts. Masked probabilities remain exact +0.
  Require both membership and the unchanged local numeric gate.

For graph-local result equality and projection membership, +0 and −0 are
numerically equivalent; report their raw bits. All nonzero exact-store results
must match the declared bits. Existing explicit signed-zero component witnesses
and masked probability +0 requirements remain fatal. Missing operands, an
unknown stage/layout, unsupported intrinsic domain, nonfinite value,
unproved subnormal behavior or an inconclusive arithmetic bound is a hard
failure, never a skipped check or a generic “Metal-permitted” escape.

The published basis is Apple's [Metal Shading Language Specification](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf),
4 June2026, Table8.1/pages368–370 and §8.6/page376. In the applicable FP32
domains, tanh has a5-ULP bound, sin/cos4, pow16 and exp4; basic arithmetic/sqrt
are correctly rounded. ULP must follow the specification's true-result
definition, including grid/binade boundaries. These are bounds on individual
intrinsics at exact arguments, not on the final GELU/rotary/softmax output.
Generated Half subnormals remain preserved. The checker must prove domain and
rounding applicability rather than assume those brief limits cover every input.

## Checks and publication order

1. Review and commit this policy and the plan correction before checker code.
2. Review a generic independent checker and bounded runtime/resource design.
   Include analytical boundary and deliberate corruption controls: wrong
   operands/gamma/masks/stores, non-bracketing projection neighbors, intrinsic
   contract violations, missing stages and incorrect qualification labels.
3. Preserve a posthoc diagnostic on the old raw capture. It can check the new
   method but cannot qualify that historical collection prospectively.
4. Freeze reviewed source/compiler/binary/runtime identities, policy and all
   coefficient/reference hashes before one fresh bounded native collection.
   Run the independent checker and Astra evidence review before a code commit.
5. Keep exact schedule, causal/cache prefix, transaction/replay, codec/stop,
   guards, allocation, process ownership and 6GiB memory requirements. Preserve
   existing primitive/tiny qualifications and their limits.

Completion here closes only selected bounded same-artifact Mac arithmetic/state
validation. Original unquantized10,000-target/100-task adoption quality remains
B5/M8 work, as do production Chat/Hugging Face integration, supported context
controls, whole-app memory and independent iPhone qualification. There is no
arbitrary-model, full-window, exhaustive-input or performance claim.

## Saved investigation evidence

Artifacts remain local, outside the tracked source tree. Exact review/evidence
identities are retained for subsequent publication:

- Source3 coupled rejection: `new-goal-b3d-native-store-posthoc-v3/check.json`,
  SHA256 `a0d59c80c2715209e45fd3590ac25eba7e805a755b0e7c5365188fe023545f8d`.
- Saved-operand diagnosis: `new-goal-b3d-native-store-rejection-investigation-v1/diagnosis.json`,
  SHA256 `7d55e863d777572a93c63004624da7c9198b5823de22059024737867b84c7017`.
- Astra design review: `new-goal-b3d-native-store-rejection-independent-v1/policy-design-review.json`,
  SHA256 `366f04c25426c2e10777637d574cc7af415d3aaa86902396a30a57fe1688aa6d`.
- Cached Apple specification SHA256:
  `41538b30d2f1140a5b2a0c84ce0a9f7b67bf0c707e224cfea0bfe5a44aa26cf5`.

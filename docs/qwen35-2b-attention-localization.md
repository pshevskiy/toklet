# Qwen3.5-2B bounded attention localization

Execution of this local investigation is closed. None of the 24 fixed grid
policies or two additional fixed QK algorithms passes both unchanged profiles
in both selected cases. No Rust numerical change is retained. Astra approves
this bounded evidence closure with no findings. The published `f53af39b`
approval covers the earlier diagnostic code and rejection evidence only.
[Review and publication](../artifacts/qwen35-2b-attention-2026-11-13/velocity-publication-checks.json).

The original context17 result remains rejected: three of 680,841,024 profile
coordinates fail, all in ideal FP64, with zero staged FP32 failures. Exact
native traces, schedule/reset controls and causal-prefix checks do not waive
that paired numerical gate. [Original paired decision](../artifacts/qwen35-2b-payload-2026-11-13/context17-paired-decision-1.json),
[published diagnostic scope](../artifacts/qwen35-2b-payload-2026-11-13/publication-checks.json).

## Selected operands and exact recovery

The local scope is token 12, layer 15, query head 6, KV head 1, all zero-based,
for `context17` at absolute position 12 and `offset17` at position 49. Each
selected head has 13 visible keys and 256 context features. The original
failing feature is 55. These two rows do not represent every model trajectory.

| Experiment | Retained result | Evidence |
| --- | --- | --- |
| Native operands | 34,119 arrays / 340,420,512 values match both original native traces; nine schedule/reset controls pass. Sixteen selected operands retain 61,648 bytes. | [Extraction](../artifacts/qwen35-2b-attention-2026-11-13/native-operands-1/result.json) |
| Native-input replay | Scores, probabilities and all 256 context features replay bitwise in both selected cases. | [Local replay](../artifacts/qwen35-2b-attention-2026-11-13/local-replay-1/result.json) |
| Stage isolation | 24 prespecified arithmetic/storage policies run on the same native operands. This initial grid has no recovered staged acceptance target. | [Grid](../artifacts/qwen35-2b-attention-2026-11-13/stage-isolation-1/result.json) |
| Staged prefix recovery | Both 13-token prefixes reproduce all 17,394 original staged array hashes / 173,223,232 values, with zero mismatches; 78,032 operand bytes retained. | [Recovery](../artifacts/qwen35-2b-attention-2026-11-13/staged-prefix13-1/result.json) |
| Coupled grid check | All 24 policies checked against recovered staged and original ideal context targets: 24,576 profile-coordinate checks; none passes both cases. | [Both-profile check](../artifacts/qwen35-2b-attention-2026-11-13/local-coupled-policy-check-1/result.json) |
| Two fixed F32 QK algorithms | 2,048 context profile-coordinate and 104 staged score/probability checks; neither algorithm passes both cases. | [Final fixed check](../artifacts/qwen35-2b-attention-2026-11-13/two-f32-qk-check-1/result.json) |

The unchanged gate is `abs(actual - reference) <= 1e-5 + 1e-4 * abs(reference)`
for each original profile. Prefix recovery authenticates discarded staged bytes;
it is not a new full17 comparison or a replacement reference profile.

## What the local comparisons establish

Widening only probability-times-value products or their accumulation does not
remove the selected ideal-profile failures. Ideal arithmetic on the same native
Q/K/V does remove those local failures. The retained decomposition separates
local arithmetic from upstream operand differences; it supports investigating
score/probability arithmetic, without qualifying a coupled model correction.

The grid combines three QK reductions, F32/F64 score storage, F32/F64 probability
storage and two FP64 denominator sums. Probability-times-value remains rounded
F32 products with ordered F32 accumulation. After exact staged recovery, none
of these 24 policies satisfies both original context gates in both cases.

The final two algorithms use a contiguous balanced F32 reduction tree or four
striped F32 accumulators for QK, preserving the other local stages. Both pass
all selected `context17` context coordinates against both profiles. Each fails
one `offset17` staged context coordinate, despite passing its ideal context
gate and all checked staged scores/probabilities. Ideal score/probability arrays
are unavailable here; no corresponding primitive claim is made.

The exact permitted F32 context intervals overlap in the selected cases.
Rejecting these 26 algorithms therefore establishes no universal impossibility,
no output-grid infeasibility and no claim about untested mechanisms. Local
native-input substitution never qualifies the independent coupled full graph.

## Closure and remaining gates

No further variants are authorized by this experiment. Preserve its negative
results and the original three-coordinate deeper rejection; fixed255 does not
advance. No quality, representative fit, request-performance, observability
overhead or managed-adoption gate is completed by this investigation.

Any later numerical proposal needs its own justified mechanism and acceptance
evidence under the unchanged coupled profiles. Codec/caller/stops, the original
full quality comparisons and admitted-context/state/memory/performance gates
remain required. [Real-source diagnostic and limitations](qwen35-2b-real-source.md).

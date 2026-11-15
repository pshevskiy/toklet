# Huihui-derived Qwen3.5-4B numerical rejection

Reject adoption of `TheWirelessPhoenix/Qwen3.5-4B-abliterated_4bit` at
`916d9f7c2e27ed41e2b9ed39f3a2badccb2bbc7c` under the unchanged numerical gate.
Its complete [parent/child conversion witness](qwen35-alternative-correspondence.md)
passes. The inference references have incompatible permitted ranges at 372
checkpoint-coordinate observations, so no native precision variant can satisfy
both requirements there. This decision applies to this pinned candidate.

The distinct fixed-source collector reuses the existing packed-row reconstruction
and projection arithmetic. It repeats the original 17 raw IDs through 256 causal
inputs at position offset zero, separately from empty state for each profile.
Both checkpoints retain all 1,051 final-token stages, recurrent/convolution state,
K/V, attention observations and full logits. The original bound remains
`abs(actual-reference) <= 1e-5 + 1e-4*abs(reference)` for every value in both
ideal FP64 and staged FP32 profiles.

The ordered recurrence helper preserves operation/store order; the builder
checks its exact inverse source patch. Both-profile primitive and complete-trace
controls pass. The historical full-context byte proof covers the old source's
staged profile; the new capture records this limitation explicitly. Neither
source identities nor rejected old trajectories are rewritten.

| Checkpoint | Paired values | Disjoint permitted ranges |
| --- | ---: | ---: |
| 128 | 18,180,608 | 54 |
| 256 | 20,310,528 | 318 |
| Total | 38,491,136 | 372 |

Counts include committed-prefix coordinates captured in both checkpoints.
All failures are in attention K/V caches: 262 K observations and 110 V
observations. Matching token choices would not clear this elementwise gate.
No new native adapter, quality campaign, request benchmark or adoption follows.

The complete capture contains 4,204 arrays and 461,893,632 bytes. Its body SHA256
is `13da6d02d4647e9ba11f09b259d5677a963a86e96cd5045bc5ae51eddc0e5f31`.
The 8,534.24-second collection passes its 6 GiB guard, including the helper,
with 1,365,065,728 bytes maximum observed charge, unchanged source/runtime
endpoints and clean teardown. These are reference-collection observations;
native model fit and inference performance remain unqualified.

The bounded reader authenticates all array hashes, finite values, complete
geometry, body SHA and runtime/source endpoints. It checks conservative outward
FP64 envelopes with four extra radius ULPs. An independent replay uses wider
intervals and exact rational certificates for every disjoint observation:
all 372 remain separated, with zero stage-count discrepancies. It also rereads
all 32 retained witness coordinates directly from the raw body. The smallest
independently certified gap is approximately `1.3651e-8`. Wider diagnostic
intervals do not change either acceptance inequality.

Seventeen independent controls and six recurrence controls pass in both Python
modes. Required Rust checks reuse exact unchanged numerical/build inputs.
The live-log reader's partial-line failure and the independent replay's initial
copied-digest typo are retained; neither changed numerical inputs. The corrected
replay and full audit pass within separate 512 MiB guards.

The CI follow-up distinguishes current-platform recurrence equivalence from
captured Mac trace bytes. Both packed and dense paths must match the unchanged
local scalar graph bitwise on every platform. A separate macOS-arm64 check
retains every original captured-byte assertion; fixture and producer hashes are
authenticated everywhere. All seven checks pass locally in both Python modes.
The Linux fixture assertion failure is retained. Numerical implementations,
original model gates and captured real-model evidence are unchanged.

The user's closing direction accepts explicit Qwen rejection decisions. The
separate Interim Labs 9B candidate remains metadata/header-only and is not
pursued. Original [4B](qwen35-context256.md) and [9B](qwen35-9b-context256.md)
rejections and the closed searches remain binding. Positive Qwen quality,
request/performance and adoption qualification stays unmet.

Evidence: [source/runtime freeze](../artifacts/qwen35-alternative-admission-2026-11-15/envelope-reference-1/inputs.json),
[complete collection](../artifacts/qwen35-alternative-admission-2026-11-15/envelope-reference-1/collection-checks.json),
[full audit](../artifacts/qwen35-alternative-admission-2026-11-15/new4-envelope-audit-1.json),
[independent replay](../artifacts/qwen35-alternative-admission-2026-11-15/independent-envelope-replay-2.json),
[closing direction](../artifacts/qwen35-alternative-admission-2026-11-15/closure-contract-3.json).
[Astra approves](../artifacts/qwen35-alternative-admission-2026-11-15/astra-envelope-review-1/verdict.json)
scoped diagnostic and rejection retention with no findings.
[Checks and publication](../artifacts/qwen35-alternative-admission-2026-11-15/envelope-publication-1.json).

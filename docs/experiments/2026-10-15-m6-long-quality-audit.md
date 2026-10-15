# M6 long-context quality synthesis

Fourteen portable tests pass: seven new synthesis checks and seven existing
saved-data replay checks. Astra accepted the source and evidence after an
independent byte-identical replay, all 115 endpoint bindings, 15 receipt pins,
13 published copies and the six independently recomputed profile fractions.
This does not close long quality.

The per-depth collector verifies 128 teacher-forced targets against the same
selected weights with Reference and Online attention. The frozen contract also
requires all four depths and the pooled 512-target gate for each model. Passing
one collection does not establish that coverage.

`tools/experiments/m6_long_quality_audit.py` first uses the accepted saved-data
auditor to replay the exact requests, raw events, state, memory and derived
metrics, and verify consumed identities. It binds the exact parsed summary bytes
to that replay and checks all bindings, including its own source, at the final
endpoint. It requires exactly one quality record for each model/depth pair at
4K, 8K, 16K and 32K. Each depth must have 128 ordered observations; each model
must have 512 distinct `(depth, target_index)` observations. The unchanged mean
NLL increase limit of 0.05 applies to every depth and the pooled mean.

The additional saved-winner checks require matching IDs to have identical logit
pairs, valid lowest-ID tie ordering, and a Reference top-two margin no larger
than its gap to a different candidate. A reversal report records the Reference
candidate gap, Online reversed gap, each candidate's logit shift, their
differential, full-vocabulary max/RMS error and the target's NLL change. The
differential must account for the sum of the two gaps.

These are numerical observations about two candidates. The older
`margin <= 2 * max_error` flag is a necessary consistency bound for a reversal;
it cannot establish its cause. Saved full-distribution hashes identify the
worker's distributions but cannot reconstruct them or locate the first differing
layer. Any changed winner keeps the automated check's investigation and
acceptance flags false. A separate investigation must resolve those observations
without rewriting the saved audit. Suspicious or unexplained reversals can
require a separately reviewed bounded capture at the exact frozen prefix.

## Partial saved-data compatibility

The [qualification receipt](../../benchmarks/results/2026-10-15-m6-long-quality-audit-qualification-v1.json)
binds the checker, tests, fourteen-test log, bounded offline compatibility script,
three native records and their exact published copies. Exact replay covers
2,747 raw events and 115 bindings. The new winner checks additionally accept all
256 rows from the existing small-model 4K/8K quality records, which have no
changed winners. The missing six real-model quality records and each model's
complete 512-target synthesis remain pending. Synthetic unit fixtures are not
real-model quality evidence.

## First maximum-depth profile

The separate small-model worker completes three Reference and three Online
profiles at 32,768 committed tokens, with no pending token and the exact frozen
history. All six have 412 operators; the first prefix has 32,767 prefill tokens,
and subsequent profiles reuse 32,766 tokens and prefill one token. The numerical
worker exits naturally with zero status, unchanged consumed identities and no
cleanup or endpoint errors.

| Mode | Three attention fractions | Median |
| --- | --- | --- |
| Reference | 99.8257%, 99.8188%, 99.7002% | 99.8188% |
| Online | 97.2638%, 97.2967%, 97.2624% | 97.2638% |

Each fraction uses the union of that decode's attention GPU intervals divided by
the matching completed GPU command duration. The Online median exceeds the
frozen 20% threshold and triggers investigation. Stage time is not a DRAM traffic
measurement and does not automatically justify I8. These counter-enabled
profiles do not establish an ordinary latency ranking. Paired unique admission
is 2,770,507,468 bytes; maximum conservatively summed worker/driver native lifetime
charge is 2,518,502,184 bytes, both below 6,442,450,944 bytes. Actual maximum-context
application generation remains pending, as do larger-model and lower-depth
profiles and the final KV decision.

Even a passing long-quality synthesis sets `m6_complete` to false. Remaining
profiling, boundary-state, application-memory and KV decisions require their own
evidence. No runtime, numerical kernel, frozen collector or contract is changed.

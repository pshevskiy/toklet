# Explicit Gemma4 diagnostic512 context candidate

`Norms::model_for_context_probe(limit)` creates an explicitly requested real-E2B
hybrid model with a position limit from1 through512. It reuses admitted immutable
weights, geometry and producer checks. `Norms::model()` and the scalar E2B route
retain32 positions. Projection/gather calls and scratch rows remain bounded to32.
No numerical kernel, native package, ordinary model identity or app route changes.

The larger model feeds its selected limit into the existing checked KV/history/
score planner. Invalid0,513 and overflow-sized limits reject before model
allocation; the real test measures zero Rust allocations on these errors. Cold
controls independently check context versus scratch bounds, monotone sizes and
explicit capacity+1 rejection.

| Additional session storage:32 →512 | Bytes |
| --- | ---: |
| Fifteen physical K/V owners | 17,694,720 |
| Attention scores and probabilities | 983,040 |
| Token history | 1,920 |
| Total | 18,679,680 |

On Apple M1 Pro, the default session allocates245,081,448 bytes and the candidate
263,761,128 bytes, matching the plan. Candidate model-plus-session admission is
2,869,593,752 bytes: exact budget succeeds and one byte less returns the expected
budget error. The source is loaded once; the two model views share immutable
weights. Sessions run sequentially. Resident owners, source file identity and
allocated pointers/bytes stay stable.

The candidate evaluates only the same frozen first32 corpus IDs as the qualified
[32-position route](gemma-hybrid32.md). Every one of the32 full-vocabulary logit
rows and all15 valid KV prefixes agree bit-for-bit with the default model, including
reset/reuse and absolute valid-slot mappings. Four selected default-row hashes
also match the earlier authenticated capture. This is preservation evidence for
the existing32-token prefix, not wider numerical qualification.

Formatting, strict default/Metal Clippy,550 default/658 Metal-feature tests and
release build pass. The explicit real admission/parity run exits cleanly;
peak RSS is2,949,955,584 bytes and physical footprint2,943,126,240 bytes, below6GiB.
Full source/config/index, input, executable and capture identities are retained.
Combined Astra code/results review passes.

Positions32–511 still require the independent long-reference and native boundary
work. The earlier four-token near-zero failures, complete unquantized quality,
ordinary Gemma Chat and phone qualification remain open. This diagnostic API
must not upgrade capability or discovery labels. Next is the bounded longer
oracle with at most32-row gathers/projections and selected hidden rows before
the tied head, followed by a timed pilot and512-position execution.
[Evidence](../benchmarks/results/2026-11-04-gemma-context-probe-v1.json).

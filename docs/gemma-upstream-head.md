# Real full-width versus tiled BF16 tied head

Every BF16 logit matches exactly at the saved real four-token activations for
both original and changed-final-token cases:2,097,152 words, zero differences.
Byte equality was frozen before execution. A separate stdlib replay confirms
both raw output bodies and their hashes against the prior sealed capture.

The isolated worker authenticates the original checkpoint, loads only its
805,306,368-byte `[262144,1536]` tied embedding table through128-row reads,
then applies stock full-width linear and the original softcap operations.
All2,048reads/805,306,368bytes match an independently prepared source chain.
Runtime/input/source endpoints and clean owned process exit pass. On M1 Pro16GiB,
peak RSS is1,236,647,936bytes and physical footprint1,153,402,416bytes, below6GiB.
The two head intervals are0.284s/0.105s; these isolated diagnostic samples do not
establish a performance claim.

This qualifies head storage equivalence only at these real four-row shapes.
It does not qualify vocabulary tiling for arbitrary row counts, original model
quality or app adoption. Next use full-width vocabulary projection in bounded
32-position batches after one stock text-model forward. Qualify the first33
frozen loss IDs/32targets against stock causal-LM output before a1,025-position
original-quality pilot. Existing fidelity failures remain rejected.
[Evidence](../benchmarks/results/2026-11-04-gemma-upstream-head-v1.json).

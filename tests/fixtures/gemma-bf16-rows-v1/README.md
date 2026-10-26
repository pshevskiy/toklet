# Original BF16 row oracle manifest

`manifest.json` is the exact pre-target capture manifest, 776091 bytes with SHA
`a17dede82620277d77149134750aa7e51abc72cf8995d6481799291e140113cc`.
It freezes 541 cases over all 540 selected original Gemma text tensors, complete
candidate-vector lineage, two dyadic-input scalar projection references and the
target comparison policy. The producer and independent checker use only stdlib.

The source excerpts and expanded arrays remain in the ignored owned capture at
`artifacts/m8-gemma-bf16-oracle-checks-v1/oracle/`. `arrays.bin` is 18900108 bytes,
SHA `8993df7c3fb2286c5d6609d0512fbfd7f7be904a3c9e909065681d5378b824fc`;
`raw.bin` is 48473228 bytes, SHA
`20c88d250d4e28ced477e04173b0b4ad33d7600f481c91bf192e6aa8dfe7416a`.
Both are required for replay; this committed manifest alone is not a numerical
fixture or permission to claim target-reader/model support. Whole source hashes
are independently required. Do not regenerate expected values from the target.

The subsequent `target-plan.json` fixes 1366 capture schedules and 4004 ordered
reads before the target runs. It is 973550 bytes, SHA
`bb63f6e72690bfa959273b710a8315376e3958962d0b6e490febf6d81b77dc82`.
The [separate target qualification](../../../docs/gemma-bf16-reader.md) replays
saved outputs against this plan and the original oracle; it does not revise the
original expected values or qualify a model graph.

See the [method and limits](../../../docs/gemma-bf16-rows-oracle.md) and
[public qualification record](../../../benchmarks/results/2026-10-26-gemma-bf16-rows-oracle-v1.json).

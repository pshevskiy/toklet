# Q4 v2 real GPU geometry qualification

The frozen group-eight candidate passes independent real-geometry GPU checks
for both selected checkpoints on Apple M1 Pro. This qualifies numerical graph
execution at capacity 27; full-capacity workloads, held-out loss/task quality,
profiling and speed acceptance remain separate.

## Mechanism and coverage

[The explicit harness](../../tests/q4_v2_models.rs) loads one immutable resident
Q4 v2 model, with session-owned Half KV and scratch. Both frozen
`Q4Compensated` and `Q4CompensatedTiled` use compensated normalization.
The six source-bound independent cases cover whole input and chunk sizes 1, 3,
8 and 17 where applicable, including absolute offset 5, populated KV prefixes
and irregular tails. The 17-token cases activate tiled prefill. Every completed
chunk compares all elements of 51 unique stages: embedding, layers 0/1/last,
final normalization and all-row logits. Each diagnostic row checks the frozen
stored-FP32 greedy oracle. Attention score/probability arrays are outside this
selected graph inventory.

The gate remains the independently frozen absolute/relative envelope; neither
reference values nor tolerances change. The count of 7,446 per model represents
completed stage-slice comparisons, not distinct whole-reference arrays.
Model geometry, format, resident weights, source/config/packed manifest,
full/probe references and oracle helper identities are checked. Each consumed
packed matrix/reference payload is authenticated; the final receipt additionally
rechecks all previously frozen packed/reference payload identities.

## Results

| Model | Resident weights, bytes | Admitted numerical + validation, bytes | Completed native lifetime charge, bytes | Maximum gate fraction |
| --- | ---: | ---: | ---: | ---: |
| Small |630,251,436|874,062,494|758,334,616|0.872696|
| Larger |1,777,247,428|2,029,218,550|1,910,425,496|0.603282|

All 14,892 stage-slice comparisons pass, with no greedy mismatches. Queried
compensated vector/tile execution widths are 32/32 on this device.

Native own-process lifetime charges include loading and validation. Explicit
admission reserves 128 MiB for validation outside numerical ownership and counts
shared numerical allocations once. These observations complement logical
accounting; they do not establish complete shared physical residency or an
application benchmark. Sessions and models run sequentially in an isolated
optimized executable; builds, tests and large hashes do not overlap GPU work.

## Verification and review

Ordinary Metal-feature tests, all-target Clippy, formatting and optimized
explicit-test compilation pass. Astra cleared the harness before native runs.
Its scope corrections distinguish stage slices, capacity 27 and final payload
endpoint checks. Astra accepted final code, all 1,812 bindings, complete metric inventories,
native/check evidence and local links with no actionable findings. The
[receipt](../../benchmarks/results/2026-10-14-q4-v2-real-geometry-qualification-v1.json)
binds executable, exact sources, references, native reports and checks.
The frozen candidate can proceed to unchanged loss guards and quality gates;
this task makes no held-out quality, performance or M5-exit claim.

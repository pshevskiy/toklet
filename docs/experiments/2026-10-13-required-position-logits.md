# Vocabulary projection at required positions

Ordinary prefill now projects only its final normalized row into the tied
vocabulary matrix. It precreates a retained one-row Half MPS view at the correct
row offset and reuses the one-row Float result/kernel. Bias initialization,
finite validation and readback consistently select output row zero. Diagnostic
sessions still compute all rows and publish their final distribution from its
existing offset. Dependencies remain in one completed command per chunk.

The new view uses the owner's queried stride and checked offset/row/guard
bounds. Session metadata accounting includes its retained pointer for every
admitted chunk size. Final normalization and chunk-by-vocabulary Float scratch
allocation remain unchanged. This is a required-position computation code task,
not a measured latency or scratch-memory improvement claim; tiled prefill,
larger chunks and M5.3 performance qualification remain pending.

The new explicit native test covers tiny and pinned Qwen 0.5B, FP16 and Q4,
position offsets 0/5 and chunks 4/3/1 including irregular tails. It compares 76
complete ordinary distributions with the last required diagnostic row under
the existing `1e-4 + 1e-3 * abs(reference)` envelope, with identical greedy IDs.
All-row snapshots have exact dimensions and agree numerically with diagnostic
publication. The row-count change may affect vendor reduction order; no internal
precision or bit-identity claim is inferred from MPS Float storage.

The first test run incorrectly requested public logits while a whole queued
prompt still had pending input; the API correctly returned none. The corrected
harness appends and evaluates one part at a time so every compared distribution
has a fully committed prefix. Failed evidence is retained under `artifacts/`;
production code and all numerical gates were unchanged by that correction.

Five Q4 native suites, three FP16 suites and three native failure/oracle suites
pass sequentially on Apple M1 Pro, macOS 26.6.2. The frozen Q4 tiny 2,310 stage and
FP16 Qwen 612 stage comparisons remain passing; warmed Q4 steps still have zero
Rust allocations. Default 84/Metal 90 ordinary tests, formatting and strict
all-target Clippy in both configurations pass. Device suites follow completed
isolated release builds with no concurrent builds or measurement jobs.

The [qualification record](../../benchmarks/results/2026-10-13-required-position-logits-v1.json)
binds sources, GPU binaries, successful logs and protected CPU hashes. The old
continuation binary is preserved for later comparative measurements. Astra accepted the final code/check/evidence task; the M5.3 checkbox remains
open for tiled execution, chunk sweeps and performance qualification.

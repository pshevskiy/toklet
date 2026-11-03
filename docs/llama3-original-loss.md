# Original Llama full loss evaluation

Both explicit workers consume the frozen first 10,001 raw WikiText IDs from
[the input manifest](../benchmarks/llama3-original-quality-v1/manifest.json).
The ten blocks cover target indices 1 through 10,000 exactly once. Each block
resets positions and KV and overlaps its predecessor by one input token; it
contains at most 1,025 inputs and 1,024 scored targets. Row `local` predicts
`tokens[local + 1]`; the final input row is unscored. No chat template, automatic
BOS, EOS stopping, case filtering or reserve changes apply.

The native worker uses the pinned packed artifact, diagnostic loader, affine
projections, compensated normalization, Reference attention and chunk size 32.
It reads completed diagnostic rows and records stable FP64 NLL statistics.
The original worker maps the authenticated BF16 source, widens one projection
at a time and uses the accepted FP32 graph. Only scoring rows pass through the
tiled tied head. Its largest logits array is about 501 MiB; it releases that
array before the next block. The processes run sequentially under the 6 GiB
budget, with the existing 1 GiB disk reserve for the reference.

The stdlib checker joins records by global target index, validates each token,
block and local position, rejects missing/duplicate targets, and independently
recomputes NLL from the recorded maximum, exponential mass and target FP32 bits.
It verifies journal/report digests and each target-weighted mean using
`math.fsum`. These are authenticated producer statistics with row hashes;
full-vocabulary distributions are not retained for all 10,000 rows.

Before collection, the execution plan pins both prior alignment reports.
First-block targets 512, 513 and 1,024 must match the previous native numerical
row hashes and original NLLs within the frozen 1e-4 tolerance. Target 1,025 is
excluded from this check because the full loss test scores it after a reset.

The frozen gate is **native mean NLL minus original mean NLL ≤ 0.05 nats/token**.
A completed rejection remains a valid evaluation result. The separate 100-task
gate, app discovery eligibility and physical iPhone qualification remain open.
No throughput or user-visible latency claim follows from this collection.

## Mac result

On Apple M1 Pro, original mean NLL is **2.782632744148034** and native mean NLL
is **2.782652474673244**. The increase of **0.00001973052521009322** passes the
frozen 0.05 gate. All 10,000 targets and the three earlier alignment checks pass.
Native peak RSS is 1,214,087,168 bytes; its admission including app allowance is
1,543,629,376 bytes. Original peak RSS is 3,589,652,480 bytes.

Three Rust worker controls, six Python checker controls, three targeted controls
on the system interpreter, strict Metal Clippy, formatting, release build and
Astra review pass. The engine sources are unchanged from the passing 549 default
and 657 Metal regression suites. The system-Python compatibility correction
replayed the preserved worker outputs; the initial failures remain recorded.
[Evidence](../benchmarks/results/2026-11-03-llama3-original-loss-v1.json).
This closes only the loss gate; the 100-task gate remains open.

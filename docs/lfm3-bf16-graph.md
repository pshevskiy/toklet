# LFM3 BF16 eager CPU graph

The authenticated loader admits the canonical 3,457,635,448-byte BF16 candidate.
The existing file audit checks the complete SHA-256 before delivery and rechecks
the delivered bytes. The loader validates the complete 286-tensor inventory and
keeps a single contiguous BF16 bank. Typed ranges resolve tensor names before
execution. Embedding lookup and the output head share the same range.

The CPU graph composes 24 layers with seven-tap short convolution or grouped-query
attention, both sandwich norms, separate BF16 residuals, and gated feed-forward
blocks. Each step prepares all 13 history and 11 KV candidates in session
scratch. The joint state commit follows the final normalization, tied output
projection and optional diagnostic callback. A failed step preserves published
logits, token history and the committed state. A successful step publishes the
last-token logits by swapping the two preallocated buffers.

The capacity-2,048, chunk-16 ledger accounts for 3,489,337,408 owned bytes across
weights, position tables, state, scratch, token history, logits and owner
structures. Admission includes a separate 2 MiB temporary loader allowance.
The release verification run stayed under the 6 GiB limit, with peak aggregate
RSS upper bounds of 3,514,073,088 bytes and sampled physical memory peaking at
3,505,524,024 bytes. These observations cover a single raw-token graph session.
[Native guard](../artifacts/lfm3-bf16-graph-2026-11-11/verify-checks.json).

The independent Torch collector streams one layer at a time under the separate
512 MiB cap. It verifies the previously pinned 28,982 runtime files before and
after execution. Three chunk schedules over four tokens match all 366 digests
and 2,476,032 BF16 values for layer outputs, convolution history, new K/V,
final normalization and last-token logits. A late diagnostic failure on a
continuation preserves the original logits, token history and generation.
Retry matches the source. A prepared ordinary forward allocates nothing.
[Contract](../tests/fixtures/lfm3-bf16-graph-v1/contract.md),
[source fixture](../tests/fixtures/lfm3-bf16-graph-v1/fixture.json),
[reference guard](../artifacts/lfm3-bf16-graph-2026-11-11/attested-graph-checks.json).

The graph runs without codec or sampling. Text/tokenizer admission, pending
token handling in the generation adapter, deeper real-model parity and
application measurements, full quality and managed adoption remain open.

## Release baseline

Two reverse-order sweeps on M1 Pro each retain the first call, three warmups
and ten samples per cohort. The timed graph includes last-token tied logits
and transactional publication. Cache reset, prefix seeding and session
allocation run outside that boundary.

| Prepared workload | Median across the two sweeps |
| --- | ---: |
| 1 token, empty prefix | 1.242–1.249 s |
| 1 token, two-token prefix | 1.252–1.260 s |
| 4 tokens, empty prefix | 3.037–3.049 s |
| 16 tokens, empty prefix | 10.237–10.240 s |

New-process loading took 32.04–32.20 seconds. The complete authentication
prepass took 20.21–20.32 seconds. Verified BF16 delivery took 11.83–11.88
seconds. Both sweeps reproduced every output digest and sampled AC power,
Low Power Mode off and nominal thermal state throughout. They remained
under the 6 GiB memory guard.
[Release samples and load phases](../benchmarks/results/2026-11-11-lfm3-bf16-graph-v1.json).

Formatting, both strict Clippy configurations, 731 CPU tests, 877 Metal
tests and nine targeted example tests pass. The three arm64 native slices each
retain 46 exports and pass strict C linking. Existing Mac CPU and Metal
C smokes pass. The initial compile error treated the header inventory
vector as an iterator. The subsequent checks use its length.
[Checks and logs](../artifacts/lfm3-bf16-graph-2026-11-11/checks.json).

Astra approved scoped retention with no findings after verifying the checkpoint,
reference runtime, raw results, timing samples and native build evidence.
[Astra review](../artifacts/lfm3-bf16-graph-2026-11-11/astra-review.json),
[Commit](../artifacts/lfm3-bf16-graph-2026-11-11/publication-checks.json).

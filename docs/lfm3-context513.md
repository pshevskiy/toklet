# Real LFM3 BF16 sliding-window rollover

The same authenticated 24-layer checkpoint runs to position 513 under two
frozen chunk schedules. They cross the 256-token boundary and continue through
the second physical ring wrap. Both schedules use the same 513 raw token IDs
and report every layer output, convolution history, K/V proposal, final norm,
and last-row tied logits.

The independent Torch reference streams one layer at a time, with the
pinned eager parent arithmetic and its 255-row prior sliding view. It verifies
the complete checkpoint and the pinned runtime inventory at both boundaries.
The guarded capture's combined final peak RSS upper bounds total
533,643,264 bytes under the separate 512 MiB reference ceiling.
[Contract](../tests/fixtures/lfm3-context513-v1/contract.json),
[Reference guard](../artifacts/lfm3-context513-2026-11-11/attested-context513-checks.json).

The existing optional graph observer exposes the actual attention view on
key proposals. It reports the absolute key start, logical extent, window, and
ordered segment lengths. The ordinary forward retains its callback-free path.
At offset 257 each local view contains 254 old rows, one wrapped old row and
one new row. The late-abort check rejects at the logits callback, checks the
old history, generation and published distribution, then retries the same
candidate.

All 1,342 observations and 70,418,432 BF16 words match the source exactly.
All nine shared tied-logit endpoints agree across schedules in both the source
and Toklet. Forty attention views contain two old spans, and both late-abort
retries reproduce the candidate logits. The native capture records
3,508,617,216 bytes combined final peak RSS upper bounds under the 6 GiB cap.
[Native capture guard](../artifacts/lfm3-context513-2026-11-11/verify-checks.json),
[Exact stage reconciliation](../artifacts/lfm3-context513-2026-11-11/report.json).

## Release timing and checks

On M1 Pro, the isolated run loaded in 32.29 seconds and seeded 257 tokens in
157.41 seconds. It retained one first pass, three warmups and ten measured
one-token forwards for each mode.

| Timing boundary | Median |
| --- | ---: |
| Ordinary commits, successive wrapped prefixes | 1.272 s |
| Minimal observer, fixed offset 257, abort before commit | 1.270 s |
| Observer consuming values and view metadata, same fixed offset | 1.277 s |

The first ordinary call's logits match the independently captured position
258. Power and thermal checks passed. The ordinary series advances its
prefix. The observed series hold a fixed prefix by discarding each
candidate after computing logits. The different publication boundaries leave
observer overhead unqualified.
[Release report](../benchmarks/results/2026-11-11-lfm3-context513-v1.json).

Formatting, strict CPU and Metal Clippy, 733 CPU tests, 879 Metal tests,
nine targeted example tests and both release probe builds pass. Three
native arm64 slices retain 46 exports each and pass strict C linking and
the existing Mac CPU and Metal C smokes. The isolated timing build extends
only the timing function. The verification entry point and model sources
match the pinned comparison build.
[Checks](../artifacts/lfm3-context513-2026-11-11/checks.json).

Deeper parity through the required 1,025 positions, full quality and
managed adoption remain open.

Astra approved the scoped change after independently checking the source
payload, native observations, timing boundaries and test evidence.
[Astra review](../artifacts/lfm3-context513-2026-11-11/astra-review.json),
[Commit](../artifacts/lfm3-context513-2026-11-11/publication-checks.json).

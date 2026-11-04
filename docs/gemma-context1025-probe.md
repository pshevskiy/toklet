# Gemma4 diagnostic 1,025-position admission

The explicit E2B diagnostic constructor admits capacities through 1,025 using
existing immutable weights and the existing numerical graph. Ordinary construction
stays at 32 positions, with at most 32 rows per forward call. This is preparation
for the frozen quality workload, not ordinary app or declared-window support.

Twelve local KV owners remain physically capped at 512 positions; three global
owners grow. Relative to the admitted 512-position session, cache, scratch and
history add 7,356,420 bytes. Pure geometry checks cover 512, 513, 1,024 and 1,025,
including fixed row scratch, monotonic allocation and rejected extreme limits.

The explicit hardware check uses the previously qualified 512-token corpus.
It compares all 512 full-vocabulary output rows between the 512 and 1,025 models,
then every valid K/V prefix, owner geometry and absolute slot. Four baseline
selected-row hashes must match the committed 512 capture. No output beyond
absolute position 511 is executed or qualified by this admission test.

The test also checks exact-budget admission, one-byte-less rejection, allocation-
free rejection of invalid constructor limits, unchanged ordinary construction,
reset/reuse and stable session/GPU/source ownership. Reset checks every slot's
invalidation and reproduces the initial 32-row logits and valid KV prefix.

The measured session allocation is 271,117,548 bytes; model plus session needs
2,876,950,172 bytes. Native peak RSS is 2,950,561,792 bytes and physical footprint
2,986,199,824 bytes, both below the 6 GiB policy. Formatting, strict Clippy in
both configurations, 550 default/658 Metal tests and release build pass.
Combined Astra review is recorded in the evidence. The next task needs
an independent 1,025-token oracle and real local eviction/wrap verification.
Earlier four-token mathematical failures, original-BF16 quality, resident GPU
scheduling, free generation and app/device qualification remain open.

[Evidence](../benchmarks/results/2026-11-04-gemma-context1025-probe-v1.json).

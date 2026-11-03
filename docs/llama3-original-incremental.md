# Incremental original Llama reference

The explicit CPU reference retains immutable mapped BF16 weights and owns
separate FP32 K/V arrays `[layer, kv_head, capacity, head_dim]`. At the frozen
566-position task capacity they consume 37,093,376 bytes. It prefills once,
then accepts one decode token at a time. Rotated keys and unrotated values stay
in the valid prefix; each new query uses its absolute RoPE position. No weight
cache is enabled.

Tentative layer writes touch only uncommitted rows. History allocation precedes
numerical work, and history, logits and committed position publish after all
layers and finite-output checks succeed. Failure can retry from the old prefix.
Reset changes logical ownership; it does not clear the entire allocation.

All 11 frozen real-model prefixes pass full-row and target-NLL gates with exact
lowest-ID greedy agreement. Independent replay verifies 22 saved rows and their
top-two margins. Maximum logit discrepancy is 3.8862e-5 and maximum target-NLL
discrepancy is 6.8034e-6. Overflow preserves history, logits and full KV bytes;
reset matches a fresh session. Twenty-one cold controls include layer failure,
history allocation failure and tied-row selection. The previous default rotary
is preserved bit-for-bit against its captured qualified implementation.

Peak RSS on Mac M1 Pro is 3,058,221,056 bytes. A profiling probe with three warmups
and five repetitions records 20 warmed reference decodes, median 856.68 ms.
These are reference preparation costs, not app/native throughput results.
The next experiment attributes widening versus numerical compute before deciding
whether a bounded cache is useful. Astra reviewed the code and actual results.
[Evidence](../benchmarks/results/2026-11-03-llama3-original-incremental-v1.json).

The separate 100-task gate remains open. Before grading, each generated reference
trajectory must match one stateless teacher-forced replay, including a terminal
hidden stop when present and no extra prediction at the output limit. That
collection/replay adapter is the next task; this qualification does not claim
that task outcomes or app/device qualification are complete.

# Metal held-out loss collection and rejected configurations

The scoring mechanism passes its code/transport checks. Qwen 0.5B Q4 v1 fails
the unchanged held-out loss gate. M5.2 and the M5 release exit remain open; CPU
is still the default and the existing FP16 path is preserved.

## Scoring mechanism

The worker uses the frozen 10,001 token IDs, independently reset blocks of at
most 1,025 tokens, one-token overlap, and every target exactly once. Every causal
row needed for scoring is explicitly requested through all-row diagnostics.
The physical chunk commits all its input rows before scoring; a loss event's
`prefix_tokens` is its logical causal prefix, checked separately from the
`chunk_done` physical evaluated count. Each block's final target is committed
but its following distribution is not scored. Positions reset to zero per block.

FP64 stable log-sum-exp and compensated sums mirror the existing independent CPU
scorer; that CPU file and its source-inventory rules remain unchanged. The driver
independently checks all labels, chunk tails, block commitments and aggregates
with `math.fsum`. Admission includes persistent snapshots, bounded host copies
and 64 MiB application allowance. Copies drop before idle native charge reads.

Workers have owned process groups, nonblocking bounded pipes, 120-second events,
3,600-second lifetime and 32,768-event bounds, and explicit idle handshakes. Only
the driver reads its own and still-owned worker's libproc v4 charge. Cleanup
signals/reaps owned workers with bounded waits, closes every pipe, and retains
original failure/endpoint evidence. Source/config/packed identities, compiler,
build, OS and power endpoints are bound before/after execution. Exact bounded
corpus and baseline bytes are hashed before parsing; worker readiness binds the
admitted IDs, including seed tokens. Helper and pinned probe manifests are also
bound. These were Astra provenance findings.

## Native results

| Configuration | Evaluation | Result |
| --- | --- | --- |
| Frozen unquantized scalar FP32 | 10,000 targets |mean NLL 2.883725149929484 |
| Q4 v1 + Half exclusions/activations/KV | 10,000 targets, 10 blocks, 88 chunks |mean NLL 3.057166039390558 |
| Q4 increase |unchanged maximum 0.05 nats/token |+0.17344088946107394: rejected |
| FP16 batch parity probe |three targets at chunks 1/128 |max absolute NLL difference 0.004680719303589598 > 0.001: stopped |

The [Q4 record](../../benchmarks/results/2026-10-13-metal-q4-held-out-loss-v1.json)
retains four successful short-probe workers followed by complete scoring. Its
same-precision chunk1/128 maximum probe difference is 0.000006310324402569734,
within the predeclared 0.001 guard. All 10,134 raw events replay exactly. All five
workers exit naturally with status 0 and clean cleanup; identities stay unchanged.
Maximum retained worker charge is 777,553,048 bytes; conservative joint
driver-plus-worker maximum is 815,023,880 bytes, below 6 GiB. Admission and native
charge complement one another; charge counters can omit shared/clean pages.

The [FP16 record](../../benchmarks/results/2026-10-13-metal-fp16-held-out-loss-v1.json)
retains two successful workers and all 17 raw events. It correctly stops before
the full corpus after the batching guard fails. It is not an FP16 full-quality
result. Batch-dependent vendor arithmetic is a hypothesis; these probes alone
do not isolate layer projections from input-dependent vocabulary projection.
Stage-controlled comparisons are needed before attributing the discrepancy.

No tolerance, corpus, aggregation, quantizer or gate changes followed these
results. A passing task score could not rescue the failed Q4 loss gate. The 100
task cases have not been collected for these configurations. Q4 v1 on this 0.5B
checkpoint remains quality-unqualified. A single larger checkpoint can qualify
separately after payload/admission-based preselection and matched unchanged
loss/task/performance gates; its outcome cannot rehabilitate this rejection.

## Verification

Default 84/Metal 90 ordinary Rust tests, formatting and strict all-target Clippy
pass. Three Metal scorer math/block tests and three unchanged CPU scorer tests
pass explicitly after device work. 149 experiment tests pass, including six new
offline collector tests for all required positions, partial rows, block edges,
typed admission, corrupt labels/aggregates, exact parsed-byte binding and owned
failure cleanup. Native workers run sequentially after completed isolated builds,
with no overlapping builds/tests/large external hashes. Root CPU release binaries
remain unchanged. Astra independently replayed both journals and sums, reviewed
the code and accepted the scoring task and honest rejection evidence, with no
remaining findings.

The [code/check binding](../../benchmarks/results/2026-10-13-metal-loss-qualification-v1.json)
records this scoped acceptance; no M5.2 or release closure is claimed.

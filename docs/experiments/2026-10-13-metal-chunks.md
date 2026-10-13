# Explicit larger Q4 chunk admission and sweep

Metal session admission now accepts explicit chunk sizes up to 1,024 instead of
128, within the existing 2,048-position and 6 GiB limits. Default chunk size is
still 128. This task does not change arithmetic, attention kernels, weights,
normalization defaults, the CPU default or any frozen quality gate. It evaluates
the planned 128/256/512/1024 prefill choices for one fixed compensated Q4 model.

## Shapes and admission

The existing checked layouts scale activations and vocabulary Float scratch by
chunk size. Separate K/V remain `[kv_heads, capacity, head_dim]`; materialized
scores/probabilities remain `[query_heads, chunk, capacity]`. Views and MPS
multiplications are created for all possible rows before numerical execution.
No unchecked shader indexing or online/tiled attention is introduced. All size,
position, overflow and budget validation precedes session allocation.

The four-candidate worker shares one immutable 1,223,992,516-byte Q4 model and
owns separate session buffers. Each new session is admitted against the joint
remaining budget, counting weights once, four existing 64 MiB framework
allowances and a 64 MiB application allowance. Large chunk scratch includes the
unchanged chunk-by-vocabulary Float buffer even though ordinary output computes
only the final row; no scratch-memory optimization is claimed.

## Correctness and scope

An independent exact-zero Half model reaches all capacity positions at each
chunk size for capacity/offset 2048/0 and 2047/1: eight complete cases. Every
final logit has positive-zero bits. Exact-budget admission passes; one byte less
is rejected. Chunk 1025, chunk larger than capacity and an overflowing position
bound are rejected before session allocations. Reusing the complete prompt
requires no evaluation; appending beyond capacity fails.

The nonzero check uses the preselected larger Qwen Q4 artifact with explicit
`Q4CompensatedTiled` projection and compensated RMS. At rotary offset 17 it
checks reset of 1,024 tokens, append of seven tokens, and an edit at position
1,000 of the resulting 1,031-token history. Committed lengths are
1024/1031/1031, exact reused prefixes are 0/1024/1000 and required prefill is
1024/7/31. Every committed token is checked. All 151,936 final-logit bits and
greedy IDs agree across all four chunk sizes for every case. Pending sampled
IDs are discarded only after the completed request, so the report still records
one pending sample and the next check consumes the committed input history.

This is cross-chunk final-distribution refinement against chunk 128, with exact
history/state checks. It is not an independent nonzero full-array oracle or
larger-chunk qualification of every Half/Original mode. Public admission alone
does not establish those additional numerical or quality conclusions. Both
fixed Q4 configurations remain rejected by the held-out loss gate.

The [nonzero check](../../benchmarks/results/2026-10-13-metal-chunks-check-v1.json)
finishes with five exact protocol events, clean completion, unchanged endpoint
identities and empty cleanup errors. Combined admission is 3,515,980,100 bytes;
maximum joint native lifetime charge is 3,233,402,240 bytes.

| Explicit chunk | Check-mode scratch bytes | Per-session admission, including shared weights |
| --- | ---: | ---: |
| 128 | 114,783,828 | 1,465,356,644 |
| 256 | 229,263,444 | 1,579,849,572 |
| 512 | 458,222,676 | 1,808,835,428 |
| 1024 | 916,141,140 | 2,266,807,140 |

These use capacity 2031/offset 17. Performance sessions use capacity 2048/offset
zero and therefore allocate slightly more cache/scores. Rows cannot be summed
without subtracting duplicate shared weights and adding the application budget.

## Performance protocol

The fixed 1,024-token prompt is SHA-256
`8acd07e9a1a11ae7e6642247f9e7621da9668863b58060620629bbad366a1ba9`.
Warm request timing covers reset through the first token event, completed GPU
work, full final-logit readback and CPU greedy. Exactly one token is generated
per candidate; this is not the full 128-output release workload.

Three warmup rounds precede ten measured rounds. Each round rotates four
candidates through execution positions; ten samples do not give a perfectly
balanced four-position design. Complete logits are compared after all four
timers. All buffers remain fixed. Idle handshakes admit/check native worker and
driver lifetime peaks jointly below 6 GiB. Source/artifact/prompt, source code,
helper and binary identities are checked before and after collection. The
transport owns one child group, bounded pipes, 300-second event and 3,600-second
total timeouts, at most 32 events and 1 MiB stdout. Failure preserves its cause
and attempts owned kill/reap/pipe cleanup before publishing exclusive evidence.

Apple M1 Pro, 16 GiB, macOS 26.6.2 and Rust 1.90.0; Cargo release defaults,
precise existing shader arithmetic. Power/thermal endpoints are retained;
continuous sensors and energy are unavailable. No concurrent builds, tests or
large hashes run during GPU measurement. AC power is attached at both endpoints,
with AC low-power mode zero and no recorded thermal/performance warning.

| Chunk | Median TTFT | MAD | Measured range | Improvement vs. 128 |
| --- | ---: | ---: | ---: | ---: |
| 128 | 23.402353 s | 41.896 ms | 23.313049–23.492871 s | Reference |
| 256 | 22.454417 s | 8.609 ms | 22.361214–22.471075 s | 4.0506% |
| 512 | 22.787014 s | 14.499 ms | 22.740844–22.821599 s | 2.6294% |
| 1024 | 25.384823 s | 673.432 ms | 24.689093–26.078174 s | -8.4712% |

All 52 completed requests, including warmups, preserve all final-logit bits
across candidates. The ten measured rounds total 940.105753 seconds of completed
request time; this is sustained component/reset activity, not the full output
workload, application trace or release qualification. The large 1024 dispersion
and four-way order are retained rather than collapsed into a tail-latency claim.
No causal attribution to cache/register/thermal effects is established here.

Combined performance admission is 3,521,063,508 bytes; joint native lifetime peak
is 3,239,218,488 bytes. Per-session scratch for 128/256/512/1024 is
114,992,724/229,681,236/459,058,260/917,812,308 bytes. Persistent inventories are
unchanged at every sample. The [completed sweep](../../benchmarks/results/2026-10-13-metal-chunks-sweep-v1.json)
has 15 exact protocol events, clean completion, unchanged identities and empty
cleanup errors.

Retain the explicit admission and diagnostic/sweep harness. Reject all larger
chunks as default performance choices on this workload: none meets the frozen
5% threshold, so no confirming retention run is warranted. Keep default 128 and
the unchanged attention. Cancellation is observed between completed chunks;
larger explicit chunks can increase the time before responding to cancellation.
This is not quality acceptance or M5 closure.

## Reproduction

```sh
CARGO_TARGET_DIR=artifacts/m5-checks cargo build --offline --release --features metal --test metal --example metal_chunks
.venv/bin/python tools/experiments/metal_chunks.py --binary artifacts/m5-checks/release/examples/metal_chunks --output benchmarks/results/2026-10-13-metal-chunks-check-v1.json --check
.venv/bin/python tools/experiments/metal_chunks.py --binary artifacts/m5-checks/release/examples/metal_chunks --output benchmarks/results/2026-10-13-metal-chunks-sweep-v1.json
```

Use fresh exclusive output paths for reruns. All previous Original/Half/Q4
checks and unchanged default behavior remain separate regression requirements.

Final formatting and strict all-target Clippy pass in default and Metal builds,
with 87 CPU tests, 94 Metal-feature ordinary tests and 169 experiment Python
tests. Five new offline tests cover modes, geometry, shared budget, causal state,
timing/order mutations, each source/artifact/prompt pin and owned failure cleanup.
Four final native regressions pass for Original Half tiny/Qwen, the compensated
Q4 tiny oracle and Q4 session/cancellation/allocation behavior. Together with
the separate capacity test and larger check protocol, these are the final task
checks. Root CPU release binaries retain their protected identities.

The [qualification receipt](../../benchmarks/results/2026-10-13-metal-chunks-qualification-v1.json)
binds final sources, fixtures, logs, immutable measured binaries and saved check/
sweep reports. Astra's source and compact saved-data reviews found no actionable
issues; acceptance remains limited to the scopes stated above.

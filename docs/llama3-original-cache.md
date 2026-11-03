# Streamed original-reference FP32 cache

The reference now has an explicit `streamed-fp32` storage profile. It verifies the
original BF16 descriptor and tensor inventory, reads bounded ranges, widens bits
exactly and fills one C-contiguous FP32 array per tensor. The source descriptor
and mmap close before read-only weights publish. A single tied table serves both
input gathers and tiled output projections; no full resident BF16 copy remains.
The mapped default remains available.

The 146 arrays occupy 4,943,257,600 bytes. At capacity 566, separate session KV is
37,093,376 bytes. Including the runtime/scratch and conversion reserves, planned
admission is 5,584,330,752 bytes. The higher observed process peak is 5,584,928,768
bytes, slightly above the estimate; both are below the 6 GiB limit. Admission and
measured peak are distinct facts.

Both independent cache runs reproduce all 22 saved mapped-reference rows bit for
bit and all 11 greedy decisions. Numerical, state, overflow/reset and margin
checks pass. Each run also completes the largest planned task replay shape:
566 corpus inputs and 256 projected output rows. This is memory coverage, not a
generated task outcome. Twenty-four cold controls and independent stdlib replay
pass. The fixture now matches the real reader’s metadata-stripped inventory.

On the fixed warmed reference-only probe, median decode is 172.68 ms and 169.40 ms,
versus 827.95 ms uncached. Both runs clear the prospectively frozen 10% improvement
requirement. Astra approves retaining the cache for task collection. These are
CPU reference preparation measurements, not native/app throughput claims.
[Evidence](../benchmarks/results/2026-11-03-llama3-original-cache-v1.json).

The 100-task pass-rate gate remains open. Collections must preserve prompt IDs,
reserves, stop handling, literal decoded text and per-case trajectory checks.

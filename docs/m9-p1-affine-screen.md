# M9 affine prefill screen on Mac

5 November 2026. The pinned Qwen2.5 MLX8 workload ran on Apple M1 Pro
at capacity 1,152 and chunk 128. The same release binary served both
timing modes in separate owned processes on AC power. Each depth used
three warmups and ten measured requests.

The correctness worker verified all 13 prefill chunks and 381 decode
distributions and committed states bitwise. It observed 168 tile
dispatches per prefill chunk and the ordinary decode kernel at the
sampled decode boundary. Both timing streams reproduced the terminal
hashes. [Frozen contract](m9-p1-affine-prefill.md).

| Prompt tokens | Baseline prefill | Tile prefill | Change | Baseline request | Tile request |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 0.773 s | 0.633 s | −18.03% | 4.706 s | 4.484 s |
| 512 | 3.112 s | 2.662 s | −14.47% | 8.661 s | 8.224 s |
| 1,024 | 6.567 s | 5.666 s | −13.71% | 14.489 s | 13.579 s |

The primary metric passed. The 512-token decode control rose from
5.546455 s to 5.562124 s, a 15.669 ms or 0.283% increase. That exceeds
the larger median absolute deviation of 7.938 ms, within overlapping raw
ranges. The unchanged executable will repeat the comparison in reverse
process order before P2. The original samples are retained.

Sampled physical peak was 638,190,624 bytes for the baseline and
638,485,512 bytes for the candidate. Both exited cleanly under the
2.5 GiB process limit. Required source checks and Astra review pass.
The independent confirmation and adoption gates remain open.
[Evidence](../benchmarks/results/2026-11-05-m9-affine-screen-v1.json).

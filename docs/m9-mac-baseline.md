# M9 Mac baseline

5 November 2026. Apple M1 Pro, MacBookPro18,3, macOS build 25G83, AC power.
Toklet release at engine revision `35adcc6e` with the P0 example added. The
audited Qwen2.5-0.5B MLX8 source, frozen tokens, Half KV, Compensated
normalization, Online attention and FullLogits use capacity 1,152 and chunk
128. The binary and workload hashes are in the evidence.

Separate owned workers collected 75 counter profiles and 39 uninstrumented
timing requests. Each prompt depth has three warmups and ten measured
requests with 128 numerical token events and 127 decode evaluations.
The reported latency begins at the fixed-token request and ends at the
completed distribution. Tokenization, sampling and visible Chat latency
are outside this workload.

| Prompt tokens | First distribution median | Prefill tokens/s median | Decode tokens/s median | Request median |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 0.770 s | 166.15 | 32.40 | 4.690 s |
| 512 | 3.211 s | 159.46 | 22.53 | 8.848 s |
| 1,024 | 6.775 s | 151.14 | 15.83 | 14.795 s |

The profile worker verifies complete logits after every prefill and decode
evaluation. It saves every prefill chunk and decode indices 0, 31, 63 and
126. The final decode hashes match the timing worker. At 1,024 tokens,
median 128-row prefill counter intervals total 738.912 ms for layer affine
projection and 13.066 ms for normalization. Sampled one-row decode totals
9.587 ms and 11.816 ms respectively. These are grouped counter intervals,
collected separately from acceptance timing.

Sampled physical peaks were 638,059,576 bytes for timing and 783,139,992
bytes for profiling. Both workers exited cleanly under the 2.5 GiB cap.
Neither start reported a thermal warning. The post-run thermal query was
unavailable. The Mac baseline is the comparison input for P1.
[Evidence](../benchmarks/results/2026-11-05-m9-mac-p0-v1.json).

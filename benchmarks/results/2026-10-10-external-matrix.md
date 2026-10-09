# FP32-storage external workload matrix, 10 October 2026

The pinned llama.cpp Metal and MLX-LM adapters ran sequentially on Apple M1 Pro,
MacBookPro18,3, 16 GiB, macOS 26.6.2 (25G83), AC power with low-power mode
disabled. Every row retains three warmups and ten completed teacher-forced
requests with the frozen 128 output IDs. Other performance jobs were stopped;
source editing and small offline tooling checks continued. Model/tokenizer and
token hashes, exact adapter/binary identities, raw event intervals and power/
thermal reports are retained in each JSON. Source hashing warms file caches;
these are loaded-model request timings rather than cold-storage measurements.

| llama.cpp workload | TTFT median (ms) | Decode steps/s | Request median (s) |
| --- | ---: | ---: | ---: |
| [128 inputs](2026-10-10-llama-fp32-short.json) | 35.759 | 75.408 | 1.719878 |
| [512 inputs](2026-10-10-llama-fp32-medium.json) | 143.642 | 71.718 | 1.914380 |
| [1024 inputs](2026-10-10-llama-fp32-long.json) | 299.950 | 68.255 | 2.160622 |
| [1920 inputs](2026-10-10-llama-fp32-boundary.json) | 606.493 | 63.822 | 2.596163 |
| [64 appended after 512 cached](2026-10-10-llama-fp32-append.json) | 25.141 | 71.866 | 1.792286 |
| [512 fully cached](2026-10-10-llama-fp32-cached.json) | 0.000667 | 71.821 | 1.768285 |
| [One pending after 512 cached](2026-10-10-llama-fp32-pending.json) | 14.722 | 72.004 | 1.778518 |
| [Changed 512-token prompt](2026-10-10-llama-fp32-changed.json) | 143.225 | 71.907 | 1.909490 |

Cached-state preparation is outside each measured request. The sub-microsecond
fully cached TTFT uses the retained completed distribution and stages the first
fixed ID; it is an internal event without sampling, transport or visible text.
Decode has 127 completed distribution intervals. The last ID remains pending.
Changed histories replay from zero. After measured capacity requests, the
pending final token is evaluated so all 2,048 positions commit; a further token
is rejected before mutation. Native stderr is retained beside each JSON.

| FP32 MLX input length | TTFT median (ms) | Decode steps/s | Request median (s) | Physical process peak (bytes) |
| --- | ---: | ---: | ---: | ---: |
| [128](2026-10-10-mlx-fp32-128-128.json) | 60.260 | 68.967 | 1.902777 | 3342387216 |
| [512](2026-10-10-mlx-fp32-512-128.json) | 153.516 | 67.543 | 2.025936 | 3346597856 |
| [1024](2026-10-10-mlx-fp32-1024-128.json) | 254.322 | 66.389 | 2.171355 | 3606988168 |
| [1920](2026-10-10-mlx-fp32-1920-128.json) | 474.138 | 65.567 | 2.413057 | 3799270672 |

MLX weights are converted once to FP32 and actual KV dtype is checked after
the measured run. Native cache allocation grows in 256-position increments:
allocated KV is 6,291,456 / 18,874,368 / 31,457,280 / 50,331,648 bytes. Every
request commits prompt+127 positions with the final output pending. The 128
row was captured before this matrix and has its own documented conditions.
The additional MLX rows use the same adapter and dependency lock as that row;
their measured physical process peaks remain below 6 GiB. Energy, pressure and
isolated compilation/scratch are unavailable.

All rows share source weights, token IDs and FP32 weight/KV storage. Their
arithmetic, chunk scheduling and cache allocation differ. The pinned M1 llama
GEMM uses half operands despite FP32 storage. Four short small-batch probes pass
the frozen independent gate, but larger-GEMM and held-out-quality qualification
remain separate. llama.cpp's kernel footprint charge counter omits clean mapped
weights and cannot establish total mapped/shared physical memory. Its
5,088,231,424-byte admission estimate is below 6 GiB; complete physical peak is
unavailable. See the [short comparator report](2026-10-10-llama-fp32-short.md).
No general speed/quality ranking or memory advantage is claimed.

```sh
.venv/bin/python tools/bench/cpu.py --backend llama --workload boundary --output benchmarks/raw/llama-boundary-repeat.json
.venv/bin/python tools/bench/mlx_baseline.py --dtype fp32 --prompt-length 1920 --output benchmarks/raw/mlx-boundary-repeat.json
```

Use all eight llama workload names and all four MLX prompt lengths from
[benchmark instructions](../README.md) to reproduce the matrix. The scalar CPU
matrix and the application trace remain pending; M2.5 is still open.

# Independent llama.cpp Metal short baseline, 10 October 2026

[Raw samples](2026-10-10-llama-fp32-short.json) retain three warmups and ten
completed 128-input/128-output teacher-forced requests on Apple M1 Pro,
MacBookPro18,3, 16 GiB, macOS 26.6.2 (25G83), AC power with low-power mode
disabled. No thermal/performance warning was reported. The pinned upstream
revision is `3d65c90d04d337e88f2b1f7f0061f40a5324e662`; the source tree was clean.
Other performance jobs were stopped during both warmups and samples.

| Measurement | Result |
| --- | ---: |
| Warm TTFT median | 35.759 ms |
| Warm TTFT min–max | 35.067–36.001 ms |
| Prefill median | 3580.993 tokens/s |
| Decode median | 75.407526 steps/s |
| Complete request median | 1.719878 s |
| Logical FP32 weights | 1976131072 bytes |
| Allocated FP32 KV | 50331648 bytes |
| Conservative admission estimate | 5088231424 bytes |
| Total mapped/shared physical peak | unavailable |

The original pinned checkpoint is converted offline to all-F32 GGUF by the
pinned upstream converter. The [probe record](2026-10-10-llama-metal-probes.json)
retains the conversion command, 7.749885-second conversion, converter identity,
GGUF SHA256 and byte count. An independent bounded reader compares all 290
converted tensors against the source BF16 bit patterns shifted into FP32;
shapes, tensor inventory and every payload value match exactly. Model import
performs no full-weight conversion during requests. Offline conversion's
physical peak was not captured.

The [build manifest](2026-10-10-llama-build.json) identifies Apple Clang 21.0.0,
CMake 4.1.3, Ninja 1.13.0, release configuration, wrapper/CMake hashes and the
binary. It took 53.144171 seconds. Embedded Metal source compiles at runtime
with the installed SDK, without a standalone `metal` command. Backend/device
initialization and initial library compilation occur before model-load timing;
pipeline compilation is lazy and not separately timed. Three warmups precede
samples. [Native diagnostics](2026-10-10-llama-fp32-short.stderr.txt) identify
25/25 layers offloaded to Metal, FP32 KV and compatible kernels. The embedding
retains a CPU-mapped view of the tied output weights; shared physical allocations
must be counted once. Flash attention is disabled and one host thread is used.

Each request resets the existing session. Input chunks are 128 tokens; vocabulary
logits are required only at each chunk's final position. Token events follow
`llama_decode`, `llama_synchronize` and completed host logits, before staging the
next fixed ID. The final output ID remains pending, leaving 255 committed
positions. The record excludes sampling, tokenization, detokenization, transport,
admission and event allocation, matching the scalar worker's defined boundary.

All four short final-vocabulary probes pass the committed independent FP32
gates unchanged, with maximum tolerance fraction 0.08760283 and matching greedy
IDs. They exercise the small-batch path. The pinned M1 `kernel_mul_mm_f32_f32`
uses half matrix operands with float accumulation; short success does not
numerically qualify its 128-token GEMM or held-out quality. Cache/pending/replay
and rejection checks pass, including all 2,048 real-model positions committed;
that capacity check establishes execution/state rather than logit parity.

The kernel `phys_footprint_peak` counter is retained in the raw records but
omits clean mapped weights and may incompletely represent shared GPU memory.
It cannot establish total physical working set or a memory advantage. The
admission estimate is below 6 GiB, reserves two complete GGUF copies, FP32 KV and
1 GiB workspace, and is an estimate rather than physical qualification.
Scratch, memory pressure, energy and visible-text latency remain unavailable.
FP32 weight/KV storage and token hashes match the controlled CPU/MLX track;
reduction behavior and allocation policies differ. No general speed or quality
ranking is claimed.

```sh
.venv/bin/python tools/bench/llama_prepare.py --check
.venv/bin/python tools/bench/llama_check.py --output benchmarks/raw/llama-probes-repeat.json
.venv/bin/python tools/bench/cpu.py --backend llama --workload short --output benchmarks/raw/llama-short-repeat.json
```

See [preparation instructions](../README.md) for the local source/build setup.
The comparator is separate-process tooling; no llama.cpp code enters Toklet.
M2.5 remains open for the full workload matrix, application trace and CPU
capacity execution. Later lossy/longer-context quality gates remain separate.

# Resident MPS projection stages, 10 October 2026

GPU bias initialization, MPS FP16-input/FP32-output projection and a final GPU
FP16 cast pass the frozen exact matrix checks and five tiny Q-projection gates
in two runs. The bias edge returns `1.0009765625` after one FP16 rounding,
unlike the rejected direct FP16-output MPS mode. The new path encodes all three
stages into one command and waits only after that command completes. This
qualifies the isolated projection mechanism on this Mac; Rust execution and
M4 milestone status remain unchanged.

The [shader](../../tools/experiments/mps_projection.metal) has two simple
kernels: promote a resident FP16 bias into every row of an FP32 result buffer,
then store the projected FP32 values as FP16. The existing public MPS operation
runs between them with `alpha=1`, `beta=1`. Input and weight layouts remain
`[tokens,in]` and `[out,in]`; result buffers are `[tokens,out]`, with separately
queried FP32/FP16 row strides. Four UInt32 parameters carry rows, columns and
both strides; checked arithmetic bounds logical counts and physical indexing.
Rounded dispatch tails return early. Width is queried from each pipeline and
device rather than assumed from this Mac.

The [native harness](../../tools/experiments/mps_probe.swift) caches both
pipelines once per process and a matrix kernel per case. Input, weights, bias,
FP32 scratch and FP16 output allocations persist across all thirteen requests
per case. The CPU never restages bias between requests in `f16-stored` mode.
Checked combined allocation precedes buffer creation; the largest logical
footprint is `273,486,676` bytes, including actual row strides and seven end
guards for all five buffers. The initial 6 GiB ceiling is unchanged.

Shared buffers use explicit tracked hazards. Separate dependent encoders are
placed in the same retained-reference command. The GPU initializes C, MPS
reads/writes C, and the store kernel reads C and writes the FP16 output; no
intermediate CPU wait or buffer access occurs. Lexical owners and retained
command references preserve resources through successful completion. Typed
CPU access is limited to initialization or checked-complete diagnostic reads.

## Frozen numerical qualification

The original [integer/manual manifest](../../tests/fixtures/mps-matrix-v1/manifest.json)
and its reconstruction oracle are unchanged. The existing exact bias edge
also remains unchanged. Every accepted generic case checks all FP32
intermediate bits, final FP16 output bits, physical end guards, and FP16 output
row padding on every repetition. Input, weight and bias payload hashes are
checked before and after the full thirteen-request case.
The irregular `[5,7,3]` case exercises distinct FP32/FP16 strides and a rounded
threadgroup tail. The finite cancellation case retains `65504` even though an
individual product exceeds Half range. `131008` remains excluded from final
FP16 storage because it is unrepresentable; the FP32-only modes still test it.

Each run preserves thirteen tested cases, three warmups and ten measurements
per tested case: 130 measured samples. Exact output payload hashes for the
single-cell bias edge independently match little-endian FP32 and FP16 bytes
for `1+2^-10`. Repeated correct outputs with no CPU bias restaging check that
GPU initialization replaces the previous result before each multiplication.
All dyadic operands are Half-exact; these cases do not infer arbitrary FP32
arithmetic or MPS accumulation precision.

Five additional functional tiny first-layer Q checks read the actual GPU-stored
Half output. They pass the original FP32 expected arrays under unchanged FP16
gates, again at maximum fraction `0.48758482163096367`. A separate exact check
compares the GPU storage conversion with one Half rounding of its FP32
intermediate. Tiny input and weight hashes remain unchanged, as do bias hashes,
row padding and end guards. Array reads now validate byte metadata before a
bounded read of at most the expected size plus one byte, and hash those loaded
bytes before little-endian decoding.

The second run also executes the three original modes: FP32 and mixed-storage
FP32 output still pass, and direct FP16 output still rejects the first bias-edge
warmup at bits `15360` versus `15361`. That failure remains preserved in the
record and stderr. No fixture or numerical gate was changed.

## Completed-work observations

The timer includes command creation, both custom encoders, MPS encoding,
submission and successful completion. It excludes allocation, pipeline/kernel
preparation, diagnostic reads, numerical checks, hashes and transport. Both GPU
pointwise stages are included, and the excluded CPU-bias-reset time is exactly
zero. Matrix data is reused across three warmups and ten measurements; modes
are sequential rather than alternating pairs. These are isolated observations,
with no engine adoption or end-to-end performance acceptance.

| Shape `[tokens, in, out]` | Run 1 median ms | Run 2 median ms | Run 2 min–max ms |
|---|---:|---:|---:|
| `[1, 896, 896]` | 1.157 | 1.163 | 1.147–1.175 |
| `[128, 896, 896]` | 0.891 | 0.868 | 0.839–0.907 |
| `[1, 896, 128]` | 0.981 | 1.137 | 1.117–1.159 |
| `[128, 896, 128]` | 0.367 | 0.385 | 0.310–0.479 |
| `[1, 896, 4864]` | 0.717 | 0.582 | 0.550–1.360 |
| `[128, 896, 4864]` | 1.531 | 1.499 | 1.208–1.946 |
| `[1, 4864, 896]` | 1.764 | 1.760 | 1.741–1.881 |
| `[128, 4864, 896]` | 1.352 | 1.319 | 1.029–1.901 |
| `[1, 896, 151936]` | 11.590 | 9.563 | 9.524–9.602 |

The output-projection medians differ substantially between runs, and several
other shapes have broad ranges. The records retain this variation; no stable
speedup is claimed. Pipeline preparation takes 132.372 ms and 1.051 ms,
respectively, with system shader caches uncontrolled. These do not establish
uncached compilation latency. FP32 scores/logits in a decoder would require
the appropriate FP32 output path; this experiment tests FP16-stored projection
results, not a complete decoder output policy.

## Evidence, environment and checks

[Run 1](../../benchmarks/results/2026-10-10-mps-resident-projection-v1.json) and
[run 2](../../benchmarks/results/2026-10-10-mps-resident-projection-v2.json) are
exact raw copies with every final sample. Raw JSONL and stderr identities are
bound into each attempt. Compiler diagnostics and successful-mode stderr are
empty; the expected direct-FP16 rejection stderr is saved with run 2.

Both runs bind Swift SHA-256
`33231376d140c5fa6c394585188607753db0ffa350c449abab9d2ae828914c10`,
shader `78277e211a8e09fc63727241b00a38202d9d2d8926845b00d069efac99cd564e`,
binary `575e7ec849ff0b9fea5beb053d78e6d3ff4528501f3e63c073fc4d9e0cfd9f1f`,
driver `a2b8732486750072437f2d1d825dc1e76dd13d33e466e14556db7504b4819788`,
and unchanged matrix manifest
`9c02ae643d9da7890d6a65d06fa0815f303211350185d923502ac70cb29af483`.
Base revision is `9eb6e50`; dirty source identities are explicit in the records.
Shader compilation uses Metal 2.3 with safe math and precise functions.
Native compilation uses Apple Swift 6.3.3, SDK 26.5 and
`-O -warnings-as-errors -target arm64-apple-macosx15.0`.

Device is Apple M1 Pro with unified memory, macOS 26.6.2 (25G83), AC power,
AC low-power mode disabled, and no reported thermal/performance warning at
both run boundaries. Own-process physical charge peaks are `282,772,296` and
`282,395,464` bytes, below 6 GiB. Total GPU/system memory and energy remain
unavailable; no full-model memory claim follows.

The scalar benchmark numerical worker finished measured request 4 normally
and remained idle with cumulative CPU time `87:20.99` during these suites and
GPU checks. Only its transport was held; it resumed at 01:06:30 UTC. Frozen
scalar benchmark and CLI release binary hashes remain unchanged.

Formatting, Clippy with warnings denied, all 74 ordinary Rust tests and 45
offline Python checks pass. The new sixth MPS test rejects CPU restaging,
missing GPU-storage/padding qualification, bad strides and invalid/duplicate
mode selection; existing tests retain exact case/sample/gate validation,
fixture corruption/overwrite rejection and child-owned watchdog cleanup.
Root checks bind source/shader/binary/fixture identities, match raw checkpoints
to final samples, independently check bias-edge payload bytes, and verify
memory, power, the preserved rejection and unchanged release binaries.
Astra accepted the complete task after the hash-check frequency wording was
corrected. Review independently verified tracked sequencing, bounds, the final
cast, all six offline tests, both exact raw copies, identities, 130 checkpoint
samples per run, medians, edge payload hashes, tiny gates and the preserved
rejection. It ran no GPU, model, compilation or heavy tests.

```sh
.venv/bin/python tools/experiments/mps_probe.py --mode f16-stored --output benchmarks/raw/mps-resident.json --build-dir artifacts/mps-resident
.venv/bin/python -m unittest discover -s tools/experiments -p 'test_*.py'
```

Native checks must run separately from measured engine workloads. Complete
tiny/Qwen GPU forward, resident model/session ownership, numerical error
recovery, quality and explained GPU traces remain pending.

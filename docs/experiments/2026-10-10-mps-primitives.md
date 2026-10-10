# Public MPS matrix evaluation, 10 October 2026

FP16 inputs with FP32 output pass the exact matrix cases, a bias rounding edge
and five frozen tiny Q-projection gates on this M1 Pro. FP16 output fails that
edge on both final runs: it returns `1.0` (bits `15360`) where the independently
specified result is `1.0009765625` (bits `15361`). The rejection remains intact.
These checks support further qualification of the mixed-storage primitive;
they select no engine backend and complete no M4 milestone.

The [Swift probe](../../tools/experiments/mps_probe.swift) uses public
[MPSMatrixMultiplication](https://developer.apple.com/documentation/metalperformanceshaders/mpsmatrixmultiplication).
Activations are row-major `[tokens, in]`, weights `[out, in]`, and output
`[tokens, out]`; `transposeRight=true`, `alpha=1`, `beta=1`. Every output row
starts with the bias. Device-recommended row strides are used, including
padding. Three storage modes are evaluated in separate processes: FP32 inputs
and output, FP16 inputs and output, and FP16 inputs with FP32 output.
Accumulation precision and internal kernel behavior are not inferred from
storage dtype or the observed failure.

## Independent checks and unchanged gates

The [integer/manual oracle](../../tools/experiments/mps_oracle.py) and its
[frozen manifest](../../tests/fixtures/mps-matrix-v1/manifest.json) preceded GPU
evaluation. Ten synthetic cases cover the nine Qwen projection shapes below
and an irregular `[5,7,3]` tail. Inputs are `(7*token+feature)%13-6`, divided by
32; weights are `(out+3*feature)%17-8`, divided by 64; bias is `(out%7-3)/256.
An independently generated 13-by-17 integer table gives exact dot products.
All possible partial-sum numerators remain below `2^24`. These dyadic values
are also exactly representable in FP16, so the FP32-storage mode does not
qualify arithmetic on arbitrary FP32 operands.

Three manual cases check a biased projection, a finite cancellation with a
product exceeding FP16 range, and a finite FP32 result of `131008`. The latter
is skipped only in FP16-output mode because it exceeds that storage range.
Outputs must match FP32 bits or one final round-to-nearest-even FP16 cast;
there is no fitted target tolerance.

A fourth manual case uses `x=[1,1]`, `w=[1,2^-11]`, `bias=2^-11`. Its expected
result `1+2^-10` follows exact arithmetic and the single-final-cast policy,
already tested in the committed independent
[reference test](../../tools/reference/test_qwen_fp16.py). Both FP32-output
modes pass. FP16 output rejects its first warmup with the exact values above,
after 120 measured rows from earlier passing cases. Failure exit, numerical
bits, partial checkpoints and stderr are retained. No gate was relaxed and
no internal cause is asserted.

For each accepted mode, one additional functional check per frozen tiny case
uses the first layer's FP16 normalized input, Q weights and bias. Results are
stored as FP16 before comparison with the original FP32 expected Q projection
under the existing [FP16 fixture](../../tests/fixtures/fp16-v1/manifest.json)
gates. All five cases pass with maximum tolerance fraction
`0.48758482163096367` on both runs. Loaded payload bytes are hashed and decoded
explicitly as little-endian values without assuming aligned `Data` storage.
This checks one biased linear operation only; normalization, rotary, KV,
attention, complete tiny/Qwen forward and quality remain unqualified here.
The final rejected FP16-output process never reaches these five checks.

## Completed-work observations

Each generic matrix case receives three warmups and ten measured requests.
The timer includes command creation, encoding, submission and successful
completion; command status is checked before reading output. CPU bias reset,
allocation, kernel construction, hashes, checks and transport are excluded.
The same matrix is reused, fitting many cases in warm caches. The modes run
sequentially, without alternating pairs. These are isolated kernel observations,
not a performance acceptance or full-model throughput comparison.

| Shape `[tokens, in, out]` | FP32 run 1 / 2 median ms | FP16 → FP32 run 1 / 2 median ms | FP16 → FP32 run 2 min–max ms |
|---|---:|---:|---:|
| `[1, 896, 896]` | 1.218 / 1.185 | 1.113 / 1.153 | 1.109–1.172 |
| `[128, 896, 896]` | 0.758 / 0.743 | 0.731 / 0.742 | 0.725–0.788 |
| `[1, 896, 128]` | 0.660 / 0.641 | 0.945 / 1.127 | 0.988–1.233 |
| `[128, 896, 128]` | 0.360 / 0.344 | 0.302 / 0.306 | 0.280–0.332 |
| `[1, 896, 4864]` | 1.016 / 1.031 | 0.561 / 0.574 | 0.563–0.593 |
| `[128, 896, 4864]` | 1.591 / 1.584 | 1.477 / 1.498 | 1.018–1.541 |
| `[1, 4864, 896]` | 1.938 / 1.921 | 1.762 / 1.770 | 1.764–1.955 |
| `[128, 4864, 896]` | 1.099 / 1.089 | 0.954 / 0.973 | 0.959–1.107 |
| `[1, 896, 151936]` | 10.781 / 10.857 | 9.447 / 9.497 | 9.407–9.628 |

FP16-input/FP32-output performance varies by shape, including a slower small
KV projection. Each final self-contained record retains every sample and GPU
timestamp when available. A resident decoder would still need a qualified GPU
bias-initialization path and FP16 activation cast, without CPU restaging or
per-operator waits. Those extra costs are absent from this table.

## Ownership, bounds and environment

Checked arithmetic admits combined buffers before allocation under both the
queried device maximum and unchanged 6 GiB ceiling. The largest logical
allocations are `545,150,036` bytes for FP32 and `272,878,904` bytes for the
mixed-storage output projection, including strides and seven guard elements
after each physical matrix payload. Row padding is unspecified and is not
required to remain unchanged. All physical end guards survive successful
checks; input and weight payload hashes remain unchanged across repetitions.
Hashing uses bounded 64 KiB chunks.

Owned shared buffers have checked alignment and typed binding. CPU accesses
occur before submission or after checked completion. Retained command
references and lexical owners preserve matrix/buffer/kernel lifetimes. The
[driver](../../tools/experiments/mps_probe.py) refuses existing evidence/build
paths, uses a five-minute compile timeout and three-minute per-mode watchdog,
and kills/reaps only the process group it creates. Successful and Swift-failed
children wait at an idle boundary for own-process footprint inspection.

Own-process physical charge peaks for run 1 / 2 are `554,009,632` /
`554,091,504` bytes (FP32), `289,686,320` / `279,905,096` (rejected FP16), and
`281,609,032` / `289,866,568` (mixed storage). All are below 6 GiB. These count
the separate probe's shared allocations once; total system/GPU memory and
energy remain unavailable. The paused scalar worker's resident model is a
separate process and is excluded from these probe counters.

Hardware is Apple M1 Pro, unified memory, macOS 26.6.2 (25G83), Apple Swift
6.3.3 and macOS SDK 26.5. Compilation uses
`-O -warnings-as-errors -target arm64-apple-macosx15.0`. Both runs record AC
power, AC low-power mode disabled and no reported thermal/performance warning.
The benchmark transport was held between requests; the numerical worker was
never suspended, completed sample 3 before probe execution, and remained idle
with unchanged cumulative CPU time during these checks. Transport resumed at
00:51:57 UTC. The original scalar release binaries remain unchanged.

## Evidence and verification

Final [run 1](../../benchmarks/results/2026-10-10-mps-primitives-v4.json) and
[run 2](../../benchmarks/results/2026-10-10-mps-primitives-v5.json) are exact
copies of local raw records, preserving failed attempts and each successful
mode's 140 samples plus five tiny checks. Compiler diagnostics are empty;
expected FP16 rejection stderr is saved alongside each record. Exploratory
v1–v3 runs remain ignored and are not qualification evidence.

Both runs bind the same Swift source SHA-256
`3814038a858e133774fa273981f0a595abdc3dbc4ecf4a6136921d8ee19cc7db`,
binary `4a4e5e9736581e35fad96fe1f717afe2d3540a4a700088835aeaa60c2605339d`,
driver `dd33dd2397fac24b16fe892ff692b8bb2fe613fa828e390f20417f2145b4499c`,
and oracle manifest `9c02ae643d9da7890d6a65d06fa0815f303211350185d923502ac70cb29af483`.
The base revision is `da20828`; modified source identities and helper,
independent-reference, manifest and raw/stderr hashes are retained in records.

Five offline probe tests exercise independent oracle reconstruction and
corruption, overwrite admission, successful/failed idle handshakes, exact
case/sample/gate validation and real child timeout/reaping without Apple APIs.
Formatting, Clippy with warnings denied, all 74 ordinary Rust tests and 44
offline Python checks pass. Root verification also matches native checkpoints
to final samples, identities, exact raw copies, expected failure bits, own
memory peaks and power records. The Rust numerical engine is unchanged.
Astra accepted the complete task without findings, independently checking
resource safety, watchdog cleanup, all five offline tests, exact raw copies,
source/binary/fixture identities, checkpoint samples, medians, tiny gates and
rejection bits. Review ran no GPU, model, compilation or heavy tests.

```sh
.venv/bin/python tools/experiments/mps_probe.py --output benchmarks/raw/mps-primitives.json --build-dir artifacts/mps-primitives
.venv/bin/python -m unittest discover -s tools/experiments -p 'test_*.py'
```

Run native checks separately from performance jobs. Full Qwen GPU execution,
quality, resident scheduling, GPU trace capture and physical iPhone checks
remain pending.

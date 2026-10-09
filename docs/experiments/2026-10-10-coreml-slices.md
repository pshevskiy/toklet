# Public Core ML projection and persistent-KV slices, 10 October 2026

The experiment tests whether public Core ML can execute a biased projection and
a persistent causal GQA slice within the numerical gates frozen in
[coreml-slice-v1](../../tests/fixtures/coreml-slice-v1/manifest.json). The
[driver](../../tools/experiments/coreml_probe.py) builds public MIL graphs directly;
it does not change the Rust decoder. Result: FP32 projection and both attention
variants pass; FP16 projection fails. Anticipated placement prefers CPU throughout,
and actual Neural Engine placement is unobserved. This provides supported CPU
slice evidence and an inconclusive ANE experiment, with no engine adoption.

## Graphs, layout and ownership

Projection uses input `[1, 32]`, row-major weights `[32, 32]`, bias `[32]`, and
output `[1, 32]`. Separate FP32 and FP16 graphs use the same immutable first-layer
tiny-model parameters, rounded only according to their declared storage type.
Logical weights are 4,224 and 2,112 bytes respectively.

Attention accepts pre-rotated query `[2, 2, 8]` (two KV groups, two query heads per
group), one new K/V row `[2, 1, 8]`, and a one-element int32 cache position. Each
session owns separate FP16 K and V state arrays `[2, 32, 8]`, with contiguous
head dimensions and 256-element head strides. Both arrays total 2,048 bytes.
Each cache is read, updated at its supplied position, and written immediately.
QK scores are scaled by `1/sqrt(8)`, positions after the supplied cache position
are masked, softmax reduces over positions, and probabilities multiply V.
Output is FP16 `[1, 32]`. One graph uses FP16 arithmetic; the other explicitly
casts Q/K/V to FP32 for scores, softmax and context, then casts the output to FP16.
Absolute rotary offsets were applied by the independent fixture generator.

Five frozen cases cover one token, four-token prompt, a changed future token,
three tokens rotated at absolute offset five, and all 32 cache positions. Every
completed sequence checks output shape/finiteness, the original frozen numerical
gate, and every cache value including the zero invalid suffix. Interleaved states
remain independent and fresh states are zero. A separate masking check writes
exact `0.5` to both complete caches, verifies exact readback before prediction,
then confirms the future suffix does not affect the first output.

The Python state API returns FP32 arrays for these FP16 states. Comparison checks
exact promoted FP32 bits against the frozen FP16 arrays; actual values are never
rounded to FP16 to make them pass. A one-ULP FP32 error that would disappear after
half rounding fails the offline regression. The public write API accepts FP32
arrays here; `0.5` is exactly representable in either format.

## Conversion, placement and measurement scope

The [conversion manifest](../../benchmarks/results/2026-10-10-coreml-slices-build.json)
records package hashes, source/oracle/fixture/lock identities, actual input/output
and state types, dependency versions and conversion/save costs. Core ML 9.0 is
installed into a separate ignored directory with the hashed
[extras lock](../../tools/experiments/coreml-extras.lock), leaving the existing
benchmark environment unchanged. Python is 3.12.8 and NumPy is 2.3.2. The installed
Torch 2.9.1 triggers an import warning; no Torch conversion frontend is used.
Public stateful APIs require iOS 18/macOS 15; this run qualifies only macOS on the
named Mac. See [stateful Core ML models](https://apple.github.io/coremltools/docs-guides/source/stateful-models.html).

The device is Apple M1 Pro, MacBookPro18,3, 16 GiB, macOS 26.6.2 (25G83). Records
start at `7c4a7c4` with the experimental driver uncommitted; the exact driver hash
is `18596f1fb3f5e75726c622e79f25ca1a8d9bf800f755e72a5affc1732723d8c8`.
AC power was attached, AC low-power mode disabled, and no thermal or performance
warning was reported at either boundary. The scalar baseline process was stopped
during its excluded second warmup, with no other benchmark running.

Conversion takes 0.095/0.051 seconds for FP32/FP16 projection and 0.072/0.077
seconds for FP16/FP32-arithmetic attention. Package saves take 1.4–1.6 ms, measured
separately; graph creation and fixture reads are outside conversion timing.
The build's Core ML import takes 2.172 seconds. Each policy/graph runs in a new
process: import takes 1.371–1.876 seconds and model compile/load 75–105 ms.
Compilation may use system caches, so these are observed process-start costs,
not a claim of an uncached compiler benchmark.

Each case has three complete warmups and ten measured sequences. Prediction
timers cover synchronous `predict` including its Python boundary, native output
allocation, and input/model-boundary conversions. In particular, the public
Python wrapper promotes FP16 input arrays to FP32 before calling the native
runtime. Prepared input dictionaries, fixture I/O, state creation, numerical
checks and diagnostic cache readback are outside these timers. The additionally
recorded sequence wall time includes loop/output-copy overhead; the two timing
scopes overlap and must not be added. The table reports median prediction sums,
not full request TTFT, token generation, or attention-kernel-only latency.

Both `CPU_ONLY` and `CPU_AND_NE` are tested. Compute plans are available for all
packages, and every nonconstant operation anticipates `MLCPUComputeDevice` as
preferred. FP32 linear supports only CPU in these plans. Several FP16/state
operations list Neural Engine as supported, but this does not establish actual
placement. The public [MLComputePlan API](https://apple.github.io/coremltools/source/coremltools.models.html)
reports anticipated assignments/costs. Full Xcode/Instruments execution tracing
is unavailable in this command-line-tools environment; actual ANE placement
remains null. CPU-only placement is guaranteed by its explicit policy.

## Results and decision

| Graph / policy | Single, ms | Prompt, ms | Changed future, ms | Offset, ms | Capacity, ms | Maximum gate fraction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| FP32 projection / CPU | 0.055 | 0.178 | 0.236 | 0.177 | 1.811 | 0.009 |
| FP32 projection / CPU + NE | 0.051 | 0.237 | 0.233 | 0.174 | 1.633 | 0.009 |
| FP16 attention / CPU | 0.136 | 0.452 | 0.449 | 0.348 | 3.040 | 0.929 |
| FP16 attention / CPU + NE | 0.131 | 0.447 | 0.430 | 0.324 | 2.921 | 0.929 |
| FP32 attention arithmetic / CPU | 0.141 | 0.435 | 0.450 | 0.313 | 3.020 | 0.424 |
| FP32 attention arithmetic / CPU + NE | 0.140 | 0.463 | 0.455 | 0.343 | 3.016 | 0.424 |

The FP16 projection fails at the first single-token warmup under both policies:
maximum absolute error `0.0018783807754516602`, frozen tolerance fraction
`1.5727969659617609`. Its failure records have no measured samples. The original
gates remain unchanged; conversion success is not numerical qualification.
The other six runs each retain 50 measured sequences, exact cache checks where
applicable, raw per-prediction timings, min/max dispersion and placement details.

The versioned records are [FP32 CPU](../../benchmarks/results/2026-10-10-coreml-projection_fp32-cpu_only.json),
[FP32 CPU + NE](../../benchmarks/results/2026-10-10-coreml-projection_fp32-cpu_and_ne.json),
[rejected FP16 CPU](../../benchmarks/results/2026-10-10-coreml-projection_fp16-cpu_only.json),
[rejected FP16 CPU + NE](../../benchmarks/results/2026-10-10-coreml-projection_fp16-cpu_and_ne.json),
[attention CPU](../../benchmarks/results/2026-10-10-coreml-stateful_attention-cpu_only.json),
[attention CPU + NE](../../benchmarks/results/2026-10-10-coreml-stateful_attention-cpu_and_ne.json),
[FP32-arithmetic attention CPU](../../benchmarks/results/2026-10-10-coreml-stateful_attention_fp32_accum-cpu_only.json),
and [FP32-arithmetic attention CPU + NE](../../benchmarks/results/2026-10-10-coreml-stateful_attention_fp32_accum-cpu_and_ne.json).
Companion stderr files preserve the warnings and rejected-gate traces. Ignored
raw checkpoints match every retained measured sample exactly. Early converter/API
diagnostics are exploratory artifacts, with no accepted timing claims.

Own-process physical peaks are 237,290,528–242,025,576 bytes, and build peak is
233,030,616 bytes, all below 6 GiB. These counters include Python/import/runtime
support but do not establish total Core ML service/device memory, energy, or
full-model working set. State/weight logical bytes above are separate quantities.

Decision: retain the reproducible probe and its supported CPU state behavior.
Reject this FP16 projection under the existing gate. ANE acceleration remains
inconclusive; microsecond policy differences do not justify adoption or a speed
claim, and no second confirmation is needed for that negative adoption decision.
No full decoder, end-to-end generation, held-out quality or engine integration
is qualified. M2.5 and M3 remain pending while the scalar matrix and CPU candidate
qualification run in their required order.

Five offline probe tests, three frozen-oracle tests, eighteen benchmark Python
tests, formatting, Clippy with warnings denied, and all 74 ordinary Rust tests
pass. Eight native probe runs have their identities, unchanged gates, checkpoints,
summaries, exact states, power records and copied evidence checked. The scalar
decoder is unchanged, so unrelated full-model suites are not repeated.
Astra accepted the completed task with no remaining findings after independently
checking the fix, frozen gates, identities, raw copies/checkpoints, summaries,
state checks and report limitations. Its poison-installation finding was fixed
and regression-tested before regenerating the final packages and records.

```sh
.venv/bin/python -m pip install --require-hashes --no-deps --target artifacts/coreml-tools -r tools/experiments/coreml-extras.lock
PYTHONPATH=artifacts/coreml-tools .venv/bin/python tools/experiments/coreml_probe.py --prepare artifacts/coreml-slices
PYTHONPATH=artifacts/coreml-tools .venv/bin/python tools/experiments/coreml_probe.py --run benchmarks/raw/coreml-state-cpu.json --packages artifacts/coreml-slices --kind stateful_attention --policy cpu_only
PYTHONPATH=artifacts/coreml-tools .venv/bin/python tools/experiments/coreml_probe.py --run benchmarks/raw/coreml-state-ne.json --packages artifacts/coreml-slices --kind stateful_attention --policy cpu_and_ne
.venv/bin/python -m unittest discover -s tools/experiments -p 'test_*.py'
```

Repeat the two runtime commands with each of the four `--kind` values and fresh
output names. FP16 projection is expected to exit unsuccessfully under the
unchanged frozen gate. Preparation/runtime refuse existing evidence paths and
changed source/package/dependency identities. Every measured sequence is
checkpointed immediately; a ten-minute watchdog bounds only the experiment's
own process. Interrupted rows remain incomplete evidence and are never resumed
or overwritten silently. Run performance jobs sequentially.

# Stable FP16 Metal SiLU, 10 October 2026

The standalone [shader](../../tools/experiments/rust_metal_silu.metal) and
[Rust primitive](../../tools/experiments/rust_metal_silu.rs) prepare the tiny
model's MLP activation. All frozen tiny stages and independent edge cases pass
on Apple M1 Pro; this task qualifies one elementwise operation. Astra accepted
the complete task without findings after independently reproducing all exact
outputs, source gates, guards, regressions, identities, memory and final records.

Input and output are contiguous `[rows, 64]` Half arrays. One flattened lane
promotes its input to FP32, computes `e = exp(-abs(x))`, selects
`1/(1+e)` for nonnegative inputs or `e/(1+e)` for negative inputs, multiplies by
`x`, and stores one Half result. The negative branch avoids exponential
overflow for large negative finite inputs. Contraction is disabled; existing
Metal 2.3 safe-math/precise-function options are required and read back.
Logical data traffic is one Half read and one Half write per element, before
cache effects. Adjacent lanes traverse adjacent elements. Each element uses
one exponential, one division and ordinary arithmetic; no reduction, weight,
session state, gating fusion or allocation occurs in the shader.

Two distinct shared buffers hold input and output, each followed by seven Half
sentinels. Buffers and the cached pipeline/queue persist across three warmups
and ten repetitions. Output sentinels reset before each submission. One command
retains both resources, completes synchronously, and passes status/error checks
before host access. Each readback verifies immutable input bits and guards,
output guards and finite active output. Failure discards the result; there is
no session state to commit or recover. The existing private buffer owner checks
actual byte length, storage/hazard mode and scalar alignment. Binding slots
0/1 hold input/output at offset zero; slot 2 copies a four-byte, four-aligned
UInt32 count. Checked rounded launches and the shader's tail check bound access.

The loader checks at most one MiB of input, 1–32 unique bounded ASCII identifiers,
1–32 rows per case, width 64, at most 8,192 total elements and finite UInt16 Half
inputs before creating a device. Conservative admission counts shared input and
output once, one host baseline and two diagnostic readbacks, plus a fixed 64 MiB
allowance for runtime, bounded parsing and retained diagnostics. At 32 rows,
each guarded buffer has 4,110 bytes: shared storage totals 8,220 bytes, host
baseline/readbacks total 12,330 bytes and admission is 67,129,414 bytes.

The [qualification driver](../../tools/experiments/rust_metal_silu.py) uses all
ten saved `gate_projection` → `gate_silu` stages from both layers of the five
frozen tiny cases: 5,632 values. Their original FP32-source gates and Half
payload/manifest identities stay fixed. Exact agreement with the independent
Half oracle is an additional diagnostic; it does not replace the source gate.
A 17-row manual case adds 1,088 values with preselected exact Half outputs for
signed zero, small subnormals, maximum finite positive/negative inputs and
positive/negative 20. The standard-library FP64 stable formula and IEEE Half
rounding check those expectations before target execution. These selected
outputs are away from sensitive Half ties and require exact bit agreement.

The fresh isolated utility also runs existing add/tail, RMS, embedding and
residual checks, including completed residual overflow rejection. Full native
outputs are checked before compaction; source/build/binary, input, oracle,
driver, raw JSONL and stderr identities bind retained evidence. Offline tests
check corruption, finite/guard/completion/ownership policies and evidence path
separation. Tests, compiler and GPU execution ran after the scalar boundary
worker finished normally, while only its next-workload orchestration remained
held. Frozen CPU numerical binaries and qualified engine sources are unchanged.

The [completed record](../../benchmarks/results/2026-10-10-metal-silu-v1.json),
[complete native outputs](../../benchmarks/results/2026-10-10-metal-silu-v1.json.native.jsonl)
and [stderr](../../benchmarks/results/2026-10-10-metal-silu-v1.json.stderr.txt)
retain six successful native runs and three expected failures. SiLU covers
11 cases, 6,720 values and 143 completed commands, checking all 87,360 active
output values and 1,001 output guards across three warmups and ten samples.
Every frozen and manual output bit agrees with the independent Half reference.
The maximum unchanged FP32-source gate fraction is `0.4889693871776979`, with
maximum absolute error `0.0016703009605407715`; neither tolerance was changed.
Inputs, guards, finite results, completion and buffer reuse pass every repetition.
The SiLU child physical peak is 13,664,880 bytes, below conservative maximum
admission of 67,129,414 bytes and the 6 GiB ceiling. This is an own-child charge,
not an estimate of preparation-process or system-wide memory.

The same binary completes all 39 add/tail, 364 RMS, 78 embedding and 273 residual
regression commands: 897 positive commands including SiLU. Invalid shader,
insufficient budget and completed residual overflow fail at their required
stages. Isolated formatting, Clippy with warnings denied, 13 standalone Rust
tests, locked offline release compilation and all 52 offline experiment Python
tests pass. Engine code is unchanged; its formatting, Clippy and 76 ordinary
Rust tests passed in the immediately preceding CLI task in the same idle window.
Observed build tools are Rust 1.90.0, macOS SDK 26.5 and deployment target 15.0.
Command timings include host encoding, submission and completed waiting; they
are diagnostic and do not isolate kernel time or establish performance acceptance.

The completed record SHA-256 is
`72035f4bc484579ba7385189ac90804f9ec04e99deb1fa722d72f50004f091de`;
the isolated executable is
`11f2c31c8459bf26eccb039818714aab8a32bc7bafb91d5ba58ab65699d59f63`.
Native JSONL and stderr hashes are respectively
`71f54ae9dc0f9f9949bd3e09f0308e60882ad79b83ef4744b0bcf3f871a69401`
and `431a7fa958680fc0315c5b07e4b1237367e3d28b463a6698f05719636606e743`.
Copied Rust/shader sources, frozen manifests/payloads, inputs and all drivers are
verified before and after execution and pinned in the record.

Width 64 is the only admitted SiLU width. Qwen activation, elementwise gating,
MLP/decoder composition, long-context state, quality, performance and GPU traces
remain pending. No milestone completion or latency acceptance is implied.

```sh
.venv/bin/python tools/experiments/rust_metal_silu.py --output benchmarks/raw/metal-silu.json --build-dir artifacts/metal-silu --input-dir artifacts/metal-silu-inputs
```

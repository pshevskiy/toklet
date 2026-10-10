# Selected Qwen Metal RMS normalization, 10 October 2026

The standalone [Qwen driver](../../tools/experiments/rust_metal_qwen_rms.py)
extends the [tiny RMS qualification](2026-10-10-metal-rms.md) to width 896 using
the same Rust owner and serial FP32/Half-storage shader. Native validation
passes on M1 Pro. Existing source arrays, gates, Rust code and shader remain unchanged.

## Source and stage pairing

The full independent FP16 reconstruction and original Transformers FP32 source
are pinned by their manifest hashes. Their corresponding committed
[FP16](../../tests/fixtures/qwen-fp16-v1/manifest.json) and
[FP32](../../tests/fixtures/qwen-v2/manifest.json) probe manifests independently
anchor full payload hashes, shapes and gates. The driver checks these links,
every selected payload, the config and checkpoint identity before native work.
Manifest reads hash and parse the same bounded bytes. Gates are reused exactly;
they were frozen before any Metal decoder evaluation.

Six saved operations have complete saved inputs for all four short cases:

| Output | Saved input | Norm weight |
|---|---|---|
| Layer 0 input norm | Embedding | Layer 0 input norm |
| Layer 0 post-attention norm | Layer 0 attention residual | Layer 0 post-attention norm |
| Layer 1 input norm | Layer 0 output | Layer 1 input norm |
| Layer 1 post-attention norm | Layer 1 attention residual | Layer 1 post-attention norm |
| Layer 23 post-attention norm | Layer 23 attention residual | Layer 23 post-attention norm |
| Final norm | Layer 23 output | Final norm |

The four original cases have 4, 3, 4 and 1 tokens, including a changed future
and a nonzero position offset. This gives 24 stages and 64,512 active values.
The fixed native inventory bound of 8,192 elements groups them into nine inputs
without splitting rows or losing cases. Each stage gets three warmups and ten
samples. Comparisons use complete original FP32 arrays and unchanged FP16-source
gates; agreement with independent stored Half bits is diagnostic.

Layer 23 input normalization is excluded because the frozen selected artifacts
do not retain layer 22's output. This task does not reconstruct that missing
activation, port other layers or qualify a composed decoder/cache/session.

## Weight loading and ownership

The checkpoint identity is the pinned BF16 safetensors artifact. A streaming
one-MiB hash buffer avoids allocating or mapping a second full model. The driver
checks the bounded header, rejects duplicate keys and requires each selected
norm tensor to be BF16 `[896]` with an aligned, exact, in-bounds range. The six
selected ranges must not overlap. Only their 10,752 payload bytes are loaded.
Each BF16 word shifts into its exact FP32 bits and rounds once to IEEE Half with
round-to-nearest-even. Nonfinite source values and Half range overflow fail
before a GPU device is created. Source and converted payload hashes are retained.
The same open descriptor is hashed before and after small tensor reads; the
checkpoint path is hashed again after native work.

The native kernel and ownership are unchanged: contiguous `[rows,896]` input
and output, `[896]` weights, seven sentinels per allocation, queried dispatch
limits, tracked hazards, retained command resources and host reads after one
completion wait. The pipeline and three shared buffers are reused across each
case's repetitions. Serial FP32 reduction preserves the documented arithmetic
order and stores Half once. Logical data traffic is 7,168 bytes per row before
cache effects. This is a correctness reference without throughput acceptance.

At four rows, native shared allocations total 16,170 bytes; host uploads and
readbacks total 25,158 bytes. Conservative admission, including the fixed 64 MiB
native/runtime/diagnostic allowance, is 67,150,192 bytes. The Python preparation
driver additionally retains selected input/reference/diagnostic arrays outside
the native child; no complete model weights are retained. Native child footprint
records do not represent total system or preparation-process physical memory.
The engine's 6 GiB budget is unchanged.

Fresh inputs, builds and output evidence must be disjoint and outside source
fixtures/model directories. Existing paths are rejected before reading the
checkpoint. All native outputs are checked before diagnostic compaction, with
full JSONL preserved locally and bound by hash. Final checks revalidate manifests,
config, checkpoint, qualification drivers and all nine input identities before
the shared driver can declare completion. Its three existing add/tail, invalid
shader and budget-rejection jobs run too.

## Checks and scope

Six synthetic offline tests cover signed zero and Half subnormal ties, archive
identity, nonfinite/overflow conversion, malformed dtype/shape/ranges, overlap,
duplicate keys, header bounds, all 24-stage/nine-batch pairing, unchanged gates,
source/payload corruption and evidence admission. These tests require no local
model downloads or Apple hardware.

The [native record](../../benchmarks/results/2026-10-10-qwen-metal-rms-v1.json)
passes all 312 completed dispatches, 838,656 active output values and 2,184
output guards. Every input/weight payload and sentinel remains unchanged.
Maximum frozen source-gate fraction is `0.4937617230261768`; maximum absolute
source error is `1.419546127319336`. All nine child processes exit zero without
a watchdog firing; maximum child physical peak is `14,729,792` bytes, below the
conservative 67,150,192-byte native admission and 6 GiB ceiling. The three
ordinary regression jobs pass, including 39 exact add/tail dispatches and the
original structured shader-compilation/admission failure stages.

Seven of the 24 stages match the independent NumPy Half arrays exactly; the
other 17 contain 46 differing values across the 64,512-value inventory, each
one Half ULP. Maximum absolute difference is `0.00390625`. Every repetition
returns identical bits. The post-qualification
[order diagnostic](../../benchmarks/results/2026-10-10-qwen-metal-rms-order-diagnostic-v1.json)
rounds each serial product, sum, division, square root, inverse and scale to
FP32 in independent Python, then stores Half once. All 64,512 native values
match it exactly. This explains the difference from NumPy's reduction order;
it changes no source gate or acceptance rule. The exact executed
[diagnostic source snapshot](../../benchmarks/results/2026-10-10-qwen-metal-rms-order-diagnostic-v1.py.txt)
is retained with its hash.

The result binds copied native source/build inputs, the norm-only checkpoint
payloads, all manifests, input files, drivers, executable and raw JSONL. Key
identities are:

| Artifact | SHA-256 |
|---|---|
| Qwen driver | `394718d6cdfdf5a5857b11ae4e7277fd9b6aee3c247264396492730f10fb4240` |
| Executable, identical to tiny qualification | `d28c3f7df3d84122f555c454b149799ac3ab6b6ffa92c4cfa2ebf17abea78f22` |
| Full native JSONL | `b05f3ed09d284fb94104ac20473ab5c8aa9d7e4346fe6e2041b43625bb39b6f1` |
| Order diagnostic source | `6519961e889e88af436aec341de9a8cd06d3c8b646b94749a5e8b8cbdb3cc5e7` |

The host is Apple M1 Pro, macOS 26.6.2, SDK 26.5 and Rust 1.90. Native compile
policy readback remains Metal 2.3, safe math, precise functions and deployment
target macOS 15. Power is AC with AC low-power mode disabled; `pmset` reports no
recorded thermal or performance warnings. These are functional results with no
controlled latency or energy claim. Formatting/Clippy with warnings denied,
five standalone Rust tests, 75 ordinary engine tests and 66 offline Python
checks pass. The [stderr](../../benchmarks/results/2026-10-10-qwen-metal-rms-v1.json.stderr.txt)
and native record preserve isolated build commands and outcomes.

The [hold](../../benchmarks/results/2026-10-10-cpu-qwen-rms-check-window.json),
[idle](../../benchmarks/results/2026-10-10-cpu-qwen-rms-check-window-idle.json)
and [completion](../../benchmarks/results/2026-10-10-cpu-qwen-rms-check-window-completed.json)
records show that only the parent coordinator was held. Its Tiled numerical
child completed naturally; worker exit was verified at 02:50:13.26 UTC before
checkpoint hashing or native/check work. The original scalar numerical worker
remained idle with CPU time `112:18.61`. After all checks, diagnostics and large
hashing finished, frozen CPU binaries and qualified source/receipt identities
were verified again and the owned primary parent resumed at 02:52:53.57 UTC.
Astra accepted the completed task with no findings after independently checking
source/native/input/raw identities, all dispatch/value/guard proofs, unchanged
source gates, evidence compaction, the order diagnostic, memory and idle-window
records using light reads and compact arithmetic checks. It ran no tests,
checkpoint hashing, builds, model or GPU work during the active CPU comparison.

No benchmark numerical worker was paused. Qwen decoder
composition, all-layer normalization, long-context cache, failure recovery,
held-out quality, performance and GPU traces remain pending. M4 remains open.

```sh
.venv/bin/python tools/experiments/rust_metal_qwen_rms.py --output benchmarks/raw/qwen-metal-rms.json --build-dir artifacts/qwen-metal-rms --input-dir artifacts/qwen-metal-rms-inputs
```

# Rust MPS projection boundary, 12 October 2026

The standalone Rust boundary passes the unchanged tiny projection gates and
exact manual bias/stride/cancellation probes. Astra accepted the completed code and evidence after independent byte-identical
replay and a precision wording correction. This connects the earlier [Swift matrix evaluation](2026-10-10-mps-resident-projection.md)
to the Rust probe; it does not yet compose a gated MLP or inference backend.

The [host implementation](../../tools/experiments/rust_metal_projection.rs)
uses pinned `objc2-metal-performance-shaders` 0.3.2 alongside the existing
`objc2`/Metal bindings. It checks MPS support on the actual retained device.
Apple's [matrix multiplication API](https://developer.apple.com/documentation/metalperformanceshaders/mpsmatrixmultiplication)
and the [generated binding](https://docs.rs/objc2-metal-performance-shaders/0.3.2/objc2_metal_performance_shaders/struct.MPSMatrixMultiplication.html)
supply a public matrix primitive. No custom GEMM is introduced.

Activations are contiguous logical Half `[rows, in]`; immutable row-major
weights are Half `[out, in]`, passed as the transposed right matrix. Matrix
row-byte queries determine physical strides. CPU upload copies each logical
row into an exactly sized guarded buffer, without assuming a Vec's alignment.
Native allocation checks size, alignment, shared storage and tracked hazards.
Bias has a separate immutable Half allocation. A GPU initializer writes bias
into a FP32 result matrix; MPS uses alpha 1 / beta 1; a final shader stores Half
from that result. Three dependent encodings share one command, with one checked
completion wait. Matrix descriptors, kernel, pipelines and buffers survive that
wait and are reused across three warmups and ten samples.

The [shader](../../tools/experiments/rust_metal_projection.metal) explicitly
bounds logical count before mapping row strides. A common queried width respects
both shader pipelines and the device. FP32/Half output storage covers every
mapped inactive lane plus seven sentinels; allocation uses the ceiling of
rounded launch count divided by output width. The host checks the entire Half
padding/tail, all immutable input storage and the FP32 tail after the logical
matrix payload. MPS may use FP32 row padding, which is not consumed as data.
Any completed nonfinite FP32/Half output is discarded with structured failure
and exit 1. Host access is excluded while GPU work is in flight.

The parser bounds rows to 1–32 and input/output widths to 1–64, at most sixteen
cases, a 32,768-value aggregate input/weight/bias/output inventory and one-MiB
JSON input. It checks finite unsigned Half payloads and exact counts, duplicate
identities, shapes and budget arithmetic before native tensor allocation.
These are prototype bounds. Native shape qualification here covers ten query
projections with in/out 32 and rows 1, 3, 4 or 32, plus the manual shapes below;
other admitted shapes are not thereby numerically qualified.

The [qualification driver](../../tools/experiments/rust_metal_projection.py)
reads both layers of all five frozen tiny cases. Saved query outputs have layout
`[rows, 4, 8]`; their contiguous head/feature dimension is viewed as `[rows, 32]`.
Original input, weights, bias, FP16 oracle and independent FP32 source hashes
remain pinned. Each completed Half output must be the single RNE conversion of
its recorded FP32 result and pass the unchanged source gate. Manual dyadic
references are independently reconstructed before native work; saved replay
also checks their exact FP32 results.

| Manual shape `[rows,in,out]` | Exact FP32 result and Half value |
| --- | --- |
| `[1,2,1]`, bias rounding | `1.0009765625`, Half bits `0x3c01` |
| `[2,3,2]`, irregular strides | `[-3.75,14.5,-2.75,5]` |
| `[1,2,1]`, finite cancellation | `65504`, Half bits `0x7bff` |

The bias probe adds `1/2048` after a dot of `1 + 1/2048`, before final Half
storage; prematurely storing the dot as Half loses the required result. The
cancellation probe contains an individual product above finite Half range but
a finite final sum. A separate max-Half-times-two case must fail only after
completed output validation. The recorded width is 32. Irregular `[2,3,2]`
uses input/weight strides 8, FP32 stride 4, Half output stride 8 and sixteen
physical output rows, exercising distinct strides and inactive mappings.

The [final native record](../../benchmarks/results/2026-10-12-rust-mps-projection-v3.json)
completes 169 projection commands and 36,686 output values. Maximum original
FP32 source gate fraction is **0.498143005408**, maximum absolute source error
**0.002189680934**. Manual results and final FP32-to-Half conversion are exact.
36,660 values match frozen Half bits. The other 26 observations repeat one
position in each of two cases across thirteen repetitions: index 31 in layer 0
of `prompt` and `future_changed`, sharing the same activation/projection.
GPU Half bits `0x8506` differ from reference `0x8507` by one Half ULP (`2^-24` at these magnitudes).
Independent FP64 dot plus bias is `-7.670558989048004e-05`; the GPU FP32 result
is `-7.665157318115234e-05`, difference `5.4016709327697754e-08`. Feature-order
FP32 reference arithmetic gives `-7.67093151807785e-05`. This investigated
reduction difference remains within the original gate; no tolerance is changed.
FP32 result storage and observed gates are qualified. MPS internal accumulation
precision/order and bit-identical projections remain unqualified.

The same binary repeats add, original tiny gating, inactive tails and completed
gating overflow. There are 468 positive commands and four required negative
jobs in total: invalid shader, insufficient generic budget, completed projection
overflow and completed gating overflow. Largest projection conservative admission
is **67,133,842 bytes**: shared allocations 10,388, host baselines/readbacks
14,590 and the existing 64-MiB runtime allowance. Largest own-child physical
charge across all jobs is **13,075,008 bytes**. These are isolated primitive
measurements, not full-model memory, energy or end-to-end latency evidence.

The run uses Apple M1 Pro / MacBookPro18,3 / 16 GiB / macOS 26.6.2 (25G83),
Rust 1.90.0 / SDK 26.5 / macOS 15 deployment target, locked offline arm64 release
builds. Both shader pipelines use language 2.3, safe math and precise functions.
Start/end AC/policy observations retain low-power mode off and no recorded
thermal/performance warning; continuous temperature/clocks, energy and GPU
traces are unavailable. Binary SHA-256 is
`9291779eda5338a081bfe3d9f5b00727c8fe2bd6e0c218df3c916d9d700ab583`.

The [initial build](../../benchmarks/results/2026-10-12-rust-mps-projection-v1.json)
failed Clippy before GPU work on the deprecated stride query and a range lint;
its stderr and copied source are retained. The
[second run](../../benchmarks/results/2026-10-12-rust-mps-projection-v2.json)
passed correctness, then was superseded by the final device-support query and
parser tests. Earlier attempts supply no final qualification. Exact records,
stderr and copied changed sources are bound by the
[archive producer](../../benchmarks/results/2026-10-12-rust-mps-projection-archive-v1.py.txt)
and [post-completion idle proof](../../benchmarks/results/2026-10-12-rust-mps-projection-idle-v1.json),
which retain actual tool completion, absence of prior numerical chains, stopped
VM and unchanged protected model/executable endpoints. No checkpoint reads are
needed for these projection fixtures; post-completion endpoint verification
hashes the protected local assets outside native work.

The [final analysis](../../benchmarks/results/2026-10-12-rust-mps-projection-analysis-v2.json)
and [exact producer](../../benchmarks/results/2026-10-12-rust-mps-projection-analysis-v2.py.txt)
independently replay raw active FP32/Half outputs, manual arithmetic, unchanged
source gates, the two investigated differences, layouts and memory calculations.
A second replay is byte-identical. Original/copied sources, manifests, inputs,
drivers and binary hashes are verified. Large local JSONL under
`benchmarks/raw/` is hash-bound by the record; complete padding/input readbacks
are not archived, so their guard/immutability evidence is the reviewed native
check and retained Boolean transcript rather than independent raw-byte replay.

Each final isolated check passes formatting, Clippy with warnings denied,
22 Rust tests and release build. Root formatting/Clippy and 76 ordinary Rust
tests pass in an isolated target. Seventy-five experiment Python tests and 193
benchmark Python tests pass; four targeted projection tests pass again after
the final support-query policy update. Engine numerical code and frozen release
executables are unchanged. Wider projections, MLP composition, attention/KV,
full decoder/quality, scheduling traces, performance and M4 completion remain
open.

# Tiny cached-prefix Metal GQA, 12 October 2026

Irregular cached chunks and last-token decode match same-binary whole-prefix
context exactly. Replacing unused K/V capacity with maximum-finite values
leaves complete scores/probabilities/context frames unchanged. All unchanged
parent source gates pass. Astra accepted completed code and evidence without findings after byte-identical independent replay.

The [attention host](../../tools/experiments/rust_metal_attention.rs) and
[shader](../../tools/experiments/rust_metal_attention.metal) now distinguish
query rows, query cache offset and valid cache length. Checked
`offset+rows=valid<=32` defines the admitted new chunk. Query `[rows,4,8]`
uses local token indexing; K/V remain separate head-major `[2,32,8]` buffers.
A query's causal limit is absolute cache position `offset+local_token`.
Diagnostic scores/probabilities are `[4,rows,valid]`; columns at or beyond
valid cache length are never read. Masked future columns inside the new chunk
are exactly positive zero. Stable ordered FP32 reductions and final Half
context stores otherwise retain the previous primitive's behavior.

Copied shape arguments are three UInt32 values (12-byte size/4-byte host
alignment). Rounded query launch, per-query context eight-tuples and
per-query valid-length FP32 rows determine guarded storage before dispatch.
All six distinct tracked resource owners remain live through checked
completion; CPU access stays outside GPU execution. Invalid offsets/lengths,
shapes, nonfinite Half storage and allocation arithmetic fail before native
work. The public prototype still fixes query heads4/KV heads2/dimension8/
capacity32 and admits rows1–32. It does not qualify larger model shapes.

The [driver](../../tools/experiments/rust_metal_cached_attention.py) slices
both layers of the independent capacity32 fixture into `[0,1)`, `[1,4)`,
`[4,11)`, `[11,24)`, `[24,32)`, plus decode `[31,32)`. Q uses the exact saved
query slice; scores/probabilities use the matching parent query rows and
columns `[0,valid)`, context uses corresponding saved token rows. Every
original parent numerical gate is retained without recalibration. These are
read-only saved prefixes, not cache writes propagated from a running session.

Each of twelve layer/segment probes has a companion with unused K/V capacity
filled by maximum positive/negative finite Half values. Eight pairs have
nonempty unused capacity; four pairs ending at capacity32 are identical-input
controls. All twelve complete first frames, including guards, match their
companions exactly. Assembling each five-chunk partition yields exact raw
same-binary full-prefix context; last-token decode equals its final32 values.
These comparisons use native results, not saved-reference equality alone.

| First cached frame | Values | Equal frozen bits | Maximum original parent gate fraction | Maximum source absolute error |
| --- | ---: | ---: | ---: | ---: |
| FP32 scores | 11,040 | 11,040 | 0.481112446897 | 0.002561926842 |
| FP32 probabilities | 11,040 | 8,720 | 0.472539753035 | 0.000396460295 |
| Half context | 4,224 | 4,222 | 0.482209842916 | 0.001033604145 |

The [native record](../../benchmarks/results/2026-10-12-rust-metal-cached-attention-v1.json)
retains **312 cached attention commands and54,912 final context values**.
First complete frames include14,136 guard values independently replayed;
the host requires all twelve later complete frames to equal the first. Later
nonfinal frames and complete input readbacks are not archived. All final
contexts are retained and independently compared.

Every score matches independent ordered FP32 reconstruction and each first
context matches ordered FP32 reduction using recorded probabilities followed
by Half RNE. The only Half mismatch is the previously investigated capacity/
layer0/token18/head3/feature5: local index253 of `[11,24)`, GPU `0x1cab` versus
frozen `0x1caa`. It appears once in each stale-pair member and preserves the
same ordered FP32 value0.004556663334369659 and Half midpoint explanation
from the [full-prefix task](2026-10-12-rust-metal-attention.md). Small FP32
probability differences retain that task's scope; no tolerance changes or
transcendental bit-identity claims.

Same-binary full-prefix, cache-layout, rotary, composed MLP and required
failure checks total **1,066 positive commands and five negative jobs**.
Invalid cached offset is rejected before device work. Fourteen build checks
include formatting, Clippy with warnings denied,31 standalone Rust tests and
locked offline arm64 release build. Binary SHA-256 is
`eafba342a35d0d9581093f3f5cd41ba6c4da4cda8aa6c303d7324550b8c77310`.

Maximum cached-shape admission is **67,158,352 bytes**; the full-prefix
regression retains its67,225,936-byte maximum. Shared physical allocations
are counted once, with conservative host baselines/readbacks and first/current
frames separately plus64MiB runtime/fixture/JSON allowance. Peak own-child
physical charge across all jobs is **22,676,080 bytes**. Persistent resources
are reused, diagnostic output guard fills occur outside command timing and
no inference latency, allocation-free decode or full-model memory claim is made.

Hardware remains M1 Pro / MacBookPro18,3 /16GiB / macOS26.6.2(25G83), Rust1.90.0,
SDK26.5, minimum macOS15, shader language2.3 and safe math/precise functions.
Start/end observations retain AC power, configured AC low-power mode off and
no thermal/performance warning. Continuous temperature/clocks, energy, total
system/device memory and GPU traces remain unavailable.

The [idle proof](../../benchmarks/results/2026-10-12-rust-metal-cached-attention-idle-v1.json)
and [archive producer](../../benchmarks/results/2026-10-12-rust-metal-cached-attention-archive-v1.py.txt)
retain natural completion, prior numerical-chain absence, stopped VM and
protected model/executable endpoints unchanged. All native children are reaped
before offline checks/hashing. The [analysis](../../benchmarks/results/2026-10-12-rust-metal-cached-attention-analysis-v1.json)
and [exact producer](../../benchmarks/results/2026-10-12-rust-metal-cached-attention-analysis-v1.py.txt)
replay all validators, original sliced gates, causal masks, guarded layouts/
admission, every finalcontext, independent ordered scores/context stores,
stale-pair/full-prefix equality and source/copied-source/input/driver/
manifests/binary identities. Two new targeted tests, five attention tests and
full suites pass:96 experiment Python,193 benchmark Python, root formatting/
Clippy and76 ordinary Rust tests in an isolated debug target. Real-model
suites were not rerun; root release binaries remain unchanged.

Retain bounded cached-prefix GQA. Complete GPU-layer/decoder propagation,
integrated session commitment/cancellation, Qwen/backend quality, scheduling
and M4 completion remain open.

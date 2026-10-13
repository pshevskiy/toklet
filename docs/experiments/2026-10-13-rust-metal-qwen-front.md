# Qwen layer0 input normalization and Q/K/V, 13 October 2026

The resident GPU path from saved embedding Half values through layer0 input
normalization and all three Q/K/V projections passes every unchanged source
FP32 gate. Normalization runs once, and Q/K/V consume its actual stored Half
output in one command with one completion wait. Four short Qwen prefixes pass
three warmups and ten samples each. Qwen rotary, KV, attention and the complete
layer remain pending; this task does not close M4.

The [native host](../../tools/experiments/rust_metal_qwen_front.rs) loads an
exact bounded profile: rows1–4, hidden896, Q896/K128/V128 and epsilon1e-6.
Input is `[tokens,896]` saved embedding; norm weights are `[896]`; row-major
projection weights are `[out,896]` and biases `[out]`. There is no saved
normalization activation in the input schema. Projection operands share one
MPS left matrix over the GPU normalization buffer. The unchanged RMS shader
uses a serial FP32 square-sum, division, precise square root, inverse, ordered
products and final Half store. Existing MPS encoding uses Half operands,
FP32 result storage, resident bias initialization with alpha1/beta1, right
transpose and one Half store. MPS internal accumulation/order remains
unqualified. No host read, copy or wait separates normalization and projections.

The loader bounds JSON to one MiB, requires exact finite Half arrays, validates
identity/rows/epsilon before external weight reads and reuses the
[qualified binary weight reader](2026-10-13-rust-metal-qwen-projections.md).
The existing converter gains an optional layer0 norm tensor, preserving its
six-tensor default. Seven pinned BF16 tensors convert once to Half with exact
shape/dtype/range checks and one RNE cast. Full checkpoint, converted tensor,
case, driver and copied build-source identities bind the run; independent
reconstruction verifies all1,034,240 converted words. CPU model expansion is
not introduced. The [driver](../../tools/experiments/rust_metal_qwen_front.py)
checks all four original per-case gates before compacting evidence.

Eight immutable buffers own embedding, norm, Q/K/V weights and biases. Four
Half scratch buffers own normalized activations and projections; three FP32
buffers own MPS results. Layout planning checks allocation arithmetic before
buffer creation. Queried Half input/normalized strides must be contiguous896.
Normalized storage covers rounded RMS row dispatch; projection storage covers
rounded scalar grids. Pipeline widths respect all queried execution/device
limits. Owners, matrices, kernels and pipelines stay live until checked GPU
completion. CPU buffer access happens before submission or after completion.
No allocations of numerical GPU buffers occur between repetitions.

| First-frame stage | Values | Equal frozen Half bits | Maximum original source gate fraction |
| --- | ---: | ---: | ---: |
| Input normalization | 10,752 | 10,731 | 0.388420865145 |
| Q projection | 10,752 | 10,596 | 0.424460127745 |
| K projection | 1,536 | 1,512 | 0.446592680019 |
| V projection | 1,536 | 1,406 | 0.379296200303 |

All10,752 GPU normalization values exactly match an independent ordered FP32
RMS reconstruction. The21 normalization differences from frozen Half are
one normal Half ULP, consistent with serial reduction versus the frozen NumPy
mean reduction. Propagation changes309 Q/K/V Half values compared with the
same binary's saved-normalization projection path. The changed values are
retained rather than replaced with reference activations. Every stored
projection Half value is the exact rounding of its native FP32 result.
The FP64 dot of actual GPU normalized Half values and converted Half
weights/biases differs from native FP32 by at most4.544900e-6 for Q,
6.990132e-6 for K and6.818300e-8 for V. These are diagnostics of actual operands,
not new numerical acceptance gates or an internal MPS precision claim.
All original gates and tolerances remain unchanged.

First four Half and three FP32 arrays are retained. Each of twelve later
complete frames is checked bit identically in native code, while all thirteen
Q/K/V Half output sets are retained in raw evidence:179,712 repeated output
values across52 front commands. Immutable inputs, Half padding/inactive
storage and tails/guards are checked after each completion. MPS FP32 row
padding within logical rows is allowed, while storage after those rows is
guarded. Full raw guard bytes and later normalization/FP32 frames are not
archived. Changing the fourth token preserves the first three tokens of all
four stages exactly.

Same-binary regressions cover twelve saved-normalization Qwen projections,
thirteen tiny/manual projections and nine complete tiny decoders. Their
active arrays match the preceding qualification bit for bit. Together with
39 add/tail checks, the final run has533 successful commands across27 positive
jobs. Four negative jobs cover invalid shader, insufficient budget, ordinary
projection overflow and propagated normalization overflow. The latter uses
finite inputs/norm weights that overflow stored Half normalization; the command
finishes, then readback rejects and discards the complete result. No GPU session
state, cancellation, rollback or pending-token semantics are claimed.

Maximum front admission is73,591,540bytes and largest measured own-child
physical charge is18,383,496bytes, below6GiB. Admission counts each shared
allocation once, plus two shared-byte allowances for first snapshots/current
readback and the established64MiB runtime allowance. The fixed allowance also
covers bounded JSON/Case values and temporary weight conversion/read buffers;
loading bounded fixtures precedes numerical buffer admission. Preparation and
system-wide memory are not inferred from native child peaks. Runtime objects
are reused; host diagnostic allocations are outside an allocation-free
steady-state claim. Timings are diagnostic, with no latency acceptance.

Both native attempts pass18 build checks, including formatting, strict Clippy,
41 standalone Rust tests and release compilation. The first run passes its
numerical gates, but Astra finds that a CLI `__main__` driver was omitted from
its own source-hash inventory. Final v2 explicitly binds and rechecks
`Path(__file__)`, with a regression exercising that CLI-name case. Its first
frames match v1 exactly. The superseded v1 record and exact original driver
are retained; final acceptance uses corrected v2 provenance. Offline checks
pass113 experiment Python tests,193 benchmark Python tests and76 ordinary
engine Rust tests, plus root formatting/Clippy. Root checks use an isolated
debug target; protected release binaries remain unchanged. Numerical work
runs with prior workers absent and VM stopped, without concurrent test/build/
model activity. The final binary SHA256 is
`9c1c991070b002c48b2f9b979f7ac5b913b876a2086273c870a879f35ecd740a`.

Evidence is [v2](../../benchmarks/results/2026-10-13-rust-metal-qwen-front-v2.json)
and [stderr](../../benchmarks/results/2026-10-13-rust-metal-qwen-front-v2.json.stderr.txt),
with [superseded v1](../../benchmarks/results/2026-10-13-rust-metal-qwen-front-v1.json)
and [original driver](../../benchmarks/results/2026-10-13-rust-metal-qwen-front-superseded-driver-v1.py.txt).
The [archive producer](../../benchmarks/results/2026-10-13-rust-metal-qwen-front-archive-v1.py.txt)
and [idle proof](../../benchmarks/results/2026-10-13-rust-metal-qwen-front-idle-v1.json)
retain natural completion, stopped VM and protected CPU/checkpoint/tokenizer
identities. The [analysis producer](../../benchmarks/results/2026-10-13-rust-metal-qwen-front-analysis-v1.py.txt)
and [result](../../benchmarks/results/2026-10-13-rust-metal-qwen-front-analysis-v1.json)
independently reconstruct ordered normalization and actual projection dots,
verify conversion and provenance, and replay all31 raw-to-compact events byte
identically. Raw JSONL and converted inputs remain locally at the recorded
`benchmarks/raw` and `artifacts` paths.

Astra accepted the complete task with no remaining findings after reviewing
code and report, passing four targeted tests, and independently replaying the
archived analysis byte identically.

Next is a bounded Qwen split-half rotary qualification for14 query heads and
two KV heads at head dimension64, followed by composition with this graph.
Qwen cache/GQA/O, MLP/residual/full-layer/full-decoder and engine session
integration, quality/performance/trace/mobile qualification and M4 closure
remain pending.

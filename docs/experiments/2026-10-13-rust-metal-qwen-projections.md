# Selected Qwen Q/K/V projections, 13 October 2026

All twelve layer0 Qwen Q/K/V projections pass their unchanged pre-GPU source
FP32 gates. The qualified mechanism is a resident Half projection with GPU
bias initialization, MPS FP32 result storage and one Half store. Inputs are
saved normalized Half activations; normalization, rotary, attention and a
complete Qwen layer are not composed by this task. This establishes the
larger projection shapes needed for the next forward-pass slice.

The [native loader](../../tools/experiments/rust_metal_projection.rs) adds a
separate `toklet-metal-qwen-projection-input-v1` profile: exactly one case,
rows1–4, input width896 and output width896 or128. The original tiny format,
shape limits and32,768-word inventory are unchanged. Qwen inventory is bounded
by819,200 words including logical inputs, weights, bias and output. JSON stays
bounded to one MiB; weights use an exact-length little-endian Half file,
maximum1,605,632bytes. Its filename is a bounded ASCII basename, with no slash,
`.` or `..`. The loader rejects nonregular/symlink entries, binds the path and
opened descriptor's device/inode/length/modification time before and after a
bounded read, rejects trailing/truncated data and requires every Half word
finite. This detects ordinary file mutation; the driver additionally binds
complete byte hashes before and after execution.
The [adapter](../../tools/experiments/rust_metal_qwen_projections.py) reads only
six layer0 tensors from the pinned988,097,824-byte BF16 checkpoint: Q/K/V
weights and biases. It streams the complete checkpoint hash before/after
conversion and after the last native job completes. Header size, unique keys,
exact shapes/dtype/ranges/alignment and nonoverlap are validated. BF16 words
expand exactly to FP32, followed by one IEEE Half round-to-nearest-even cast.
Temporary conversion buffers are bounded to one tensor; the complete model
is never expanded. Three raw converted weight files are reused across cases;
bias arrays and activations remain small JSON values. Full checkpoint,
converted tensor, driver, case and copied build-source identities are retained.
Independent post-idle reconstruction verifies all1,033,344 weight/bias words.

Four cases are future_changed4, offset3, short4 and single1. Q is896→896
(`[tokens,896]`, equivalently14 heads×64); K/V are896→128 (two heads×64).
All matrices are row-major, with queried MPS strides and explicit right
transpose. No head expansion or transpose copy is introduced. Offset-case
activations are saved reference values; no rotary offset is consumed by this
projection primitive. Immutable weights/bias/input load once per child and
remain resident across three warmups and ten samples. Existing tracked
bias→MPS→Half-store encoders use one command/completion wait. FP32 output
storage is qualified; MPS internal accumulation precision/order is not.

| Projection | First-frame values | Equal frozen Half bits | Maximum original source gate fraction |
| --- | ---: | ---: | ---: |
| Q | 10,752 | 10,751 | 0.424460127745 |
| K | 1,536 | 1,536 | 0.446592680019 |
| V | 1,536 | 1,533 | 0.379296200303 |

Four differences are one Half ULP at normal values near rounding boundaries.
For Q, short-case index3438 is0x2a0d versus frozen0x2a0c; GPU FP32 is
0.0472570657730, the FP64 dot of actual Half operands is0.0472566322132,
and the Half midpoint is0.0472564697266. For V, index246 in both short and
future_changed is0x10da versus0x10db: GPU FP32 lands exactly on midpoint
0.000592470169067, while the FP64 dot is0.000592470459196. V short index391
is0x272e versus0x272d; GPU FP32 is0.0280380472541 and midpoint0.0280380249023.
The maximum native FP32 difference from an FP64 dot is5.411961e-6 for Q,
6.990132e-6 for K and6.818300e-8 for V. These are explanatory diagnostics;
acceptance uses only the original source gates, with no tolerance change.

All stored Half values are exact casts of their retained native FP32 results.
Every Qwen FP32 and Half frame repeats bit identically; changing the fourth
prompt token preserves the first three Q/K/V tokens exactly. Native checks
cover immutable inputs, Half padding/inactive storage and all tails/guards.
MPS FP32 padding within logical rows is allowed; FP32 storage beyond logical
rows is guarded. Raw active arrays are retained for every repetition; raw full
padding/guard bytes are checked natively but not archived.

There are156 Qwen commands and179,712 repeated active Half values. Same-binary
regressions add169 tiny projection commands,117 complete tiny decoder commands
and39 add/tail commands:481 successful commands across23 positive jobs.
Three expected negative jobs retain shader failure, insufficient budget and
completed finite-input projection overflow/discard. All nine full tiny decoder
first frames match the preceding qualified binary bit for bit, including all
layers, embedding, final normalization and logits.

Maximum Qwen admission is71,995,858bytes. The largest own-child physical charge
across all native jobs is17,875,520bytes, below6GiB. Admission counts resident
buffers, immutable host baselines and diagnostic readbacks, with the existing
64MiB allowance covering bounded JSON/Case values, temporary weight reads and
runtime objects. It occurs before numerical buffer allocation; loading bounded
fixture data precedes it. Preparation-process/system memory is not inferred
from native child peaks. Timings are diagnostic, with no latency acceptance.

The first native attempt passes all17 build checks, including formatting,
Clippy with warnings denied,40 standalone Rust tests and release compilation.
Offline checks pass109 experiment Python tests,193 benchmark Python tests and
76 ordinary engine Rust tests, plus root formatting/Clippy. Root checks use
an isolated debug target; protected release binaries remain unchanged. GPU
qualification occurs with numerical workers absent and the VM stopped, with
no concurrent tests/builds/model work. The final native binary SHA256 is
`0d4668f44eae87fdf9f614a1790b1ab34f0936a9161b2f13097ab443e54b07f8`.

Evidence is the [native record](../../benchmarks/results/2026-10-13-rust-metal-qwen-projections-v1.json)
and [compiler transcript](../../benchmarks/results/2026-10-13-rust-metal-qwen-projections-v1.json.stderr.txt),
with [archive producer](../../benchmarks/results/2026-10-13-rust-metal-qwen-projections-archive-v1.py.txt)
and [idle/protected identity proof](../../benchmarks/results/2026-10-13-rust-metal-qwen-projections-idle-v1.json).
The [analysis producer](../../benchmarks/results/2026-10-13-rust-metal-qwen-projections-analysis-v1.py.txt)
and [final analysis](../../benchmarks/results/2026-10-13-rust-metal-qwen-projections-analysis-v2.json)
replay all raw-to-compact events byte identically, reconstruct the converted
tensors and diagnose the four rounding differences. Analysis v2 adds rounding
diagnostics to the local v1 summary; native evidence is a single successful
attempt. Raw JSONL and converted tensor/case files remain locally in the
recorded `benchmarks/raw` and `artifacts` directories.

Astra accepted the complete task with no remaining findings after code and
report review, three targeted tests and independent byte-identical replay of
the archived final analysis.

Qwen propagated normalization, rotary/cache/GQA/O, MLP/residual/full-layer and
decoder/session integration, quality/performance/trace qualification and M4
closure remain pending. The next bounded step is composing layer0 input
normalization with these Q/K/V projections under the same frozen gates.

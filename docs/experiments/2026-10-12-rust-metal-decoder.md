# Full tiny Metal decoder, 12 October 2026

The complete tiny numerical decoder passes its frozen gates. GPU embedding,
both layers, final RMS normalization and tied FP32 logits execute in one
command with one completion wait. All five saved prefixes and four fixed
greedy histories pass thirteen repetitions each. The four greedy IDs are
`[47, 47, 47, 47]`, matching both independent references. This is a standalone
qualification program; the engine backend and GPU session API remain pending.

The [host](../../tools/experiments/rust_metal_decoder.rs) loads one bounded
case with resident Half weights. The [layer refactor](../../tools/experiments/rust_metal_layer.rs)
separates checked layout planning, resident resources, reset, encoding and
readback. The standalone layer retains its original output schema and all ten
regression cases. Its encoder accepts a borrowed incoming GPU buffer: layer0
reads the embedding result, and layer1 reads layer0's final residual directly.
The decoder input schema forbids saved layer activations. The layer parser's
zero placeholder is only a small host validation value and is never uploaded
or used in encoding. No intermediate host read or wait separates operations.

Final RMSNorm uses the unchanged serial FP32 arithmetic, epsilon1e-6 and Half
store. Output projection uses MPS Half operands, row-major `[64,32]` weights
transposed, FP32 `[tokens,64]` result storage, alpha1/beta1 and resident zero
initialization. The right matrix shares the identical embedding buffer; there
is no second output table. Logits have no Half store. CPU greedy selects the
lowest token ID on ties and rejects nonfinite logits. MPS internal arithmetic
and accumulation order remain unqualified.

The tiny topology is hidden32/intermediate64/two layers/four query heads/two KV
heads/head dimension8/vocabulary64. Each command rebuilds the full prefix with
zeroed head-major capacity32 caches. Absolute rotary offsets0 and5 are
qualified independently of local cache placement0. Rows1–32 are accepted;
the checked parser limit `position+rows<=2048` is not a numerical qualification
of every accepted offset. All shader layouts, rounded launch tails, cache
strides, causal masks, grouped-head mapping, split-half rotary conventions and
stored Half boundaries retain the preceding layer qualification.

The [driver](../../tools/experiments/rust_metal_decoder.py) uses five immutable
prefixes: capacity32, changed future4, offset3 at5, prompt4 and single1. Each
layer's original eighteen source FP32 gates are applied to actual propagated
GPU values. Cache anchors remain layout diagnostics; acceptance checks exact
copies of actual GPU rotated K/projected V, including zero invalid capacity.
Projection casts, residual sums and gated products consume actual stored
operands. First-frame arrays are retained in raw evidence. All twelve later
complete frames match exactly in the native host; only last-row logits and
small completion records are retained for each repetition. Raw full guard
bytes and later nonfinal arrays are not archived. All immutable buffers, Half padding/inactive storage and buffer tails/guards
are checked natively after every successful completion. MPS FP32 row padding
within logical rows may be written by MPS and is deliberately allowed by
the readback helper; FP32 storage after those rows remains guarded.

Four additional histories have lengths4,5,6,7. The reference freezes only
last-row greedy logits, so numerical gates are applied only there. Other
recorded arrays still receive complete shape/type/finite checks, native
repeat/guard checks and independent actual-operand replay. Histories are
source FP32 teacher-forced. Because every target-selected ID equals its source
ID, these histories also equal the generated target history for all four
qualified steps. Stop is fixed four steps with no EOS semantics. This does
not qualify incremental KV sessions, pending sampled state, cancellation or
prefix reuse.

| Five-prefix outer stage | First-frame values | Equal frozen bits | Maximum original source gate fraction |
| --- | ---: | ---: | ---: |
| Embedding | 1,408 | 1,408 | 0.442093905507 |
| Final normalization | 1,408 | 1,408 | 0.498069062402 |
| FP32 logits | 2,816 | 2,466 | 0.497773477259 |

Capacity, offset and single logits match all frozen bits. Prompt and changed
future have 176 and174 FP32 differences respectively, while preserving their
original source gates. Last-row greedy step0 matches18/64 frozen bits; steps1–3
match64/64 each. Greedy gate fractions range0.474571–0.490146. The independent
FP64 dot of actual Half normalization/table operands differs from native
logits by at most7.67409801483e-7; this is a diagnostic, not a replacement gate.
There are eight known Half attention differences across the five prefixes,
identical to the [layer record](2026-10-12-rust-metal-layer.md), absorbed at the
first residual. All downstream layer Half outputs and final norms match the
frozen bits. Gates and tolerances are unchanged.

Independent post-idle replay verifies10,560 normalization values exactly,
9,536 ordered FP32 attention scores exactly, all ordered context Half stores,
actual cache copies, projection casts, residual sums and gated products.
Changing the fourth prompt token preserves the first three tokens through
both layer outputs, final normalization and logits bit for bit. There are **7,488 retained repeated last-row logits values** across117 completed
decoder commands. Ten standalone layer regressions add130 commands and the
base add/tail regression adds39, totaling286 successful commands. Five
expected negative jobs cover invalid shader, insufficient budget, invalid
vocabulary index and two completed finite-input normalization overflows.
Overflow results are discarded after successful GPU completion with the
existing structured layer-validation error.

Combined admission precedes every numerical buffer allocation. It counts
both resident layer graphs, one tied table, final norm, zero bias, token buffer,
embedding/finalnorm scratch and FP32 logits, including guards. Persistent
physical buffers are counted once; two shared-byte allowances cover first
snapshots and current readback, plus the established64MiB runtime allowance.
Maximum admission is67,909,198bytes; largest measured native child's physical
charge is14,402,184bytes, below6GiB. JSON/parser/pipeline/matrix runtime costs
are covered by the fixed allowance, not claimed as individually measured.
Own-child peak does not measure preparation or system-wide memory. Pipelines,
matrices, MPS kernels and initialized owners survive the checked completion;
CPU buffer access happens before submission or after completion. Four bounded
GPU residual blits preserve diagnostic snapshots. No performance acceptance
is inferred from these diagnostic encode/commit/wait timings.

The isolated final build passes17 checks, including formatting, Clippy with
warnings denied,38 Rust tests and optimized release compilation. Offline
checks pass106 experiment Python tests,193 benchmark Python tests and76
ordinary engine Rust tests, plus root formatting/Clippy. Engine tests use the
existing isolated debug target; root release binaries are preserved. Initial
benchmark test invocations used the wrong directory and then an environment
without jsonschema/owned-process permissions; the configured `.venv` and
required process access pass the complete suite. No code change was made for
those environment failures.

Two fresh native attempts fail before GPU work: v1 a JSON-macro parse error,
v2 strict Clippy unused-variable/type-complexity errors. Final v3 fixes these
and Astra's two review findings: malformed layer entries now return a fixture
error rather than panic, and all greedy arrays receive complete validation
before compaction. Native compiler/test transcripts, both failed records and
copied Rust sources are retained. The failed Python driver bytes were not
archived; their recorded hashes are provenance only. No native evidence comes
from either failed attempt. The final binary SHA256 is
`3c9de9c6b4875519eacf9ef1266a1f0696c3df4063db086736255d409398372f`.

Final evidence is [v3](../../benchmarks/results/2026-10-12-rust-metal-decoder-v3.json)
and [stderr](../../benchmarks/results/2026-10-12-rust-metal-decoder-v3.json.stderr.txt),
with retained [v1](../../benchmarks/results/2026-10-12-rust-metal-decoder-v1.json)
and [v2](../../benchmarks/results/2026-10-12-rust-metal-decoder-v2.json)
build failures. The [archive producer](../../benchmarks/results/2026-10-12-rust-metal-decoder-archive-v1.py.txt)
and [idle proof](../../benchmarks/results/2026-10-12-rust-metal-decoder-idle-v1.json)
verify natural completion, prior workers absent, VM stopped and protected CPU,
checkpoint and tokenizer identities unchanged. The [analysis producer](../../benchmarks/results/2026-10-12-rust-metal-decoder-analysis-v1.py.txt)
and [result](../../benchmarks/results/2026-10-12-rust-metal-decoder-analysis-v1.json)
replay raw-to-compact records byte identically and independently reconstruct
actual numerical boundaries. Raw JSONL is retained locally under
`benchmarks/raw/2026-10-12-rust-metal-decoder-v3.json.native.jsonl`.

Astra accepted the complete task with no remaining findings after reviewing
code, archived evidence and this report, independently replaying the analysis
byte identically and passing four targeted decoder tests.

M4 remains open: Qwen propagation, engine backend integration, resident GPU
session behavior, recovery boundaries, GPU trace, end-to-end quality and
performance acceptance, and physical iPhone qualification are still pending.

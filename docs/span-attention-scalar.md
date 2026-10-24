# Attention over borrowed ring spans

`cpu::span_attention` consumes three concrete borrowed K/V segments: up to two
old ring intervals plus the new-token chunk. Logical origins concatenate in
array order. Each nonempty segment holds separate `[kv_heads,keys,D]` data with
an explicit head stride; its backing slice covers exactly
`(kv_heads-1)*head_stride + keys*D` elements. Head gaps are initialized Rust
storage, but are neither validated numerically nor read. Empty segments require
empty K/V and ignore stride/origin metadata. Only f32 alignment is assumed.

Q and context remain `[queries,query_heads,D]`; scratch scores/probabilities are
`[query_heads,queries,keys]` in logical key order. Concrete segment selection sits
outside inner dot-product loops. There is no gathered/linearized KV buffer,
allocation, dynamic dispatch, variant match per scalar or cache mutation.
The existing readable contiguous scalar remains unchanged as a reference.

Admission checks all dimensions, lengths, stride/nonoverlap, ordered origins,
position/size arithmetic, finite used rows and every query's nonempty visible
range before writes. Future and expired positions receive exactly zero
probabilities. Their diagnostic scores still compute and can reject overflow.
Dot products/scaling use FP64 before FP32 scores; stable FP64 softmax stores
FP32 probabilities. One FP64 context accumulator visits included keys in logical
order across all segments, then rounds once to FP32. Arithmetic failure leaves
discardable output/scratch and immutable inputs for replay.

`KvView::segments` derives physical starts and full owner head strides from
validated cache geometry. New tokens have their own head stride. Its shape
helper supplies logical origins/window and cache geometry; the CPU entry point
checks query heads and scale. Borrowed lifetimes prevent cache mutation during
attention. Old ring keys needed by early chunk queries remain available until
successful commit; no consumer copies or appends shared K/V.

Eight methods authenticate the prior
[attention witness](gemma-local-attention-reference.md) and
[KV state witness](gemma-kv-reference.md). They compare frozen scores,
probabilities and original FP64 contexts under unchanged tolerance, plus bitwise
agreement with contiguous scalar execution. All thirty cache-witness updates
run through actual borrowed ring views, including Hkv2 wrap and512-window
transition; pointer checks verify each head's physical span and new input.
Actual256/512 head widths and the512-window transition remain separately bounded
cases. Tests also cover empty leading/middle/trailing spans, NaN and negative-zero
padding preservation, late invalid/overflowing spans before writes, invalid
geometry/fully masked later rows, masked-score and rounded-probability context
overflow/replay, numerical failure without cache commit, and hand GQA at the
inclusive maximum usize position. That position boundary is a scalar arithmetic
check, not model/context/rotary qualification.

Formatting, strict default/Metal Clippy,245 CPU/267 Metal-feature tests and both
iOS cross-checks pass. Hardware suites remain ignored in these general checks.
The [receipt](../benchmarks/results/2026-10-24-span-attention-scalar-v1.json)
binds exact source, fixtures, checks and unchanged other engine/shader/protected
binary hashes. Astra reviews final evidence before commit.

This is a scalar numerical/cache-view mechanism. Whole-model cancellation,
shared-layer numerical composition, Gemma graph/weights/tokenizer/stops, rotary
parity, resident Metal, quality/performance and real-model/device memory fit
remain open in the [Gemma4 plan](gemma4-implementation-plan.md). No native ABI,
engine capabilities, app limits or installed package changes.

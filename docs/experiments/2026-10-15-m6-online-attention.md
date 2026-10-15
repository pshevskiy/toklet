# M6 bounded online attention component

15 October 2026. Component qualification only. M6 remains in progress; no
longer-context configuration, real-model quality or performance improvement is
qualified by this task. The score-matrix Reference path remains the default.

The hypothesis is that keeping only a key tile and running softmax state removes
attention's depth-sized score/probability workspace and permits cooperative
contiguous feature reads. `AttentionMode::Online` selects the implementation
before session admission; existing constructors select Reference.

Each queried SIMD group owns one query head. Decode evaluates one row; prefill
evaluates two rows and reuses each K load. V traversal remains per query. Queries
and outputs are `[rows, query_heads, head_dim]`; separate resident FP16 K/V remain
`[kv_heads, capacity, head_dim]`. GQA maps contiguous query heads to their shared
KV head without expansion. Cache placement is relative to the session; rotary
positions already include the absolute offset. The causal bound is placement
plus query row plus one, independently of allocated capacity.

The kernel keeps 16 scores per row, FP32 maximum, mass and weighted numerator.
When a tile raises the maximum, it rescales both mass and numerator before
adding that tile. It stores Half once at the output. Features are striped over
queried lanes; supported dimensions are at most128 and Online checks a SIMD
width of at least32, requiring at most four feature slots per lane. There are no
barriers, full score matrices, intermediate host waits or new unsafe Rust.
The existing ordered command completion and transactional publication remain.

Online replaces the two allocated reference matrices with two eight-word guards
(64 bytes total), independent of depth. Other scratch and KV allocations remain
admitted. At capacity2048/chunk128 this saves29,360,120 bytes for the small
geometry and25,165,816 bytes for the larger geometry, calculated from the exact
allocation formula including guards. This is allocated scratch reduction, not a
measurement of process footprint or system-wide memory.

The [component evidence](../../benchmarks/results/2026-10-15-m6-online-component-v1.json)
binds source snapshots, fixture manifests and raw check logs. Explicit native
checks on the available Apple GPU pass:

- Independent FP64 softmax over exact stored Half values:202,048 outputs across
  dimensions8/64/128, both GQA geometries, irregular rows and cache offsets up to
  32,751. The primitive envelope, frozen before collection, is
  `abs(error) <= 1e-4 + 1e-3 * abs(reference)`; maximum observed absolute error is
  0.0006552625687139546. This primitive envelope does not replace model gates.
- Eighty-one uniform, changing-maximum, dominant-key, finite-extreme and active
  NaN/Inf cases;15/16/17/31/32/33 boundaries; even/odd rows; poisoned future K/V;
  earlier-query independence; direct decode/prefill agreement; guards and input
  preservation.
- All35 unique tiny diagnostic stages for every chunk in all five independent
  fixture cases, including absolute rotary offsets:173,184 values under the
  unchanged FP16-vs-source gates. Cached/edited prefixes match fresh replay.
- Cancellation, stop/output failure, pending continuation, both readbacks,
  finite failure, injected completion failure/replay, diagnostics/profile
  validity, exact admission boundaries and zero warmed Rust allocations.

Formatting, strict CPU and Metal all-target Clippy,95 CPU tests and107 ordinary
Metal-feature tests pass. Native library checks pass four tests; the final tiny
integration check passes separately. A sandbox preflight could not access Metal.
An initial integration attempt passed numerical stages then failed because its
legacy admission helper recreated Reference using the smaller Online budget.
The helper now recreates the actual mode; the replacement check passes. Finalv4 additionally asserts each actual chunk size against the independent requested size. The
original failing log is retained. No tolerance or numerical implementation was
changed to address that helper error.

Astra accepted the final code and bound evidence with no remaining findings.
Retain as an explicit component for further qualification. Full held-out
loss/tasks, paired end-to-end measurements, extended admission, independent
high-position rotary checks, long-context behavior/memory and the conditional KV
precision decision remain before M6 closure. M5's selected configurations and
CPU/Metal/readback/chunk defaults remain unchanged. M7 is pending.

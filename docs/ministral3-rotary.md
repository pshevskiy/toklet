# Ministral 3 cached CPU YaRN primitive

Status: independent fixture generation, required Rust checks and release/
allocation controls pass. Astra approves scoped retention with no findings;
[review and publication](../artifacts/ministral3-yarn-2026-11-13/publication-checks.json). This standalone Half primitive does not admit a Ministral model,
construct a graph or qualify Metal, quality, latency, overhead or adoption.

The [prospective profile](../tests/fixtures/ministral3-rotary-v1/profile.json)
has SHA256 `0f8d25bec77f2f6901e86830b7da22b75808fccb7800ce939f17f386206f4c9f`.
Its source identities and numerical gates were frozen before reference or native
outputs. The selected original/Heretic configurations use 128-dimensional
split-half rotation, theta 1,000,000, YaRN factor 16, beta-fast 32, beta-slow one,
original position limit 16,384, and source maximum 262,144. Both source mscale
parameters are one, so their ratio is exactly one. The additional query beta is
0.1. Source weights are BF16; this is explicit prospective Half child arithmetic,
with no claim of BF16 model or MLX implementation emulation.

The primitive keeps source-form period denominators. Correction bounds use
FP64 logarithms, floor/ceil and the source clamp. FP32 power results and the
expanded FP32 numerator/denominator blend precede a stored FP32 period. The
cache computes the FP32 absolute-position/period angle, stores FP32 cosine/sine,
and separately stores the Half query gain. It does not replace the source
expression with an inverse-frequency simplification.

`Rotary` owns inline periods, `[capacity,64,{cos,sin}]` FP32 coefficients and
`[capacity]` Half gains. Its absolute `base` lets an eight-row boundary probe
allocate eight rows rather than a prefix ending near 262,144. All heads and
layers can borrow the immutable table. Caller-owned `[tokens,heads,128]` Half
inputs, outputs, scratch and KV are separate owners. Checked planned bytes
include the inline object and both requested vectors; actual reserved capacities
are checked again after reserve. These are owner bytes, not measured full-model
residency or allocator bookkeeping.

Keys widen Half values, execute separate FP32 products/add/subtract and store
Half once. Queries first perform the same Half rotation, then multiply by a gain
already cast to Half, and store Half again. Separate statically specialized
query/key paths keep selection outside scalar work. Steady-state apply has no
allocation, trigonometry, dynamic dispatch or profile lookup. Span, shape,
arithmetic overflow and finite inputs validate before writes; numerical Half
store overflow may leave partial output for the caller to discard. The immutable
owner never publishes session or KV state.

The [fixture producer](../tools/reference/ministral3_rotary_fixture.py) uses only
the Python standard library. Its main profile simulates every FP32 operation;
the independent Decimal80 profile uses separate trigonometric reduction/series
and exact rational nearest-even conversion at storage boundaries. Both retain
explicit FP32 period/coefficient and Half rotation/gain stores. Ordinary tests
consume tracked data and producer hashes without model downloads or an Apple
GPU. Genuine generation checks the pinned external source files without importing
MLX or constructing a model.

Eleven cases cover positions zero, 16,383/16,384, 32,767/32,768 and 262,143;
one/two/three/eight-row spans, actual 32-query/eight-KV-head shapes, signed zeros
and subnormal inputs. The fixed input recipe and flat word layouts are in the
profile. Periods and query gains must match the main profile's exact words.
For every other exact case/array separately, native values must pass against
both profiles with `atol=max(1e-4,2*max_abs(main-wide))` and `rtol=1e-3`.
No native-derived expansion or pooling across arrays/cases is allowed. Repeated,
chunked and rebased-table applications require exact Half-byte equivalence.

The [genuine fixture](../tests/fixtures/ministral3-rotary-v1/fixture.json) contains
195,776 numeric words in 1,212,502 bytes, below the prospective 2 MiB bound.
Its SHA256 is `0e1eb4f6a9056bf1725d8265e905ec7fe8ddc3c690e8a802f58121d96f36c427`.
The [generation receipt](../artifacts/ministral3-yarn-2026-11-13/fixture-generation-1.json)
records unchanged source hashes, 88,539,136 bytes including terminal accounting
within the 512 MiB guard, and empty owned cleanup. Twenty-four portable producer
controls pass in normal and optimized Python modes. An incorrect hand-entered
expected gain and a symlinked temporary-directory test path were corrected in
tests; no source arithmetic or gate changed.

The [complete Rust check queue](../artifacts/ministral3-yarn-2026-11-13/rust-checks-1.json)
passes formatting, strict CPU/Metal Clippy, 744 CPU tests, 894 Metal-feature tests,
six targeted debug tests, six targeted release tests and the release allocation
control. CPU/Metal suites retain 52/340 explicitly ignored tests; feature-enabled
coverage is distinct from hardware qualification. Each frozen-fixture test
checks 134,592 native/reference coordinates plus 704 exact main period words,
32 exact gains and 62,464 independently constructed input words. The
[optimized Python reproduction](../artifacts/ministral3-yarn-2026-11-13/fixture-reproduction-result-1.json)
is byte-identical to the original.

The release allocation control records zero Rust allocations across 24 warmed
key calls, 24 query calls and 40 rejected calls on its own measured thread.
This is no whole-process, GPU or model allocation claim. Full compiler/test
observations stay below the 3 GiB compiler cap; the largest observed charge is
2,812,444,672 bytes including reserve. Owned cleanup is empty. Sampled observations
and terminal RSS do not establish a continuous allocation cap. Test command
clocks include building/diagnostics and are not inference-performance samples.

The existing scalar references, `PositionPlan`, source adapters, graph execution
and ordinary 512-position Metal cap are unchanged. Boundary arithmetic at a
larger absolute position grants no model/context admission. A complete Ministral
route still needs strict source binding, authenticated payload/conversion,
independent full-graph and original 10,000-target/100-task quality gates, exact
codec/stops, admitted working sets and completed request/overhead measurements.
The selected stale Q4 index remains rejected. iPhone and interactive capture
remain deferred.

# Independent KV staging and ring-state witness

This G2 witness fixes normal global and local-window cache semantics before
native cache implementation. Immutable FP32 Q is `[query_heads,new_tokens,D]`;
K/V inputs and states are `[kv_heads,rows,D]`. Each case defines absolute
committed positions and independent FP64 logical-prefix attention at scale1.
Query heads map contiguously to KV heads. The frozen
[fixture](../tests/fixtures/gemma-kv-v1/math.json) contains seven cases, thirty
updates,314 arrays and23306 values.

A local window W retains the last W committed tokens in fixed slots
`absolute_position % W`. During a chunk beginning at offset O, attention needs
the union from `max(0,O-W+1)` through the end of the new chunk. Each query q
includes only `[max(0,q-W+1),q+1)`. The final committed ring is a different view:
early queries may need old tokens already absent from that final ring. The
witness represents needed old tokens with at most two physical spans, followed
by the new immutable chunk. Global attention retains the full prefix within
its reserved capacity. Empty state, repeated single-token wrap, oversized
prefill, concatenation after wrap, window1, global and actual window512 boundary
are covered separately. Tiny head widths bound the reference memory; these are
cache semantics rather than full Gemma head-shape qualification.

The explicit adapter authenticates MLX-LM revision
`0d4d3d04f8ab7ce4a51e7209e8615755a48ba94d`, using the pinned mask/attention
functions from `base.py` and `_BaseCache`, `KVCache`, `RotatingKVCache` and mask
helper from `cache.py`. Only selected captured definitions execute. Deferring
annotations avoids importing unrelated class names; executable bodies are
unchanged. The source's physical phase can depend on prior chunking and need
not equal the proposed native fixed-slot layout. Encoded positions let the
adapter compare the exact logical key union, every K/V value and per-query
mask independently of that physical permutation.

Seven portable methods cover ring spans/union/visibility, premature-overwrite
failure, required abort/retry policy, logical chunk equivalence and grouped
heads, invalid geometry/dtype/finiteness/external identity, unique source
selection and retention of a completed raw mask if update fails. Three owned
process regressions and three cold-runtime admission methods also pass. The
first system-Python attempt failed preflight before any child. It remains
separate; the complete capture uses the repository virtual environment.

Admitted NumPy2.3.2/MLX0.32.3 CPU and GPU each retain101 raw arrays and pass all30
context comparisons. K/V and masks are exact. Maximum context error/policy
ratios are about0.00193 CPU and0.00230 GPU under unchanged
`1e-5 + 1e-4*abs(expected)`. These describe numerical error, not latency.
Completed arrays are retained before subsequent calls or validation can fail.
Independent stdlib replay checks all seeded arrays, ring states, required spans,
visible ranges, FP64 contexts and saved source observations. The
[receipt](../benchmarks/results/2026-10-24-gemma-kv-reference-v1.json) binds the
sources, runtime, captures and replay. Astra reviews exact final evidence before
commit; numerical engine sources and protected binaries remain unchanged.

Aborting a proposed native update must preserve exact prior K/V, positions and
generation, and retry must append once. This is required policy illustrated by
the witness, not execution of a native transaction or upstream rollback.
The [standalone cache owner](window-cache-scalar.md) now passes native cache
batch commit/discard checks. Shared-producer reuse, model/session integration,
ring-aware numerical
attention, graph/Metal integration, rotary parity, model quality, memory fit,
performance and phone qualification remain open in the
[Gemma4 plan](gemma4-implementation-plan.md). No capability or limit changes.

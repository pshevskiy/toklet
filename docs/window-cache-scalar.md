# Scalar global and sliding-window KV ownership

`model::window_cache` is a standalone FP32 cache owner, separate from existing
decoder/session paths. K/V have fixed initialized `[kv_heads,capacity,head_dim]`
storage. A local cache has capacity equal to its positive window W and stores
position p at slot `p % W`; global cache retains the full prefix up to capacity.
Maximum positions also bound local lifetime. Constructor arithmetic includes
both buffers and the owner object; actual reporting uses vector capacities.
Borrowed new-token staging, outer collections and numerical scratch remain
separate memory owners.

`prepare` validates token count, total positions, generation overflow, lengths,
allocation arithmetic and every finite input before returning a private frame.
New K/V are borrowed `[kv_heads,new_tokens,head_dim]` slices; committed state is
unchanged. Constructor-only atomic IDs distinguish cache owners without adding
per-token atomics. Owners cannot be cloned; moving preserves identity. Frames
carry owner, generation and starting offset, rejecting stale/foreign reuse.

A checked view exposes at most two borrowed old spans plus the immutable new
chunk per KV head, in logical order. It also supplies each query's causal/window
range. Earlier queries can therefore use old keys that would be overwritten by
prematurely committing the chunk. Neither the old cache nor the whole attention
union is copied. Safe Rust lifetimes prevent mutation while a view is used.
This task supplies storage/views; ring-aware numerical attention remains a
separate integration task.

`commit_batch` requires one matching frame for every supplied owner. It validates
all identities, generations, offsets, admission arithmetic and lengths before
any write. The remaining copies/indexing/offset advances are infallible under
those invariants. Only new rows are written; local chunks larger than W write
only their final W rows. Discarding a frame or a rejected batch preserves exact
committed K/V, positions and generation. Retrying after discard appends once;
reusing a successfully committed frame rejects. The caller must invoke commit
only after the entire numerical step succeeds. This establishes cache batch
commit/discard behavior, not a whole-model transaction.

Six methods authenticate the previously reviewed
[314-array witness](gemma-kv-reference.md), exercise all thirty updates, exact
ring state, span origins/pointers and visibility, discard/retry, last-entry batch
failure, missing/reordered/duplicate/foreign/stale frames, signed zeros, oversized
local chunks, global limits, invalid inputs/arithmetic, moved owners and exact
chunk-equivalent retained state. Pointer/capacity checks confirm cache buffers
are reused, without claiming a measured allocation or performance result.

Formatting, strict default/Metal Clippy,229 CPU/251 Metal-feature tests and both
iOS cross-checks pass. Hardware tests remain explicitly ignored in these suites.
The [receipt](../benchmarks/results/2026-10-24-window-cache-scalar-v1.json) binds
exact source, fixture, checks and unchanged prior engine/shader/protected binary
hashes. Astra reviews final source/evidence before commit.

No native ABI, capabilities, app limits or installed phone package change.
Shared producer/consumer integration, full session rollback, ring-aware attention,
Metal ownership/residency, rotary parity, Gemma graph/weights/tokenizer/quality,
real-model timing and device memory fit remain open in the
[Gemma4 plan](gemma4-implementation-plan.md).

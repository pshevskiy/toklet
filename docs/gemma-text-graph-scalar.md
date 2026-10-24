# Bounded scalar Gemma reference graph

The Rust reference now composes the complete frozen synthetic Gemma text graph.
It accepts exactly the witness's hidden8/vocabulary17/four-layer geometry and
32 total positions. This is a G3 foundation; no real Gemma checkpoint, search
eligibility, native app capability or longer context is added. The independent
[mathematical and CPU source witness](gemma-text-graph-reference.md) remains
separate from both unresolved upstream GPU gaps.

## Model, session and numerical path

The named `model::gemma_reference` module accepts concrete immutable FP32 slices.
Tensor names, loaders, tokenizers, chat formatting and UI stay outside it.
Construction validates every table, projection, direct learned norm, producer
assignment, required PLE gate and finite scalar. Layers0/1 own K/V projections;
layers2/3 have none. The single embedding table also supplies the tied output
projection. The reference accepts no quantized backend or expanded real-model
weight fallback.

Each session borrows one exact model; forward accepts tokens, with no separate
model parameter that could substitute an identically shaped model. Separate
producer caches, new-token staging and reusable numerical scratch belong to
the session. Forward uses the existing embedding/scaling/PLE, RMS, rotary,
borrowed-span GQA, dense layer finish and in-place softcap operations through
ordinary static calls. There is no numerical trait dispatch or unsafe kernel.
K/V retain the documented `[kv_heads,capacity,head_dim]` layout. Here KV heads=1,
so the contiguous projected `[tokens,head_dim]` rows need no transpose.

The two producer proposals live on the stack. Each borrows its own persistent
new K/V buffers until both shared consumers and final norm/head/softcap succeed.
All old-ring/new-chunk attention is borrowed, without linearizing committed KV.
The batch commit validates every producer before the first cache write. Only
infallible token-history copying and a public-logit buffer swap follow. Any
earlier error leaves the committed KV bytes/positions/generations, history and
previous public logits unchanged. Scratch and staging may change and are rebuilt
on retry. Two bounded logit buffers preserve the previous published chunk while
the next candidate is computed.

Reset preflights every cache generation before invalidating the complete prefix.
It preserves physical buffers and allocation ownership; old proposals become
stale. Public history and logits become empty. Sampling, pending sampled tokens,
stop/tokenizer semantics, cancellation callbacks and exact-prefix reuse remain
later session mechanisms.

## Admission and checks

Nonempty chunks must fit32 total committed positions, with IDs below17. Logical
model/session admission precedes cache, scratch and staging allocation. Actual
Vec capacities are checked again, with both logit buffers and all producer
storage counted. These are conservative numerical reservation figures, not
whole-process RSS/physical peak or device fit. Caller-owned container/application
metadata remains separate. The preliminary unit-test build reports18,856 logical
model bytes and68,408 session bytes; the test-only fault field affects that layout.
No production/mobile memory claim follows from those synthetic figures.

Five graph methods authenticate the frozen fixture/receipt and check all three
schedules against both ideal and staged contracts. At each chunk, test-only
stops compare every just-finished layer's output, Q norm/rotary, attention/context,
PLE and retained FF stages before retrying from the same committed prefix.
Producer staging and committed physical slots are checked against independent
absolute-position values. Final norm/logits, immutable weights, fixed storage
pointers and reset/replay also pass. Selected intermediate comparisons explicitly
include layer1, where the upstream source GPU witness first exceeded policy.

Five injected late failure boundaries preserve prior public state and retry
exactly; a real finite-weight scalar overflow also leaves the empty committed
prefix intact after producing new K/V. Invalid/empty/out-of-range/over-capacity
requests and malformed model boundaries reject. A separate cache-reset method
checks stale proposals, retained bytes/pointers, all-invalid logical slots and
generation-exhaustion atomicity. The test-only System allocator shim measures
zero Rust allocations across96 warmed decode steps, three resets/accounting
calls and separate budget-minus-one rejection. Fixture/setup allocations are
outside that measurement.

Formatting, strict default/Metal Clippy,272 CPU/294 Metal-feature tests and both
iOS compile targets pass. CPU tests remain offline; ignored real-model/GPU suites
are not represented as executed. The [receipt](../benchmarks/results/2026-10-25-gemma-text-graph-scalar-v1.json)
binds source/fixture/helper inventories, raw logs, actual CLI executable identities,
empty default-harness proof and protected binaries. Astra reviews exact final
source/evidence before commit.

The first preliminary helper call completed package cleanup but failed while
formatting relative log paths into its result. Its raw logs and unaccepted error
record remain retained; no format/compile/test command ran in that attempt.
Corrected absolute paths complete the preliminary checks, and the authoritative
guarded run starts with another package-only clean in the private copied target.
Neither numerical code nor tolerances changed to recover that collection error.

The native artifact adapter, actual E2B tensor/quantized representation, general
config routing, broader session mechanisms, resident Metal, held-out quality,
real loading/generation/timing/memory and physical iPhone checks remain open.
The short upstream GPU intermediate rejection and all eight long rotary failures
stay unwaived. Existing Chat256/raw128/total512 app limits and installed packages
remain outside this task; a131072-output reply still needs its separate route.

# Concrete Gemma3 CPU composition

Toklet now has a separate packed Gemma3 FP32 numerical graph and transactional
CPU window session. It passes the [frozen independent fixtures](gemma3-cpu-reference.md).
The [real checkpoint plan](gemma3-local-global.md) is still cold metadata support:
this task does not add payload loading, native/Half execution or Chat eligibility.

## Blocks, layout and ownership

The graph reuses existing packed projection/gather, RMSNorm, ordinary rotary,
span attention, tanh-GeGLU and residual operators. Packed operations move to a
small sibling module with previous Gemma4 paths reexported; their numerical
method bodies are unchanged. There is no graph interpreter, hot tensor-name
lookup or per-element virtual dispatch.

A model borrows immutable packed matrices. One table supplies embedding and
output; row-major projections keep residual width separate from query width.
Construction consumes raw BF16 gamma vectors and derives owned FP32 `1+gamma`
coefficients once. It never folds them into Half. A precise2^-12-gamma control
requires coefficient bits0x3f800800, distinguishing both BF16 and Half folding.
Embedding uses BF16-rounded sqrt(hidden),8 for the tiny graph and25.25 for640.

Per layer: input norm; Q/K/V projections; Q/K norm; layer-specific ordinary RoPE;
local/full GQA; output projection; post-attention norm plus residual; pre-FF
norm; gate/up/tanh-GeGLU/down; post-FF norm plus residual. V is not normalized.
Final norm feeds the same packed table. Scalar reductions use FP64 with FP32
operator stores, an explicit CPU profile rather than upstream BF16/Half parity.

A session owns initialized FP32 scratch, history, published logits, independent
per-layer window/full KV owners and new-row staging. `SharedKvPlan` has zero
shared layers. Local views borrow old ring spans plus the immutable proposal;
old rows needed by early queries remain available until the entire chunk ends.
All layers, final head and backend readiness succeed before one batch cache
commit, followed by non-fallible history/logit publication. Scratch is discardable
on failure; prior committed caches, history and published logits remain valid.
Reset/replay is explicit. Pending sampled tokens and arbitrary prefix rewinds
are not implemented by this new-chunk primitive.

Initial implementation bounds are32 layers, one KV head,544 total positions and
32 rows per call. Window must fit admitted capacity. These are backend admission
bounds, not qualification of every shape or the real candidate at those limits.
The existing Gemma4 fifteen-producer bound stays unchanged; this graph uses32
inline proposal slots and separately checks an18-layer synthetic path.

## Numerical and state checks

All428,832 comparisons across2,400 observations pass the unchanged
`1e-5 + 1e-4*abs(reference)` gate. Maximum absolute error is
0.000002384185791015625; maximum gate fraction0.0753252425036715.
One-shot calls cover all192 source groups/108,816 values completely. Chunks1/3/7
compare only their actual absolute-key attention intervals; unavailable evicted
or future scores are not fabricated. Every chunk must emit all48 stages exactly
once with matching shapes. Ordinary and diagnostic forward, full logits across
schedules and the valid eleven-token causal prefix agree exactly as numbers.
Passing target intermediates are checked in Rust and are not retained for an
external raw rescore; the independent source arrays remain available.

Twenty-four injected layer/final-norm/head failure points in two- and18-layer
synthetic models preserve the full prior state. Failure, retry and reset perform
zero allocations across72 measured phases. Persistent owners remain stable;
the hidden/residual pair may exchange roles. Four exact-budget constructions
match planned versus allocated persistent ownership; one byte less rejects
before allocation. Capacity, scratch and token errors match precisely and leave
state unchanged.

An actual head256/window512 CPU cache/operator witness evaluates query512,
commits513 rows and checks the next ring step to514. It distinguishes local
exclusion of key0 from global retention while preserving key1/current visibility.
This is not a complete513-token model or a resident GPU ring qualification.

Formatting, strict default/Metal all-target Clippy,420 default tests,
298 Metal-library tests, five targeted CPU tests, five metadata controls and
19 old packed-graph regression tests pass; suites overlap. Eight owned groups
close. Initial compile failures on a feature-scoped compatibility import and
checked error conversion were preserved and corrected before target results;
fixtures, arithmetic and gates never changed. Prospective review also fixed
consumer inventory completeness and buffer-owner comparison before outcomes.

Persistent model/session accounting includes borrowed tensor extents, owned
FP32 norm capacities, cache/scratch/history/staging and metadata. Caller source
padding, construction/conversion overlap, diagnostic retention, other sessions
and process/driver/application owners remain separate. No whole-process fit or
performance qualification follows from the tiny allocation checks.

Astra authenticates3,927 current source bindings,20 retained output files and
all closed groups, approving bounded CPU retention and commit. Next is B3c's resident packed/Half
composition and explicitly qualified rolling state, then the bounded real model
and text/resource work. See [compact evidence](../benchmarks/results/2026-10-28-gemma3-cpu-composition-v1.json),
[TODO](../TODO.md) and [improvement record](improvements.md).

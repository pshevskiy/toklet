# Gemma session context, cache and history ownership

This follows [chunk-bounded host scratch](gemma-chunk-scratch.md) by separating
actual session context capacity from the model's executable position ceiling.
The [model input/output contract](long-reply-engine-plan.md) still targets the
selected model's complete shared window. This mechanism keeps Gemma's current
32-position executable ceiling and grants no new app eligibility or real
whole32 qualification.

## Shapes and ownership

The raw [model API](../src/model/gemma_packed.rs) adds
`planned_session_bytes_with_capacity(C, chunk_rows)` and
`new_session_with_capacity(C, chunk_rows, budget)`; `capacity()` reports the
admitted lifetime. Both require `1 <= chunk_rows <= C <= layout.positions`
before allocation. Existing constructors select the current model ceiling.
Generation's existing capacity option now selects its physical host ownership
as well as its request bound.

Each session owns a resolved [KV plan](../src/model/shared_kv.rs). Full-attention
producer buffers retain C positions; local producers retain `min(model_window,C)`.
K/V storage remains separate `[kv_heads, physical_capacity, head_dim]` buffers.
Shared consumer layers refer to their original producer. Narrowing does not
change the original model window used by numerical attention, absolute position
offsets, causal masking, head mapping or arithmetic. The wider immutable model
plan remains separately counted. Routes are copied once after admission, without
name lookup or dynamic dispatch in numerical loops.

Private/published logits, activation and producer staging buffers use chunk rows;
attention scratch uses `query_heads * chunk_rows * C` for each of its two buffers.
Raw and generation token histories each own C u32 slots. Logical sampled,
committed and pending lengths remain distinct from allocated capacity. No history
growth or full-cache copy is required during steady numerical execution.

Cold admission checks sizes and the aggregate budget before allocating routes,
histories, caches or scratch. Actual accounting includes Vec capacities, route
heap storage, both history heaps and inline metadata. Generation wrapper bytes
include its extra history allocation, and hybrid preadmission uses the same
calculation. Existing GPU pools remain32-row allocations, fully charged.
They are not made dynamically sized by this task.

## Independent verification contract

The [source witness](../tests/fixtures/gemma-session-capacity-v1/README.md) freezes
exact dyadic K/V data and19 capacity/chunk schedules before the native changes.
A W512/global32 plan is compared with admitted capacities1/2/3/4/5/17/32 across
35 consumers and15 producers with two KV heads. The
[native cache test](../src/model/shared_kv/tests/capacity.rs) emits logical head
bits and the actual segments consumed by attention: key positions/counts,
head strides, per-head content digests and every query's visible range.
The [independent reader](../tools/reference/check_gemma_session_capacity.py)
reconstructs them from frozen source bytes. Unused physical ring slots are not
compared as logical state. Foreign/stale proposals and overflow must preserve
the committed prefix.

The [raw numerical tests](../src/model/gemma_packed/tests/capacity.rs) compare
every admitted tiny prefix against both existing frozen graph contracts,
including the local W4 boundary. Invalid capacities, exact/minus-one budgets,
context exhaustion, reset, exact late failure and retry check ownership and
transaction semantics. The
[generation tests](../src/model/gemma_packed/generation/tests/capacity.rs) check
greedy selection against those frozen rows, final pending token at C, failure
with C-1 committed positions, retry, full replay and frozen-prompt restoration.
The sampled-token full replay is a native state-consistency check, not a new
independent numerical oracle for that changed sequence. At C1, one input plus
one requested output rejects; a manually appended pending input exercises the
same final-slot failure path without claiming a sampled token.

The reader independently reconstructs23 scratch dimensions, raw/generation
ownership and actual E2B-geometry cache payloads. The E2B geometry test allocates
only the15 caches at C1/5/32, without model weights or graph execution.
[Saved-output corruption controls](../tools/reference/test_check_gemma_session_capacity.py)
cover route/head/segment/visible-position errors, numerical rows, coherent memory
changes, pending histories, model-scope claims and exact test provenance.
Legacy chunk and hybrid readers retain their historical default interpretation;
new owned-context interpretation is explicit.

## Native evidence, 26 October 2026

Independent replay passes6860 logical/segmented consumer views and723184 exact
K/V words across19 schedules. The raw graph emits98 records and7956 scalar
comparisons against both frozen contracts; maximum error-to-bound ratio is
`0.017896043200363743`. All23 new saved-output controls reject their intended
errors. No numerical tolerance or quality gate changed.

| Admitted context C | Chunk rows | Tiny raw session bytes | Raw history bytes | Both generation histories bytes |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 1 | 15948 | 4 | 8 |
| 3 | 3 | 45540 | 12 | 24 |
| 5 | 3 | 48204 | 20 | 40 |
| 17 | 3 | 61116 | 68 | 136 |
| 32 | 3 | 77256 | 128 | 256 |
| 32 | 32 | 474672 | 128 | 256 |

The raw numbers include the instrumented tiny test-build session and actual
route/history/cache/staging/scratch ownership. They exclude immutable model,
caller/process and the outer generation wrapper. They are not release E2B or
phone footprint estimates. At C32 the new owned route/header accounting adds
216 bytes relative to the prior chunk-scratch test build. Smaller contexts
physically reduce storage rather than merely rejecting longer requests.

The separate E2B-geometry cache allocation witness records36864/184320/1179648
payload bytes at C1/5/32, respectively, across15 producer caches. Both local
and full caches retain C at these lifetimes; the original numerical window512
is unchanged. This is actual cache allocation without weights, graph or whole
process fit qualification.

Formatting and strict all-target Clippy pass for default and Metal builds.
The default CPU suite passes360 tests with16 ignored. The scoped Metal suite
passes18 tests with3 ignored, followed by the explicitly selected synthetic GPU
control. Nine portable hybrid-reader methods and six legacy chunk-transcript
controls also pass. The chunk regression repeats4352 comparisons unchanged.
The GPU control checks35 matrices/105 unchanged owners, exact/minus-one budgets
and two injected pre-command failures; in-flight failure remains unqualified.

On MacBookPro18,3 / Apple M1 Pro /16 GiB /macOS26.6.2(25G83), direct KV/raw/
generation/E2B-cache processes peak at3440976/4866408/4555112/8143232 physical
bytes, respectively. Their RSS peaks are7405568/15286272/14991360/13238272.
The chunk/GPU regressions peak at4407656/14156352 physical bytes and14778368/
35504128 RSS. Each fits the512 MiB diagnostic policy, excluding compilation.
No latency, real-model process fit or iPhone claim follows. All26 owned groups
close; protected release binaries are unchanged. The initial strict-Clippy
test-loop failure is retained separately and corrected without changing gates.
See the [public evidence](../benchmarks/results/2026-10-26-gemma-session-capacity-v1.json).

## Remaining delivery gates

This does not qualify larger numerical contexts, the real whole32 boundary,
511/512/513 or1024/1025 boundaries, the full unchanged quality workload, phone
fit or the model's declared131072-position shared window. The eleven unresolved
original BF16 cross-precision failures and earlier packed/GPU/rotary gaps remain
unwaived. Follow [the active Gemma queue](gemma4-implementation-plan.md): expand
bounded reference/native capacity, resolve numerical/quality requirements,
measure complete representation/session/process ownership, and qualify useful
partial app/device delivery before progressively expanding toward the full
model window. Existing Qwen3 phone qualification is separate.

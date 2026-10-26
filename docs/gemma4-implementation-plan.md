# Gemma4 delivery and model context support

**Active correction, 26 October 2026:** the human replaced the fixed 131072-output
requirement with support for the selected model's input/output context window.
Follow the [model context contract](long-reply-engine-plan.md). Gemma E2B's target
is its declared shared total window; short support remains explicitly partial.
The [Astra-assisted review](m8-plan-review-2026-10-26.md) corrects stale statuses
and puts a complete short Mac path before further standalone prerequisites.

G4.1 [complete immutable storage](gemma-resident-weights.md) and the first G4.2
[real short hybrid execution/generation](gemma-hybrid-real.md) now pass their
stated Mac scopes. The complete raw graph passes staged FP32 while retaining
its27 mathematical-logit failures per forward; selected raw/formatted/streamed
generation rows pass both contracts. This is a short functional diagnostic,
not product quality, a whole32-position boundary, resident performance or app
support. Next: **G2.1/G5 precision and quality decision → G4.3 resident scheduling
and progressive context.** Integrate model-driven Mac Chat/Expert only after
required admission/quality checks; decide phone fit before installation.
The [frozen-reference precision localization](gemma-precision-localization.md)
now identifies all57 cross-precision failures and their selected endpoint
contributions. It supports deferring an isolated head/norm change. The
[full quality-input contract](gemma-quality-contract.md) now freezes all 100
tasks/10000 loss targets and passes independent source/Rust codec replay.
The [official unquantized header/layout](gemma-unquantized-layout.md) now verifies
all2011 descriptors and the540 selected/60 unused text tensors. The
[complete official acquisition](gemma-unquantized-checkpoint.md) now authenticates
the full file/header. The [bounded BF16 file owner](bf16-source.md) now passes
fixture ownership, exact conversion and greater-than4GiB offset checks. The
[independent original BF16 row/scalar references](gemma-bf16-rows-oracle.md) now
pass 541 cases, all 262 vector lineages and coherent corruption controls. The
[positive target row/layout and bounded projection checks](gemma-bf16-reader.md)
now pass saved-output/read-chain replay. The
[original BF16 short mathematical graph](gemma-bf16-text-reference.md) now has
two identical complete captures, causal checks and selected scalar boundaries.
Eleven cross-precision failures across seven comparisons remain; full independent
graph/upstream BF16 agreement, model fit and quality are open. The
[native host chunk-scratch mechanism](gemma-chunk-scratch.md) now has tiny32-prefix,
transaction and synthetic GPU admission checks. The
[session-capacity mechanism](gemma-session-capacity.md) now sizes actual KV,
routes and both histories for admitted C1–32 with independent chunk bounds.
Logical/segmented KV, tiny numerical/state and budget checks pass; the executable
ceiling remains32 and real whole32 is unqualified. Next expand bounded reference
and native capacity before executing the full1025-position demand. Held-out/source quality and
remaining BF16/GPU/rotary discrepancies stay open; no workload or gate is reduced.
Each code task needs Astra review, applicable checks and a commit; keep
[improvements](improvements.md) current.

The G2–G7 rows below are umbrella work packages. Their mechanisms may be
implemented while numerical/product gates remain open. Historical progress
sections retain the scope and limitations at their publication dates; this
current queue and table supersede their stale next-step instructions.

The installed iPhone baseline is the reviewed `0dffb16` Qwen3 UI with the
immutable native package from `40bf67f`; its
[physical Qwen3 checks](mobile-qwen3-downloads.md) do not add Gemma4. Official and
MLX Gemma4 E2B config-only app probes reject unknown architecture and create no
native Gemma model handle. The [Mac storage load](gemma4-checkpoint.md) now
admits pinned Gemma text weights; the short scalar diagnostic is checked as
listed below, while full qualification and phone fit remain open.

## Current status and execution tasks

| Package | Implemented and checked | Remaining acceptance |
| --- | --- | --- |
| G1 — mapped/acquired | [Source/config/header mapping](gemma4-e2b-inspection.md), [Mac payload admission](gemma4-checkpoint.md), named layout and scalar adapter. | Any newly selected artifact/representation needs its own admission; mapping is not graph/device support. |
| G2 — mechanisms implemented, parity open | Scalar local/rotary/norm/PLE/dense/softcap operations, tiny graphs, shared KV/cache and [real FP64/staged-gap localization](gemma-precision-localization.md). | G2.1: remaining source-BF16/GPU/long-rotary investigation and product precision decision with held-out/source quality; existing gates remain unwaived. |
| G3 — bounded scalar graph/session implemented | [Real E2B scalar execution](gemma4-short-scalar.md), [generation state](gemma-generation-state.md), raw/formatted/streamed schedules at 32 positions. | Broader session/quality qualification and useful/full model context; short diagnostics grant no public app capability. |
| G4 — storage and short hybrid diagnostic implemented | Original resident weights, [tiny graph/session](gemma-hybrid-graph.md), [real E2B vector-copy/graph and raw/formatted/streamed schedules](gemma-hybrid-real.md), [bounded host chunk scratch](gemma-chunk-scratch.md) and [session context/cache/history ownership](gemma-session-capacity.md). | Required precision/quality decision remains open; G4.3 larger bounded reference/native capacity, resident graph/KV/scheduling, real whole32-position boundary and progressive context/memory qualification. |
| G5 — quality inputs/reference preparation; scoring open | [Basic-hello generation](gemma4-basic-chat-generation.md), [bounded stream composition](gemma4-stream-generation.md), [full quality-input/source/Rust codec replay](gemma-quality-contract.md) and [original BF16 short mathematical reference preparation](gemma-bf16-text-reference.md) on Mac. | Preserve eleven new cross-precision failures and earlier gaps; qualify full independent graph/upstream agreement, expand bounded numerical capacity to the complete1025-position suite, then collect checkpoint-specific held-out quality, required parity, representative completed timings and real-context evidence. |
| G6 — app integration open | Existing app search/download/chat infrastructure can be reused. | Engine-owned compatibility/capacities, exact model/codec identity and pinned Gemma acquisition, offline Chat/Expert, stop/cancel/reopen/model-switch and truthful partial/full-window labels. |
| G7 — phone route open | Existing Qwen3 baseline and rollback only. | Independent whole-process fit/representation decision first; then immutable package and physical iPhone 14 Pro numerical/state/context/UI/lifecycle evidence. |

G4.2 reuses the existing CPU forward ordering and transaction semantics while
GPU packed weights serve projection/gather. Use a small concrete adapter or
coarse static provider, without per-element dynamic dispatch or a second full
CPU model. Count bounded vector copies and CPU/GPU staging/readback. Keep the
scalar implementation as oracle. Hybrid timings do not qualify fully resident
performance. Require whole-graph feedback before more general prerequisites.

G4.3 checks the whole32-position boundary before claiming that current capacity,
then a useful512-position candidate and model-specific larger
boundaries. A short candidate can be integrated as explicitly partial support
after its required numerical/quality checks pass. It cannot close the full
model-window requirement. The original cross-precision, upstream BF16, dense/
packed GPU and long-rotary discrepancies remain unwaived.

## Shared model window replaces fixed output length

The [pinned official E2B text config](https://huggingface.co/google/gemma-4-E2B-it/raw/3e22461f65e89153144f8adb70e3b8c2cc9845a7/config.json)
declares 131072 total positions. Chat's rendered prompt/history and requested
continuation share that capacity. The engine must admit both together, subject
to qualified positions and current device memory, with any distinct model output
limit applied separately. A generation default is not such a hard limit.

The former requirement for 131072 output plus a nonempty prompt is superseded;
an extended-context Gemma or different architecture is no longer a prerequisite
for this target. Full-window numerical/quality/device qualification remains
open. Preserve historical fixed-output decoder fixtures and receipts as scoped
evidence, without carrying their old product requirement forward.

## Mobile handoff contract

Send the mobile work the committed engine source/ABI, immutable native and signed
package manifests with hashes, admitted model/tokenizer/revision, actual raw/Chat
output, total-context and byte capabilities, checked phone memory admission,
Mac real generation/state evidence and remaining limitations. Distinguish declared
model windows from qualified and device-admitted capacity; do not send a candidate
based on metadata alone. Preserve the
passing installed baseline and its rollback artifact until that candidate is
ready. Numerical operators or config inspection alone are not an install-ready
Gemma4 app or full-context candidate. The mobile work owns all phone actions and shared-index
coordination remains explicit.

## Historical scoped progress

Latest G4 complete-model storage preparation, 26 October 2026:
[the independent source/guard inventory](gemma-resident-storage-reference.md)
now freezes all 278 matrices/262 vectors, 1096 text components, 308 packed chunks
and 1186 expected guarded allocations before a new native loader. Two identical
captures, complete independent replays and 34 portable/actual reader controls
pass. New bundle/vector ABI and aggregate model admission remain unknown; no GPU
model or new graph/app/device eligibility is qualified by this reference.

Scoped G4 projection-session qualification, 26 October 2026:
[reusable fixed-geometry projection scratch](gemma-projection-session.md) now
borrows independent immutable weights per request and shares those weights with
lookup sessions after construction Context teardown. Tiny 88-step ownership,
preflight/poison/recovery checks and seven complete real-head reuse calls pass.
New real reuse is bitwise frozen staged-FP32; the separate legacy head retains
its ordered/scalar numerical gates. Full graph, source/precision, app/phone and
long-output requirements remain open; no capability or limits change.

Scoped G4 ownership qualification, 26 October 2026:
[immutable packed GPU weights](gemma-immutable-weights.md) now outlive their
construction Context and feed fresh same-device lookup sessions. Tiny teardown,
recovery/width-rejection and legacy controls pass; both complete original tables
pass all 12 schedules/1102080 exact returned values with 102 live buffer identities.
This removes per-matrix activation/status ownership from shared weights; it does
not close G2–G7, source/precision, whole graph, app/phone or long-output gates.

Latest actual G4 table qualification, 26 October 2026:
[complete real embedding/PLE upload and lookup](gemma-real-tables.md) keeps both
original tables resident, checks all 32 chunk-boundary rows and repeated/reordered
requests, and matches 1102080 returned values exactly. Aggregate policy and actual
Mac process peaks pass; whole Gemma graph/model fit, G2–G7, app/phone and long-output
gates remain open. No source/precision failures or limits are waived.

Latest standalone G4 loading progress, 26 October 2026:
[bounded packed upload](gemma-stream-upload.md) reuses one source chunk while
resident GPU chunks accumulate. Tiny independent fixtures qualify peak admission,
source/error/unwind ownership and recovery. The subsequent actual-table result
above covers those two loading/gather boundaries; whole-model fit and G2–G7 remain open.

Latest reference preparation, 26 October 2026: [complete real lookup rows](gemma-lookup-reference.md)
freeze all 41984 embedding/PLE values from the accepted independent trace for
later original-table GPU tests. Extraction does not run a model or qualify large
table loading, graph execution, phone fit or any historical precision gate.

Latest standalone G4 progress, 26 October 2026: [resident GPU table gathers](gemma-table-gather-gpu.md)
now reuse the same original packed chunks as affine projection. Tiny independent
fixtures qualify repeated/reordered rows, complete buffer identity, preflight,
transactional failure and workspace recovery. This closes one concrete primitive;
subsequent actual head/PLE selections are recorded above; whole-model admission
and G2–G7 remain open.

## G2 progress

The [tanh-GELU/GeGLU witness](gemma-gelu-reference.md) was frozen before its native
functions. The [standalone scalar target](gemma-gelu-scalar.md) now passes its
offline fixtures and required Rust/iOS checks. The
[proportional rotary diagnostic](gemma-rotary-precision.md) exposes short/long
position precision differences under the unchanged tolerance. The
[independent mathematical contract](gemma-rotary-math.md) now freezes broad
shape/chunk/offset witnesses before the
[standalone scalar reference](gemma-rotary-scalar.md), which now passes the
required checks. Upstream rotary qualification remains open. Independent
local/sliding attention now has a
[frozen numerical witness](gemma-local-attention-reference.md), including the512
window boundary. Its [standalone scalar target](window-attention-scalar.md)
passes the required checks. The [KV staging/ring witness](gemma-kv-reference.md)
now freezes normal cache state and chunk visibility. Its
[standalone cache owner](window-cache-scalar.md) passes batch commit/discard
checks. The [shared-KV routing witness](gemma-shared-kv-reference.md) now freezes
current-step producer identity and source forwarding. The
[native routing/publication plan](shared-kv-scalar.md) now passes ordered-prefix
availability and complete cache-batch checks. [Borrowed-span scalar attention](span-attention-scalar.md)
now passes frozen numerical/ring-view checks without linearizing KV.
G2 now also has a [per-layer lookup/projection witness](gemma-per-layer-reference.md)
with separate FP64 and staged FP32 contracts. The
[native FP32 composition](per-layer-scalar.md) now passes both frozen contracts.
The [dense decoder-order witness](gemma-decoder-order-reference.md) now freezes
the per-layer gate and exact norm/residual/scalar order at an explicit external
attention boundary. Its [native FP32 finish](dense-decoder-scalar.md) now
passes the frozen contracts. Quantized projection/fetch and whole-layer graph
integration and whole-graph embedding/output-head scaling remain open, along with the
upstream rotary precision gate. A new [unlearned V-norm witness](gemma-v-norm-reference.md)
freezes per-head mean-square RMS. The [existing scalar RMS recipe](gemma-v-norm-scalar.md)
now passes that fixture; adapter/storage integration remains open. The
[final-logit softcap witness](gemma-softcap-reference.md) now freezes ideal and
staged FP32 contracts. Its [native in-place transform](gemma-softcap-scalar.md)
now passes them; full-vocabulary graph integration remains open. All other
listed G2 mechanisms, whole graph/session and later real-model/app/device
stages remain open.

A [complete tiny token-to-logit witness](gemma-text-graph-reference.md) now
combines embedding/PLE, four dense local/global/shared-KV layers and the tied
output head at every position0..31. The original source CPU graph passes all
7,462 comparisons; its GPU graph completes but fails15 intermediate comparisons
under unchanged gates. Both the rejection and passing final logits are retained.
This freezes an independent synthetic short CPU reference for bounded scalar
work, not G2 completion, checkpoint admission or GPU/device qualification.
Whole native graph/session integration and the original long-rotary gate remain
open; the GPU discrepancy does not identify a specific kernel root cause.

The [bounded native scalar reference graph](gemma-text-graph-scalar.md) now
composes the exact four-layer fixture, with model-bound sessions, persistent
producer staging and atomic KV/history/logit publication. All three schedules
and selected states from every layer pass both frozen contracts; failure/reset
and allocation checks pass. This is a32-position synthetic G3 foundation.
The actual artifact/config adapter, general generation/session semantics and
real-checkpoint/Metal/app/device stages remain open, as do both upstream GPU gaps.

A [cold E2B config/layout adapter](gemma4-config-layout.md) now describes the
audited real text graph and checks540 logical/1096 physical tensor descriptors
against the original header/index. It requires explicit modality exclusion and
does not join the executable dense registry. Actual payload/value loading,
real-geometry numerical execution and all subsequent qualification remain open.

The [packed table/projection witness](gemma-packed-reference.md) now freezes
independent FP32 reconstruction and separate BF16 rounding contracts at small
and actual embedding widths. All required FP32/tied comparisons pass on source
CPU/GPU. BF16 gathers differ between backends and BF16 projection diagnostics
reject under unchanged gates; their raw evidence remains visible. Native packed
graph composition and BF16/checkpoint/device qualification remain open.

A [native packed row fetch](mlx-affine.md#native-row-fetch) now fills caller-owned
FP32 storage with one dtype/bit dispatch and complete-row failure preflight.
It matches the frozen gather contracts and preserves signed zero/storage on
boundary or late-group overflow. This is a G3 primitive, with real graph/session,
payload and BF16 qualification still open.

The [complete tiny packed graph witness](gemma-packed-graph-reference.md) now
freezes group64-compatible embedding/PLE/attention/MLP/head matrices, with original
BF16 coefficient bits and explicit promoted-FP32 source execution. All7,462 CPU
comparisons pass;15 GPU intermediate comparisons reject unchanged gates while
all final logits pass. This is the checked reference for native packed model/
session composition, with real E2B and original-BF16 qualification still open.

The [native borrowed packed graph/session](gemma-packed-model-scalar.md) now
executes the exact H64/four-layer witness without expanded matrices. Its public
constructor admits only the frozen geometry; the private layout drives the
execution. All three schedules, selected states, shared cache ownership, late
failure/retry and allocation checks are covered. This is a short scalar G3
foundation. Actual E2B payload/value admission and real geometry are next;
generation/tokenizer, Metal, quality, performance, device and app gates stay
open. Original-BF16 and all upstream GPU/long-rotary discrepancies remain
unqualified; no catalog or ABI capability changes.

A [cold payload store](gemma4-payload-storage.md) now streams an externally pinned
container into immutable text-only packed storage, checks finite vectors and
all-code affine bounds, and admits live conversion/capacity overlap before
publication. Synthetic success and public real-metadata budget rejection are
covered separately. This does not acquire or qualify the real payload. Actual
E2B acquisition/load and its15-producer geometry remain next, followed by the
unchanged generation/Metal/quality/device/app gates.

The [immutable E2B acquisition and native Mac storage load](gemma4-checkpoint.md)
now authenticate all nine selected files and admit actual text values through
all540 cold views. The original failed transfer remains rejected; recovery uses
only independently verified successful ranges. A fresh process passes within
the existing desktop logical allowance. This closes acquisition/storage only
within G5: the executable35-layer/15-producer geometry, source numerical parity,
tokenizer/chat/generation, quality, representative performance, Metal, phone fit
and app integration remain open. No whole G2/G3/G5 gate is marked complete.

The [bounded multi-producer staging mechanism](gemma-producer-staging.md) now
prepares up to15 borrowed KV proposals without heap spill and preserves atomic
publication through every shared consumer and the final head. The public graph
remains the original four-layer witness. A private35-layer structural test
covers15 owners, routes13/14 and all38 failure/retry boundaries; it does not
qualify real E2B dimensions or numerical parity. The named checkpoint execution
adapter and independent real-shape witnesses remain next.

The [bounded four-bit real-data reference](affine4-tiled-reference.md) now
checks834 selected rows across all278 packed text matrices and two complete
projections against independent scalar decoding/reductions. All eight unchanged
projection gates pass within a separate256MiB reference-process allowance.
This validates the streaming reference mechanism, without full E2B graph or
original-BF16/source qualification. Complete real-shape token-to-logit witnesses
and the named executable checkpoint adapter remain next.


The [complete four-token E2B mathematical reference](gemma4-text-math.md) now
streams all713 states in each of two precision contracts over the actual pinned
checkpoint. Every1,426 future-prefix comparison preserves exact typed bytes.
Independent scalar replay verifies complete cross-precision observations and
compact probes within a separate512MiB reference-process allowance. Five arrays
reject57 cross-precision values under unchanged tolerances; that gap remains
explicit. This advances real-shape reference evidence only. The named native
adapter, checkpoint numerical parity, generation, Metal, quality, device and app
qualification remain open, with all earlier precision failures unwaived.


The [named short native E2B adapter](gemma4-short-scalar.md) now borrows the
admitted payload into the existing packed scalar graph. Physical512-slot local
rings are separate from the unchanged32-position execution limit, and retained
store capacity/metadata are counted once. Real four-token batched, one-token
and mixed schedules produce identical logits and pass every staged-FP32 value.
Each retains27 FP64-ideal logit failures from the reference precision gap;
overall numerical acceptance remains false. Native short raw-logit execution
is now measured on Mac, with generation/state, source-BF16, precision gaps,
Metal, quality/performance, phone fit and app integration still open. G3 is not
marked complete and no capability/catalog/position limit changes.


The [bounded packed generation-state wrapper](gemma-generation-state.md) now
connects the same model-bound raw session to the existing greedy generator.
Separate pending history, exact entire committed-prefix reuse, conservative
fork/shortening replay, deferred physical reset failures and final-row sampling
are checked offline. All302 CPU/324 Metal-feature tests, both iOS targets and
five release state methods pass; warmed state operations allocate nothing.
This does not qualify real E2B continuation or close G3. Independent continuation
witnesses, tokenizer/chat, precision/source, Metal, quality/performance, device
and app gates remain open, with no catalog/ABI/position changes.

The [independent four-step greedy reference](gemma4-greedy-reference.md) now
freezes raw selected IDs107,100,45518,107 before native continuation testing.
All24 complete selected-row arrays and12 unchanged cross-precision gates are
independently replayed; FP64 is teacher-forced on staged-FP32 history. Step0
matches the earlier four-token reference bitwise. This advances continuation
witnesses only: native generation is next, and all earlier full-graph precision,
source, tokenizer/chat, Metal, quality/performance, device/app and long-output
gates remain open. No G2/G3/G4 completion or capability/limit change.

The [real short raw continuation probe](gemma4-raw-generation.md) now checks
uninterrupted, split, cancellation/resume and explicit caller-stop/continue
schedules on Mac. All32 complete-logit gates pass over8,388,608 pairs and all
four schedules preserve bitwise-identical distributions, seven committed IDs
and one pending sampled token. Required Rust/default/Metal/iOS checks pass and
the measured process fits6GiB. This advances short raw generation/state only.
Tokenizer/basic-text formatting/model-stop witnesses are next; all earlier
full-graph precision/source failures and Metal, quality/performance, device/app
and long-output gates remain open. G3 is not marked complete.

The [pinned tokenizer/basic-text witness](gemma4-text-codec-reference.md) now
freezes97 exact records in two fresh byte-identical upstream captures, with
manual prompt-layout and scalar complete/prefix-decode replay. The strict
user/assistant subset, all24 control literals, no-Strip byte-fallback behavior
and selected-artifact stops1/106/50 are explicit. Seven offline methods and
separate1GiB reference-process checks pass. Native text-profile implementation
and formatted model continuation are next; no app capability, streaming, G3,
numerical/source, quality/performance, device or long-output gate is closed.

The [dedicated native basic-text codec](gemma4-native-codec.md) now matches all97
frozen records and285 prefix decodes, without adding an app Profile or numerical
capability. Same-buffer pinned loading, duplicate-safe rendering and preallocated
decode-byte admission pass offline and real-tokenizer checks. All307 CPU/329
Metal-feature tests, both iOS targets and the release codec checks pass; the
isolated text process fits its separate1GiB allowance. Formatted model execution
and append-only text ownership are next. Numerical/source gaps, full G3, Metal,
quality/performance, combined/mobile admission, app and long-output gates remain
open, with the32-position numerical limit unchanged.


The [formatted basic-hello mathematical reference](gemma4-basic-chat-reference.md)
now composes the frozen ten-token text prompt with the complete713-state graph.
Both precision paths select the same11 IDs, ending naturally at hidden pending
stop106; ten visible IDs decode to “Hi! How can I help you today? 😊”. All33
selected-row gates pass over5,784,064 pairs. Independent replay checks exact
prompt/selection policy, complete arrays, text and committed/pending/stop state;
the separate512MiB reference-process gate passes. This advances a frozen
formatted continuation witness. Native formatted execution, streaming, full G3,
earlier numerical/source gaps, Metal, quality/performance, combined/mobile fit,
app integration and131072-output qualification remain open. No limits change.


The [native formatted basic-hello diagnostic](gemma4-basic-chat-generation.md)
now composes the pinned codec, named scalar E2B model and generation owner with
combined admission. Uninterrupted and split2+remaining schedules produce the
same eleven samples and exact reply, naturally stopping at hidden pending106.
All22 logit rows are bitwise identical between schedules and all44 reference
gates pass over11,534,336 pairs. Required307 CPU/329 Metal-feature tests, both
iOS targets and seven release diagnostic methods pass. The whole Mac process
fits6GiB with3,855,781,740 combined admitted bytes. Append-only streaming is next;
broader text quality, full G3, earlier numerical/source gaps, resident Metal,
representative performance, mobile fit/app integration and131072-output gates
remain open. No app profile, ABI, catalog or numerical-limit change.


The [pinned no-Strip Gemma text stream](gemma4-native-stream.md) now reuses a
private owned decoder state behind a borrowed tokenizer facade. A separately
admitted 4 MiB stream owner preserves whole fallback runs, skipped specials,
terminal flushing and poisoned-state semantics. All 310 CPU/332 Metal-feature
tests, both iOS targets, 22 release text methods and four failure-retention
methods pass. Existing real profiles retain parity across 233,585 vocabulary IDs.
The actual Gemma probe matches 262,144 isolated IDs, 67 small cases/four partitions
and five synthetic long patterns/three partitions; all 7,675,150 raw bytes replay
against the scalar oracle. Text-process admission passes. This closes the
bounded text-stream mechanism only; full G3, quality/precision/source gaps,
Metal, representative performance, mobile fit/ABI/app integration and 131072
LLM-output qualification remain open, with no numerical-limit changes.


The [short native Gemma streamed reply](gemma4-stream-generation.md) now composes
the accepted scalar generation and bounded text owners in an explicit diagnostic.
Uninterrupted and split runs stream the same exact reply with hidden stop 106;
cancellation after two visible IDs flushes “Hi!” once and preserves 11 committed
plus one pending ID. All 24 full rows match bitwise across applicable schedules;
48 unchanged gates pass over 12,582,912 values. All 310 CPU/332 Metal-feature tests,
both iOS targets and 12 release methods pass. Combined admission is 3,859,976,044
bytes and the measured Mac process fits 6 GiB. This closes short diagnostic
composition only. Full G3, quality/precision/source gaps, resident Metal,
representative performance, qualified identity/ABI/app integration, mobile fit
and 131072-output gates remain open, with no capability or numerical-limit change.


[Loaded Gemma content identity](gemma4-content-identity.md) now retains hashes of
the exact admitted config/index and completely consumed container, and copies
that evidence into the named short model. Late failures publish no store;
tiny/manual constructors retain absent identity. All 312 CPU/334 Metal-feature
tests, both iOS targets and seven payload/15 model release methods pass. A fresh
Mac cold load checks all 540 views and actual store/model identity without forward
execution; its measured process fits 6 GiB. This is content binding only. Trusted
repository/tokenizer pairing, native capabilities/ABI/app integration and the
remaining G2/G3/G4, quality/performance, phone and long-output gates stay open.

## Native capability prerequisite — consumed-byte reference

The [maintained Smol witness tools](smol-load-identity.md) now pass a fresh
relocated two-capture/two-replay execution with 28 synthetic methods. Their
694-range/tied-read evidence is an independent prerequisite for the later native
loaded-model identity boundary. They add no Gemma4 loader or app grant and do
not close any G2–G7, phone or 131072-output gate.

## G4 foundation — standalone resident rotary

The [fixed-shape Metal proportional rotary](gemma-metal-rotary.md) now matches
independent mathematical witnesses on M1 Pro: eleven cases/two repeats, 86,640
comparisons, twelve exact token partitions and explicit rounding/guard/overflow
checks. All 312 CPU/335 Metal-feature tests, both iOS targets and release geometry
checks pass. Buffers and coefficients remain resident for fixed offsets; new
offsets require separately admitted owners. Earlier source/precision failures,
resident whole-graph scheduling, quality/performance, mobile memory/ABI/app and
131072-output gates remain open. G4 is not complete and no capability changes.

## Native source-identity prerequisite

The [opt-in Smol source audit](smol-native-content.md) now authenticates consumed
metadata/ranges and retains historical source evidence with sticky failure
semantics. Seventeen source/consumer methods and all 329 CPU/352 Metal-feature
tests pass, as do both iOS targets. A real source-only Mac read checks 694 ranges
and six repeated visits within its separate budget. This does not construct a
live model or grant tokenizer/native/app capability. All Gemma G2–G7 and
131072-output gates remain open; binding a successful resident model is next at
this integration boundary.

## Resident-model identity prerequisite

The [audited Smol resident import](smol-model-content.md) now retains historical
consumed-source evidence only after complete upload and successful assembly.
Legacy imports remain unbound. All required 329 CPU/353 Metal-feature and iOS
checks pass; six separate legacy/audited real regressions pass the existing frozen
numerical/state/capacity witnesses. Codec/profile/device admission and native ABI/
capability/app boundaries remain separate. This does not add Gemma execution,
close earlier numerical/source failures or qualify 131072 output.


## Consumed codec-content prerequisite

The [audited Smol codec](smol-codec-content.md) now authenticates bounded consumed
config/tokenizer buffers against supplied source evidence, then publishes immutable
pairing only after the shared parser/policy succeeds. Legacy loaders stay unbound.
All 337 CPU/361 Metal-feature tests, both iOS targets and eight release methods
pass; a real Mac load matches every vocabulary ID, the two independent complete
chat cases and 102 same-implementation prefixes. Completed process admission
passes. This closes historical codec-content pairing only; an engine-owned live
model/codec boundary, native capabilities/response production, app/device support
and the broader architecture, Gemma/source/precision and 131072-output gates remain
open. Existing text/numerical limits are unchanged.


## Live native model/codec ownership prerequisite

The [audited native Smol factories](smol-native-pair.md) now publish validated text
assets with the exact supplied live model allocation, retaining it until the
final audited chat/session owner ends. Legacy factories keep their behavior.
All 343 CPU/367 Metal-feature tests, both iOS targets, six release methods and
three-target strict C links/export checks pass. Separate Mac Rust/C probes verify
owner lifetime, rejected factories, wrong-thread preservation and frozen text
results within their explicit memory policies. This closes native pairing and
retention only. Capability/response endpoints, app/device integration, broader
architectures, Gemma/source/precision/GPU gates and 131072-output-plus-prompt
qualification remain separate; no text/numerical limit or phone grant changes.

## G4 FP32 RMS foundation

The [standalone resident FP32 RMS operator](gemma-metal-rms.md) now passes its
pre-target independent mathematical fixture, scalar checks and completed M1 Pro
GPU diagnostic. Direct weighted/unweighted forms cover actual local/global head,
hidden and PLE widths, including amplified tiny values. Fixed-shape admission,
guards, transactionally published output, preflight reuse and completed overflow/
poison/recreation checks pass. It remains separate from the model graph and
existing rotary context; joint resident weights/KV/scratch/command scheduling,
source/cross-precision gates and physical phone fit remain open. No model or
application eligibility, context limit or 131072-output grant changes.

## G4 packed FP32 projection foundation

The [standalone resident packed projection](gemma-metal-affine.md) now passes
new exact/ordered witnesses and four unchanged historical projection fixtures
with FP32 inputs/outputs. Same-owner batch reuse, padding/guards/inactive capacity,
constructor/input admission and valid recovery checks pass on M1 Pro. The fixed
numerical subset rejects incompatible ordered/exact sums explicitly. Actual
Gemma activation fit, whole-model storage/upload, FP32 attention/KV/pointwise
composition, scheduling, quality/performance and phone admission remain open.
All prior graph/source/cross-precision and 131072-output gates remain unwaived.

## Real packed-projection admission

The [streaming checkpoint/trace audit](gemma-affine-admission.md) checks all 278
original text codebooks and all 277 four-token staged-FP32 projection inputs.
It proves 260 calls' numerical/geometry admission: 171 direct, 35 exact residual
and 54 sufficient product envelopes. Sixteen FF-down product envelopes remain
inconclusive; the full tied head and both whole table owners exceed the existing
row bound. This is independent reference admission, without GPU execution or
native activation parity. Resolve the products and table/head storage before
graph composition; all previous G2/G3/G4, source/precision, device and long-output
requirements remain open.

## Coordinate product admission refinement

The [coordinate-correlated audit](gemma-affine-admission.md#coordinate-correlated-product-refinement)
proves nine additional FF-down envelopes, bringing the short frozen trace to
269 admitted calls. Seven envelopes at layers 0..6 remain not proved, and the
head/table geometry limits remain. The original default mode preserves all 277
accepted records and its earlier 260/16/1 classification. This changes reference
envelopes only; operator S/U/geometry, native/GPU graph, source/precision and
later device/app/long-output gates are unchanged.

## Independent L1 admission reference

The [pre-target per-token L1 witnesses](gemma-affine-l1.md) freeze 19 synthetic
cases and independently replay all 277 captured-input calls. The proposed
combined proof preserves 269 legacy passes and proves the remaining seven
FF-down envelopes under the existing error allowance. One full-head geometry
rejection remains, as do whole-table storage limits. The engine is unchanged.
Thirteen portable and two retention methods, two identical captures and both
independent replays pass; the first failed test attempt remains preserved.
A separate reviewed engine fallback and actual GPU qualification must precede
graph use. All native/source/precision, graph, app/device and long-output
requirements remain open.

## G4 L1 fallback checked on Metal

The [standalone packed projection](gemma-metal-affine.md#per-token-l1-fallback-qualification)
now tries the frozen per-token L1 proof only after its original max proof
rejects. Limits, shader and pre-staging rejection behavior are preserved.
All 347 CPU / 374 Metal-feature tests, both iOS compile targets and optimized
admission pass. Actual M1 Pro execution passes 15 new cases twice under both
references, partitions/reuse, four rejection recoveries and a later-row
failure preserving all device buffers; old GPU cases and ten capture-checker
regressions pass. This closes the standalone fallback subset only. Actual
real-model projection/native parity, table/head storage, full graph and all
prior source/precision, app/device and long-output requirements remain open.

## Complete real-projection reference arrays

The [complete projection reference](gemma-projection-trace.md) freezes 552
arrays / 26,333,184 bytes for 276 fitting calls in the four-token trace.
Separate 206 captured-state reconstructions from 70 acknowledged mathematical
GELU products; outputs are complete rounded staged-FP32 linear states before
norms/scales/residuals. Independent mappings/rounding/raw-byte replays agree,
including signed zeros, FF-width transition and max-first/L1 admission.
Eleven portable and two retention methods, two identical captures/replays
and six owned process/resource closures pass. The head remains geometry
rejected. These are pre-native reference inputs, with actual projection/GPU,
native activation parity, table/head storage, resident graph and earlier
source/precision, app/device and long-output gates still open.

## Actual real-checkpoint projections on Metal

The [bounded real-projection diagnostic](gemma-real-projections.md) passes all
276 fitting calls / 1932 GPU steps with original packed/BF16 weights. Complete
outputs, independent dual-rounded-reference gates, native ordered rows, full
buffer digests, partitions/resume and 18 checker regressions pass. The largest
single owner is 10,838,428 bytes; measured diagnostic RSS/physical peaks stay
below its separate policy. All required Rust/iOS checks pass after a retained
strict-lint-only first attempt. The head must retain its exact geometry error.
This closes the standalone four-token projection qualification, not G2/G3/G4.
Next is bounded table/head storage and numerical qualification before resident
graph composition; prior source/precision, app/device, quality/performance and
131072-output-plus-prompt requirements remain open.

## Table/head storage reference before native implementation

The [small packed-table reference](gemma-table-reference.md) freezes complete
19×128 and 11×64 source tables, repeated gathers, partition coverage, exact
padding and signed zeros. Sixteen portable/two retention methods and two
identical captures with independent replays pass. Late overflow or missing
source rows leave the entire gather destination unchanged. This is a pre-target
witness; native row access, admitted chunked head/table execution and resident
graph composition are next. The old full-head geometry rejection and all prior
source/precision, device/app, performance and long-output gates remain unchanged.

## Native immutable packed row views

[Concrete borrowed row ranges](mlx-row-views.md) now share validated original
codes and metadata without allocating or reconstructing tensors. All 54 existing
storage combinations preserve source pointers, offsets and lifetimes. Frozen
scalar checks preserve 960 gathered values / 192 negative zeros and 360
partitioned outputs; invalid/extreme and nested ranges pass. Required CPU/Metal,
iOS and optimized checks pass. Existing numerical loops and GPU constructors
are unchanged. Admitted chunked head/table execution and transactional multirow
gather are next, with full graph/source/device/quality/long-output gates open.

## G4 bounded resident projection chunks

The [partitioned affine owner](gemma-partitioned-affine.md) now admits simultaneous
original packed chunks and complete parent/child staging before upload. All-child
numerical preflight preserves warmed state on rejection; caller publication waits
for every chunk. A test-only failure after two completed children leaves changed
partial staging, untouched caller output and sticky parent poison; recreating the
same geometry succeeds. Independent complete-buffer replay passes 60 steps across
10 owners and 22 mutation/reader tests. Required source checks pass (351 CPU,
381 Metal-feature tests, iOS targets and optimized plan/row-view checks). This is
a tiny correctness qualification with explicit per-chunk waits/readbacks, not
actual full-head, table-gather, graph, performance or phone qualification.
Context/caller/source-loading overlap needs separate model-wide admission.
Earlier source/precision failures and all open G2–G7 and long-output gates remain.

## Complete real tied-head reference before native qualification

The [full-head extraction witness](gemma-head-reference.md) now supplies all 6144
staged final-normalization inputs and 1048576 pre-softcap head outputs from the
accepted independent mathematical trace. Two captures match byte-for-byte; two
independent complete-byte replays, 17 mutation/reader and 3 failure-retention tests
pass. This verifies extraction of an existing numerical reference without
reevaluating the graph or executing an engine/GPU. Original source/precision
failures remain unwaived. Actual full-head residency and execution, complete
graph and later device/app/performance/long-output gates remain open.

## G4 complete real tied-head qualification

The [original full tied head](gemma-real-head.md) now passes seven standalone
Mac GPU calls with all sixteen packed chunks resident together, using the frozen
complete four-token reference. All 1048576 scalar/mathematical and 8388608 dual
GPU/reference comparisons pass with zero difference against rounded FP32 arrays.
Complete device outputs are retained separately from caller results; all 96
allocation/buffer identities, guards, inactive capacity and parent/chunk staging
pass independent replay. Aggregate owner/workload policies and actual process
peaks pass; original CPU packed sources are dropped before reuse. All 24
mutation/reader tests and required source/iOS/optimized checks pass. This closes
the fixed standalone head boundary only. It does not close G4, large table gather,
resident graph, historical source/precision discrepancies, performance, phone/app
or 131072-output-plus-nonempty-prompt gates.

## Transactional scalar table gather

The [native multirow gather](mlx-table-gather.md) now preflights all selected
indices and finite reconstructions before publishing any FP32 value, over the
original immutable packed source. The pre-target witness checks all 960 canonical
values / 192 negative zeros, duplicate/order semantics, successful 1/32-ID
boundaries, whole-output rejection and recovery after late overflow. Shared
private helpers preserve the old single-row numerical loops and error semantics.
Required source/iOS checks and optimized 54-format control plus all four fixture
methods pass (356 CPU / 387 Metal-feature tests). Actual large table selections,
GPU gather/storage, full graph and earlier source/precision, performance, app/phone
and long-output gates remain open. No capabilities or limits change.

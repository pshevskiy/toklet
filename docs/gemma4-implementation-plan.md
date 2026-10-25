# Gemma4 and 128K reply priorities

Human instruction relayed through the authorized mobile work on 24 October2026:
prioritize native Gemma4 support and 131072 output tokens in one Chat/Expert
reply. Finish the current small normalization reference, then perform the tasks
below ahead of unrelated architecture expansion. The remaining
[engine/Hugging Face plan](engine-expansion-plan.md) remains open. Astra reviews
each code task and it is committed after its checks pass.

The installed iPhone baseline is the reviewed `0dffb16` Qwen3 UI with the
immutable native package from `40bf67f`; its
[physical Qwen3 checks](mobile-qwen3-downloads.md) do not add Gemma4. Official and
MLX Gemma4 E2B config-only app probes reject unknown architecture and create no
native Gemma model handle. The [Mac storage load](gemma4-checkpoint.md) now
admits pinned Gemma text weights; numerical execution and phone fit remain open.

## Ordered tasks

| Task | Work and acceptance | Status |
| --- | --- | --- |
| G1 | Authenticate the pinned Gemma4 wrapper/text source, official E2B config and quantized candidate metadata. Map every graph option and tensor/tokenizer/chat requirement. Inspect bounded index/tensor headers and actual representation; enumerate missing mechanisms and rejected variants. Separate effective parameter names from actual stored weights. Derive allocation requirements before downloading a large payload. | [Mapped](gemma4-e2b-inspection.md); [Mac payload admitted](gemma4-checkpoint.md), inference/fit open |
| G2 — active | Freeze independent tiny witnesses for each missing concrete mechanism: local/global attention geometry and position rules, sliding masks, shared KV ownership, per-layer embedding/input paths, exact norm/scale/activation/residual order and logit capping where required by G1. Keep old families' fixtures passing. | Open |
| G3 | Implement a small named Gemma4 config/weight adapter and scalar text graph/session using concrete validated layers. Qualify state commit/replay, pending sampled tokens, exact prefix reuse, stop semantics and overflow. Reject every unimplemented graph/modality/representation rather than aliasing an existing family. | Open |
| G4 | Add resident Metal execution and allocation plans for actual weights, shared/rotating KV, scratch, staging and temporary overlap. Compare scalar/upstream numerical states and outputs. Qualify independent Mac/phone admission and fail before allocation on non-fitting requests. | Open |
| G5 | Acquire an immutable small real checkpoint only after G1 admission. Freeze independent numerical and tokenizer/chat witnesses; qualify real generation/state, quality and completed Mac timing/memory. Preserve old download/chat profiles. | Open |
| G6 | Integrate engine-owned compatibility/capabilities into search, required-file downloads, offline loading, Chat and Expert. Bind source/model/tokenizer revisions and explicit raw/Chat/context/byte limits. Exercise stop/reopen/offline/model-switch and failure recovery. | Open |
| G7 | Deliver a committed native/signed package and complete evidence checklist to the mobile work. Install and run physical iPhone14Pro tests with rollback, history preservation and explicit unsupported/over-budget outcomes. | Open |

Independent engine operators already checked for other graphs remain available,
including gated-delta math and scalar causal convolution. Those do not qualify
Gemma4. The current [gated normalization reference](gated-norm-reference.md) is
finished as a small separate witness before the priority switch; no further
unrelated hybrid graph work takes precedence over G1.

## 131072 output tokens is a separate acceptance gate

The pinned official E2B text config at
[`3e22461f65e89153144f8adb70e3b8c2cc9845a7`](https://huggingface.co/google/gemma-4-E2B-it/raw/3e22461f65e89153144f8adb70e3b8c2cc9845a7/config.json)
declares 131072 total positions. It cannot accommodate 131072 output tokens plus
a nonempty prompt under those unchanged position semantics. Google describes
smaller and larger family context sizes in its
[official overview](https://ai.google.dev/gemma/docs/core); those published sizes
are not Toklet artifact or device qualification.

Gemma4 functionality may first run shorter replies while the
[long-reply engine plan](long-reply-engine-plan.md) qualifies its distinct graph,
position and memory route. Determine whether an explicitly supported extended
context Gemma4 configuration or another small long-context checkpoint can meet
actual prompt+output capacity and both device budgets. Any context extension
needs independent numerical/position and long-context quality evidence. A model
name, advertised mobile format, constant change or hypothetical KV calculation
cannot grant support. Larger model weights are not assumed to fit this Mac's
6GiB working-set policy or the phone's changing allowance.

## Mobile handoff contract

Send the mobile work the committed engine source/ABI, immutable native and signed
package manifests with hashes, admitted model/tokenizer/revision, actual raw/Chat
output, total-context and byte capabilities, checked phone memory admission,
Mac real generation/state evidence and remaining limitations. Preserve the
passing installed baseline and its rollback artifact until that candidate is
ready. Numerical operators or config inspection alone are not an install-ready
Gemma4 or 128K candidate. The mobile work owns all phone actions and shared-index
coordination remains explicit.

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

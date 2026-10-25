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

# Gemma4 and 128K reply priorities

Human instruction relayed through the authorized mobile work on 24 October2026:
prioritize native Gemma4 support and 131072 output tokens in one Chat/Expert
reply. Finish the current small normalization reference, then perform the tasks
below ahead of unrelated architecture expansion. The remaining
[engine/Hugging Face plan](engine-expansion-plan.md) remains open. Astra reviews
each code task and it is committed after its checks pass.

The installed iPhone baseline is the reviewed `17fbc4f` UI with native
`2d7d35b0`; its [six physical discovery/UI checks](mobile-model-ranking.md) and
the earlier [four Gemma checks](mobile-gemma4-testing.md) do not add Gemma4.
Official and MLX Gemma4 E2B config-only probes reject unknown architecture and
create no native model handle. No Gemma weights or phone memory fit are admitted.

## Ordered tasks

| Task | Work and acceptance | Status |
| --- | --- | --- |
| G1 | Authenticate the pinned Gemma4 wrapper/text source, official E2B config and quantized candidate metadata. Map every graph option and tensor/tokenizer/chat requirement. Inspect bounded index/tensor headers and actual representation; enumerate missing mechanisms and rejected variants. Separate effective parameter names from actual stored weights. Derive allocation requirements before downloading a large payload. | [Mapped](gemma4-e2b-inspection.md); full payload/fit not qualified |
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
integration, embedding/logit scales and capping remain open, along with the
upstream rotary precision gate. A new [unlearned V-norm witness](gemma-v-norm-reference.md)
freezes per-head mean-square RMS. The [existing scalar RMS recipe](gemma-v-norm-scalar.md)
now passes that fixture; adapter/storage integration remains open. The
[final-logit softcap witness](gemma-softcap-reference.md) now freezes ideal and
staged FP32 contracts before the native transform. All other
listed G2 mechanisms, whole graph/session and later real-model/app/device
stages remain open.

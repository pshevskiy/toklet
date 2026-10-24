# Gemma4 and 128K reply priorities

Human instruction relayed through the authorized mobile work on 24 October2026:
prioritize native Gemma4 support and 131072 output tokens in one Chat/Expert
reply. Finish the current small normalization reference, then perform the tasks
below ahead of unrelated architecture expansion. The remaining
[engine/Hugging Face plan](engine-expansion-plan.md) remains open. Astra reviews
each code task and it is committed after its checks pass.

The installed iPhone baseline remains the reviewed `abe661e` UI with native
`2d7d35b0`; its [four physical checks](mobile-gemma4-testing.md) do not add Gemma4.
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
position precision differences under the unchanged tolerance. Its independent
frequency/phase contract and broader witnesses are next; rotary qualification
remains open. All other listed G2 mechanisms, whole graph/session and later
real-model/app/device stages remain open.

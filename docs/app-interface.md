# One Toklet app: Chat and Expert mode

23 October 2026. Toklet Harness now opens a friendly local chat interface inspired
by the neighboring InferencePhone project. That earlier project is a separate
app/runtime; this change consolidates the requested interface in Toklet, using
Toklet's existing Rust/Metal engine and one serial native worker. It does not
merge or install the neighboring runtime.

The optional [native Mac window add-on](mac-app.md) now shares these views and
adds Mac folder access and joined window/quit lifecycle. Its visible functional
checks are separate from the earlier iPhone/Simulator evidence below.

## Use on the iPhone

1. Open **Toklet Harness**, then **Choose a model**.
2. Tap **Use model** beside a saved SmolLM2 or Qwen model, or **Download and use**
   for a new download. **Active** identifies the loaded model. Models closes
   automatically after loading. Search results have the same direct actions.
   You can also open a copied local model folder or choose its complete folder
   in Files. See the [download guide](model-hub.md) for progress/cancellation and
   offline opening.
3. Type a message and press Send. **Writing…** remains visible until the reply
   completes; Stop requests cancellation and preserves text decoded from the
   completed prefix. Replies are currently displayed at completion. Inline Markdown, fenced code blocks, syntax highlighting and Copy are available in replies, including code fences cut short by a stopped reply.
4. Open the conversations icon for saved chats and **Settings**. Choose
   **Expert mode** to use the original raw-token controls; **Back to Chat**
   returns to the friendly interface. The gear icon also opens Settings directly. Expert mode explains each generation and lifecycle action and includes a copyable tiny-fixture example; its output is raw token IDs.

History, drafts and the response limit are saved locally. Changing modes releases
the current numerical model/session before switching. Closing the Expert worker
is terminal for that worker; returning to Chat creates a new worker. Chat history
is separate from native KV state. A background interruption releases the session;
a subsequent explicit Send can recreate it. A memory warning releases weights
as well, requiring explicit model opening again. Foregrounding never submits a
prompt automatically.

## Current support and limits

The [mobile dashboard](mobile-dashboard.md) adds RAM used/allowed, first-token
time and decode rate, a supported-model fit shortlist, an offline timing
benchmark and explicit single/all conversation deletion. These changes have
their own local verification scope; they do not imply a new iPhone installation
or an increased output limit.

Text chat supports the selected **SmolLM2-135M-Instruct eight-bit** (Llama family),
**Qwen2.5-0.5B-Instruct eight-bit**, and the selected **Mistral-7B-Instruct-v0.3
four-bit** snapshot. Admission checks exact model
configuration, supported loaded geometry and pinned tokenizer/template identities
in the [text profile](app-chat-profile.md); local text admission does not
authenticate every weight payload. Downloaded snapshots additionally hash weights.
SmolLM2 and Qwen have physical iPhone14Pro controller and live UI evidence in
the [model-switching follow-up](mobile-model-switching.md). Mistral phone Chat
remains outside the admitted phone budget.
Other qualified numerical models remain available through Expert mode. A model
architecture being implemented does not establish tokenizer/template support for
every checkpoint. Hugging Face search and verified downloads are available for these qualified
profiles through [Models](model-hub.md). Other search results explain their
unsupported status. Supported results use the same Use model action as the
saved-model list. Opening Models reads a lightweight presence inventory; Use
checks full snapshot integrity. A newer Use can replace pending verification;
stale results cannot open the previous selection. Downloads keep their model
selection fixed until completion or cancellation. Opening the already active model preserves its session. The
Models sheet closes after a successful load and remains open on failure. Wider downloadable/text profiles remain open in
[M8.4](mlx-implementation-plan.md). Downloaded snapshots additionally authenticate
the six required file hashes, including weights, before publication and opening.

All three selected chat profiles admit 512 total token positions and chunk32. SmolLM2 reserves
512 MiB numerical plus 64 MiB text; Qwen reserves 1 GiB numerical plus 128 MiB
text. Mistral reserves 5 GiB numerical plus 64 MiB text, with a separate
5.5 GiB current-process admission threshold; see [Mistral integration](mlx-mistral-app.md).
Profiles check current-process admission before loading. Prompt plus requested
output must fit before generation resets KV; there is no silent truncation.
New workspaces default to a 128-token reply limit. Settings offers 32, 64, 128
and 256-token choices, and older saved limits are preserved. The native text
bridge supports up to 256 output tokens; Expert raw-ID generation remains capped
at 128. A reply that reaches its limit is labelled explicitly; a model may also
finish earlier. Text formatting accepts up to63 alternating
messages, at most64KiB combined UTF-8 input. Long saved conversations can exceed
the model's context and require a new chat.

Local history is bounded to20 conversations,64 messages per conversation and a
64MiB file, with16KiB drafts and8MiB stored message text. The
[long-reply preparation](long-reply-mobile.md) describes storage reservations;
these bounds do not increase the current generation or conversation limits. Malformed/oversized history
blocks editing until a successful retry, avoiding accidental replacement. Failed
writes keep the in-memory draft. Interrupted placeholders restore as stopped;
restoring history does not run inference. External Files folder access is held
for the loaded model's lifetime; no persistent security bookmark is stored, so
choose an external folder again after relaunch.

This tiny model's answer quality is limited. A functional two-turn probe returned
“Two is a positive integer.” to an arithmetic question. The text smoke test is
not a quality acceptance result or a recommendation for factual answers.

## Verification and boundaries

Astra reviewed the text boundary and each interface code task. Review fixes cover
queued-chat cancellation before dequeue, terminal result retention through cleanup,
storage recovery, folder scope lifetime and rejected-load identity. The final
strict Swift6 Mac runner passes workspace/storage/mode/cancellation regressions,
real SmolLM2 text inference and the original raw CPU/Metal worker checks. Both
iOS SDK app builds pass. A fresh signed physical app contains the final code.

The physical iPhone14Pro text smoke test and Mac run return identical token IDs
and independently decoded text for first turn, second turn and explicit resume.
They also check context rejection, native cancellation, release and session
recreation. The original physical raw CPU/Metal checks pass. These are functional
checks; this interface task adds no latency, sustained rate, memory-limit or
broader M8.5 qualification claim.

The earlier interface task’s visible Simulator checks cover
Chat/Models/history/settings, opening and cancelling
Files, Expert raw generation (`47,47`), terminal Close, Back to Chat and draft
entry. The physical default screen was inspected in dark appearance. The
Simulator correctly rejects this MLX model without a unified-memory MPS device.
The original record did not qualify physical GUI Send or model switching. The
[24 October follow-up](mobile-model-switching.md) now records repeated physical
SmolLM2/Qwen switching, real GUI Send, the 256-token setting/limit notice, Chat
code and Expert output. Search/download taps, GUI Stop, successful Files folder
selection and largest accessibility text layouts remain outside that new run;
native and controller tests retain their separate scope. The
[interface receipt](../benchmarks/results/2026-10-23-app-interface-v1.json) preserves
exact executed-source identities, including the earlier physical smoke build.

The [Models integration receipt](../benchmarks/results/2026-10-23-model-hub-app-v1.json)
adds a fresh signed iPhone 14 Pro download and full offline-controller generation
with identical Mac token IDs/text, while preserving the visual limits above.

The [24 October mobile-interface follow-up](mobile-interface-fixes.md) records
responsive model selection, longer replies, Expert help and code rendering,
with separate build, controller and visual-validation boundaries.

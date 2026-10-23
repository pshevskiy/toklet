# One Toklet app: Chat and Expert mode

23 October 2026. Toklet Harness now opens a friendly local chat interface inspired
by the neighboring InferencePhone project. That earlier project is a separate
app/runtime; this change consolidates the requested interface in Toklet, using
Toklet's existing Rust/Metal engine and one serial native worker. It does not
merge or install the neighboring runtime.

## Use on the iPhone

1. Open **Toklet Harness**, then **Choose a model**.
2. Search Hugging Face or choose the tested SmolLM2 card, then **Download model**
   and **Open downloaded model**. You can also open a copied local model folder
   or choose its complete folder in Files. Wait for loading to finish, then close
   Models. See the [download guide](model-hub.md) for progress/cancellation and
   offline opening.
3. Type a message and press Send. **Writing…** remains visible until the reply
   completes; Stop requests cancellation and preserves text decoded from the
   completed prefix. Replies are currently displayed at completion.
4. Open the conversations icon for saved chats and **Settings**. Choose
   **Expert mode** to use the original raw-token controls; **Back to Chat**
   returns to the friendly interface.

History, drafts and the response limit are saved locally. Changing modes releases
the current numerical model/session before switching. Closing the Expert worker
is terminal for that worker; returning to Chat creates a new worker. Chat history
is separate from native KV state. A background interruption releases the session;
a subsequent explicit Send can recreate it. A memory warning releases weights
as well, requiring explicit model opening again. Foregrounding never submits a
prompt automatically.

## Current support and limits

Text chat is qualified for the selected **SmolLM2-135M-Instruct eight-bit**
snapshot. Admission checks supported model geometry and exact tokenizer/template
identities in the [text profile](app-chat-profile.md); it does not authenticate
the weight payload against that snapshot’s SHA.
Other qualified numerical models remain available through Expert mode. A model
architecture being implemented does not establish tokenizer/template support for
every checkpoint. Hugging Face search and verified downloads are available for that qualified
profile through [Models](model-hub.md). Other search results explain their
unsupported status. Wider downloadable/text profiles remain open in
[M8.4](mlx-implementation-plan.md). Downloaded snapshots additionally authenticate
the six required file hashes, including weights, before publication and opening.

The chat configuration admits 512 total token positions, chunk32, a 512MiB
numerical allowance and a separate64MiB text allowance. Prompt plus requested
output must fit before generation resets KV; there is no silent truncation.
Output limits are32,64 or128 tokens. Text formatting accepts up to63 alternating
messages, at most64KiB combined UTF-8 input. Long saved conversations can exceed
the model's context and require a new chat.

Local history is bounded to20 conversations,64 messages per conversation and a
2MiB file, with16KiB drafts and64KiB message text. Malformed/oversized history
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
The new search/download tap flow, physical GUI Send/Stop, successful Files folder
selection, the on-screen keyboard
and largest accessibility text layouts have not been visually qualified; native
and controller tests are separate evidence. The
[interface receipt](../benchmarks/results/2026-10-23-app-interface-v1.json) preserves
exact executed-source identities, including the earlier physical smoke build.

The [Models integration receipt](../benchmarks/results/2026-10-23-model-hub-app-v1.json)
adds a fresh signed iPhone 14 Pro download and full offline-controller generation
with identical Mac token IDs/text, while preserving the visual limits above.

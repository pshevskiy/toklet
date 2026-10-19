# Native raw-token boundary, ABI 1

The [C header](../apple/include/toklet.h) exposes the existing CPU scalar model,
transactional session and greedy generator. This first component takes local
UTF-8 model paths and raw token IDs. Tokenization, chat templates, text decoding,
model downloads and UI remain outside numerical execution. CPU capacity remains
at most 2,048; Metal/iOS integration and physical-device limits are separate tasks.

Build the library with `cargo build --locked --offline --lib`. The package emits
both the ordinary Rust library and `libtoklet.a`. Import `CToklet` through the
adjacent module map for Swift, or include `toklet.h` from C. This does not itself
package or qualify an iOS library.

Create model and session handles on a dedicated inference thread. All their
operations, including release, remain on that thread. A serial dispatch queue
alone does not guarantee this thread affinity. The model is immutable; every
session retains its allocation independently, so releasing the model handle
before an idle/live session does not invalidate weights. Each session owns its
KV, chunk scratch, distribution buffers and bounded history. The private owning
wrapper drops its borrowed session before the retained model. No model tensor
or session reference escapes through C.

`TokletOptions` supplies ABI version, exact record size, capacity, prefill chunk
and an explicit memory budget. The bridge reserves 4,096 bytes before numerical
load/session admission and reports it in session `admitted_bytes`. Engine
admission includes weights once per single-model/session view; multiple live
sessions, caller token/event/text buffers, tokenizer and UI need additional
accounting. No iPhone budget is inferred from the 6 GiB Mac ceiling. Input tokens
and stop IDs are borrowed for the synchronous request and not retained as caller
pointers. Internal history copies IDs into its admitted capacity.

Operations are prompt, append, continue and reset+prompt. Prompt/stop IDs and
requested output capacity are validated before session mutation. There are no
implicit EOS IDs: the frontend resolves and passes every stop ID. Stop IDs enter
history but are hidden from callbacks. Final sampled IDs stay pending until
continued; history and committed KV lengths are distinct in `TokletReport`.
Reset clears logical history/KV validity while retaining allocated resources.

Callbacks receive POD token events by value, synchronously on the inference
thread. The caller context remains live until generation returns. Callback
nonzero returns `TOKLET_CALLBACK_ERROR`, preserving the sampled pending token
and accurately reporting prior successful emissions. Reentrant state/mutation
or session release returns `TOKLET_BUSY`; no session reference or collection is
exposed. Callbacks must return normally, without foreign exceptions or unwinding.

An independent cancellation handle contains an atomic flag. Setting it is the
only supported concurrent action during inference. It stops at existing chunk
or token boundaries, after completed GPU work when that backend is added. Clear
it explicitly while idle before a new request; cancellation is never cleared
implicitly. Keep the flag alive until generation and all concurrent setters
return, then release it. Cancellation does not free resources or interrupt a
kernel in progress.

Every native entrypoint contains Rust unwinding. A panic during session work
poisons that session; subsequent work returns `TOKLET_POISONED` and the caller
must release/recreate it. Allocation aborts, invalid foreign pointers and
foreign exceptions cannot be contained by this contract. Each non-null pointer
must name live aligned storage for its complete initialized extent; writable
outputs/error records must not overlap inputs or one another. The ABI checks
null, alignment, lengths, sizes and integer bounds, but cannot prove foreign
pointer provenance. Never reuse or release a handle twice. NULL release is a
no-op. Errors use a fixed caller-owned record with a stable numeric code and a
NUL-terminated UTF-8 message; passing NULL suppresses the error text.

The [native component report](experiments/2026-10-19-m7-native-abi.md) records
source review, portable checks and a real C-to-Rust loader/generation smoke test.
The frozen tiny fixture supplies an independent greedy oracle. These checks
establish ABI behavior on the named host, without phone integration, mobile
quality, memory, throughput or lifecycle qualification.

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

## Explicit Metal models

`toklet_model_load_metal` is an additive entrypoint; ABI 1 record layouts and
CPU behavior remain unchanged. A zero-length packed path selects compensated
Half weights with Original normalization. An explicit packed directory selects
Q4 v2 compensated tiled projection with Compensated normalization; v1 is rejected
before a model handle is published. Sessions use Online attention, FP16 KV and
FullLogits. Backend selection happens once per generation request before the
statically dispatched numerical loop. No automatic CPU fallback occurs.

The same retained owner, dedicated-thread requirement, busy guard, error/panic
policy and cancellation flag apply. Boxed model/session allocations are fixed
at creation and covered by the checked bridge reservation. Multiple sessions
share the immutable model allocation and retain separate KV/scratch. Device and
API capability checks still apply; this entrypoint compiles for arm64 macOS15+ and iOS18+ Metal builds, with
explicit unsupported errors in other builds. The [Apple package report](experiments/2026-10-19-m7-apple-package.md)
records separate Mac/device/simulator XCFramework slices, exact C export/linkage
and deployment checks. Mac Metal and simulator CPU integration pass; physical
iPhone Metal runtime, quality and supported limits remain unqualified.

The [Metal bridge report](experiments/2026-10-19-m7-metal-abi.md) separates
integration/state/combined logical admission checks from performance and mobile
qualification. No new latency or physical-memory claim follows from that test.

## History and replay

`toklet_session_history` returns exact sampled IDs, including pending samples and
hidden stops, into caller-owned storage on the session thread. NULL output with
zero capacity queries count. A small buffer returns Invalid with required count
and no partial token copy. Keep output/count/error storage separate; reserve
application memory for the bounded history copy.

After releasing a session, a new session with the same retained immutable model
can eagerly replay all copied IDs using prompt plus zero requested output.
History is preserved; pending IDs become committed. Continue then evaluates from
that exact token prefix. Reusing a path after replacing/reloading weights is not
proof of model identity. A frontend must preserve the same model owner or
independently validate identity before it offers replay.

## Read-only next-token scoring

`toklet_session_score` queries the final fully committed next-token distribution
on the owning thread. Its additive48-byte `TokletScore` record returns target
and lowest-ID greedy tokens, vocabulary/evaluated counts, exactly widened FP32
logits and a stable compensated FP64 negative log likelihood. Existing ABI1
records keep their layouts. The query borrows the resident logits and allocates
no logits vector or model copy; backend selection precedes the reduction.

Generate with zero requested output to evaluate an exact prompt or append
without sampling. Score before appending each target. Only the current final
row is available; a multi-token prefill cannot retrospectively score its earlier
rows. Pending sampled IDs, reset or invalid distributions reject. Cancellation
alone preserves an earlier valid committed prefix, so a caller must verify exact
history/evaluated counts after any cancelled generation before assigning a score
to a target. This read-only query changes neither prefix nor cancellation flag.

The normal owner, busy, poison, pointer and non-overlap contracts apply. Output
is untouched unless the call succeeds. The [component report](experiments/2026-10-21-m7-score-api.md)
records independent numerical checks, selected-model Metal comparisons and
three-platform C build/link checks. Full phone corpus/task quality collection
and device qualification remain separate work.

# Mobile capability and response consumer foundation

This is an isolated L3/L5 consumer proposal, agreed with the engine owner on
25 October 2026. It is not wired to existing model profiles, settings, native
handles or generation. No ABI/header, app eligibility or installed package is
changed. The engine must publish and qualify the matching native producer,
lifetime and memory contract before integration. Gemma and single-reply
131,072-token generation remain separate, unqualified acceptance gates.

## Strict capability metadata

[GenerationCapabilities](../apple/Harness/GenerationCapabilities.swift) accepts
version1, an immutable Hugging Face repository/40-hex revision and two 64-hex
config/tokenizer hashes. Both `raw` and `chat` mode records must be explicit.
An unavailable mode has a reason and no runnable limits. A runnable mode has
positive total-context, prompt/output, input/decoded-byte and tokenizer-allowance
limits plus batch-token, batch-UTF8 and fallback-token bounds. Unknown keys,
versions, duplicate modes or JSON keys, missing fields, Boolean/fractional/
negative counts and overflow reject. These facts reserve no resources and do
not prove graph eligibility or numerical qualification.

Actual prompt+requested output is checked separately from the two independent
maxima. Their maxima need not fit together. Prompt includes the complete active
history, including inherited pending tokens. A131072-position model cannot
admit131072 new tokens plus a nonempty prompt. Native resource admission remains
required after this metadata-only check; no app limit is silently raised.

## One-owner bounded responses

[GenerationResponse](../apple/Harness/GenerationResponse.swift) owns fixed-size
UTF8 storage and exposes small summaries. There is no queue, full token-ID
history or timing array. The consumer accepts only its UUID and ordered sequence
starting0. A valid text batch contributes emitted IDs and/or nonempty UTF8;
zero-token text drains support delayed fallback/long-token output. A terminal
batch must assert `terminal_flush=true`, include exact totals and then seals the
owner. Empty terminal batches are valid. Malformed, stale, duplicate, reordered,
oversized or inconsistent events reject before text/counters change.

Generated sampled/committed/pending totals are relative to this request,
excluding prompt and inherited pending tokens. Their relation is
sampled=committed+pending; generated pending is at most1 for this greedy contract.
Emitted equals all accepted batch token counts and cannot exceed sampled or the
requested output. `output_limit` requires exactly the requested count. A hidden
stop requires one additional sampled token and a `stop_token_id` from the
constructor's admitted set. Failed/cancelled responses may also have un-emitted
sampled tokens and are not treated as successful output-limit replies.

Terminal `session_committed_tokens`, `session_pending_tokens` and
`replay_required` describe numerical state separately. Without replay, session
history must equal the original prompt/history plus newly sampled tokens. Once
any new token is sampled, the prompt must be fully committed: session committed
equals prompt plus generated committed, and session pending equals generated
pending. Unsampled partial prefill may still retain pending prompt tokens.
With explicit replay, a shorter surviving session prefix is allowed; its checked
total cannot exceed prompt plus sampled tokens or context. The consumer cannot
independently verify a producer's carry is
drained or numerical state is truthful. Those are required native tests, not
properties established by a JSON flag.

## Ownership and memory bounds

The caller explicitly chooses stored UTF8 bytes R, at most8MiB and no more than
the mode's decoded-byte bound. Storage initializes once and never grows. Swift's
allocator can reserve more than R; the checker records actual Array capacity,
which must be included in future process admission with allocator headroom.
No allocation uses untrusted context, output or tokenizer maxima directly.

Let B=min(batch UTF8 bound,R). Event input is bounded to6B+2048 bytes, including
worst-case JSON escaping and a bounded envelope. Before Foundation parses it,
a byte scanner limits decoded strings and their aggregate to B+2048, nesting8, containers16, keys64, key
UTF8 bytes64, separators64 and scalar lexemes32 bytes, and rejects repeated keys.
Numeric lexemes must be unsigned digits fitting UInt64; decimals/exponents never
reach a floating conversion that could round away a fraction. Unicode
escapes/surrogate pairs are counted before large String construction;
Foundation still validates the complete grammar and raw UTF8. Delta UTF8 is at
most B; arbitrary arrays/objects cannot expand without a structural bound.

Logical live owners include the response Array capacity, caller's encoded event
Data up to6B+2048, the parsed Foundation strings (allow up to2(B+2048) for UTF16) and bounded
metadata/parser objects. Applying a delta iterates its UTF8 into the existing
buffer without building another byte array. Explicit `text()` materializes a
complete additional String; budget up to2R if bridged through UTF16. Polling must
use the small summary rather than calling it every token. This foundation does
not measure Foundation's scratch, object headers, allocator caches or physical
peak, so this owner accounting is not a hard process-memory admission result.

Saving a response can need up to6R JSON bytes before metadata; the existing
workspace reserves48MiB+16KiB for an8MiB completion and bounds encoded history
at64MiB. Existing decoded history, encoding buffers, returned String and disk
Data may overlap. Integration must account for these together with tokenizer,
native cursor/carry, weights, KV/recurrent state, numerical scratch and conversion
owners. Fallback storage belongs to the future producer; its reported token
bound does not prove its byte allocation or lifecycle. Phone admission and peak
measurements must be established independently before enabling the capability.

## Synthetic checks and remaining evidence

[The standalone checker](../apple/tests/swift/GenerationContractMain.swift) uses
[explicitly synthetic metadata/events](../apple/tests/fixtures/generation-contract-v1/capabilities.json).
It exercises real Swift parsing/storage and rejects invalid schemas, identity,
numeric bounds, event order, counter scope, hidden stops and text limits while
checking atomic rejection. A generated synthetic stream supplies131072 token
counts and4MiB of escaped NUL text in1024 bounded events; no model produces
these events. Storage capacity must remain unchanged and text must reassemble
exactly. This is neither numerical generation nor a physical-device test.

Strict Swift6 builds and standalone execution are recorded separately from
iOS/simulator module compilation. The artifact checks retain failed trials and
bind source/fixture/log hashes. Before production integration, obtain engine
and Astra review, qualify the real native cursor/producer, freeze package/model
identities, run Mac generation/state/admission checks and then physical iPhone
tests with owner-history preservation and the current Qwen3 rollback. Real
early stops, unsupported/over-budget outcomes and actual emitted counts must
be reported directly; synthetic counts cannot satisfy the131072-output request.

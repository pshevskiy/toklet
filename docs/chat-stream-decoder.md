# Bounded Rust response text decoder

This is the L2 text mechanism in the [long-reply plan](long-reply-engine-plan.md).
It is checked against the independent text witnesses and current pinned
tokenizers. The user resumed testing after the recorded interruption; the
simulator check completed in a fresh capture. This text mechanism does not close
the remaining architecture, generation or app plan.

## API and ownership

`Codec::response_stream(ResponseLimits)` borrows an already admitted immutable
codec. `ResponseLimits::new` rejects zero or excessive limits; hard maxima are
131,072 response IDs and8MiB decoded UTF-8. `push` accepts at most128 IDs and
returns only the new text chunk, borrowed until the next call. Consumers can
append that chunk without asking the decoder to rebuild or copy previous text.
The decoder retains no full token history or response. These text-only maxima
grant no numerical context or native/app generation capability.

Call `finish` at ordinary completion, hidden stop or cancellation to flush
pending text. Every failed mutation poisons the stream; construct a fresh stream
and replay the response IDs to recover. A failed push returns no text and rolls
back public counters to the last successfully returned chunk. The private
incomplete state is unusable after failure. A text failure does not mutate any
numerical session, sampled history or committed KV position.

Current native/app Chat and raw Expert limits stay256 and128, respectively;
Chat context stays512 and conversation JSON/whole-prefix decoding stay64KiB.
The old bounded whole-prefix codec and foreign-buffer API are unchanged. The new
Rust API has no C stream owner, cursor, callback queue or app integration yet.

## Decoder mechanisms

- SmolLM2/Qwen ByteLevel maps token characters to bytes, using the whole token's
  original UTF-8 when any character lies outside that alphabet. A four-byte
  scratch array holds at most three incomplete UTF-8 bytes between inputs.
  Valid scalars and irreparable errors are published once; final incomplete
  input produces the same replacement text as complete lossy decoding.
- Mistral preserves Replace→ByteFallback→Fuse→initial-space Strip. It retains a
  whole consecutive fallback-byte run, bounded by one byte per admitted ID.
  A normal token or finish validates that run once. Invalid runs produce one
  replacement character per byte. The decoder can deliberately publish no text
  during a long run: early publication would violate the frozen rewrite witness.
  Skipped controls and absent IDs do not terminate the run.

Special IDs count toward token limits and emit no text. Absent IDs below the
profile vocabulary bound are skipped, as in the original tokenizer; IDs at or
above that bound reject. Raw token text is additionally capped at4096 bytes.
All current pinned vocabulary entries measured at most256 bytes in the completed
Mac text check. New tokenizer profiles need explicit decoder admission.

Profile selection happens once at construction/batch dispatch, outside numerical
kernels. Reserved storage bounds a maximum per-call delta plus the possible
Mistral run; no complete output is stored. Token lookup still allocates a String
per token, and Mistral replacement has temporary ownership too. This is not an
allocation-free or measured-latency claim. The reserved buffers, temporary token,
borrowed codec, consumer-owned output and app/history state must all be included
in later process admission before integration.

## Verification and limits

The independent [L1 witnesses](chat-stream-oracle.md) were committed before this
target implementation. Ordinary Rust tests compare33 frozen cases across four
batch partitions and eight exact131072-ID cases at batch1/128, hashing output
without retaining a whole response. They check stable emitted prefixes, terminal
flush, carry bounds, unchanged reserved capacities, poison/replay requirements,
byte/token/batch limits and unchanged old codec limits. A separate bounded UTF-8
check covers65,536 byte pairs,10,000 deterministic four-byte cases and key invalid
and boundary sequences against standard lossy decoding.

At the stop, captured formatting, strict default/Metal Clippy,191 CPU tests and213
Metal-feature tests passed (15/107 ignored). An explicit release Mac text check
passed all233,585 vocabulary IDs across the three exact original tokenizers,
all33 short and8 long witnesses, and the4096-byte raw-token bound. The device
Rust cross-check also passed. The simulator cross-check was stopped by SIGTERM
at the user's request. After explicit testing resumption, the same unchanged
source passed the simulator cross-check in a fresh capture. The seven earlier
successful commands are inherited only after authenticating their complete
source/input/log bindings. The interrupted command remains a recorded failure.
Owned runner/child processes
and groups were confirmed absent, and captured endpoint/cleanup checks reported
no errors. Raw partial results and the stop record remain in
`artifacts/m8-chat-stream-v1/`; the resumed result is in
`artifacts/m8-chat-stream-v2/`.

Astra source review found no blocker; the
[receipt](../benchmarks/results/2026-10-24-chat-stream-decoder-v1.json)
binds the completed check records and exact scope. No new physical app, native package, numerical generation,
128K response quality, speed or device-memory qualification follows. L3/L4/L5/L6
remain open.

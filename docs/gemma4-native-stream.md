# Pinned Gemma4 append-only text streaming

The dedicated [Gemma4 codec](gemma4-native-codec.md) now exposes a bounded
no-Strip response stream. Existing Smol/Qwen/Qwen3 ByteLevel and Mistral
Replace/ByteFallback/Fuse routes retain their public API and prior behavior.
This is a text decoder; it does not extend numerical context/output limits,
add a model to the app, or execute an LLM for 131072 tokens.

## Ownership and behavior

A small `ResponseStream` facade borrows the pinned tokenizer and owns private
`ResponseState`: resolved vocabulary bound, decoder choice, special IDs, token
and byte counters, UTF-8 carry, fallback run and one delivery buffer. State
cannot escape or be paired with another tokenizer through a public API. The
cold factory selects ByteFallback with leading-space stripping for Mistral and
without stripping for Gemma. Numerical kernels and model/session ownership are
unchanged. Future native cursor integration still needs explicit codec/model
identity, lifetime and admission rules.

`push` returns only a borrowed new chunk, valid until the next mutation. It
retains no complete response ID history or text, and never decodes the full
prefix on each token. Skipped special tokens do not flush a fallback run. A
later invalid byte can change an entire formerly valid consecutive byte run,
so the stream delays that run until a normal token or `finish`. All 131072
fallback IDs may legitimately remain invisible until finish; no per-token
visible-latency guarantee is made. Always finish after stop/cancellation using
only admitted visible IDs; hidden model stop IDs are not delivered as text.

Failed pushes expose no partial new chunk, restore published counters and
poison the state. Terminal/error reuse requires a fresh stream and replay.
The earlier ByteLevel carry and Mistral strip behavior remain checked.

| Owner | Separate logical allowance |
| --- | ---: |
| Pinned codec, tokenizer construction/tables and bounded operations | 1 GiB |
| Gemma stream state/facade, capacities and bounded temporary token text | 4 MiB |
| Explicit probe fixtures, comparisons, serialization and file writer | 16 MiB |

The Gemma factory rejects insufficient stream allowance before allocation and
rechecks actual capacities with checked arithmetic. `allocated_bytes` counts
State/facade metadata plus delta, fallback and special-vector capacities,
excluding the borrowed codec. `id_to_token`/Replace still create temporary
per-token Strings; this is not allocation-free decoding. Caller-retained chunks
need separate memory. The existing text-only ceiling remains 131072 IDs / 8 MiB,
with batches at most 128; complete `Codec::decode` retains its separate 64 KiB
bound. Neither text API grants numerical or app capacity.

## Frozen data and verification

Before decoder changes, a guarded isolated Python extractor authenticates the
upstream text receipt, exact vocabulary and earlier scalar decoder. The
[decode-only fixture](../tests/fixtures/gemma4-stream-v1/manifest.json) preserves
317 actual sparse IDs, all 24 added-token flags and the unchanged decoder
settings; merges are removed, so encoding is not qualified. Its 67 small cases
contain 59 upstream decode-bearing records, the frozen generated reply and seven
scalar-derived edges. Five synthetic 131072-ID patterns include a valid fallback
run followed by one invalid byte across batch boundaries. Extractor resident/
physical observations 370,819,072/281,084,696 bytes fit its separate 1 GiB gate.
Historical input hashes remain bound to the pre-change Git sources. Python
runtime aliases and their resolved targets were both byte-pinned; no upstream
package/tokenizer code is executed by this data extractor.

All 310 CPU/332 Metal-feature tests, formatting, both strict checks, both iOS
targets, 22 release text methods and four release probe fault methods pass.
The explicit legacy real-tokenizer suite also passes 233,585 vocabulary IDs,
33 short and eight long cases across three profiles. The initial compile
attempt's type/test-call errors are retained without test credit. Later
preliminary snapshots preserve the core pass, three fault tests and final four
fault tests separately. Prefix mismatch, expected-decode failure, partial
vocabulary data and publication I/O failure retain actual output and its active
stage while preserving the original error.

The actual Gemma probe checks 268 small case/partition records with 472 retained
prefix chunks, 15 long case/partition records, and all 262,144 isolated vocabulary
IDs using one-token limits. All 7,675,150 binary bytes are retained: complete long
UTF-8 results and ID/length/UTF-8 vocabulary frames. An independent scalar oracle
replays every output against the authenticated full vocabulary. A regression of
the actual verifier and success classification rejects seven corrupt packets;
a rejected comparison cannot set parity true.

Maximum observed state/facade ownership is 2,097,419 bytes; isolated single-ID
ownership is 4,361 bytes. Native process resident/physical peaks are
544,669,696/489,227,032 bytes under 1,094,713,344 combined admitted bytes. This
3.87-second load/verification/I/O observation is not an LLM latency benchmark.
See the [receipt](../benchmarks/results/2026-10-25-gemma4-native-stream-v1.json).

## Remaining gates

The [short formatted model diagnostic](gemma4-basic-chat-generation.md) remains
separate accepted evidence. Broader quality/full G3, earlier precision/source,
GPU intermediate and long-rotary gaps remain unwaived. Resident Metal,
representative performance, phone memory/execution, native ABI/app integration
and 131072 output tokens remain open. No app profile, catalog or numerical
limit changes. Official E2B 131072 total positions cannot hold 131072 output
tokens plus a nonempty prompt.

# Qwen3.5 bounded text codec

Q1.H implements the pinned basic-text boundary for
`avan-ag/Qwen3.5-4B-Uncensored-MLX-4bit`, revision
`fb98fbc59c71a567f3e23435386c9e93951261db`. The numerical graph, sampling,
controller and ordinary model/native app eligibility remain separate. This
prerequisite supplies exact text semantics for later request metrics.

The tokenizer's named EOS is `<|im_end|>` at 248046; the model and generation
configs declare `<|endoftext|>` at 248044. The selected MLX-LM 0.32 tokenizer
wrapper uses their union. Native `stop_ids()` explicitly returns both. Two
unmodified upstream constructor statements execute in the closed reference's
source-slice witness. A separate tokenizer-only MLX-LM 0.32.0/Transformers 5.7.0
corroboration observes the same pair without loading weights; its 619 MB peak
and logs are retained. That corroboration is not the admitted fixture oracle.
No metadata is rewritten and no whole upstream model equivalence is claimed.

Six plain chat cases freeze the original template, including three leading
newlines, assistant-message suffix whitespace and the explicit empty closed
thinking block. The supported subset alternates nonempty user/assistant text,
ends in user, and requires `enable_thinking=false` and
`add_generation_prompt=true`. Typed JSON rejects duplicate/unknown fields,
system/tool/media/reasoning messages, all 33 reserved input literals and C0
controls other than tab/CR/LF. Escaped literals are checked after JSON decoding.
Production executes no Jinja or repository Python.

The exact NFC/ByteLevel tokenizer has 248044 base IDs and 33 added IDs: 21
special and 12 ordinary. Its text vocabulary is contiguous through 248076;
the model is padded to 248320. Decode and stream admission reject padded IDs
explicitly. This grants no logit mask or numerical sampling policy. Raw text
encoding may intentionally contain control spellings; chat rendering has the
stricter typed input policy. Special-token skipping preserves ordinary added
spelling such as `<think>`.

A codec requires a 512 MiB allowance before opening any asset. Five config,
template and tokenizer files use the existing nonblocking, no-follow regular
opener before size/hash checks, so FIFOs and final symlinks reject. Ordinary IO
error kinds are preserved. Assets must be exactly sized and authenticated;
parsing consumes the same captured tokenizer buffer. A cold per-ID table
bounds ByteLevel decode expansion, including three-byte replacements for
invalid UTF-8. Limits are 64 KiB for wire/canonical/rendered/decoded text,
65536 codec IDs and 63 messages. Conservative decode admission may reject a
sequence whose actual output would fit. These are text safety limits, not
qualified numerical context or output capacity. The allowance covers captured
JSON, construction overlap, tokenizer tables, metadata and bounded operations;
it is policy, not an owner-by-owner tokenizer residency measurement.

The existing append-only ByteLevel decoder has a new cold constructor for this
codec. Each stream separately admits 4 MiB; caller-retained output needs its own
budget. Incomplete UTF-8 is delayed, finish flushes it, and any failed mutation
poisons the stream. Replay uses a fresh stream. Whole-batch padding rejection
publishes no partial text and leaves returned counters unchanged. No existing
profile or decoder algorithm changes.

Before native implementation, two owned, externally admitted Jinja2 3.1.6 /
Tokenizers 0.22.2 reference processes produced identical 100-record witnesses.
Their measured lifetime peaks are 295,797,480 / 308,921,112 B. The reference
pins Python sources/native images and checks both runtime/source endpoints;
its EOS source slice has a separate explicit scope. The frozen fixture covers
full rendered text/IDs, NFC, control flags, decode prefixes, incomplete/invalid
byte sequences, bounded-profile rejection and upstream empty padding decode.

The initial native codec passes all text/ID/decode checks and 165 stream
schedules at a 290,079,440 B lifetime peak. Reconstructing it 13 times in one
process then reaches 686,376,232 B and rejects the 512 MiB component process
cap. The original failed receipt remains intact. An idle heap scan reports
zero leaks and about 19 MiB of live allocations; VM snapshots show large empty
allocator regions. Worker/parser/pressure-relief experiments do not meet the
cap and add no mechanisms to the engine.

The qualified diagnostic process profile supplies `MallocLargeCache=0` before
exec, with the other studied allocator settings absent. Apple's pinned
[allocator initialization source](https://github.com/apple-oss-distributions/libmalloc/blob/c49dafa25f1efe8607701ae6014a663ad2ee437f/src/malloc.c#L1837)
reads this setting. Native reports independently echo the effective child
settings. The parent may inherit `MallocNanoZone=0`; the runner records it and
isolates only its owned child's environment. This fixes the interrupted
pre-launch assertion without changing parent, app or library defaults.
The hardened profile replay passes the same 100 records/165 schedules at 260,735,648 B.
Default repeated reconstruction remains unqualified under the component cap.
The 6 GiB engine ceiling and aggregate model/session/application admission
remain separate, unchanged gates.

Two fresh, sequential processes each execute three warmups and ten measured
trials. Filesystem caches are warm; each trial constructs a fresh tokenizer/BPE
cache, encodes a three-turn code prompt and decodes/streams a fixed Unicode
sequence. Stream time includes construction and finalization. Raw samples and
dispersion are retained; no request, speedup or tail claim follows.

| Cohort | Construction (ms) | Encode (µs) | Decode (µs) | Stream (µs) | Process peak (MB) |
| --- | --- | --- | --- | --- | --- |
| 0 | 359.644 | 279.813 | 4.667 | 5.208 | 261.309 |
| 1 | 357.793 | 309.292 | 4.897 | 5.438 | 261.309 |

Seven download-free CPU controls cover typed/admission/identity/byte bounds and
atomic stream failure. Fresh formatting, strict CPU/Metal all-target Clippy and
638 CPU/772 Metal tests pass under the controlled profile (41/315 ignored).
Native56 freshly builds three arm64 slices/46 exports and both C smokes pass.
Earlier native55/test-only reuse remains historical; the final nonblocking-open
fix has fresh checks and a fresh package. Twenty Python tests and 32 resealed
evidence controls pass. Astra approves this codec and restricted diagnostic
profile with no actionable findings;
[checks/review/commit](../artifacts/qwen35-text-codec-2026-11-08/publication-checks.json).

[Contract](../artifacts/qwen35-text-codec-2026-11-08/acceptance-contract.json),
[frozen witness](../artifacts/qwen35-text-codec-2026-11-08/frozen-witness.json),
[profile text fidelity](../artifacts/qwen35-text-codec-2026-11-08/fidelity-hardened-final.json),
[phase baseline](../artifacts/qwen35-text-codec-2026-11-08/baseline-hardened-final.json),
[memory investigation](../artifacts/qwen35-text-codec-2026-11-08/memory-investigation.json),
[retry audit](../artifacts/qwen35-text-codec-2026-11-08/retry-environment-audit.json).

Controller pending/committed state, exact-prefix/replay semantics, request
summaries/overhead, deeper actual contexts, full-request and held-out quality
remain subsequent gates. Neither this codec nor source metadata admits a
managed route. The full selected improvement goal stays active.

# Independent response-decoding witnesses

L1 of the [single-reply128K plan](long-reply-engine-plan.md) freezes text behavior
before implementing incremental decoding. The producer reads only three already
acquired, exact tokenizers. It imports Python tokenizers0.22.2 and no Toklet code;
expected strings come from complete-prefix/full decoding through that binding.
This is independent of the target streaming implementation, not an independently
implemented tokenizer library or a numerical-generation reference.

The [fixture](../tests/fixtures/chat-stream-v1/reference.json) has33 short cases
across the existing SmolLM2, Qwen2.5 and Mistral tokenizer identities. Every token
prefix has exact text, including empty input, skipped controls, split/incomplete
UTF-8, whitespace, code, literal replacement characters and invalid byte runs.
Eight long cases contain exactly131,072 IDs, represented as repeat patterns and
a tail. Each records the complete UTF-8 byte count and SHA256. These are text
inputs, not generated responses or a model-context qualification.

The Mistral decoder's ByteFallback stage treats a consecutive byte-token run as
one UTF-8 sequence. A129-byte ASCII run decodes as129 As; appending a standalone
continuation byte changes the whole130-byte run to130 replacement characters.
SmolLM2/Qwen ByteLevel decoding preserves the ASCII prefix in the same witness.
Consequently append-only stable text cannot be assumed across all admitted
profiles. Literal replacement characters also differ from incomplete byte input;
a strategy that waits whenever text ends in a replacement character needs an
explicit bounded-carry policy. Terminal incomplete input must match full decoding.

Three compact decode-only tokenizer files retain original IDs and decoder
configuration, remove unused vocabulary and merges, and disable encoding
machinery. Every short prefix and complete long output is checked against the
original tokenizer before publication. They are not valid production tokenizers,
chat templates or reduced model assets. Ordinary future Rust tests can use them
without downloads or Apple GPU access. Real-tokenizer regression remains separate.

## Evidence

The [receipt](../benchmarks/results/2026-10-24-chat-stream-oracle-v1.json)
binds producer, tests, original/compact tokenizer identities, runtime and raw
commands. The final rerun pins all tokenizer Python/native package files and
version metadata before import and uses a fresh bytecode-cache root; the earlier
passing run retains its narrower runtime closure. An export to a fresh artifact directory reproduces every frozen byte;
explicit verification against the repository fixture and eight reference tests
pass. Tests independently assert the130-byte rewrite, Unicode/control semantics,
exact long ID/byte/hash counts, bounded pin admission, exclusive publication,
inventory changes and rejected tokenizer policies. Two exploratory producer
assumptions were corrected before freezing: SmolLM2 has no0xFF byte token, so the
shared invalid witness uses standalone0x80; a literal replacement character can
encode to multiple IDs, so long inputs use repeat patterns with exact-count tails.

Retain these witnesses. No target decoder, C ABI, app limit, engine/context,
model download eligibility, quality, latency or physical-device support changes.
L2 implementation and all later long-reply gates remain open.

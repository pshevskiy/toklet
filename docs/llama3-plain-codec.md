# Pinned Llama3 plain text codec

`app_chat::llama3::Codec` is a cold Rust text entry point for
`mlx-community/Llama-3.2-1B-Instruct-8bit` revision
`d48cdf0a4ea22d893b7c63a99d6a693e24822795`. It requires the exact tokenizer,
tokenizer configuration and special-token map. It does not activate native
Chat factories, app discovery or a model session.

The bounded renderer owns one BOS and the fixed system date 26 Jul 2024, accepts
only alternating user/assistant strings ending with user, applies Python-style
trimming, and rejects every reserved tokenizer literal in message content.
Automatic special insertion is disabled. Decoding replaces the whole visible
prefix, skips special IDs and performs no Transformers space cleanup. The
three hidden stops are 128001/128008/128009. Shared 64 KiB / 63-message input, 32768
rendered-token and 256 response-token safety limits remain separate from actual
model/session admission.

Ordinary CPU tests compare rendered bytes with the independent frozen
[Jinja/tokenizers fixture](../tests/fixtures/llama3-plain-text-v1/fixture.json)
and reject malformed/unsupported conversations. An explicit ignored check uses
the cached pinned tokenizer to compare exact IDs, Unicode roundtrips, all 256
reserved inputs, single BOS and all three hidden stops. It downloads nothing
and executes no model.

Native 512-token prompt/reply boundaries, cancellation/recovery, audited owner pairing,
app activation and original full-adoption quality remain subsequent gates.

# Independent Qwen3 plain Chat witness

The selected public candidate is `mlx-community/Qwen3-0.6B-8bit` at revision
`11de96878523501bcaa86104e3c186de07ff9068`. Its existing numerical/reference
qualification is described in [real Mac execution](mlx-qwen3-real-metal.md).
This additional text witness does not enable the app or qualify phone fit.

The [independent producer](../tools/reference/app_chat_qwen3.py) authenticates
exact model configuration, tokenizer and tokenizer-configuration bytes before
using the original template in Jinja2 3.1.6 and Python tokenizers 0.22.2. No
Toklet rendering or numerical code is imported. The runtime app will not execute
remote Jinja.

The [frozen fixture](../tests/fixtures/app-chat-qwen3-v1/reference.json) records
five plain alternating user/assistant conversations: greeting, Unicode multi-turn,
Python code, five turns with preserved newlines, and Chinese/combining/emoji text.
It includes complete rendered strings, complete token IDs and decoded text.
The explicit non-thinking branch has no invented system message and appends
`<|im_start|>assistant\n<think>\n\n</think>\n\n`. A separate thinking-enabled
counterexample ends at the assistant marker, ensuring the modes cannot silently
be confused. Tool, system and reasoning-message inputs are outside this profile.

The original model and tokenizer agree that EOS is `<|im_end|>` (151645).
Thinking delimiters are 151667 and 151668. All 26 added-token literal/ID/special-flag pairs
are frozen for cold input validation. Twelve have `special=false`, including
thinking/tool delimiters, and must remain visible during ordinary decoding.
Input denial must cover all 26 without changing their decoder flags. No new
numerical semantics follow from those identities.

Four independent response-text cases retain every decoded token prefix,
including incomplete Unicode and visible thinking/tool delimiters. Qwen3's original tokenizer normalizes decomposed
`e` plus combining acute to precomposed `é`; the witness preserves that observed
normalization. An initial extra verification assertion incorrectly expected
input/decode byte identity, then the verifier was corrected to check the captured
normalization. Neither the producer nor captured fixture was changed to hide it.

Two original-template capture/replay invocations match exactly. The supplemental
stdlib checks validate role endings, bounded IDs, no-system/non-thinking markers,
EOS/thinking IDs, literal uniqueness and final decoded prefixes. Reference
identity is recorded in the accompanying receipt. Target codec/C bridge,
immutable native package, UI download/open, memory and iPhone tests follow as
separate steps; no fourth model is runnable in the installed app yet.

## Guarded execution and provenance

The first unguarded capture/check JSON is retained as preliminary and unaccepted.
Its too-strong byte-identity assertion is reproduced as an explicitly expected
nonzero command with retained stderr; the original fixture values stay intact.
The final fixture adds original special flags, one delimiter decoding case and
complete runtime identities while preserving all previous conversation and
Unicode-case values.

The [owned capture driver](../tools/experiments/qwen3_chat_reference.py) admits
exact producer, helper, Python executable/shared-library and Jinja2/MarkupSafe/
tokenizers package file identities before external imports. It uses fresh caches,
exclusive asset copies, external manifest hashes, owned process groups, separate
stdout/stderr, joined cleanup and endpoint checks. Attempts are recorded before
the helper call; both streams and endpoints are captured independently in finally,
including helper exceptions. Returned owner records and primary, capture and
endpoint errors remain separate. Eleven portable negative and
positive runtime checks and four ownership-failure checks pass without importing third-party libraries. Ordinary
unconfigured discovery skips those explicit runtime tests; no model download is
required by ordinary CPU tests. Capture and replay both pass with zero cleanup
or endpoint errors. The reference exports text only; no new numerical engine,
model quality, memory budget, app integration or phone qualification is claimed.

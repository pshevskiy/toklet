# LFM2 text reference

B4e2b freezes the tokenizer/chat contract for the pinned
`mlx-community/LFM2-350M-4bit` revision
`18dc72abf3b2337f9123cfd6eeeb58dfa7947066`. Eight portable controls and the
explicit codec export/replay pass. This is a reference fixture; LFM2 app/Chat
admission and numerical model execution remain later tasks.
The [fixture](../tests/fixtures/lfm2-text-v1/README.md) and
[result record](../benchmarks/results/2026-10-29-lfm2-text-reference-v1.json)
preserve exact bytes and evidence pins. Astra audits all saved records/runtime
identities and independently reconstructs 29 decoded strings from 382 ID
occurrences using the original vocabulary, without importing a tokenizer.
Saved audit SHA256: `73a6507499ed47a517825ded9e38ceb073b1b8f9ae57bc92f715def24ab601f1`.

The manual renderer starts with `<|startoftext|>`, preserves each message's
content, and appends `<|im_start|>role\ncontent<|im_end|>\n`. The optional
generation prefix is `<|im_start|>assistant\n`. It applies no trimming,
normalization, default system message or extra EOS. Ten fixed cases compare
manual bytes with the separately pinned 209-byte Jinja template, then retain
actual encoded IDs and whole-prefix decoded strings. This independently checks
rendering, rather than implementing a second BPE tokenizer.

| Fixed message sequence | Tokens without generation prefix | Tokens with generation prefix |
| --- | ---: | ---: |
| Empty user | 6 | 9 |
| Plain user | 10 | 13 |
| System then user | 22 | 25 |
| User/assistant/user history | 25 | 28 |
| Unicode and newline | 16 | 19 |

All ten observed decodes equal their rendered text; the contract records the
actual strings and does not assume this for arbitrary input. Evidence decoding
uses Tokenizers' raw ByteLevel decoder with `skip_special_tokens=False` and no
Transformers cleanup. Transformers is not executed or claimed equivalent.

The bounded manual schema accepts exactly string `role`/`content` fields, an
optional first system message, then alternating user/assistant turns. Empty
content and assistant-ending conversations are accepted. Duplicate JSON keys,
unknown fields, wrong roles/types/order, system-only/empty lists and reserved
special-token literals reject. Limits are 64 messages, 64 KiB input JSON and
rendered UTF-8, and 512 codec IDs. These are text-boundary limits, not a claim
of a 512-token numerical trajectory.

Template text encodes with `add_special_tokens=False`, preserving BOS1 exactly
once. A diagnostic postprocessor run proves that enabling it would insert a
second BOS. Plain text prepends `[1]` to encoding without special postprocessing.
The stop union is exactly `{7}`; token 2 remains non-stop. The 64,400 mapped IDs
are distinct from the 65,536 numerical head slots: IDs 64400–65535 fail explicitly
before decode. No padded-head masking or remapping is introduced.

The authenticated metadata has 505 reserved special literals and two ordinary
added entries, `Mathias` and `python`. All 505 injection checks reject, while
both ordinary entries encode/decode with their actual mapped IDs. Controls also
cover accepted/rejected message and token-count boundaries, whitespace/Unicode,
strict JSON and both generation flags.

The stable 23,078-byte fixture and 123,852-byte raw capture replay identically;
each export retains all 539 records. Fresh cache prefixes, module/native origins
and resource observations are separate per-run files. The owner pins 1,996
inputs, including its 1,971-file text runtime, and runs three CPU jobs with a
120 s deadline each, 1 GiB observed ceiling, 1 MiB stdout/stderr caps and bounded
process-group teardown. OS maxRSS is 41,730,048/187,367,424/195,379,200 bytes for
controls/export/replay. These are codec-worker measurements, not whole-app fit
or phone results; system shared-cache paths are observed rather than hashed.

All six repository checks pass: formatting, strict default/Metal all-target
Clippy, default 450 and Metal 545 tests, and the release Metal test build.
Counts overlap; 20/181 checks remain explicitly ignored. Rust/Metal numerical
code, existing fixtures and comparison gates are unchanged. The full real LFM2
reference and native/session qualification remain B4e2c/B4e3.

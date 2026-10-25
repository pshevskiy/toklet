# Pinned Gemma4 tokenizer and basic-text witness

Two fresh captures now freeze text behavior for the
[immutable E2B checkpoint](gemma4-checkpoint.md), before implementing its native
text profile. This runs the pinned upstream tokenizer and exact authenticated
Jinja template, with no template loader or repository Python execution. Toklet
model inference, app profiles and append-only response streaming do not run.

## Admitted subset and source policy

The first subset accepts a nonempty odd list of at most63 alternating
user/assistant string messages, ending with a user. The UTF-8 JSON request is
bounded to64KiB and every content string must remain nonempty after trimming.
Only `enable_thinking=false, add_generation_prompt=true` is admitted. System,
developer, tool, reasoning, multimodal/content-array and assistant-continuation
options are rejected. All24 special literals are rejected inside content, as
are C0 controls below U+0020 other than tab, carriage return and newline.

The exact plain generation prefix is:

```text
<bos><|turn>user
Hi<turn|>
<|turn>model
```

It encodes to `[2,105,2364,107,10979,106,107,105,4368,107]`.
Each historical assistant uses the role label `model`; trimmed turns end with
`<turn|>\n`. Six complete conversations cover whitespace, Unicode/combining
characters, a literal `▁`, multiple turns and Unicode trimming. They contain
10,10,15,10,24,22 tokens respectively. Counts describe the witness; they grant
no capacity beyond the existing32-position numerical limit.

The actual selected generation config and converted wrapper both declare
stops `[1,106,50]`: `<eos>`, `<turn|>`, `<|tool_response>`. The nested text config
names only1. The official1/106 policy remains distinct provenance in the
[inspection](gemma4-e2b-inspection.md). This witness freezes the selected
artifact's policy explicitly. Its declared default stochastic sampling is not
qualified by the existing explicit greedy engine checks.

## Normalization and decoding

The tokenizer has262144 BPE entries,514906 merges and24 special added tokens.
Normalization replaces spaces with `▁`; decoding applies Replace, ByteFallback
and Fuse, with no initial-space Strip. A literal `▁` consequently decodes as a
space. Both `add_special_tokens` settings produce identical captured IDs: the
postprocessor is identity, so the formatter supplies the single explicit BOS.
All24 control literals resolve to their exact declared IDs and disappear when
special-token decoding is skipped.

Eighteen byte-fallback variants cover valid two/three/four-byte UTF-8, incomplete
and invalid runs, skipped special IDs inside a run, and normal-token boundaries.
For example, `E2 82 AC` decodes to `€`, but adding `FF` changes the whole four-byte
run to four replacement characters. An incomplete `E2 82` produces two. Complete
prefix decodes at checkpoints1/3/128 are retained, including those rewrites.
They do not qualify append-only streaming; a native stream must retain a whole
fallback run until a boundary or explicit finish.

Three upstream macro diagnostics separately capture trimming/thinking-channel
behavior. U+001C/U+001F are trimmed by Python and differ from Rust's ordinary
Unicode trimming; the first basic subset rejects those controls. Thinking
markers remain rejected content. These observations do not admit advanced
template features.

## Evidence and limitations

The committed [fixture](../tests/fixtures/gemma4-text-codec-v1/manifest.json)
contains all97 records/36,960 bytes: one structure, six chats,33 rejections,
11 raw texts,24 controls,18 fallback variants, one stop policy and three macro
diagnostics. Every JSONL span has its own offset, length and SHA256. Both cold
captures are byte-identical. A separate scalar vocabulary decoder independently
replays Replace/ByteFallback/Fuse and every prefix; manually assembled turn
layout matches all rendered prompt bytes. The upstream tokenizer uses the same
tokenizers0.22.2 release as the Rust dependency; this is an integration witness,
not a separately implemented BPE oracle.

Cold admission pins1,970 runtime files, including Python source/native files,
tokenizers0.22.2, Jinja2 3.1.6 and MarkupSafe3.0.4. Source/native loader types are
resolved before package imports, actual module/native-image origins are checked
afterward, and MarkupSafe's native extension is required explicitly. Fresh cache
prefixes, disabled user-site/bytecode writes and endpoint identity checks apply
to each run. The separate1GiB reference-process gate passes: maximum resident
memory is649,592,832/671,023,104 bytes and peak physical footprint is
380,700,976/391,661,968 bytes. Combined observations2.10/1.94 seconds are not
tokenizer performance benchmarks or engine/device memory admission.

Seven offline methods pass, covering strict input rejection, typed raw capture,
short writes/flush failures, cold hash/size/version/import rejection, loader
shadowing, interrupted production/endpoint failures and raw-read/publication
failures with restored signal handlers. The first attempt failed because its
test used an unresolved Darwin temporary-path alias; it remains failed. The
corrected five-method attempt and final expanded seven-method attempt remain
separate. Rust sources/tests/examples are unchanged from `3b2a300`; no unrelated
engine builds are rerun. See the [receipt](../benchmarks/results/2026-10-25-gemma4-text-codec-reference-v1.json).

Native text-profile parity, formatted model continuation, append-only streaming,
full G3, original-BF16/source and existing numerical gaps, resident Metal,
quality, representative performance, phone fit, app integration and131072
output tokens remain open. No ABI, catalog, family or position-limit change.

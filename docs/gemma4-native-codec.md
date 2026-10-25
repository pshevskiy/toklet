# Dedicated native Gemma4 basic-text codec

The new [codec module](../src/app_chat/gemma4.rs) implements the
[frozen basic-text witness](gemma4-text-codec-reference.md) outside numerical
kernels. It owns its tokenizer separately from model weights and session state.
One module declaration connects it to the application-text namespace; existing
`Profile`, CLI, ABI, catalog and app eligibility remain unchanged.

## Admission and behavior

Loading requires a conservative1GiB text allowance before opening any file.
Four small pinned assets select the exact config/tokenizer/template/stop policy
before the32,169,626-byte tokenizer is read. Hashing and tokenization parse the
same captured buffer. The allowance includes tokenizer construction overlapping
that buffer, persistent token/merge tables, metadata, a precomputed u16
token-length table and bounded operation buffers. Returned responses/history
retained by the caller need additional application admission. This is separate
from numerical model/session and device fit.

The renderer accepts only typed, bounded alternating user/assistant string
messages and explicit non-thinking generation options. Direct serde parsing
rejects unknown and duplicate fields. It rejects all24 reserved literals after
JSON unescaping and C0 controls below U+0020 except tab/CR/LF; DEL is outside that
set. This excludes the Python/Rust C0 trimming discrepancy before ordinary Rust
trimming. It checks the witness's canonical spaced UTF-8 message size and exact
rendered size before reserving the output String. No Jinja executes natively.

Raw encoding intentionally permits reserved literals for diagnostic parity.
Raw input, rendered prompts and decoded text each have separate64KiB byte
bounds. Complete decoding first validates every ID and admits a conservative
byte sum from the cold token-length table, before asking the tokenizer to
allocate output. Replace cannot expand ordinary token spellings; a six-byte
fallback spelling produces at most three replacement bytes. Skipped controls
contribute zero. This upper-bound policy can reject a request whose eventual
decoded text would be smaller. The262144-ID codec cap is a text diagnostic
bound, not numerical context or output support.

Selected-artifact stops1/106/50 are exposed explicitly. They do not change
generation defaults or automatically configure an app session. Thinking-channel
macro diagnostics stay excluded; there is no production thinking stripper.
Complete-prefix decoding can rewrite a byte-fallback run, so append-only
streaming remains a separate task.

## Checks and actual Mac results

Five new offline methods authenticate the fixture manifest/JSONL, match all six
frozen rendered prompts and33 rejections, exercise typed duplicate/escaped
controls, UTF-8/message/canonical-byte limits, admission before file access and
decode preflight. Inclusive64KiB upper-bound and262144 skipped-ID success cases
pass, with one additional byte/ID rejected. No model download is needed for
ordinary tests. Formatting, strict default/Metal checks,307 CPU/329 Metal-feature
tests, both iOS compile targets, the release example and five release codec
methods pass. The16/109 ignored hardware/model cases remain explicit. The first
strict check rejected one style issue before any tests ran; its failed attempt
is preserved separately from the corrected passing run.

The explicit [real-codec probe](../examples/gemma4_codec.rs) loads the actual
pinned tokenizer once. All97 observed records match, including285 prefix
decodes, both encode modes, exact prompt bytes/IDs, all control literals,
normalization, fallback rewrites, selected stops and unsupported diagnostic
exclusions. Actual per-record values are flushed before comparison. The
controller retains partial JSONL even if final JSON is absent. Three mutation
regressions execute its actual verification/flag block and prove that complete
97-row inventories with changed prompt, prefix or diagnostic fields cannot
publish a passing parity flag.

On MacBookPro18,3 / Apple M1 Pro /16GiB, macOS26.6.2 build25G83, maximum resident
memory is515,948,544 bytes and peak physical footprint488,866,560 bytes, below
the separate1GiB codec gate. The1.73-second combined load/97-record/output
observation is not a performance benchmark. Its23,632-byte actual JSONL and exact
executed release binary are bound by the
[receipt](../benchmarks/results/2026-10-25-gemma4-native-codec-v1.json).

Formatted model continuation, append-only streaming, full G3, original-BF16/
source qualification and earlier numerical gaps, resident Metal, quality,
representative performance, combined engine/application admission, phone fit,
app integration and131072 output tokens remain open. Numerical execution stays
at32 positions. This codec alone grants none of those capabilities.

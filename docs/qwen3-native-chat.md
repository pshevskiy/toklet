# Qwen3 cold Chat profile

The exact `mlx-community/Qwen3-0.6B-8bit` revision
`11de96878523501bcaa86104e3c186de07ff9068` now has an isolated, checked native
text profile. This extends the already qualified numerical geometry; it adds no
kernels, backend selection, model geometry or context capacity. The installed
phone app is unchanged until a separately pinned package and app pass admission.

The [independent text witness](qwen3-chat-reference.md) freezes the original
non-thinking template, complete IDs, Unicode prefixes and added-token flags.
The Rust profile checks exact model/tokenizer/configuration identities, renders
that branch without an invented system message and stops only at EOS 151645.
All 26 reserved added-token literals are rejected in user input. The twelve
ordinary added tokens retain their original visible decoding behavior; reasoning
and tool message inputs remain unsupported. Streaming uses the existing bounded
ByteLevel decoder. Tokenization and formatting remain outside numerical kernels.

The caller must reserve 256 MiB of logical text allowance separately from
numerical buffers. This is a conservative reservation, not a measured physical
peak. The new C API test checks allowance-minus-one rejection, exact-allowance
creation, stop queries, ownership after releasing the model handle, eight real
generated tokens and a forced hidden first-token stop. No C declarations or ABI
changed; the header comment lists the additional profile and allowance.

The [native receipt](../benchmarks/results/2026-10-25-qwen3-native-chat-v1.json)
binds the isolated base, exact changed sources, accepted independent reference,
all command records, raw logs and input/capture pins. Both ordinary Rust test
configurations, strict Clippy configurations, formatting and both iOS compile
targets pass. All four real tokenizer regressions, four public native Chat
trials and the five existing frozen Qwen3 GPU numerical/state/admission tests
pass on Apple M1 Pro. The latter retain their original gates and explicit
512-position boundary, cached/pending/reset behavior and callback recovery.

These isolated runners did not pre/post bind the integration-test sources and
the complete included fixture/helper closure. Their numerical results retain
that evidence limitation. A later comparison found all 3,316 original test,
fixture and example files byte-identical to the archived base; that comparison
is retrospective and does not replace execution-time guards. The subsequent
[shared-tree integration collection](../benchmarks/results/2026-10-25-qwen3-native-chat-integration-v1.json)
is authoritative: it binds the complete Rust, test, example, fixture and header
inventories, all imported project helpers, all four model inventories and the
external Qwen3 numerical array payload before and after every owned command.
All 266 CPU and 288 Metal-feature ordinary tests, strict checks, both iOS targets,
four codecs, four native Chat trials and five frozen Qwen3 GPU tests pass. Two
sets of actual CLI binaries are retained; seven spawning harnesses identify the
current target executable, and the disabled default MLX harness is proved empty.
All owned processes/groups were joined, with no cleanup/capture/endpoint errors.

The first release run found an old ignored Mistral test asserting rejection at
129 decoded tokens, although the existing runtime cap was already 256. Its
failed command, source snapshot and logs are preserved. The test now checks
acceptance at `MAX_RESPONSE_TOKENS` and rejection at that constant plus one.
No runtime cap, algorithm or numerical tolerance changed. All checks were then
rerun against the final source and header identities.

This native task does not qualify whole-app memory, a new Apple package, iPhone
download/open/generation, full held-out quality, or performance. Those app/device
checks follow separately. Chat output remains 256 tokens, Expert raw output
128, and admitted numerical context 512 positions. Gemma 4 and a real 131,072-token
single reply remain separate engine work; neither is enabled by this profile.

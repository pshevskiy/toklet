# Bounded 256-token Chat output

The native text decoder and shared app worker now support up to 256 visible Chat
output IDs. `TOKLET_MAX_CHAT_OUTPUT_TOKENS` and
`NativeSettings.maximumChatOutputTokens` expose the bound to app callers. Existing
requests through 128 tokens keep their behavior. The Swift raw/Expert commands
still admit at most 128 output IDs per command; their existing multi-command
response collector is unchanged.

Total Chat context stays 512 tokens for the current app profiles. Prompt plus
requested output must fit before reset or generation starts. A 256-token reply
therefore leaves at most 256 prompt tokens. Natural stop IDs can end a reply
sooner. Decoded text still has a 64 KiB bound; no tokenizer, template, stop policy,
model geometry, numerical kernel or C record layout changes.

IDs and timing arrays reserve the larger bounded Chat capacity, and lifecycle
traces accept that visible prefix. Their application storage remains separate
from numerical admission. `NativeSnapshot.emitted` reports the actual completed
C callback count, reset to zero when generation never ran. The app worker's
sampled history, committed KV and pending token remain separate.

A defaulted diagnostic `autoCancelAfter` on the Chat command permits precise
callback-boundary checks; normal callers leave it zero. Create, callbacks,
decode and owner cleanup keep the existing inference-thread contract. The
private Chat path selects the larger bound once per request; raw requests cannot
select it.

## Evidence

Portable codec and C-boundary tests check exact 128/129/256 decoding, special and
invalid IDs, 257 rejection, count queries and transactional short buffers. They
use an existing tiny tokenizer fixture and need no download or GPU.

The independently rendered [long input](../tests/fixtures/chat-long-v1/reference.json)
uses the pinned Qwen tokenizer and original template. Its 65 IDs are frozen
before the long target run. The selected real Qwen eight-bit model on M1 Pro
emits all 256 requested IDs: sampled history 321, committed 320, pending one.
The saved prefix's first 65 IDs match the independent fixture exactly.

Deterministic cancellation after emitted IDs 128 and 129 leaves history
193/194, committed 192/193 and one pending token. Each visible prefix matches the
uninterrupted result. Oversized output, prompt+output overflow and raw diagnostic
requests above their existing limits reject without changing the entire saved
history. Fresh Chat replay matches the full original output and state exactly.
These are functional/state witnesses, not a long-context numerical oracle,
task-quality score or speed measurement.

Formatting, both strict all-target Clippy configurations, 180 CPU and 200
Metal-feature tests pass (14/101 ignored in the ordinary suites). The explicit
runtime checks pass 89 portable checker methods, C compile/link/caller, strict
Swift long Chat and metadata probes, CPU/Half-Metal/MLX worker regressions and
strict iOS/simulator modules. A fresh unsigned native package builds and links
all three Apple slices, retaining 23 public symbols each.

Short SmolLM2, Qwen and Mistral Chat regressions also pass; complete replies and
flags match the preceding worker receipt exactly. Immutable source snapshots
separate these checks from the parallel mobile UI changes. The
[output-bound receipt](../benchmarks/results/2026-10-24-chat-output-v1.json)
records exact sources, identities and raw captures. Previously admitted Cargo
outputs are preserved by using a fresh target directory.

Retain this bounded Chat extension. The separate mobile UI task owns the 256
option, persisted settings and its app builds. Physical iPhone behavior at the
new bound remains unqualified. Larger context, general downloaded text profiles,
runtime planning, receipt-based compatibility and the remaining engine inventory
stay open.

# M8 capability and admission audit

**6 November 2026 update. Apple Silicon Mac has seven managed profiles and
iOS has four. M8 remains open for the full public Gemma BF16 transfer and
context controls, plus the deferred physical iPhone 14 Pro checks.**
The official BF16 path passed corrected numerical and original-quality gates.
Its Mac offline managed process peaked at 4.93 GB. [Coverage](model-coverage.md).

The [coverage table](model-coverage.md) is the current artifact/route/device
index. Match it against `ModelHub.swift` and `NativeAcceptedIdentity.swift`,
using exact model and codec identities. The catalog contains Qwen2.5 0.5B,
Qwen3, SmolLM2, Mistral and three Mac-only profiles, Llama3, Qwen2.5 1.5B
and official Gemma 4 E2B BF16.
The five earlier profiles have production fresh download/Stop/offline evidence.
The sixth has a linked Mac product and isolated controller evidence.
Official Gemma has a signed Mac build, verified cached publication and
network-denied Chat. [Gemma Hub evidence](../benchmarks/results/2026-11-06-gemma-bf16-mac-hub-v1.json).
[Production journeys](mac-hub-journeys.md).
[Qwen2.5 1.5B controller](../benchmarks/results/2026-11-06-qwen15-mac-app-v1.json).

| Decision | Current scope / evidence |
| --- | --- |
| Qwen2.5 managed | Mac growth/Chat through1024; model declaration32768 remains separate. Initial KV512, chosen upper512/1024. |
| Qwen2.5 1.5B managed, Mac only | Original10k loss and100-task gates pass. Reference attention up to512, verified download, Apply256 and network-denied controller restart. |
| Qwen3 conditional managed | Mac512, explicit experimental accuracy notice. Original-relative full quality passes; separate C512 fidelity gate remains rejected. |
| SmolLM2 and Mistral managed | Mac512 routes, identity-bound preferences and fresh/offline journeys. Mistral remains separately memory-admitted. |
| Llama3 managed, Mac only | Original10k NLL and100-task gates pass; Mac512 managed adoption/download/offline evidence is complete. No131072-window or iPhone claim. |
| Phi3, Gemma3, LFM2 diagnostic | Bounded adapter/block/state/reference qualifications exist. Original full adoption quality and ordinary app admission do not; they stay out of the catalog. |
| Gemma4 Q4 rejected for adoption | Original/native full collections and grading complete. NLL passes, tasks52→43/100 fail. Historical mathematical-logit/KV failures remain; four stock-MLX decisions agree with Toklet but cannot waive those failures. |
| Official Gemma 4 E2B BF16 managed, Mac only | Corrected numerical and original10k/100 quality gates pass. Shared512 window, exact six assets, cached publication and offline Chat pass. Full public transfer timed out; retry remains open. |
| Gemma4 alternative weights | OptiQ passes stock trajectories and fails the task limit at49/100 vs original52/100. Uniform Q6 and oQ6 reject on a proposal cycle. Separate stock-only Q8 screen proceeds under a measured 6GiB gate. Native/app Q8 still exceeds its estimated allowance. |
| Physical iPhone14Pro outstanding | Four-profile catalog is not new-device evidence. Current Settings, journeys, wider context and diagnostic adapters lack a fresh phone run; Llama3 is excluded. |

[Context/KV controls](context-app-integration.md) calculate capacity from the
actual backend/layout and preserve model-declared versus loaded/selected limits.
Mac Settings Preview/Apply, physical KV ceiling, authenticated preferences,
owner reporting and conditional load-peak reporting have separate checked
receipts indexed by [coverage](model-coverage.md). No selectable quantized KV
format or full model-declared window is implied. Reply limits and prompt/history
share the selected context; displayed positions are not tokens per second.

This audit fixes stale completion bookkeeping: Mistral C6's three children and
Llama's original-quality children are complete; Gemma collection/grading is
complete with a rejected acceptance gate. It does not mark rejected numerical
checks passed or silently remove them from the checklist. Diagnostic full-quality
work is a prerequisite for future adoption, not a new ordinary support claim.

The 5 November audit used550 default/658 Metal tests, formatting, strict
all-target Clippy for both configurations and release library test build in
`4467b86`. Subsequent work changes reference/benchmark tooling and documentation;
its task-specific checks and Astra reviews are recorded per commit. There is no
new engine binary or performance qualification from the audit. Subsequent
M9 decisions and their measurements are indexed separately. [M9 plan](next-performance-milestone.md).

Astra approves the publication/admission reconciliation, while identifying one
remaining desktop acceptance gap: a fresh production Mac app load→send/reply
journey needs externally observed process memory, exact bundle/profile/config
and budget. Existing helper/conditional load peaks and controller-only journeys
do not close it. Prepare an isolated run preserving the user workspace. The
read-only host device inventory also shows the intended14Pro disconnected.

Catalog/identity agreement, six approved result flags,303 local links, unchanged
Rust inputs and whitespace pass. This publishes decisions without declaring
M8 complete. [Audit evidence](../benchmarks/results/2026-11-05-m8-admission-audit-v1.json).

Update: the [production Mac Qwen2 witness](mac-production-memory.md) closes the
representative whole-app requirement after Astra review. Other profiles’
production peaks remain unmeasured; no phone or new Gemma admission follows.

Subsequent scoped work includes [Granite3.1 MoE Mac512](granite-moe-execution.md)
and the [M9 Mac baseline](m9-mac-baseline.md). The MoE test covers pinned
engine decisions, state and process memory. Broader numerical qualification,
catalog adoption and deferred phone verification remain open.

The 6 November review verified all six catalog and identity pairs, the selected
41-export package and current rejection receipts. The package was built before
the later test-only Gemma capture edits. Rebuild for the next source-matched
Apple package validation. The remaining Gemma and phone gates keep the
aggregate TODO open.
[Current audit evidence](../benchmarks/results/2026-11-06-m8-mac-audit-v1.json).

The later source-matched refresh selects a new 41-export package after the
chunk-pool change. It passes an offline Mac controller run and both iOS SDK
builds. [Refresh evidence](../benchmarks/results/2026-11-06-chunk-pools-native41-v1.json).

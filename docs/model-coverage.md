# Exact model coverage

Coverage is artifact-, backend-, route- and device-specific. The ordinary Mac
catalog has six pinned profiles. iOS has four. Catalog membership selects a
managed route, while consumed-byte model/codec identity and fresh admission
still authorize loading. Other Hugging Face results remain discoverable without
becoming executable. Model declarations are shared input/history/output windows,
not tokens per second or a promise that the engine can use that whole window.

| Pinned artifact (revision prefix) | Backend / route | Declared positions | Qualified or tested scope | Ordinary app |
| --- | --- | ---: | --- | --- |
| Qwen2.5-0.5B-Instruct-8bit (`6474d4c`) | Native Metal dense, managed | 32,768 | Mac load/session growth and rendered Chat through1,024; larger values require separate qualification | Mac/iOS catalog; next-load upper512 or1,024, initial KV512 |
| Qwen2.5-1.5B-Instruct-8bit (`c7e1ff2`) | Native Metal dense, managed Reference | 32,768 | Mac512/chunks1,32, original loss and task gates, fresh download and network-denied Chat controller | Mac catalog, up to512, 3GiB process allowance |
| Qwen3-0.6B-8bit (`11de968`) | Native Metal dense with Q/K norm, managed | 40,960 | Mac512 route; original relative loss/task gates accepted, but separate C512 numerical fidelity gate remains rejected | Mac/iOS catalog,512; experimental accuracy label |
| SmolLM2-135M-Instruct-8bit (`0f0d9b8`) | Native Metal dense Llama-family, managed | 8,192 | Mac512, fresh download and network-denied offline Chat; authenticated preferences | Mac/iOS catalog,512 |
| Mistral-7B-Instruct-v0.3-4bit (`a4b8f87`) | Native Metal dense, managed | 32,768 | Recorded bounded Mac route and authenticated preferences; no family-wide/window claim | Mac/iOS catalog,512; independent memory admission |
| Llama-3.2-1B-Instruct-8bit (`d48cdf0`) | Native Metal scaled RoPE, managed | 131,072 | Original10k-target/100-task quality passes; diagnostic loss1,025/task566; actual Mac512 UI and fresh/offline journey | Default Mac catalog,512; excluded from iOS |
| SmolLM3-3B-4bit (`d3a7e05`) | Native Metal dense with per-layer RoPE bypass, diagnostic | 65,536 | Mac512/chunks1,32; seven stock greedy decisions, state/reset/overflow and process-memory checks | No managed Chat/catalog adoption |
| granite-3.3-2b-instruct-8bit (`37b9855`) | Native Metal dense with checkpoint scales, diagnostic | 131,072 | Mac512/chunks1,32; stock decisions, full-cache/reset/overflow and process-memory checks | No managed Chat/catalog adoption |
| Phi-3-mini-4k-instruct-4bit (`5b3819e`) | Native Metal fused-source dense, diagnostic | 4,096 | Selected Mac512 state/schedule and reference checks | No ordinary catalog/Chat adoption |
| gemma-3-270m-it-4bit (`ff1143e`) | Native Metal local/global Gemma3, diagnostic | 32,768 | Selected capacity544, absolute positions0–543; bounded native conformance policy; historical precision rejections retained | No ordinary catalog/Chat adoption |
| LFM2-350M-4bit (`18dc72a`) | Native Metal attention/causal-convolution, diagnostic | 128,000 |64 trajectory positions; capacity512 admitted separately | No ordinary catalog/Chat adoption |
| gemma-4-e2b-it-4bit (`2387675`) | Packed Metal projections with CPU shared hybrid graph, diagnostic | 131,072 | 512 selected-logit evidence;1,025 state/wrap passes, mathematical-logit/stagedKV qualification rejected; historical numerical failures remain; full task gate rejected52→43/100 | No ordinary catalog/Chat adoption |

The full revisions, layouts, gates and numerical policies are in the artifact
reports: [Qwen2 context](context-app-integration.md),
[Qwen3](mlx-qwen3-real-metal.md), [Qwen3 full relative quality](qwen3-full-quality.md), [Smol transfer](smol-download-offline.md),
[Mistral identity/preferences](mistral-managed-preferences.md),
[Llama adoption](llama3-mac-adoption.md), [Phi3](phi3-real-model.md),
[Gemma3](gemma3-real-native.md), [Gemma3 conformance](gemma3-native-conformance.md),
[LFM2](lfm2-real-native.md), [Gemma4 hybrid](gemma-hybrid-real.md), [Gemma512 boundary](gemma-hybrid512.md),
[Gemma1,025 admission](gemma-context1025-probe.md), [Gemma1,025 diagnostics](gemma-hybrid1025.md).
This table indexes those scopes; it does not expand them.

[Current M8 admission audit](m8-admission-audit.md) records all managed, conditional
and diagnostic decisions. Gemma4 Q4 remains rejected by its
[completed task gate](gemma-full-tasks.md); the
[four-decision stock witness](gemma-mlx-decisions.md) does not waive it.
Gemma E2B8-bit is excluded by the [current-owner memory estimate](gemma8-feasibility.md),
with no payload or runtime qualification.

Mac Settings separates model-declared, loaded and selected limits. Capacity,
chunk, reply and numerical budget are saved under authenticated artifact/backend/
load/text/planning identities; Apply rechecks fresh admission and preserves old
state and drafts on rejection. The physical KV-byte ceiling is intentionally
per-Apply. KV storage remains the current implemented format, with no implied
quantization choice. Chat output remains bounded to256 tokens and existing text,
byte and event limits apply. Prompt plus reply must fit the selected shared
window; a configured reply maximum is not a separate model window.

Memory reports count resident weights, physical KV owners, scratch and bridge/
application allowances. Selected-plan admission, allocated session owners and
observed process memory are distinct. A load-only peak can be inferred only if
that interval establishes a new lifetime high-water mark; otherwise it remains
unknown. A numerical fit is not a reservation or a whole-device guarantee.
[Context/settings](../benchmarks/results/2026-11-03-mac-settings-rendered-v1.json),
[owner reporting](../benchmarks/results/2026-11-02-session-owner-memory-v1.json),
[load peak](../benchmarks/results/2026-11-03-load-memory-checkpoints-v1.json).

Physical iPhone14Pro is currently unavailable. Earlier device receipts keep their
original scope; the new Settings, Llama, wider context and diagnostic adapters
have no new phone qualification. Mac compilation or a different Pro Max cannot
substitute for it. The five earlier Mac profiles have fresh production
transfer/Stop/offline evidence. [Journey report](mac-hub-journeys.md).
The sixth profile has isolated Mac controller evidence and a linked Mac product.
[Qwen2.5-1.5B evidence](../benchmarks/results/2026-11-06-qwen15-mac-app-v1.json).

The [representative production Mac Qwen2 journey](mac-production-memory.md)
measures lifetimeRSS816,709,632bytes and greatest observed pre-exit physical
high-water728,090,064bytes below its1.5GiB profile allowance at512/chunk32/
reply128. The Qwen2.5-1.5B controller's fresh-download journey measured peak
RSS2,007,515,136bytes and sampled physical high-water1,884,473,456bytes.

The initial [Qwen2.5-1.5B execution](qwen15-engine-execution.md) adds a second
real dense geometry. The adopted profile also passes the
[10,000-target loss](../benchmarks/results/2026-11-06-qwen15-full-loss-v1.json)
and [100-task quality](../benchmarks/results/2026-11-06-qwen15-task-quality-v1.json)
gates against the matched original FP32 reference.

[SmolLM3 real execution](smollm3-engine-execution.md) adds the validated per-layer
rotary mask to the shared dense blocks. Stock BF16/native Half row differences
remain descriptive; full fidelity and held-out quality remain unqualified.

Additional diagnostic: [Granite3.3 execution](granite33-engine-execution.md).

[Granite3.1 sparse-MoE execution](granite-moe-execution.md) verifies the pinned
original BF16 checkpoint converted to resident Half on the M1 Pro. Its two
frozen decisions and final-layer selected sets match stock. Native capacity
512, chunk 1/32, overflow and reset pass. This is an engine diagnostic.
Wider context, held-out quality and managed Chat remain open.

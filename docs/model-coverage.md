# Exact model coverage

Coverage is artifact-, backend-, route- and device-specific. The ordinary Mac
catalog has five pinned profiles; iOS has four. Catalog membership selects a
managed route, while consumed-byte model/codec identity and fresh admission
still authorize loading. Other Hugging Face results remain discoverable without
becoming executable. Model declarations are shared input/history/output windows,
not tokens per second or a promise that the engine can use that whole window.

| Pinned artifact (revision prefix) | Backend / route | Declared positions | Qualified or tested scope | Ordinary app |
| --- | --- | ---: | --- | --- |
| Qwen2.5-0.5B-Instruct-8bit (`6474d4c`) | Native Metal dense, managed | 32,768 | Mac load/session growth and rendered Chat through1,024; larger values require separate qualification | Mac/iOS catalog; next-load upper512 or1,024, initial KV512 |
| Qwen3-0.6B-8bit (`11de968`) | Native Metal dense with Q/K norm, managed | 40,960 | Mac512 route; original relative loss/task gates accepted, but separate C512 numerical fidelity gate remains rejected | Mac/iOS catalog,512; experimental accuracy label |
| SmolLM2-135M-Instruct-8bit (`0f0d9b8`) | Native Metal dense Llama-family, managed | 8,192 | Mac512, fresh download and network-denied offline Chat; authenticated preferences | Mac/iOS catalog,512 |
| Mistral-7B-Instruct-v0.3-4bit (`a4b8f87`) | Native Metal dense, managed | 32,768 | Recorded bounded Mac route and authenticated preferences; no family-wide/window claim | Mac/iOS catalog,512; independent memory admission |
| Llama-3.2-1B-Instruct-8bit (`d48cdf0`) | Native Metal scaled RoPE, managed | 131,072 | Original10k-target/100-task quality passes; diagnostic loss1,025/task566; actual Mac512 UI and fresh/offline journey | Default Mac catalog,512; excluded from iOS |
| Phi-3-mini-4k-instruct-4bit (`5b3819e`) | Native Metal fused-source dense, diagnostic | 4,096 | Selected Mac512 state/schedule and reference checks | No ordinary catalog/Chat adoption |
| gemma-3-270m-it-4bit (`ff1143e`) | Native Metal local/global Gemma3, diagnostic | 32,768 | Selected capacity544, absolute positions0–543; bounded native conformance policy; historical precision rejections retained | No ordinary catalog/Chat adoption |
| LFM2-350M-4bit (`18dc72a`) | Native Metal attention/causal-convolution, diagnostic | 128,000 |64 trajectory positions; capacity512 admitted separately | No ordinary catalog/Chat adoption |
| gemma-4-e2b-it-4bit (`2387675`) | Packed Metal projections with CPU shared hybrid graph, diagnostic | 131,072 | Short graph/generation only; implemented32-position boundary still requires qualification; original precision/quality gaps retained | No ordinary catalog/Chat adoption |

The full revisions, layouts, gates and numerical policies are in the artifact
reports: [Qwen2 context](context-app-integration.md),
[Qwen3](mlx-qwen3-real-metal.md), [Qwen3 full relative quality](qwen3-full-quality.md), [Smol transfer](smol-download-offline.md),
[Mistral identity/preferences](mistral-managed-preferences.md),
[Llama adoption](llama3-mac-adoption.md), [Phi3](phi3-real-model.md),
[Gemma3](gemma3-real-native.md), [Gemma3 conformance](gemma3-native-conformance.md),
[LFM2](lfm2-real-native.md), [Gemma4 hybrid](gemma-hybrid-real.md).
This table indexes those scopes; it does not expand them.

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
substitute for it. Fresh production transfer/offline evidence is directly linked
for Smol and Llama; remaining per-profile journeys must be reconciled separately.

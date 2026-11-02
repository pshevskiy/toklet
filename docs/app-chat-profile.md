# Toklet application text profile

The user requested the friendly InferencePhone interface in Toklet Harness,
with the existing raw-token interface retained as Expert mode. InferencePhone
belongs to the neighboring `llm_inference` project and has its own runtime and
mobile restrictions. This implementation uses Toklet's engine and one worker;
the older app is a visual reference, not a second backend.

The first text profile supports the already acquired
[SmolLM2-135M eight-bit snapshot](mlx-smollm-checkpoint.md). It validates exact
tokenizer JSON and tokenizer-configuration SHA256 identities, including the
original chat template and special IDs 0/1/2. This is tokenizer/template
compatibility with the admitted model geometry, not verification of every
checkpoint payload or a new quality claim. Other tokenizers reject clearly.
No downloaded Jinja is executed by the app.

The independent [reference](../tests/fixtures/app-chat-v1/reference.json) was
captured with Python tokenizers0.22.2 and Jinja2 3.1.6 before target text changes.
Its two cases include a32-token user turn and a68-token multi-turn Unicode
conversation. [The producer](../tools/reference/app_chat.py) reproduces exact
rendered text, IDs and complete decoded text from the pinned local assets.
Ordinary tests need no download. Real-tokenizer and Mac GPU C-bridge checks
are explicit opt-ins.

`app_chat` stays outside numerical kernels and the CLI keeps its existing
formatting behavior. The additive C text handle is immutable and thread-affine.
It loads from the model's frozen absolute originating directory, owns no model
borrow and can outlive model release. Query calls return required counts;
undersized buffers copy no elements. JSON is limited to64KiB/63 alternating
user/assistant messages. The original encoded-input limit was512 IDs; the
[current codec ceiling](../benchmarks/results/2026-11-02-rendered-codec-ceiling-v1.json)
is32768, with actual rendered prompt plus reply still admitted against the
loaded model and active session before generation. The later
[bounded output extension](chat-output-limit.md) raises visible Chat output
from128 to256 IDs; the original receipt below retains its original128-ID scope.
Chat control delimiters in message content reject instead of creating roles.
Decoding always replaces the complete visible prefix, avoiding concatenation
of separately decoded byte fragments. Swift must independently reserve output
within the admitted numerical session capacity before changing session state.

The application reserves64MiB for the exact tokenizer/container and bounded
text buffers in addition to numerical admission. This is a conservative logical
allowance, not a measured process peak or an allocator-enforced physical cap.
Retaining the originating path increases the native bridge reservation from
4KiB to20KiB; its absolute path is bounded to16KiB and exact owned capacity.
Existing numerical kernels, precision, layouts and CLI behavior are unchanged.

Worker integration, the default interface and actual phone text trials follow
as separate reviewed tasks. Hugging Face search/downloads remain M8.4 work.

The [check receipt](../benchmarks/results/2026-10-23-app-chat-profile-v1.json)
binds the final source, reference, exact command logs and retained source copies.
Rust1.90 formatting, strict all-target Metal Clippy, ordinary CPU/Metal suites,
explicit local tokenizer/C-handle checks, independent Python replay and the
actual Mac GPU public create/query/decode/release check pass. Protected CPU
binaries remain unchanged. A preliminary log-wrapper path error is observed in
tool output; only its formatting logs are retained, with no preliminary test
credit. Subsequent complete runs are retained separately. Astra accepts this
bounded text-profile task after source and evidence review.

## Qwen text profile extension — 23 October 2026

The native text boundary additionally supports the already qualified
[Qwen2.5-0.5B eight-bit snapshot](mlx8-checkpoint.md). A separate independent
[Python/Jinja export](../tools/reference/app_chat_qwen.py) freezes the original
plain conversation branch, including Qwen’s default system message, complete
IDs and decoded Unicode text. The app never executes downloaded Jinja or enables
tools. Alternating text-only user/assistant messages retain the existing bounds.

Text creation now checks the exact model configuration SHA as well as its loaded
geometry and exact tokenizer/config identities. Qwen reserves a separate 128 MiB
logical text allowance; SmolLM2 retains 64 MiB. These are conservative admission
reservations, not measured physical peaks. A new immutable C stop-ID query returns
`151643,151645` for Qwen and `0,2` for SmolLM2, so the app can select stops outside
numerical generation. Existing raw-ID APIs keep explicit caller stop semantics.
Control-token strings in user/assistant content reject before encoding.

This native task does not enable Qwen in the Swift Models screen by itself. That
integration, updated native package and actual phone Chat checks follow in a
separate reviewed task. Broader Mistral execution and text support also remain
separate. No new answer-quality or latency claim follows from this extension.

The [Qwen text receipt](../benchmarks/results/2026-10-23-qwen-app-chat-profile-v1.json)
retains final Rust formatting/strict Clippy, ordinary CPU/Metal suites, explicit
local codecs/C bridge generation and independent Python replay. Earlier exports
and check versions remain bound through exact source snapshots.

## Mistral text profile — 23 October 2026

The selected [Mistral snapshot](mlx-mistral-checkpoint.md) now has a native text
profile. Its exact model configuration, tokenizer and tokenizer configuration
are pinned independently of the numerical graph. The larger configuration has a
256 KiB input bound; the tokenizer has a 4 MiB bound. The separate logical text
allowance is 64 MiB. This reservation is not a measured physical peak.

The original `default` template renders one BOS, user instruction blocks and
assistant text followed by EOS. It adds no system message or trailing assistant
marker. The [independent fixture](../tests/fixtures/app-chat-mistral-v1/reference.json)
checks greeting and Unicode multi-turn rendering, complete IDs and decoding.
All 771 added-special literals from the exact tokenizer are rejected in message
content after JSON decoding, including tool/control markers and escaped forms.
The only generation stop ID is **2**. No downloaded template is executed.

Formatting, strict Clippy, CPU/Metal baselines, all three local tokenizer profiles
and independent Python template replay pass. The first sandboxed native trial
returned model-load errors for all three GPU models; its failed receipt and logs
are retained. The unchanged source passes all four native text tests with Mac
GPU access. Mistral returns `[23325,29576,2370,1309,1083,2084,1136,3922]`, independently
decoded as **“Hello! How can I help you today”**. Qwen retains token 0 and its
previous exact greeting. Allowance rejection, ownership after model release,
full-prefix decoding and hidden forced stops pass.

The [text receipt](../benchmarks/results/2026-10-23-mistral-app-chat-profile-v1.json)
binds final sources, successful and failed diagnostics, independent decoding and
unchanged protected binaries. This task changes no numerical kernel or ABI
layout. App catalog/package integration follows separately. Performance,
held-out quality, physical text peaks and phone Chat remain unqualified.

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
user/assistant messages, encoded input to512 IDs, and visible output to128 IDs.
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

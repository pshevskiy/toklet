# Bounded Mistral reference preparation

The selected public candidate is `mlx-community/Mistral-7B-Instruct-v0.3-4bit`
at immutable revision `a4b8f870474b0eb527f466a03fbc187830d271f5`.
Its configuration describes 32 layers, hidden4096/intermediate14336, 32 query
heads mapped to eight KV heads of dimension128, vocabulary32768, epsilon1e-5
and ordinary split-half rotary theta1e6. `sliding_window:null` selects full
causal attention. This task does not implement sliding attention or admit other
Mistral configurations.

The [independent exporter](../tools/reference/mlx_mistral_real.py) validates the
exact743-tensor inventory and reconstructs low-nibble-first four-bit/group64
weights with F16 scale/bias metadata through FP64 into FP32. A matrix view expands
at most1,024 requested rows, internally decoding128-row stripes. Embedding
lookups gather only requested rows and round once to Half. The untied output
head retains original reconstructed FP32 values for projection. No full FP32
model or embedding table is retained. The [graph](../tools/reference/mistral_fp16.py)
uses the existing independent NumPy primitives, Half activations/KV and FP32
scores/logits, with separate FP32 and FP64 accumulation runs.

Six authored cases cover lengths1/3/4/17, future-token changes, offsets0/5 and
layers0/1/31. Four greedy and three scoring distributions make349 planned arrays.
The numerical formula remains
`max(1e-4, 2*pooled_FP32_FP64_max_abs, 2*equivalent_prefix_max_abs) + 1e-3*abs(reference)`.
The eventual measured envelopes and external manifest SHA must be frozen before
native target edits or evaluation. They do not replace held-out quality checks.

Nine portable tests pass, including literal packing, bounded row/gather behavior,
actual projection across the1,024-row boundary in both precisions against scalar
nibble reconstruction, an independent untied-head toy graph, exact349-stage
schema and gate mutations, external SHA, descriptor substitution, truncation,
nonfinite data and exclusive publication. The logical reservation stays below
the6GiB Mac budget. Lifetime-footprint observations reject an exceeded budget
after each case/greedy/scoring pair and before publication; they are observed
acceptance checks, not an operating-system allocation cap. Actual export and
physical peak remain pending until the complete checkpoint is admitted.

The separate [chat exporter](../tools/reference/app_chat_mistral.py) reproduces
the original pinned `default` template from the tokenizer configuration's named
list. It renders `<s> [INST] Hello! [/INST]` to
`[1,29473,3,23325,29576,29473,4]`. Unicode multi-turn formatting is also frozen in
the [fixture](../tests/fixtures/app-chat-mistral-v1/reference.json). This is
independent tokenizer evidence, without native chat support or tool-template
execution.

The [preparation receipt](../benchmarks/results/2026-10-23-mistral-reference-tools-v1.json)
binds the portable checks, exact source copies and verified small metadata.
Astra cleared the source after the added row-boundary and verifier mutation
coverage. The4,077,480,215-byte weight transfer and numerical export remain
separate pending steps; no engine, download UI, latency, quality or device
qualification is claimed here.

During acquisition, a reused ignored inspection directory mixed an older
TinyMistral candidate's metadata with a new7B inventory. That inspection is
marked unusable; the collision and a separately named fresh metadata response
are retained. No model or accepted test result was overwritten. An initial
serial transfer ended with curl18 after178,536,295 bytes, published no model,
and its five owned process groups were subsequently observed absent. The fresh
bounded range retry retains its own independent logs and final admission.

iPhone work is paused at the user's request because the phone is temporarily
off. Further implementation and qualification in this session target Mac.
Qwen's pending physical Chat check remains pending.

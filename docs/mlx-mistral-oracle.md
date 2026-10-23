# Independent Mistral full-attention reference

The selected public candidate is `mlx-community/Mistral-7B-Instruct-v0.3-4bit`
at immutable revision `a4b8f870474b0eb527f466a03fbc187830d271f5`.
Its configuration describes 32 layers, hidden size 4096 and intermediate size 14336, 32 query
heads mapped to eight KV heads of dimension 128, vocabulary 32768, epsilon 1e-5
and ordinary split-half rotary theta 1e6. `sliding_window:null` selects full
causal attention. This task does not implement sliding attention or admit other
Mistral configurations.

The [independent exporter](../tools/reference/mlx_mistral_real.py) validates the
exact 743-tensor inventory and reconstructs low-nibble-first four-bit/group64
weights with F16 scale/bias metadata through FP64 into FP32. A matrix view expands
at most 1,024 requested rows, internally decoding 128-row stripes. Embedding
lookups gather only requested rows and round once to Half. The untied output
head retains original reconstructed FP32 values for projection. No full FP32
model or embedding table is retained. The [graph](../tools/reference/mistral_fp16.py)
uses the existing independent NumPy primitives, Half activations/KV and FP32
scores/logits, with separate FP32 and FP64 accumulation runs.

Six authored cases cover lengths 1/3/4/17, future-token changes, offsets 0/5 and
layers 0/1/31. Four greedy and three scoring distributions make 349 arrays.
The numerical formula remains
`max(1e-4, 2*pooled_FP32_FP64_max_abs, 2*equivalent_prefix_max_abs) + 1e-3*abs(reference)`.
The completed measured envelopes and external manifest SHA are frozen before
native target edits or evaluation. They do not replace held-out quality checks.

Nine portable tests pass, including literal packing, bounded row/gather behavior,
actual projection across the 1,024-row boundary in both precisions against scalar
nibble reconstruction, an independent untied-head toy graph, exact 349-stage
schema and gate mutations, external SHA, descriptor substitution, truncation,
nonfinite data and exclusive publication. The logical reservation stays below
the 6 GiB Mac budget. Lifetime-footprint observations reject an exceeded budget
after each case/greedy/scoring pair and before publication; they are observed
acceptance checks, not an operating-system allocation cap. The completed export
and its separate charged-memory observations are recorded below; logical admission also includes the clean source mapping.

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
coverage. The tools receipt records the earlier preparation-only step. The
[complete checkpoint](mlx-mistral-checkpoint.md) and completed CPU export below
remain separate from engine, download UI, latency, quality or device qualification.

During acquisition, a reused ignored inspection directory mixed an older
TinyMistral candidate's metadata with a new 7B inventory. That inspection is
marked unusable; the collision and a separately named fresh metadata response
are retained. No model or accepted test result was overwritten. An initial
serial transfer ended with curl18 after 178,536,295 bytes, published no model,
and its five owned process groups were subsequently observed absent. The fresh
bounded range retry retains its own independent logs and final admission.

iPhone work is paused at the user's request because the phone is temporarily
off. Further implementation and qualification in this session target Mac.
Qwen's pending physical Chat check remains pending.

## Completed independent CPU export

The complete reviewed checkpoint produces **349 arrays / 35,025,920 bytes**.
The [frozen manifest](../tests/fixtures/mlx-mistral-real-v1/manifest.json) has SHA256
`b6b59b1b763aea83d8f75da33c40989162b842020493a0526790167fc6e71e1d`;
the arrays have SHA256
`3c0d0c7d968e4e69940f7fff8abc7f7a51c23cf17d667463214426c771cac5bb`.
An external identity file is published before any native Mistral target edit or
evaluation, followed by independent stdlib-only verification without NumPy.
Both owned commands pass with unchanged source/package/data endpoints and no
cleanup or diagnostic-capture errors.

The six cases, four greedy distributions and three scoring distributions retain
the original policy. Maximum frozen absolute allowance is **0.5**; the logits
allowance is **0.08046722412109375**, each combined with relative 1e-3.
FP32 and FP64 choose the same four IDs, `[781,781,29505,29510]`, under the
FP64 teacher-forced history. Their FP64 top-two margins are approximately
`[0.201707,3.973089,2.122173,0.918957]`. These are raw-ID numerical probes, not
instruction-following or held-out quality evidence.

Logical admission is **5,486,766,359 bytes**, including the complete clean
read-only weight mapping and bounded matrix/BLAS, trace, framework and tokenizer
reservations. Observed lifetime charged physical peak is **293,752,696 bytes**.
The charged counter can omit clean mapped pages and does not replace logical
working-set accounting or establish a phone limit. No full FP32 model is
retained. The [export receipt](../benchmarks/results/2026-10-23-mistral-real-oracle-v1.json)
binds the admitted checkpoint, frozen manifest/arrays, exact producers/packages,
source copies, protected CPU binaries and command diagnostics. Native Mistral,
app Chat, quality and performance remain separate pending tasks.

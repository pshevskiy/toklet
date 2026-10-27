# Qwen3 native arithmetic qualification

On27 October2026 the human approved Astra's recommendation: qualify a specified
native arithmetic profile through independent implementation/state conformance
and the original complete relative-quality policy. Preserve strict CPU-trajectory
fidelity as a separately reported result. Make technical decisions with Astra
and continue autonomously.

The frozen [v1 contract](../benchmarks/qwen3-native-profile-v1/contract.json)
selects `mlx-community/Qwen3-0.6B-8bit` revision
`11de96878523501bcaa86104e3c186de07ff9068`: MlxAffine projection, Compensated
normalization, Reference attention, Half activations/KV and FP32 logits on the
Apple M1 Pro Mac. CompensatedReference remains an opt-in local experiment.

## Fixed meaning and acceptance

Affine reconstruction/accumulation, compensated normalization, FP32 rotary and
transcendentals, ordered FP32 attention, explicit FP32/Half stores, causal GQA,
tokenization and stop/accounting semantics remain specified by the existing
implementation and independent fixtures. This contract introduces no numerical
kernel, error-derived envelope, new tolerance or near-tie exception.

- Run unchanged independent affine/projection/gather, tied/untied miniature graph
  and real short-checkpoint gates against the current build.
- Add exact analytical controls at Qwen3 RMS widths128/1024 and Reference
  attention16:8/head128 through512 keys, including masking and guards.
- After review/freeze, run all eight reserved inputs from three texts at
  chunk32/128. Require finite complete rows, exact distribution/greedy identity
  across schedules and same-source causal prefixes, exact history/commit/pending,
  cached reuse/replay and atomic capacity rejection. These are composition/state
  checks, not an independent whole-graph accuracy oracle.
- Replay the current native worker on every original10000 loss target and100
  tasks, at diagnostic capacity1025 without clipping. Exact per-target
  distribution hashes/bits/losses and deterministic token/stop/accounting records
  must match the authenticated historical native run to reuse its baseline and
  grades. Only named timing/resource observations are excluded from identity.
  A mismatch requires separate investigation/scoring and isolated grading of
  changed code outputs; no generated code runs on the Mac.
- Preserve the original mean-loss increase≤0.05 nats/token and task-score
  loss≤2 percentage points, current memory accounting/6GiB process bound and
  complete resource ownership/cleanup.

The execution manifest freezes sources, compiler, current binaries, fixtures,
exact analytical expectations and comparison fields before execution. Supported
qualification modes are chunk32/128 and the explicitly checked FullLogits and
GreedyToken paths; no blanket claim covers every accepted chunk configuration.

## Separate statuses

Native-profile qualification is **in progress**. There is no default or capacity
change from adopting this target. The historical FP32-stage and uniform0.05
FP64-log-probability contracts remain **rejected**, including the later compensated
attention experiment. Their values, references and decisions are preserved.

C512 is partial support for the declared40960-position model. Full model-window
and independent iPhone qualification remain open. Historical strict task scores
are3/100 versus5/100, exactly the permitted relative regression; no strong
instruction-following or absolute-quality claim follows. A changed candidate
cannot inherit those results without the complete current-output bridge.

See the [active queue](qwen3-mac-delivery-plan.md),
[complete historical quality](qwen3-full-quality.md),
[fidelity rejection](qwen3-compensated-output-screen.md) and
[improvement record](improvements.md).

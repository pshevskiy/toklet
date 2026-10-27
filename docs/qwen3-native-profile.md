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
The eight-case complete row comparisons use FullLogits at32/128. GreedyToken has
the independent selection controls and current tiny/real readback checks, with
the earlier bounded C512 app journeys. This is compositional coverage, not an
exhaustive new cross-product of every readback, chunk and context configuration.

## Separate statuses

Native-profile qualification is **complete at C512 on the named M1 Pro Mac** for
the tested chunk32/128 and FullLogits/GreedyToken paths. There is no default or
capacity change from this decision. The historical FP32-stage and uniform0.05
FP64-log-probability contracts remain **rejected**, including the later compensated
attention experiment. Their values, references and decisions are preserved.

C512 is partial support for the declared40960-position model. Full model-window
and independent iPhone qualification remain open. Historical strict task scores
are3/100 versus5/100, exactly the permitted relative regression; no strong
instruction-following or absolute-quality claim follows. A changed candidate
cannot inherit those results without the complete current-output bridge.

## Completed qualification — 28 October2026

All18688 analytical Half outputs match their exact expectations, with unchanged
guards/status. The affine/gather, tied/untied miniature graphs, real short graph,
pending/readback and1025 lifecycle checks pass. All eight reserved inputs pass:
2175 rows per schedule,4350 row hashes, five nested prefixes,16 cached/replay checks
and32 exact atomic capacity rejections. These remain composition/state evidence,
not a new independent whole-graph trajectory oracle.

The complete current-native10000-target loss journal is byte-identical to the
historical native run. All5640 task events match after removing only the five
explicit timing fields and readiness resource observations: every task token,
stop, count and deterministic field agrees. The unchanged baseline and exact
graded programs/outcomes therefore apply to the current binary without executing
generated code again. Mean loss remains3.252649075779492 versus3.248645013938721;
increase0.004004061840770845 passes0.05. Tasks remain3/100 versus5/100, exactly
the two-percentage-point limit.

Formatting, strict default/Metal Clippy,369 default CPU tests,243 Metal-feature
library tests and six portable continuity controls pass; the counts overlap and
must not be added as unique tests. Eleven explicit GPU tests pass. All282 frozen
bindings,73 evidence pins and nine reader endpoints verify, and20 successful owned
process groups are absent afterward. Astra independently reviews the contract,
code, all analytical outputs, all validation journals and full quality continuity.

| Current process | Peak physical bytes | Maximum RSS bytes |
| --- | ---: | ---: |
|1025 lifecycle|1126778152|898793472|
|Eight-case validation|1052427560|1059946496|
|Complete native loss|1033094368|1039745024|
|100 native tasks|864945280|873529344|
|Continuity reader|41976360|51658752|

Native processes fit6GiB and the reader fits512MiB. These are completed lifetime
observations, not throughput qualification. The frozen contract's pending flags
describe its pre-execution state; this completed decision supplies the native
qualification. Historical fidelity and full40960 statuses remain separate.

An initial release test build unexpectedly replaced the protected historical CLI;
its guard stopped before GPU/model execution and the original exact bytes were
restored. Successful builds use an isolated target. An extra CPU suite's first
memory observer was blocked by the sandbox after all tests passed; one identical
permitted retry supplies the resource measurement. Both failed observations are
retained. No numerical candidate, fixture or threshold was adjusted to pass.

See [completed evidence](../benchmarks/results/2026-10-28-qwen3-native-profile-v1.json)
and the [next full-window tasks](qwen3-full-window-plan.md).

See the [active queue](qwen3-mac-delivery-plan.md),
[complete historical quality](qwen3-full-quality.md),
[fidelity rejection](qwen3-compensated-output-screen.md) and
[improvement record](improvements.md).

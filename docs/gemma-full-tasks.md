# Gemma paired task collection and rejected Q4 quality gate

Both roles now have100 ordered, admitted records under the frozen prompt,
reserve, stop and pending-token contract. The original BF16 run verifies every
sampled ID and hidden stop against stateless replay, with approved pilot reuse
at0/17/40/70. Six cases require one deterministic suffix repair each: indices
17/18/19/20/33/82. The last is `lambada/row-12`. Rejected proposals remain
rejected; the accepted trajectories pass the unchanged exact gate.

The original source read chain has27,435 reads/4,876,720,198 bytes, independently
replayed from the recorded model-input-call journal. Source/runtime endpoint
hashes and clean owned exits pass. PeakRSS is5,166,071,808bytes and physical
peak4,987,457,832, below6GiB. Native100 uses the existing generation API and
its approved three-case reuse. Both journals pass exact identity/cap/stop/state
admission. Astra independently approves the complete collections for reuse.

The first isolated grading window stops at baseline HumanEval/39 after40 code
records. Its valid readiness frame is followed by exit137 without a terminal
result, OOM, output limit or independent wall watchdog. The4.191-second elapsed
time does not identify process CPU consumption or the signal's cause. The
classifier leaves this case **unresolved** and the outer supervisor publishes
no score. Both VM restoration and lifetime-owner completion pass.

Preserve that first window. The reviewed [process-CPU witness](gemma-cpu-limit-witness.md)
then permits a fresh ordinary-first window: only unknown137 receives one
identical observed retry. HumanEval/39 is the sole retry, with4,000,417,166ns of
process PROF and matching final SIGKILL. Its original attempt remains retained;
the new window resolves that case as a qualified retry resource failure.

All200 outcomes resolve and replay independently:80 primary code attempts,
one observed rerun and120 text outcomes. VM restoration, lifetime-owner
completion and outer process-group absence pass. Astra approves the evidence
and **rejects this Q4 candidate under the frozen task-quality gate**.

| Category | Original BF16 | Native Q4 |
| --- | ---: | ---: |
| Code | 33/40 | 27/40 |
| Prose | 19/30 | 16/30 |
| Continuation | 0/30 | 0/30 |
| Total | 52/100 | 43/100 |

Twelve original-only and three native-only wins give a9-point loss against the
unchanged2-point limit. The passing [10k NLL gate](gemma-full-loss.md) does not
override this rejection. Historical numerical rejections, ordinary Mac app
admission and phone qualification remain open.

Next, freeze a same-checkpoint stock-MLX decision witness: first original-only
and native-only win in each affected category, namely HumanEval/1, HumanEval/24,
truthfulqa/row-0 and truthfulqa/row-3. At their first sampled-ID disagreement,
score the identical shared prefix with stock strict-loaded MLX, using the
original prefill followed by agreed single-token steps. Record candidate
logits/greedy/margin/dtype/hash under6GiB. This distinguishes a concrete engine
disagreement from differences also present outside Toklet; four decisions alone
cannot prove quantization caused the full regression. No regrading, tolerance
change or8-bit implementation follows before this bounded experiment.
[Collection and failure evidence](../benchmarks/results/2026-11-04-gemma-full-task-collection-v1.json).
[Completed grading evidence](../benchmarks/results/2026-11-04-gemma-task-quality-v1.json).

# Full matched Half task quality and native grading windows

Both explicit Half configurations meet the unchanged frozen100-task precision-loss
gate against their unquantized FP32 references. Small Half scores2/100 against2/100;
larger Half scores11/100 against its new matched larger FP32 reference11/100. Both
lose0percentage points, within the predeclared2pp limit. Original100 IDs/order,
40 code/30 prose/30 continuation cases, prompts,256/32/16 caps, greedy/EOS policy,
literal source construction, text rules and equal-weight aggregation remain fixed.
No model, quantizer, task, quality gate, source arithmetic or default changes.

| Checkpoint/configuration | Code /40 | Prose /30 | Continuation /30 | Total /100 | Drop against FP32 |
| --- | ---: | ---: | ---: | ---: | ---: |
|Small FP32, frozen Tiled reference|0|1|1|2|—|
|Small HalfCompensated + Original RMS|0|1|1|2|0pp|
|Larger FP32, bounded raw reference|0|10|1|11|—|
|Larger HalfCompensated + Original RMS|0|10|1|11|0pp|

Absolute scores are low. All40 code cases per configuration resolve as model failures;
small has40 SyntaxErrors; each larger configuration has39 SyntaxErrors and one
AssertionError. Literal completions are not repaired. No compatibility failure is counted as a model
answer: all300 outcomes are resolved, with zero infrastructure-unresolved cases.
The small comparison reuses the existing accepted frozen FP32 result; larger now has
its complete matched reference. This validates the declared precision-loss policy
on the frozen suites, not broader model coding quality or an external speed ranking.

The separately reviewed backend-aware window executes three sequential real native
attempts only after all model workers naturally exit and fresh PID/group absence,
exact collection/input/binary identities and pinned literal/runtime receipts pass.
The qualified stopped machine, arm64 Linux image and explicit API binding are used.
Each window grades100 outcomes and40 literal code programs in the unchanged qualified
controller/classifier/isolation, removes owned containers, verifies an empty live
container inventory, restores stopped state and finishes owned helper pipes. All
three succeed with no failure/restoration errors. A fresh final machine inventory
also confirms stopped/not-starting. The earlier sandbox lockfile read failure changed
no VM state; the authorized native windows use the required machine access.

Saved evidence is independently replayed: every literal program/completion hash and
all120 classifier results, text outcomes and100-case aggregates match reconstruction
from the admitted full responses. Collection/raw hashes remain the admitted values;
current runner/window/executor sources and completed restoration identities match.
No generated program runs on the host. The original dataset programs remain in the
ignored frozen payload; published evidence retains binding hashes and literal results.

Combined with [full10,000-target Half loss](2026-10-14-half-full-loss.md), both selected
HalfCompensated/Original configurations now pass both unchanged frozen quality gates.
The Original Half consistency failures and both Q4 v1 full-loss rejections remain
recorded; passing Half cannot waive the explicit Q4 exit. CPU/scalar stays default.
Controlled/application performance and sustained final-configuration evidence remain
outstanding; no speed, energy or performance release adoption follows from scoring.

Source parent `3784280`; no source change accompanies this data task. The reviewed
worker/collector/adapter checks remain applicable; full native admission/classifier/
aggregate replay, fresh process/machine checks, content/link and whitespace checks
are the task-specific validation. The [qualification receipt](../../benchmarks/results/2026-10-14-half-task-quality-qualification-v1.json)
binds exact original native records and byte-identical published copies:

- [Small Half grading](../../benchmarks/results/2026-10-14-backend-task-small-half-tasks-full-v1-probe.json)
  and [window](../../benchmarks/results/2026-10-14-backend-task-small-half-tasks-full-v1-window.json).
- [Larger Half grading](../../benchmarks/results/2026-10-14-backend-task-larger-half-tasks-full-v1-probe.json)
  and [window](../../benchmarks/results/2026-10-14-backend-task-larger-half-tasks-full-v1-window.json).
- [Larger FP32 grading](../../benchmarks/results/2026-10-14-backend-task-larger-cpu-task-reference-full-v1-probe.json)
  and [window](../../benchmarks/results/2026-10-14-backend-task-larger-cpu-task-reference-full-v1-window.json).
- [Small FP32 reference](../../benchmarks/results/2026-10-11-held-out-task-tiled-grading-v1-probe.json).

Original ignored raw paths remain on this host, so internal native receipts keep their
original paths and hashes. Published copies preserve exact bytes rather than rewriting
native receipts. M5 remains open with the unchanged Q4 quality blocker and remaining
performance/sustained gates.

# Full scalar held-out task grading, 11 October 2026

The completed [100-response scalar collection](2026-10-11-held-out-task-scalar.md)
has now been graded under the frozen literal task protocol. All 100 outcomes are
resolved: **2 passed out of 100 (2%)**. Code is 0/40, prose 1/30 and continuation
1/30. This is the scalar baseline for the fixed equal-weight comparison, not a
general capability estimate or acceptance of a candidate. Scalar remains the
default; tiled task responses and their separately graded comparison remain open.

The [checked integration](2026-10-11-held-out-task-grading.md) ran from clean
revision `ca0edff28af1646b12fa1a4c01b955dd1c90fb68`. The
[launch receipt](../../benchmarks/results/2026-10-11-held-out-task-scalar-grading-v1-launch.json)
and [exact launcher](../../benchmarks/results/2026-10-11-held-out-task-scalar-grading-v1-launch.py.txt)
bind the completed collection, natural-exit idle proof and passed code checks.
No model process, test suite or build ran during grading. The original held
orchestration remained held.

Each of the forty HumanEval programs used the source prefix, exact finalized
model completion, source tests and literal `check(entry_point)` without trimming
fences, repairing indentation, adding signatures or replacing a completion.
All forty recorded failures are `SyntaxError` under that literal composition.
The protected prose scorer requires the whole trimmed uppercase option letter;
the continuation scorer requires the exact first word, preserving case and
punctuation. The low score is retained under those frozen semantics. Neither
response repair nor a relaxed quality gate is justified by this result.

The [raw grading receipt](../../benchmarks/results/2026-10-11-held-out-task-scalar-grading-v1-probe.json)
retains all forty controller results and all 100 outcomes. Every code program
hash and completion hash was reconstructed from the frozen payload and collected
response; every classifier result and every text grade replays exactly.
There are no unresolved infrastructure cases and no resource-limit outcomes.
The aggregate uses all 100 equal-weight binary outcomes. Original dataset
programs remain in the ignored frozen payload. This controlled source-test
execution does not provide adversarial grading attestation.

The [runtime window](../../benchmarks/results/2026-10-11-held-out-task-scalar-grading-v1-window.json)
started at `15:55:15.793510 UTC` and finished at `15:56:18.866618 UTC`.
It used the previously qualified arm64 Python 3.12.15 image and rootless Linux
runtime, exact cached image/API identities, unchanged limits, denied network,
read-only container filesystem and trusted bootstrap over bounded stdin.
All forty distinct containers were removed. Healthy inventory was empty before
the owned machine stopped. Persistent helper pipes completed after shutdown.
This interval includes admission, VM startup, grading and restoration; it is
not inference latency or a performance acceptance measurement.

The [owned observation](../../benchmarks/results/2026-10-11-held-out-task-scalar-grading-v1-owned.json)
records launcher 2394, window 2398 and persistent helpers 2445/2446 in startup
group 2444 after its startup client had exited. The first observer expected that
short-lived client still to be alive and failed before publication; its original
source remains in ignored artifacts. The corrected observer records the actual
persistent ownership without changing the run or runtime policy.
The [actual native completion](../../benchmarks/results/2026-10-11-held-out-task-scalar-grading-v1-session-completion.json)
records session 42097 exiting zero. The
[idle proof](../../benchmarks/results/2026-10-11-held-out-task-scalar-grading-v1-idle.json)
at `15:57:12.021437 UTC` verifies all five PIDs absent, both owned groups empty,
the VM stopped and protected root binaries/model/tokenizer identities unchanged.
It binds eight exact archived receipts/producers.

The [analysis](../../benchmarks/results/2026-10-11-held-out-task-scalar-grading-v1-analysis.json)
and [exact producer](../../benchmarks/results/2026-10-11-held-out-task-scalar-grading-v1-analysis.py.txt)
replay collection provenance, literal programs, all classifications, text scores,
aggregate, chronology, source/revision identities and restoration. The replay
executes no generated program and starts no VM. Astra independently reconstructed
all programs and outcomes, checked container control/configuration and ownership,
and confirmed the same score. The exact analysis replay is byte-identical; Astra
accepted the complete evidence and final report without findings.

The full 10,000-target scalar/tiled loss comparison already passes with identical
FP32 logits and NLL events. Actual tiled held-out task collection and grading
must still pass the unchanged maximum two-percentage-point pass-rate drop gate.
M2.5 remains open; this scalar task baseline does not close performance,
application, full-model GPU or external baseline gates.

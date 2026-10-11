# Full scalar/tiled held-out task quality, 11 October 2026

The opt-in tiled candidate passes the unchanged frozen task-quality gate with
**zero percentage-point pass-rate drop**. Both scalar and tiled resolve all 100
cases at **2/100 (2%)**: code 0/40, prose 1/30 and continuation 1/30. All 100
visible texts, emitted token sequences and stop reasons match exactly; all 100
binary outcomes match. This qualifies this fixed task comparison, not a general
capability ranking. Scalar stays default until the remaining candidate workload
and application measurements support an explicit retention decision.

The [full tiled collection](2026-10-11-held-out-task-tiled.md) was committed and
reviewed before actual grading. The existing [checked integration](2026-10-11-held-out-task-grading.md)
ran from clean revision `0a903c93e695867b68ac640a2608fe452c64be47`.
The [launch receipt](../../benchmarks/results/2026-10-11-held-out-task-tiled-grading-v1-launch.json)
and [exact launcher](../../benchmarks/results/2026-10-11-held-out-task-tiled-grading-v1-launch.py.txt)
bind the completed collection, idle proof and unchanged passed code checks.
No numerical/model process, test suite or build ran during grading. The original
held orchestration remained held.

Each HumanEval program preserves the source prefix, exact finalized completion,
source tests and literal `check(entry_point)`. No fences, indentation, signatures,
tests or completions are repaired. All forty recorded code outcomes are
`SyntaxError`, as in the [scalar baseline](2026-10-11-held-out-task-scalar-grading.md).
Prose uses the entire trimmed exact uppercase letter; continuation uses the
exact first word, preserving case and punctuation. There are no unresolved
infrastructure cases and no resource-limit outcomes. The low baseline is
retained under the frozen semantics. No gate, input or scorer was changed to
accept the candidate.

The [raw grading receipt](../../benchmarks/results/2026-10-11-held-out-task-tiled-grading-v1-probe.json)
retains all forty native controller results and all 100 outcomes. Program and
completion hashes are reconstructed from the frozen payload and finalized
responses. Classifier replay and all sixty protected text grades reproduce
the aggregate exactly. Original dataset programs stay in the ignored frozen
payload. This controlled source-test mechanism does not provide adversarial
grading attestation.

The [runtime window](../../benchmarks/results/2026-10-11-held-out-task-tiled-grading-v1-window.json)
started at `17:04:05.466202 UTC` and finished at `17:05:08.730738 UTC`.
It used the same qualified cached arm64 Python 3.12.15 image, rootless Linux
runtime, fixed limits and machine/API identities. All forty distinct containers
were removed; healthy inventory was empty before the owned VM stopped.
Persistent helper pipe lifetime completed after verified shutdown. The window
includes admission, VM startup, grading and restoration; it is not inference
latency or a comparative timing result.

The [owned observation](../../benchmarks/results/2026-10-11-held-out-task-tiled-grading-v1-owned.json)
records launcher 23954, window 23958 and persistent helpers 24014/24015 in
startup group 24013. Its bounded observation admits the startup client either
still alive or already naturally exited and records its snapshot when present.
The [actual native completion](../../benchmarks/results/2026-10-11-held-out-task-tiled-grading-v1-session-completion.json)
records session 74369 exiting zero. The
[idle proof](../../benchmarks/results/2026-10-11-held-out-task-tiled-grading-v1-idle.json)
at `17:05:22.437315 UTC` verifies all five PIDs absent, both groups empty,
the VM stopped and protected root binaries/model/tokenizer identities unchanged.
It binds eight exact archived receipts and producers.

The [analysis](../../benchmarks/results/2026-10-11-held-out-task-tiled-grading-v1-analysis.json)
and [exact producer](../../benchmarks/results/2026-10-11-held-out-task-tiled-grading-v1-analysis.py.txt)
verify chronology, archived identities, clean source/revision bindings,
collection provenance, literal programs, classifications, text scores and
runtime restoration. The accepted scalar grading/analysis are anchored in the
recorded Git revision and their saved evidence hashes are verified. Both
collections are replayed before comparing response text, token IDs and stops.
The fixed 100-case denominator makes each passed case one percentage point;
the gate uses the exact integer pass-count difference against two points.
The replay executes no program and starts no VM. Astra independently reproduced
the analysis byte for byte, checked container control/configuration and accepted
the complete evidence and report without findings.

The [full 10,000-target loss comparison](2026-10-11-held-out-tiled-loss.md)
already has identical full-logit hashes and NLL events, with zero mean increase
against the unchanged 0.05-nats/token gate. Together these two frozen quality
gates pass for the tiled candidate. Quality alone does not close M3.1:
medium, long, boundary, append, fully cached, pending and changed-prefix tiled
measurements plus the full tiled application trace and final retention review
remain required. The existing paired primary timing already has a confirming
second run; its measured scope and limitations remain unchanged.

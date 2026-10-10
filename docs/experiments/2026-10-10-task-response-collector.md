# Frozen task responses and deterministic scorers, 10 October 2026

The completed collector passes offline checks and four authored native requests.
Astra accepted the complete task without findings. No quality score is claimed. This task implements
one application-side mechanism: exact admission, response finalization and
frozen scoring rules for the existing 100 cases.

The [case module](../../tools/bench/task_cases.py) checks the existing payload
SHA, ordered identities, 40 code / 30 prose / 30 continuation cases, greedy
sampling, binary equal weights and unchanged two-percentage-point gate. It
keeps the prepared prompt IDs and the fixed 256/32/16 sampled output caps.
Those IDs already contain the pinned chat template where required; requests
use raw token input without another formatting or tokenization pass. Payload
reads are bounded at 1 MiB and responses at 64 KiB of UTF-8; exceeding a
transport bound is an infrastructure failure, not a failed model answer.

A response owns ordered visible token IDs, finalized text, its SHA, native
statistics and the separate stop reason. An independent request starts with
an empty session. Output-limit responses retain one pending sampled token;
reset-time decoder text still belongs to the current response. EOS is sampled
and pending without appearing in the visible IDs; its text finalizes before
`done`. The collector requires reset acknowledgement before grading and rejects
duplicated, incomplete or misordered events, unexpected CPU mode, repeated
formatting/tokenization, invalid sample/commit counts and timing metadata.
Numerical model/session code and protected release binaries are unchanged.

TruthfulQA compares the entire trimmed completion with the expected uppercase
letter. LAMBADA compares only the first whitespace-delimited word, preserving
case and punctuation. HumanEval retains literal prefix, completion and all
source tests and explicitly invokes `check(entry_point)`; it performs no fence
removal or answer repair. The module does not execute this program. A separately
qualified isolated executor must run code cases and attest test completion
before any code result can contribute to full quality evidence. A zero process
exit alone cannot establish passing tests.

Aggregation requires every ordered case to have a consistent binary model
pass/fail result. Missing cases or unresolved transport/executor failures stop
aggregation. Category counts are reported alongside the 100 equal-weight
overall result; infrastructure errors are never removed from the denominator
to produce an apparent successful run.

The [offline fixtures](../../tools/bench/test_task_cases.py) cover exact text
rules, literal code composition, pending EOS, reset-time Unicode text,
admission/inventory corruption, bounded payloads, event order and binary
aggregation without model downloads. The
[transport fixtures](../../tools/bench/test_task_cases_probe.py) use a tiny fake
CLI to exercise resident requests, malformed/nonfinite output, missing reset,
worker error/trailing output and owned-group watchdog cleanup.

The [authored native probe](../../tools/bench/task_cases_probe.py) requires a
complete saved cached-workload idle proof and a currently held orchestration
parent before checks or model work. It reuses the already qualified isolated
CLI in scalar and tiled modes with two short frozen numerical inputs, with
32 and 16 sampled output caps and exact independent first-token checks. It
requires scalar/tiled equality of finalized text, visible IDs and stop reasons.
It validates all prepared task metadata, but executes no held-out task prompt and
produces no task score. Raw JSON lines, errors, semantic events, source/helper/
binary/model identities and conservative joint CLI-plus-driver physical peaks
are retained. Successful completion requires both processes' accounting to
remain below 6 GiB. The output path must be fresh and inside raw evidence.

The [completed record](../../benchmarks/results/2026-10-10-task-response-probe-v2.json),
[raw native lines](../../benchmarks/results/2026-10-10-task-response-probe-v2.json.native.jsonl)
and [stderr](../../benchmarks/results/2026-10-10-task-response-probe-v2.json.stderr.txt)
retain two resident workers and four requests. Both modes emit 48 visible and
48 sampled tokens across the two requests; all four reach their output limits.
All 106 raw events replay exactly, including reset finalization, and scalar/tiled
IDs, text, text SHA and stop reasons match. The first token of every request
matches the independent frozen numerical argmax. The
[analysis](../../benchmarks/results/2026-10-10-task-response-probe-analysis-v1.json)
reconstructs text/IDs independently from raw events and checks all identities,
with its exact [producer](../../benchmarks/results/2026-10-10-task-response-probe-analysis-v2.py.txt).

Maximum CLI physical peak is 2,125,301,896 bytes; maximum driver peak is
25,264,704 bytes. Maximum conservative sum is 2,150,566,600 bytes below
6 GiB. The isolated CLI SHA remains
`8e1be695229fb4a46e49725f06050b3296f6fb398e718fca8567bf0c6b400b43`;
the existing qualification receipt binds its compiler and unchanged numerical
sources. Probe launch checkout is `3338af4` with this task uncommitted. Source,
helper, model, tokenizer, contract and binary identities are checked before
and after execution. These functional probes establish no speed result.

The probe preserves failure records and exit/timeout/cleanup evidence. Each
worker has its own process group; failed leaders and their descendants are
terminated and reaped. The 20-minute functional watchdog is separate from
future generated-code CPU/memory/wall/output limits. Source preparation and
AST checks occur during the cached run; tests, compilation, native execution
and large/binary/model hashing began only after
[verified idle](../../benchmarks/results/2026-10-10-cpu-task-cases-check-window-idle.json)
at 09:58:13 UTC, with the exact
[idle producer](../../benchmarks/results/2026-10-10-cpu-task-cases-check-window-idle-v2.py.txt)
retained. The original cached numerical worker was never stopped. The
[resumption producer](../../benchmarks/results/2026-10-10-cpu-task-cases-check-window-resume-v2.py.txt)
requires clean reviewed commits, rechecks protected binaries, assets and all
dependent evidence, then signals only the owned orchestration parent.

The first idle probe attempt stopped at offline tests and preserved a failure
record in ignored raw evidence; no model worker ran. A cleanup regression exposed
`PermissionError` when signaling an already dead process group. The helper
retains that error unless the leader has exited and process metadata proves
there is no live member. A live group or failed status query remains a cleanup
failure. The added regression covers dead, absent and surviving group members.
The fresh v2 probe passes root formatting, Clippy with warnings denied, all
76 ordinary Rust tests and all 54 offline benchmark tests, including 19 collector
and transport tests. Model-dependent ordinary tests remain explicitly ignored;
the native authored requests run separately. No root release binary is rebuilt.

Astra independently ran all 19 collector/transport tests with process inspection
enabled and reproduced the retained saved analysis exactly. The review accepted
the frozen graders, pending/reset semantics, bounds, cleanup, source/evidence
identities, physical accounting and prepared resumption sequence without findings.

Full 10,000-target loss, isolated HumanEval execution, all 100 generated
responses, candidate/reference quality comparison, performance qualification
and M2.5/M3 acceptance remain pending.

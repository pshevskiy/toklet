# Full task collector preparation, 11 October 2026

The [collector](../../tools/bench/held_out_tasks.py) can collect finalized
responses for all 100 frozen tasks in one qualified CLI process. It preserves
prepared token IDs, greedy EOS policy, category output caps and explicit reset
between cases. Immediate hidden EOS with empty visible output is valid; the
collector has no authored first-token requirement. This is checked collection
preparation. Held-out model responses, isolated grading, pass rates, candidate
workload acceptance and M2.5 remain pending.

The frozen inventory contains 40 code, 30 prose and 30 continuation cases.
Each generation uses capacity 2,048 and prefill chunk 128. The existing
[task semantics](../../tools/bench/task_cases.py) validate ordered tokens,
finalized text, sampled versus committed history, one pending token, stopping
and reset acknowledgement. Raw events, semantic responses and hashes are
retained. The collector records case identity/category/prompt length and raw
requests; original dataset text stays in the ignored prepared payload.
It never compiles or executes generated code and never starts the Linux VM.

Readiness is bounded to 1,200 seconds, each case independently to 1,800 seconds,
and shutdown to 20 seconds. Timers are canceled and joined before changing
phase; final joined timeout state invalidates success. These are failure
watchdogs, not performance thresholds. The transport rejects oversized lines,
duplicate JSON fields and nonfinite values. Output-limit text finalized at
reset remains part of the response; EOS text finalizes before completion.

Each invocation owns its new worker session group. Failures attempt to terminate
that group and reap the leader, with separate bounded cleanup waits. Descendants found after normal leader exit also
trigger owned-group termination and up to twenty bounded rechecks. Initial and
final inventories are saved; an unexpected descendant leaves the collection
failed even when cleanup succeeds. Unknown process inventories and cleanup
errors prohibit success. A completed record requires natural exit zero,
no watchdog expiry, complete reset responses and empty final group inventory.

Before collection, a completed workload idle receipt is checked against fresh
PID/group absence, the original held orchestration, protected identities and
the stopped VM. The qualified CLI, frozen task payload, source/model/tokenizer
identities and EOS configuration are verified. Helper, binary, qualification,
manifest, input and asset identities are checked again after the worker exits.
Conservative joint CLI and driver physical peak charge must remain within
6 GiB. This collector does not establish an end-to-end timing comparison.

The [eleven targeted tests](../../tools/bench/test_held_out_tasks.py) cover
all 100 immediate EOS/reset responses, visible finalization, output-limit reset
text, missing reset, invalid pending state, late tokens, trailing output,
invalid admission, expired phase timers, the late shutdown-timeout boundary,
and successful or unresolved descendant cleanup. A real tiny fake CLI consumes
all 100 requests, returns 201 semantic events and exits naturally; it downloads
no model and executes no generated program.

The [full check record](../../benchmarks/results/2026-10-11-held-out-task-collector-checks-v1.json)
and [exact producer](../../benchmarks/results/2026-10-11-held-out-task-collector-checks-v1.py.txt)
retain all 172 offline benchmark tests, formatting, locked offline Clippy with
warnings denied and 76 ordinary Rust tests. All pass in the isolated target;
root release binaries and protected assets remain unchanged. The
[observed check completion](../../benchmarks/results/2026-10-11-held-out-task-collector-checks-completion-v1.json)
records session 93116 exiting zero. No held-out model or VM run occurred.

Astra found two cleanup bugs during review: late shutdown timeout could leave
success marked true, and descendants detected after normal exit were reported
without termination. Both were corrected before the full checks and have
independent regression coverage. The initial check producer also rejected a
string passed to a Path-only hash helper before any checks or model launch;
the corrected producer and collector use explicit Path values. The initial
producer remains in ignored artifacts. No policy or quality threshold changed.

Astra independently passed the three cleanup boundary regressions, verified the
final source/check/completion bindings and documented scope, and accepted the
complete code task with no remaining actionable findings. All 138 local links
across this report, README and PLAN resolve; whitespace checks pass.

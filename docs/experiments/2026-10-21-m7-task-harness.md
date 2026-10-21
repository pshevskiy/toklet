# M7 sealed raw task harness

The app now admits a sealed bounded task set and streams validated native chunks
and completed raw responses. This is preparation for the frozen100-task phone
collection; the checks here use four tiny cases, not100 real-model responses.

The exact12-field configuration carries the admitted model/packed inventory,
backend, capacity/chunk, numerical/process budgets, task path/SHA and case count.
The task file has exactly version/vocabulary/stops/cases; each case has exactly
id/promptTokens/maximum. Both JSON inputs use a bounded512KiB read and SHA over
the same consumed buffer, reject extra fields and symlinks, and keep tasks outside
model inventories. Cases are ordered with unique bounded ASCII IDs, at most100
cases,306 prompt IDs and256 sampled outputs. Whole-request capacity must fit;
capacity is at most562 and prefill chunk at most128. IDs/stops/duplicates, budget
arithmetic and Int.max output/count values are checked before native ownership.
The later host controller must bind the exact frozen100-case derivative, original
caps and EOS IDs; the generic Swift admission also supports shorter tiny cases.

The off-main runner retains immutable model ownership across cases, passes the
persistent notification/foreground latch into the bounded response coordinator,
and observes actual positive native admission within the configured numerical
budget. Each streamed chunk includes case identity, command-local events, caller
submission/completion uptime, admitted bytes, process current/lifetime peak and
thermal/power/foreground state. A completed case includes the pre-reset response
and an explicit empty reset acknowledgement, including for the final case.
Partial chunks or failing output sinks cannot complete a case or a whole result.
Model/task proofs run before load and after close; memory phases cover proof,
load, each completed case, unload, close and final proof. Release is explicit and
weak-owner checked. Process peaks are observed rejection gates rather than OS
hard caps. The runner performs no tokenization or grading.

The diagnostic --tasks mode requires exactly config path and SHA after the mode,
waits for initial foreground activation, disables the idle timer, records signed
installation proofs and emits separate chunk/case/result markers. It owns no
interactive-store worker. The project now includes NativeTaskResponse.swift and
NativeTasks.swift; the new runner builds in both full iOS SDK apps.

Thirty-one native test-program cases cover CPU/Metal completion, malformed
config/task policies and inventories, duplicate IDs/stops, token/capacity bounds,
extra fields, wrong SHA, symlink/oversize rejection, failed chunk/case output and
changed model/task endpoints. Each backend completes two cases: prompt[1,7,13]
with cap3 and prompt[1,7,13,3] with cap2. Both first greedy IDs are47 in the frozen
independent FP32 prompt fixture; all10 observed visible IDs are47, matching the
existing qualified tiny generation traces. Exact response/prefix/reset states,
actual admission, process observations and clean close/release are replayed.
Five saved failing results remain rejected, including an overflowing case count
that creates no native owner. The256-token protocol was checked by the preceding
coordinator component; this task does not claim256-token tiny-model coverage.

Astra found two checker gaps before native execution: a later command submission
must follow the previous completion rather than its last callback, and nested
receipt SHA must bind the same bounded parsed buffer. Both are fixed and have
specific overlap/replacement regressions. The clock guard spans both backend
runs in this one test process; peak observations are replayed chronologically.
An initial controlled preflight then caught my mistaken expectation that[1,7]
also greedily emits47: its frozen row actually emits7. No tool/native commands ran
in that failed attempt. The test was corrected to the already frozen four-token
prompt, with correct6/5/1 history/evaluated/pending counts. The failed receipt and
three exact old source snapshots remain published; no gate/tolerance was changed.

The [qualification receipt](../../benchmarks/results/2026-10-21-m7-task-harness-qualification-v1.json)
binds253 paths, six changed code files, two parent-source mappings, three failed-
preflight source mappings and131 exact raw copies. Six task-check commands pass;
nested existing quality, measurement/hash and both SDK app checks add7+5+9 clean
commands. All27 complete without primary, cleanup or endpoint errors. The nested
quality suite retains its independent CPU loss checks and19-case protocol;
measurement retains17 cases/39 requests and the256MiB bounded hashing regression.
Strict optimized Swift checks complete concurrency and treats warnings as errors.

Twelve final targeted parser methods and ten package methods pass. All140 portable
M7 methods pass across15 isolated whole-module Python processes against one
before/after verified source closure. This is not a combined-suite pass: three
combined attempts fail the unchanged timeout-cleanup method on host process-group
EPERM, with the other139 methods passing. All failures remain. Eleven isolated
modules pass by qualified module name; four require their standard single-file
discovery import path and pass that way. Their preceding import failures are also
retained. No assertion, source or fail-closed cleanup policy was relaxed. Earlier
parser logs predate final checker/prompt corrections and are historical output,
not final-source qualification. The unresolved combined-run EPERM remains an
explicit test-environment limitation, rather than being counted as a pass.

Astra cleared final source and accepted completed code/evidence/index/privacy/
link review with no findings. Retain the
sealed runner and checks. Physical signed-app collection, exact frozen100-case
admission, post-exit tokenizer decoding, literal grading, full10,000-target loss
and physical lifecycle/sustained/base iPhone14 qualification remain open.
Numerical kernels, modes, defaults and acceptance thresholds stay unchanged.

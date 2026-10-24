# Mobile monitoring, model fits, benchmark and dialog deletion

24 October 2026. This mobile change adds a monitoring strip and **Find best
model** and **Benchmark** actions to the Chat screen. Expert shows the same
monitoring strip. The reviewed app was subsequently installed and tested on
iPhone 14 Pro at the user's request; see the
[physical UI checks and Gemma 4 limitations](mobile-gemma4-testing.md).

## Measurements

RAM used is `TASK_VM_INFO.phys_footprint` for the app process, not model file
size or the engine's logical allocation plan. On iOS, allowed is footprint plus
`os_proc_available_memory()`, sampled close together. It is an instantaneous OS
allowance, can change under pressure, and reserves no memory. On Mac, allowed
is the configured process cap `min(physical RAM / 2, 6 GiB)`. Read failures show
Unavailable. Samples update at most once per second, independently of native
snapshot revisions.

First-token time uses monotonic uptime from accepted worker submission to the
first emitted token callback. It includes queueing, Chat encoding and prefill,
but excludes model loading and time spent editing a draft. Decode tok/s is
`(callback count - 1) / (last callback time - first callback time)`; zero or one
token has no rate. Hidden stop IDs are not emitted callbacks. Terminal duration
ends after FFI, decode and worker state inspection return, before UI publication.
These are current/last-generation observations, not a quality or sustained
performance qualification. UI polling is not the measurement clock.

## Model shortlist

Find best model filters only the existing pinned, downloadable Chat catalog
through the current device admission policy, then shows up to five results
ordered by the selected task's capacity, estimated speed and memory headroom
scores. [Task-based ranking](mobile-model-ranking.md) documents the weights and
assumptions; these estimates do not claim measured answer quality or speed.
The presently resident Chat profile can
remain listed without requiring space for a second copy. Other profiles must fit
the allowance remaining now; unload the active model to evaluate replacements
with more headroom. The screen explains the basis and why fewer than five can
be returned. The current catalog has three profiles; an iPhone allowance can
admit fewer. Unsupported search results never become runnable recommendations.

Use/download retain pinned revision, file hash and native model validation,
download storage checks and fresh post-release memory admission. A fit estimate
does not bypass those checks or qualify a new model or architecture.

## Local benchmark

Benchmark uses the already loaded Chat model and the ordinary serial native
worker. The built-in offline workload is the fixed seasonal-description prompt
in `FriendlyChatStore.benchmarkPrompt`, with at most 64 output tokens per run.
It runs one warm-up then three measured independent Chat requests. Each resets
the numerical prompt through the existing Chat path; a subsequent ordinary Send
also resets/replays its own conversation. No network or benchmark dataset is
required, and owner dialogs, drafts, output preference and history files are not
modified. Results name the model and report each actual output count, stop
reason, TTFT, decode rate and complete-command duration. Early model stops are
allowed; cancellation, rejection and failed runs are excluded from samples.

The benchmark disables conflicting dialog/model changes while running. Stop
requests native cancellation; cleanup remains on the owning worker thread. A
background interruption or model release stops the loop instead of automatically
starting the next request. Keep the app open. This short session check does not
replace the repeated, controlled workloads required by PLAN.md for engine
performance claims.

## Dialog deletion

Every conversation has a visible trash button, swipe action and context-menu
action. Delete all conversations has a separate destructive confirmation.
Deletion uses the existing bounded history writer and is blocked while work,
model opening or storage recovery is active. Deleting all resets messages and
drafts to one empty conversation while preserving the active model identity and
reply preference. Downloaded models are kept. Pending draft saves are cancelled
before writing; late response IDs cannot restore deleted conversations. A failed
write rolls the proposed deletion back in memory, keeps the saved file intact
and shows the error in History; retry deletion after storage recovers.

## Validation scope

The local Swift check exercises timing arithmetic and missing measurements,
admission/ranking and resident-model cases, selection after deletion, persistence
and stale-result isolation. Explicit real-model checks exercise the warm-up,
three measured generations and cancellation without changing dialog bytes.
The [receipt](../benchmarks/results/2026-10-24-mobile-dashboard-v1.json) binds
the final source snapshots and immutable native package. Eight final commands
pass: strict Swift 6 compilation, controller regressions, CPU worker checks,
light/dark rendering, real SmolLM2 benchmark/Chat checks, SmolLM2/Qwen replacement
admission, and device/Simulator module compilation. Full unsigned Release device
and Simulator app builds pass from the same sources.

The final iPhone 17 Simulator XCUITest passes with no failures or skips. It checks
visible metrics, model-fit explanation, benchmark readiness without a loaded
model, creating three dialogs, cancelling and confirming single deletion,
cancelling and confirming delete-all, Expert monitoring and return to Chat.
The isolated temporary workspace never reads owner dialogs. Numerical benchmark
execution is covered separately on the Mac, since the Simulator does not admit
the MLX Metal model. This GUI run does not test a successful download or every
accessibility layout.

Earlier failed runs and their exact sources/logs are retained. They exposed a
container accessibility identifier overriding metric identifiers, and a
single-delete confirmation that read mutable selection after dismissal. The
confirmation now captures its presented UUID. The test also waits for the
updated row count. An additional test failure assumed a Cancel row existed;
iOS can present a popover cancelled by tapping outside. The final test uses its
observed frame to dismiss it and verifies that the dialogs remain.

The subsequent [physical suite](mobile-gemma4-testing.md) passes dashboard,
benchmark, deletion, model switching and Chat/Expert rendering checks on the
installed app. Its short device timing observations do not establish controlled
performance or 128K numerical generation. Current native Chat/raw/context limits
remain 256/128/512; the [128K engine plan](long-reply-engine-plan.md) is separate.

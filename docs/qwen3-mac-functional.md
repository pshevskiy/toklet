# Experimental Qwen3 Mac app audit — 27 October 2026

The pinned Qwen3-0.6B eight-bit profile now carries the same explicit experimental
accuracy status in Models, the active Chat dashboard and Expert instructions.
The message names this exact profile. Other model rows and active Chat profiles
do not inherit it; Expert's general instructions keep it visible independently
of transient compatibility inspections or session suspension.

The audit also fixes a demonstrated stale-error defect: a valid retry previously
streamed while the preceding “Conversation is full” error remained visible.
An accepted worker command now clears the previous status/message under the
existing queue lock. A rejected submission keeps the snapshot untouched.

These changes preserve the 512-position shared window,
the separately displayed model declaration of 40960, reply/text limits,
download identity checks and native execution. The
[output-screen rejection](qwen3-output-screen.md) and weak absolute
[quality result](qwen3-full-quality.md) remain explicit. This does not establish
qualified Q1–Q5 delivery.

## Checks completed

The native Mac app compiles with strict Swift6 concurrency and warnings as errors,
then passes ad-hoc signing and strict signature verification. The source set
also passes strict iOS SDK typechecking without contacting the phone. The first
iOS check command omitted the existing WorkerChecks source; that failed command
is retained, and the corrected invocation passes. No product code changed to
repair the invocation.

The existing Hugging Face controller and CPU worker/context checks pass against
the current Swift snapshot and admitted C512 Mac archive. A deterministic retry
regression holds valid work before execution: the previous error clears while
queued, a rejected busy submission changes nothing, the literal `[1,7,13,3]`
prefix survives failure and retry, and successful shutdown is explicitly joined.
The archive and source/package identities remain bound. No Rust numerical code
or qualification criterion changes.

Interactive checks use the app's documented isolated `--data-root`, preserving
the owner's normal conversations and model cache. The current Models and Expert
notices are visually and accessibly present. Public search returns the pinned
Qwen3 model and labels other checkpoints unavailable. Cancel returns the picker
to an operable state. Entering Expert also cancels an active download through
the existing model-choice policy; returning to Chat works. A clean retry in Chat
completes the 644,927,148-byte pinned snapshot and opens Qwen3. “Hello!” receives
a reply; Stop retains partial text with a Stopped marker. A 3,600-byte oversized
draft rejects with all four existing message objects unchanged and the entire
draft restored. A shorter retry works and exposes the stale-error defect above.
The label visibility observations use the retained pre-fix build; the corrected
worker build passes compilation and the deterministic regression.

The existing self-test then reuses the downloaded snapshot with the corrected
worker. Public metadata search and saved-file verification pass. Its separate
offline controller rejects every network request while reopening the saved
model, generating a reply and unloading. Native functional checks also confirm
context rejection preserves the prefix, cancellation and explicit Send after
suspend. All workers release. The process peaks at 932,463,960 physical bytes /
1,005,862,912 maximum RSS, below 6 GiB. The earlier visible pre-fix process has
a separately observed lifetime physical peak of 944,817,664 bytes. These are
functional resources, not throughput or whole-phase memory qualification.

**The final visible replay of the worker fix and interactive switching to another
model remain pending because the Mac locked.** The user has been asked once to
unlock it. This prevents that observation; it does not invalidate the passing
compiled/fixture checks or establish complete Q5 acceptance. No phone is contacted.
The idle corrected lab window was stopped with SIGTERM after the lock; its
exit−15 is recorded as cleanup, not a successful visual replay. All26 owned
build/check/audit process groups are absent. The isolated snapshot is retained.

## Existing evidence reused

The unchanged [context integration](context-app-integration.md) already verifies
385 rendered input plus 127 requested output positions, rejects one additional
position in both modes without altering the prefix, and checks cancellation,
suspend/replay, fresh model/session identities and rejected-draft restoration.
Its admission witness cancels after one output; it is not completed 512-position
app execution. The [native C512 trial](qwen3-context512-native.md) supplies separate
complete cache/chunk/state witnesses while retaining rejected numerical gates.

The bounded audit retains these distinctions. Larger app/codec/output delivery,
the declared full model window and numerical qualification remain open in the
[delivery queue](qwen3-mac-delivery-plan.md). See the
[public receipt](../benchmarks/results/2026-10-27-qwen3-mac-functional-v1.json).

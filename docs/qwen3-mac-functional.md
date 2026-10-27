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

The first visible replay was blocked by Mac lock. Its idle lab window was stopped
with SIGTERM; that exit−15 remains cleanup evidence. After the user unlocked the
Mac, **the corrected visible replay and cross-model switching passed**. A valid
story retry streamed after the oversized rejection with the previous error
removed, then completed its 128-token limit. Qwen3 → local pinned SmolLM → saved
Qwen3 switching worked: the Qwen3 notice disappeared for SmolLM and returned for
Qwen3, which generated normally. Intervening human input was preserved and is
excluded from controlled response results. The user yielded the window before
the final switch. The later audit app quit normally with clean source/bundle
endpoints; its lifetime physical peak was 952,174,128 bytes. No phone is contacted.

## Completed raw Expert C512 witness

The actual Expert window completed 384 authored input IDs plus 128 output IDs
without hidden stops. Displayed output was exactly32 repetitions of
`9707,11,1879,0`, a functional replay expectation rather than an independent
numerical oracle. It reached history512/committed511/pending1. Requesting position
513 rejected with those counts unchanged; Continue(0) committed the pending token.
Suspend retained the 512-token prefix, and Replay restored512 committed positions
in a new session with the same model.

An explicit `--raw-window` mode in the existing Swift context checker covers
values the UI cannot expose. Eleven retained snapshots precede assertions. Both
pending and committed overflow reject while preserving the exact literal512-ID
prefix; the available scalar diagnostic is unchanged across committed overflow.
Replay restores the exact saved IDs and makes the scalar diagnostic available.
This does **not** assert equality of every logit across replay schedules. Closure
is explicitly joined and native owners are released. Strict Swift compilation
and the existing CPU worker/context regressions pass. The one real-model process
peaks at751,109,224 physical/766,738,432 RSS bytes, below6GiB; all four new owned
UI/build/check groups are absent.

This closes the raw Expert C512 functional gap. Numerical qualification and
full40960 delivery remain open.
See the [follow-up receipt](../benchmarks/results/2026-10-27-qwen3-mac-visible-v1.json).

## Completed Chat-worker C512 witness

The remaining Chat boundary now passes through the ordinary app worker and codec.
An authored long-story conversation was prepared separately using the native
renderer: exactly385 input IDs and127 requested outputs. Its7,312-byte frozen
input has SHA
`0b0aca92794419fc5d1d0cd3488b8b56b31ef409cb29079de717993e328bcd1a`.
Preparation generated no tokens. Execution consumed those exact bytes/IDs with
normal Chat stops unchanged; no stop override or post-result prompt tuning.

The one request emits127 visible IDs, terminates at its normal output limit and
reaches history512/committed511/pending1. Actual admission reports385+127=512.
Saved tokens exactly equal the frozen rendered IDs plus emitted IDs. Requesting
128 outputs for the same conversation rejects513 before changing that prefix or
session facts. A separately labelled raw Continue(0) flush commits the last
pending token, and the full-context scalar diagnostic is available. Six snapshots
precede assertions; closure is joined and native owners release.

Strict Swift compilation, existing CPU fixture/context checks and the separately
prepared input pass. The one Chat process peaks at898,041,080 physical/
821,886,976 RSS bytes; preparation peaks at813,909,192 physical bytes, each below
6GiB. All four phase groups exit with source/model/archive/input endpoints
unchanged. These are functional resources, not a throughput benchmark.

**The bounded experimental C512 Mac functional milestone is complete.** This
combines the visible recovery/model-switch journey, pinned download/offline path,
and full Expert and Chat-worker shared-window checks. It is not numerical or
full40960 qualification. See the
[Chat receipt](../benchmarks/results/2026-10-27-qwen3-chat-window-v1.json).

## Existing evidence reused

The unchanged [context integration](context-app-integration.md) already verifies
385 rendered input plus 127 requested output positions, rejects one additional
position in both modes without altering the prefix, and checks cancellation,
suspend/replay, fresh model/session identities and rejected-draft restoration.
That historical Chat admission witness cancels after one output; the newer
Chat-worker witness above supplies completed512-position execution. The
[native C512 trial](qwen3-context512-native.md) supplies separate
complete cache/chunk/state witnesses while retaining rejected numerical gates.

The bounded audit retains these distinctions. Larger app/codec/output delivery,
the declared full model window and numerical qualification remain open in the
[delivery queue](qwen3-mac-delivery-plan.md). See the
[public receipt](../benchmarks/results/2026-10-27-qwen3-mac-functional-v1.json).

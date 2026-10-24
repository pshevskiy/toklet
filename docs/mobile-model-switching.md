# Direct model switching on iPhone

24 October 2026. The user reported that switching between models remained
unusable in the installed app. Earlier controller checks did not establish that
the actual phone interaction worked.

## Problem and change

The installed Models screen used a dropdown which changed a download profile,
then required a separate Open action below the initial viewport. Selecting a
profile did not switch the active inference model. Opening the screen also
started full model-file verification. A physical XCUITest reproduction captured
that screen and a 60-second animation-idle wait after tapping its native picker.
That is an automation wait observation, not a generation-latency benchmark.

Models now shows a direct **Use model** action for each saved model and
**Download and use** for another model. **Active** identifies the loaded model;
choosing it returns to Chat without reloading. Search results use the same
controls. Progress identifies the requested model while the previous model is
released. A newer Use choice can replace pending verification; cancelled or stale
verification cannot open the previous choice. An active download stays exclusive.

Opening Models reads a small inventory for the “saved” labels. Presence is not an
integrity claim: Use still performs complete receipt/type/size/SHA validation.
Downloaded files and the native loader's validation remain unchanged. The Mac
view retains Show models in Finder for damaged-download recovery.

A replacement no longer compares the new model's full allowance against memory
remaining while the previous model is resident. Its freshly released state is
checked in `FriendlyChatStore.beginLoad` before loading the next model. Initial
opens and downloads retain their existing admission gates. No memory allowance,
model catalog, context or output limit is increased.

The worker now numbers snapshot changes under its existing condition lock.
`HarnessStore.poll` publishes only a changed revision, instead of rebuilding the
UI ten times per second while idle. A separate counter avoids retaining the old
snapshot's token arrays on every callback. The ordinary callback, cancellation,
cleanup and terminal-close ownership rules remain in force.

## Verification

The [receipt](../benchmarks/results/2026-10-24-mobile-model-switch-v1.json)
binds the final source snapshots, native policy package, Release builds and
physical XCUITest run. Strict Swift checks cover replacing a pending Use, stale
completion isolation, idle publication suppression and revision changes through
submit, callbacks, suspend/replay, unload and close. Explicit real-model tests
simulate unavailable memory while the old owner is resident and sufficient
memory after release; SmolLM2 → Qwen → SmolLM2 receives three fresh admissions.

The live test uses the production screens with `--ui-test-history`, which creates
a unique temporary workspace rather than reading or changing owner conversations.
It switches repeatedly in both directions, sends real prompts, checks a visible
code Copy action, and reselects the active model. The host compares owner history
before/after and returns the passing installed app to its normal launch.
Screenshots and failure logs are retained. The first new-flow run reached Qwen
and generated highlighted code, then failed because the test selected an
off-screen first Copy control while another code block was visible. The test now
checks the visible code control. Failed build attempts retain the stale-package
and precompiled-header diagnostics; fresh products use the admitted policy
package throughout.

Both final phone tests pass on iPhone14Pro. The additional test selects the
256-token reply setting, observes a long Qwen reply's limit notice, opens Expert,
loads the CPU tiny fixture, checks highlighted `47, 47` with its Copy control,
and returns to Chat. Owner history bytes match before and after both runs. The
passing updated app is left installed and launched normally.

The result does not qualify every model, low-memory pressure case, 128K output,
new device or performance improvement. Runnable Chat output remains 256 tokens,
Expert raw output 128, and the Chat context 512 positions. The
[128K single-reply engine work](long-reply-engine-plan.md) remains separate.

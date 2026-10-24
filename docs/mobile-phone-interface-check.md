# Physical mobile interface checks — 24 October 2026

The user requested installation and tests of model switching, longer replies,
and code highlighting in Chat and Expert. The connected, paired iPhone 14 Pro
(`iPhone15,2`, iOS 27.0.1) runs the frozen mobile changes from `18c9101` with the
reviewed `c2d9c5ca…` native package. Concurrent engine development is excluded.

## Installed and functionally checked

The reviewed unsigned app was signed using the existing device/team profile,
its signature was verified, and it was installed as an update. A private backup
of saved chats was taken first. The chat file remained byte-for-byte unchanged
through installation and the ordinary device checks. Existing model folders
were retained; the saved SmolLM2 and Qwen snapshots were verified and reused.

The installed production app passes its CPU tiny, Metal tiny, SmolLM2 Hub/Chat,
and Qwen Hub/Chat checks. Before/after installed-bundle proofs match the signed
app files in each run.

An instrumented clone adds only a diagnostic launch branch and the
[phone controller check](../apple/tests/swift/PhoneInterfaceChecks.swift).
The Chat store, model Hub, workspace, renderer and native engine remain the
frozen production sources. This check uses a separate temporary history file
and does not read or replace the owner's chats. On the physical phone it passes:

- SmolLM2 → Qwen → SmolLM2 using the same Chat controller/worker, with separate
  conversations across model changes.
- A Qwen reply emitting **256** IDs, with **321** sampled-history positions,
  **320** committed positions and **one** pending token, stopping at the limit.
- A real generated Python fenced code block, with exact text preserved by the
  highlighting spans.
- Entry into Expert mode, release of the Chat model, CPU fixture generation of
  **47, 47**, exact highlighted-ID text, and explicit resource release.

The reviewed production app is restored after instrumentation. These are
functional/controller and text-rendering-data checks, not task-quality, latency,
memory-limit or newer-engine qualification.

## Live UI validation

The owner explicitly authorized XCUITest because Device Hub's computer-control
calls stalled. The test runner was built and signed after the Apple Account was
made available. The device's three free-development app slots were already
occupied, so the existing UI-test-runner slot was reused rather than removing
Toklet or Inference. A stale generated test-host identifier was corrected to
match that runner's actual signed bundle identifier.

The runner installs and launches, but its first corrected run timed out while
enabling device automation. The owner has been asked to enable UI Automation and
keep the phone unlocked. **Live tap-through and screenshot/color inspection are
not yet accepted.** Code-block parsing and attributed text preservation above do
not substitute for that visual check. The complete XCUITest flow is built and
ready to test Settings, search/model selection, a long reply, Chat code, Expert
IDs/help, and returning to Chat.

The [receipt](../benchmarks/results/2026-10-24-mobile-phone-interface-v1.json)
separates passed physical functional checks from this pending UI gate. Private
device, signing and saved-chat evidence stays in ignored local artifacts.

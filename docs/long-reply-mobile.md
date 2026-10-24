# Mobile preparation for 128K-token replies

Requested 24 October 2026. The user explicitly means **128K in one reply**,
rather than a 128K total conversation. The target preference is 131,072 output
tokens. Prompt, template and prior history need additional context positions.
The parallel engine task owns codec, generation, context/model support, resource
admission and physical qualification.

## Current runnable boundary

This change prepares the mobile presentation and persistence layers. It does
**not** implement or qualify 128K generation. The native contract still exposes
256 Chat output tokens, 512 Chat context positions and 64 KiB conversation JSON.
The existing default remains 128 output tokens. The installed iPhone app and its
[physical checks](mobile-phone-interface-check.md) retain their original scope.

Settings derives its offered limits from `NativeSettings.maximumChatOutputTokens`.
The workspace can preserve preferences through 131,072 for a future compatible
engine. A saved preference above the current engine limit is shown as unavailable
and disables Send until the user chooses a supported value. It is never silently
shortened. Increasing a preference or storage bound establishes no model support.

## Mobile changes

- A stored message may contain 8 MiB of UTF-8; the bounded history file may
  contain 64 MiB. Original version-1 histories and reply settings still restore.
  Before Send, the workspace reserves 48 MiB plus 16 KiB for a maximum completion
  and metadata, including worst-case six-byte JSON escaping. The current 64 KiB
  conversation JSON and 16 KiB draft limits remain until the engine publishes
  their replacements. Large stored replies do not bypass native admission.
- Long prose is divided into pieces of at most 8,192 Swift Characters for lazy
  text layout. Long code uses Previous/Next parts with that same bound on each
  highlighting input. Copy uses the entire original code block. Characters,
  Unicode combining sequences and newlines survive splitting exactly.
- Unchanged assistant text uses an equatable view, avoiding repeated parsing
  when unrelated interface state changes. Splitting/parsing the full document
  still visits the whole text; this is not a constant-memory streaming renderer.
  The basic lexical highlighter operates on each displayed part independently;
  it is not a stateful language parser across part boundaries.

The larger disk/text bounds need to be included in the future app memory
allowance. They do not qualify a larger numerical or process budget on iPhone.

## Verification

The [receipt](../benchmarks/results/2026-10-24-mobile-long-replies-v1.json)
records snapshotted sources and the immutable native policy package used for
strict Swift 6 builds, controller checks, highlighted-text previews and separate
iOS/simulator module checks. Full unsigned Release iOS and simulator apps also
build against the same package, including the in-flight engine resource
preflight captured in their source snapshots. Large-text fixtures cover Unicode, long lines,
32,768 code lines, exact reassembly, large-reply save/restore, maximum escaped
completion storage, preference preservation and native-limit filtering. These
are synthetic text/controller probes; repeated words are not claimed to be
131,072 model tokens.

No new physical-device UI, 128K generation, model quality, throughput, cancellation
at that length or process-memory qualification follows from these checks.
The [earlier UI automation prerequisite](mobile-phone-interface-check.md)
also remains pending.

## Remaining integration

The engine must publish bounded conversation/decoded-text constants, admitted
context and runnable output capabilities, and implement a model/context path
with room for the requested reply plus prompt/history. Generation, cancellation,
prefix recovery and memory/quality checks must cover those limits. The app can
then expose the admitted reply choices and replace its remaining short-context
help and input bounds. The complete updated package must be installed and tested
on the owner's iPhone before claiming device support.

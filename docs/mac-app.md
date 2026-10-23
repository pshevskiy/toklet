# Native Mac Chat and Models add-on

23 October 2026. A separate macOS app entry point adds a native Toklet window to
the existing Harness. It shares the friendly Chat, Models, saved conversations
and Expert controls with the iPhone source and uses the existing Rust/Metal
engine. The iPhone target remains available. Physical phone work is paused at the
user’s request.

## Open and use

The checked local bundle is `artifacts/m8-mac-window-build-v7/Toklet.app`.
Open it in Finder. Choose a model, search Hugging Face or select an available
pinned profile, then Download and Open downloaded model. A complete local model
folder can also be selected through the native folder picker. Close Models and
send a message. Command-Return sends, or requests Stop while a reply is running.
Replies appear at completion; Stop preserves the completed prefix.

Conversations contains saved chats and Settings. Settings opens Expert mode,
which retains the original raw-token/backend/session controls; Back to Chat
returns to the friendly view. Changing modes unloads the numerical model. The
bundled CPU Tiny fixture is available for integration checks.

Normal launches use `~/Library/Application Support/Toklet`, with `Models/` and
`chat-v1.json`. Show models in Finder opens the actual download directory.
Existing Documents or experiment caches are not moved; use Choose local folder
to open them. External folders must be selected again after relaunch.

Switching to another Mac app keeps work running. Closing the last window cancels
and joins downloads/native work, saves history and releases the worker while the
application stays running. Shift-Command-N opens Toklet again with a fresh worker
and saved draft/history. Quit also joins work before exiting. A save failure
keeps the draft and prevents quit so it can be retried. No chat is automatically
submitted on reopening.

This is a local, ad-hoc signed developer bundle, not a notarized installer. The
repeatable build helper takes an already admitted native package and a fresh
artifact directory:

```sh
python3 tools/experiments/m8_macos_harness_build.py \
  artifacts/m8-mistral-chat-apple-package-v1 \
  37f255ca04d4c839f51ece740bda1d8d2fa0bd026b505a1a551cf50975e3eb04 \
  artifacts/my-new-mac-build
```

That command also requires the prepared, hash-checked Tiny fixture under
`artifacts/m7-native-tiny-v2`. It compiles verified source copies, records command
and bundle hashes, and does not install globally or access an Apple account.

## Checks and evidence

The [receipt](../benchmarks/results/2026-10-23-macos-window-v1.json) binds the final
Mac build, strict Swift coordinator/real-model checks, six Python build-failure
checks, local simulator/unsigned-device SDK builds, and the owned visible app
process. Rust numerical kernels and the admitted native library are unchanged.
Astra reviewed the code and evidence before commit.

The final visible M1 Pro trial used isolated synthetic data. Public search,
partial-download cancellation and a fresh 146,607,552-byte SmolLM2 download passed;
all six files were independently hashed. It opened that model and produced a
real greeting. Native folder selection then opened existing pinned Qwen and the
actual newly downloaded Mistral cache, with complete visible greetings from both.
The Mistral greeting prefix matches the earlier independently decoded controller
result. Stop during another Mistral reply displayed a terminal prefix. Expert
CPU generation produced `47,47`. Closing/reopening preserved a synthetic draft;
quit joined successfully. All four saved conversations were terminal, and the
final run emitted zero stderr bytes. The primary agent observed AX/screenshots;
no stored screenshot files or independent reviewer GUI execution are claimed.

Earlier trials remain recorded: the first automation call outlasted the owned
process watchdog and then reopened a normal-storage app, which was closed without
entering data. Two intermediate UI runs recorded AttributeGraph warnings. A
controlled close trace proved that a later accessibility inspection itself
reopened the window. The final trial uses explicit close traces before reopening.
Zero warnings in its particular action sequence does not establish a universal
SwiftUI warning fix. The history Settings row needed a screenshot coordinate
click during automation; Models controls expose actual accessible button roles.

## Remaining limits

Only the selected pinned SmolLM2 eight-bit, Qwen eight-bit and full-attention
Mistral four-bit text profiles are supported in Models. Search results do not
make arbitrary checkpoints executable. See [Models](model-hub.md) and the
[full summary](final-summary.md) for model-specific budgets, context and numerical
qualification. Wider architectures, held-out Mistral/Smol quality, full performance
matrices and physical Qwen/Mistral phone Chat remain separate open work.

This task qualifies the described Mac functional flow, not UI latency, sustained
performance, all accessibility sizes, or active-generation GUI Quit. The native
coordinator suite separately covers active-chat shutdown, joined download
cancellation, rapid close/reopen, save-failure recovery and damaged-history
preservation. No iPhone was contacted during this add-on task.

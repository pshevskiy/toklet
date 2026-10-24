# Mobile interface follow-up — 24 October 2026

The user reported unreliable model selection/search, short replies, missing
Expert control descriptions, and unformatted code. This task owns the Swift UI,
chat workspace and Hub controllers. The parallel engine chat owns the
[256-token native output extension](chat-output-limit.md) and compatibility
preflight. Both sets of changes share the same checkout; engine files and their
commits are kept separate from this interface change.

## Result

- Automatic model verification no longer locks profile selection or search.
  Replacing a check cancels and joins it before starting the latest operation.
  Stale completions cannot replace the selected profile or clear newer work.
  Downloads and explicit Open keep their captured choice until they finish.
  Cancellation clears operation ownership so another selection remains possible.
- Search immediately includes matching supported profiles, then adds public
  results. Supported matches have a selection action and selected status.
  Unsupported public results explain why they cannot run; searching does not
  establish engine/tokenizer compatibility or expand the pinned catalog.
- Model loading has an explicit state, prevents history changes during a switch,
  and closes Models after success. Reopening the already active model preserves
  its native token prefix and conversation. An invalid folder preserves the
  working model and reports the error.
- A new workspace defaults to 128 reply tokens. Settings offers 32, 64, 128 and
  256, with older saved limits preserved. Output-limit completion has an explicit
  notice, and the toolbar has a Settings shortcut. Prompt/history/output still
  share **512 positions**. Context overflow rejects before generation; no hidden
  truncation or context-policy expansion was added. Models may naturally stop
  earlier. Expert raw generation retains its 128-ID bound.
- Replies support inline Markdown and fenced code, language labels, horizontal
  scrolling, basic lexical syntax highlighting and Copy. Unclosed fences remain
  readable code after cancellation or a response limit. Highlighting preserves
  exact text, including Unicode; it is presentation, not a language parser.
- Expert mode explains model fields, Prompt, Append, Continue, Reset, Cancel,
  Suspend, Replay, Unload and Close. It includes a copyable tiny-fixture example,
  highlighted/copyable emitted IDs and readable stop reasons. The engine's
  metadata inspection section is retained.

## Verification and limits

`tools/experiments/mobile_interface_check.py` snapshots sources, accepts an
explicit native archive, and uses strict Swift 6 concurrency and warnings as
errors. Its ordinary tests need no model download or GPU. Tests cover replacing
a blocked verification, stale results, cancellation cleanup, exclusive/cancelled
Open, case-insensitive supported matches, saved limits, completion notices,
fences and exact highlighted text. An explicit local SmolLM2 run checks actual
load/Send, loading completion, repeated Open and invalid-folder preservation.

Both full Release simulator and unsigned iOS apps build against the same native
package as the controller checks. The root project’s generated XCFramework link
now selects that package too, and the admitted normal-project build helper passes
both platforms (ad-hoc simulator signing and unsigned device output). The build also checks the project registration
of both the engine compatibility module and the new content renderer. The UI
checker emits iOS and simulator modules as separate platform checks.

The [receipt](../benchmarks/results/2026-10-24-mobile-interface-v1.json) links
the exact source/native/app snapshots and logs. Earlier failed attempts are kept:
the Command Line Tools compiler crashed during an actor-method reabstraction,
sandboxed Xcode macros were unavailable, and a full build exposed Swift 5/new SDK
attribute-key and function-reference issues. Explicit closures and attribute
keys resolve those build errors without weakening concurrency checks.

An earlier 128-cap UI simulator build installed/launched, then the isolated
simulator was shut down. Final 256-cap app evidence is build/controller evidence,
not a final installed-app tap-through.
Device Hub's live inspection stalled, so this follow-up does **not** claim a
completed tap-through, physical-iPhone test, largest-text layout check, latency,
memory-limit or answer-quality qualification. Offscreen light/dark samples inspect
highlighted text at phone width only; ImageRenderer omits native scroll/button
surfaces, so these exports are not full app screenshots. Earlier omitted-surface
exports remain in the failed/partial diagnostic history.

For reproduction with a fresh output directory:

```sh
python3 tools/experiments/mobile_interface_check.py \
  artifacts/m8-chat-output-v1/package/Toklet.xcframework/macos-arm64/libtoklet.a \
  artifacts/mobile-interface-check-new \
  --model models/MLX-SmolLM2-135M-Instruct-8bit-0f0d9b8
```

Omit `--model` for controller/rendering/platform checks without real-model work.
Xcode's compiler plugin process and an explicit GPU trial need the corresponding
local execution access. This task makes no numerical-kernel or model-quality
change.

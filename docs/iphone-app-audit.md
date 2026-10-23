# iPhone app function and button audit

23 October 2026. The user requested a check of every function and button on the
paired iPhone 14 Pro. This audit is **in progress**. It separates controller
checks, older installed-app observations and current-build physical verification.
No existing owner chat or model cache has been removed or exported.

The installed phone initially showed the older Smol-only UI. Device Hub visual
checks opened Models, verified/opened its saved Smol snapshot, created a test
conversation while preserving the previous draft, submitted a synthetic prompt
and observed a real reply, and opened History, response-limit options, Settings
and Expert mode. A second Send/Stop tap arrived after the short reply finished;
it is not evidence of stopping an active reply. Remote hardware input used the
phone’s Russian layout, so that synthetic greeting is not an English oracle.

A definite Expert bug was reproduced: tapping Tiny fixture changed the directory
from the app bundle to the app’s Documents path. Default Form styling activated
both buttons in the same row. This also endangered Prompt/Append/Continue and
Reset/Cancel. The corrected row style gives each action an independent bordered
button, and stacks actions at accessibility text sizes.

The [shared controller fixes](app-recovery.md) are committed separately. Further
UI fixes keep model controls above search results, dismiss the search keyboard
on submission, show folder errors in Models, ignore ordinary Files cancellation,
present delete confirmation from the visible History sheet, and expose History
New/Settings as individual Mac buttons. Model headings truncate visually while
retaining their full accessible name. Conversation scrolling starts at the
latest saved content, follows a new Send and completion, and avoids forcing
completion/resize scrolling during manual tracking or scroll-back.

## Physical checklist

The corrected signed app was installed in place using the reviewed physical
preparation tool, then launched normally through Device Hub. The observations
below are primary visual observations in the computer-use transcript; screenshots
were displayed there, not saved as separate artifact files. “Open” denotes a
remaining physical check. The Mac locked again after the local-folder Open action;
that action completed after unlock and displayed the loaded Smol model in Chat.

| Screen / control | Coverage so far | Corrected phone build |
| --- | --- | --- |
| Welcome Choose a model | Older installed app opens Models | Open; toolbar chooser works |
| Toolbar model chooser | Source and existing shared controller checks | Observed: opens Models |
| Toolbar New chat | Older app creates a separate draft; loaded-model association has real-model regression | Observed: new separate chat retains live Smol model |
| Composer / keyboard / Send | Older app accepted synthetic text and produced a reply | Observed: Qwen two-turn exchange; synthetic Russian greeting and story request |
| Stop during active reply | Real-model controller cancellation after emitted tokens passes | Observed on Smol: Send → Writing → immediate Stop preserves partial text and displays Stopped; earlier Qwen attempts completed before Stop |
| New message/completion scrolling | Reviewed UI implementation; no final visual claim | Observed: longer Qwen exchange overflows; subsequent Send and completion bring latest text into view |
| Scroll-back / keyboard resize / restored history | Reviewed UI implementation; no final visual claim | Observed scroll-back retains older content through completion; new Send follows latest; software-keyboard resize and long restored-history checks remain |
| Conversations / select saved chat / Close | Older list opened; controller selection tests pass | Observed: History opens, selecting saved interrupted audit turn restores partial text and Stopped |
| History New chat | Real-model association and storage checks pass | Observed: separate empty conversation retains live Smol |
| Swipe/context Delete / confirmation Cancel | Synthetic controller deletion passes; sheet presentation requires observation | Swipe and long-press context menu both present confirmation; outside tap cancels without deletion |
| Delete confirmation final action | Only a newly created audit conversation may be used; no owner chat deletion | Observed after explicit user approval: only the newly created Smol interrupted test chat removed; other rows retained |
| Settings / Close | Older installed navigation works | Settings opens; its Close button is not yet checked |
| Response limit 32 / 64 / 128 | Older menu contains all options; bounded settings storage checks pass | All32/64/128 selections reflected in Settings; original64 restored |
| Expert mode / Back to Chat | Older app enters Expert; shared mode-switch checks pass | Observed both transitions, including return after terminal Close worker |
| Models Close / recheck saved snapshot | Existing snapshot/controller validation passes | Close observed; explicit recheck returns enabled verified Qwen Open |
| Public search / keyboard Search / Search button | Real Mac service/controller tests pass; phone input/result flow needs current check | Both submit routes return public results; exact Qwen query obtained with synthetic clipboard input |
| Search Select for chat / model picker | Three exact profiles in current catalog; stale verification regression passes | Three picker choices observed; supported Qwen Select for chat tapped; selection outcome still needs top-of-sheet check |
| Download / progress / Cancel / retry | Existing Mac transport/controller and earlier Smol phone tests; current phone sweep still required | Qwen download completed and verified; progress bar observed; cancellation/retry not caught before completion |
| Open downloaded model / Unload model | Older Smol open works; current real-controller opening/release passes | Qwen Open reaches Model loaded and replies; friendly Unload removes loaded controls; Qwen reload succeeds |
| Smol, Qwen and Mistral fit labels | Exact profile admission tests pass; device denial is a valid result | Qwen estimated fit and real load; Mistral memory denial prevents Download; Smol final fit check open |
| Local folder / successful choice / cancellation / invalid folder | Invalid/stale folder controller checks pass | Files picker opens and cancels normally without error; saved test Smol folder opens successfully; invalid selection open |
| Storage retry button | Synthetic malformed-history and failed-write recovery checks pass; no owner file corruption injected | Controller-only: not injected into owner history |
| Expert Tiny fixture / Documents model | Tiny button bug physically reproduced before correction | Observed separate Documents and bundled paths; Tiny then loads successfully |
| Expert Metal / MLX toggles / packed folder | Existing numerical routing and validation checks; final UI actions needed | MLX toggle shows requirement help; Tiny resets MLX; Metal toggle loads Tiny to Ready; Metal generation and packed-path edit remain open |
| Expert capacity / chunk / budget / Load | Shared worker bounds and bundled fixture checks | CPU default32/chunk3/128MiB loads; editable16/chunk2/129MiB loads Metal Tiny to Ready (67,177,176bytes); boundary inputs remain automated-only |
| Expert prompt IDs / stop IDs / output limit | Existing native fixture contracts | Default prompt1,7,13,3 / no stops / maximum2 succeeds; other edits open |
| Expert Prompt / Append / Continue | Independent button styling added; raw generation contracts already checked | Observed independent histories6/5/1 then12/11/1 then14/13/1; each returns47,47 |
| Expert Reset / Cancel | Independent button styling added; native state contracts already checked | Idle Cancel preserves14/13/1; Reset clears to0/0/0; active Expert cancellation remains controller-only |
| Expert Suspend / Replay / Unload / Close | Existing lifecycle/coordinator checks pass | Observed release, Replay restores14/14/0, Unloaded, Closed; disabled states correct |
| Background / foreground / explicit reload | Shared native lifecycle checks; physical current-build sweep remains open | Home and foreground return retain the long conversation and scroll position; post-background new Send not yet exercised |

## Verification limits

The [UI build receipt](../benchmarks/results/2026-10-23-iphone-ui-controls-v1.json)
records exact builds and review scope. It does not claim all buttons pass from
compilation. The preparation receipt proves in-place installation; ordinary launch and
individual actions are recorded separately from automated numerical qualification. The current pinned native library and Rust numerical
kernels are unchanged. Larger accessibility sizes, software-keyboard resizing, long restored history
and invalid-folder selection remain open; user content is not used
for destructive or injected-failure tests.

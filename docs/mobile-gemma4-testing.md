# iPhone UI and Gemma 4 compatibility checks

24 October 2026. The reviewed mobile dashboard app (`abe661e`) with the frozen
native package `2d7d35b0` is installed on iPhone 14 Pro / iOS 27.0.1. Four physical
XCUITests passed, with no failures or skips. Normal launch succeeded afterward;
the owner's saved chat file was byte-identical before and after the suite.
Tests used isolated temporary histories, including deletion checks.

The [receipt](../benchmarks/results/2026-10-24-mobile-gemma4-phone-v1.json)
binds the signed app, compiled sources, native package, result bundle extracts,
screenshots and native compatibility probes. No engine implementation changed.

## Physical checks

| Check | Observed result |
| --- | --- |
| Monitoring | RAM used / allowed, first-token time and decode tok/s are visible. Timed values appear after generation. |
| Find best model | Opens the supported-device shortlist and explains the three-profile catalog. No five-model or quality ranking claim. |
| Local benchmark | SmolLM2 completes one warm-up and three measured independent requests, each producing 64 tokens. All three result rows are visible. |
| Model switching | SmolLM2 to Qwen2.5 and back, repeated replacements and selecting the active model complete; both models generate replies. |
| Chat code | Qwen generates fenced Python; a displayed block has a visible Copy action. Screenshot inspection confirms colored numbers/comments. |
| Longer reply | Qwen reaches the selected 256-token output cap and displays the response-limit notice. This is not a 128K reply. |
| Expert | Tiny CPU fixture returns `47, 47`; colored numeric output and an enabled Copy action are visible. Function/lifecycle descriptions and return to Chat remain available. |
| Dialog deletion | Creating three dialogs, cancelling/confirming one deletion, then cancelling/confirming delete-all leaves one empty dialog. |
| Gemma 4 discovery | Searching `gemma-4` shows E2B/E4B results as unavailable for Chat. Closing search preserves the loaded SmolLM2 model. |

The short benchmark's displayed samples are 364/362/361 ms to first token,
18.0/17.8/17.9 decode tok/s and 3.87/3.89/3.88 s complete-command duration.
These rounded UI observations describe this session. They do not establish
controlled performance, quality, peak memory, thermal or other-device results.
The later dashboard capture shows approximately 203 MB used / 3 GB allowed;
the OS allowance is instantaneous, not a reservation or a Gemma memory estimate.

## Gemma 4 cannot run yet

The installed engine has no native Gemma 4 graph or admitted Chat/tokenizer
profile. A supported-looking search result does not add those mechanisms.
The config-only checks pin:

- `google/gemma-4-E2B-it` at `3e22461f65e89153144f8adb70e3b8c2cc9845a7`.
- `mlx-community/gemma-4-e2b-it-4bit` at `238767527555cb75a05732a84dff5d6ba0dd6809`.

Both configurations declare `gemma4`. The actual frozen native C ABI inspector,
on scalar and Metal, reports `unknown_architecture`; load admission and model
qualification are not evaluated. Actual native loading rejects each with status
2 and creates no model handle. Model weights were not downloaded, no tokens were
generated, and neither memory fit nor numerical generation was qualified.

The pinned [official config](https://huggingface.co/google/gemma-4-E2B-it/raw/3e22461f65e89153144f8adb70e3b8c2cc9845a7/config.json)
sets the text context to 131,072 total positions. That alone cannot accommodate
131,072 reply tokens plus a nonempty prompt. Current app/native limits remain
256 Chat output tokens, 128 raw Expert output tokens and 512 Chat positions;
the [single-reply 128K work](long-reply-engine-plan.md) remains separate.

## Reproduction and retained failures

The signed `build-for-testing` snapshot and strict signature checks passed.
Run the four `PhoneUITests` methods in
`apple/tests/ui/ModelSwitchUITests.swift` with `test-without-building`, parallel
testing disabled, the paired phone destination and the snapshot's corrected
`.xctestrun`. Retained `build_phone.py` and `test_phone_v2.py` under
`artifacts/mobile-gemma4-testing-2026-10-24-v1/` record the actual commands.
Their private settings contain device/team identifiers and must stay local.
The wrapper hashes the owner history before/after and normally launches the
updated app on success; on failure it restores the previously verified app.
Future runs must refresh source/signature pins and the rollback baseline.

The initial attempt failed before any test executed because the Developer App
certificate was untrusted. Its result bundle/logs and explicit successful
owner-history comparison remain preserved. After the user trusted the app on
the phone, the fresh `attempt2/phone-ui-v2.xcresult` passed all four methods.
An initial Python HTTPS metadata request also failed local certificate-chain
validation; metadata was fetched with system curl and full TLS verification.
No certificate verification was disabled.

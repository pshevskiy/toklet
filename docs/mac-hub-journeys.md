# Adopted Mac model download, Stop and offline journeys

The shared controller journey now selects an explicit pinned profile with
`--profile smol|qwen|qwen3|mistral|llama3`. The default Smol and legacy Llama
selector remain supported. Unknown or conflicting requests fail before creating
a cache. Fourteen cold controls check the exact Hub/native profile mapping.

Fresh Mac runs pass remote exact-repository search, Hub search, device admission,
download, full-file/receipt verification, audited open and nonempty Chat:

| Profile | Verified files | Download bytes | Stop emitted tokens |
| --- | ---: | ---: | ---: |
| Qwen2.5-0.5B eight-bit |6 |532,130,488 |1 |
| Qwen3-0.6B eight-bit |6 |644,927,148 |1 |
| Mistral-7B-v0.3 four-bit |6 |4,079,633,318 |1 |
| SmolLM2-135M eight-bit |6 |146,607,552 |1 |

Each uses an absent isolated cache. Mistral exercises the production large-file
ranged assembler. After initial Chat, the production Stop action cancels an
emitting long response, preserving `history = evaluated + pending`, with at most
one pending token and an exact saved prefix. Another Chat request succeeds.
These are controller actions, without a new rendered-UI claim.

A separately launched process denies all networking. A local write and numeric
loopback EPERM probe independently establish confinement; the pinned metadata
request fails. The same saved snapshot re-verifies and Chat succeeds. Missing-file
and same-size weight corruption in private copies are rejected before the open
callback. Active identity, session, conversation and prefix stay intact; the
original cache re-verifies and recovery succeeds. All owned online/offline
processes exit cleanly and all source/binary guards pass.

Together with [Llama's fresh/offline journey](llama3-download-offline.md) and
[unflagged Stop/recovery/adoption](llama3-mac-adoption.md), these records cover
all five ordinary Mac profiles. They retain each artifact's numerical/quality
limitations. In particular, Qwen3's separate C512 fidelity rejection and visible
experimental notice remain; functional Chat does not supersede them.

Strict Mac helper compilation, existing Hub controls, selector controls and combined Astra code/results review pass.
An optimized Mac app is also built from unmodified production sources and the
coherent native40 package. Its entry has no fixture root, seed model or launch
flag. Architecture/platform and source/package checks pass. It was not launched
or installed over a user session. [Built app](../artifacts/mac-hub-journeys-2026-11-04/production/Toklet.app).

No numerical implementation or production Swift source changes in this task.
The latest [physical iPhone14Pro query](../artifacts/m8-phone-availability-2026-11-04/availability.json)
still times out; new phone Settings/lifecycle evidence remains open. These Mac
journeys provide no new whole-process peak-memory, throughput, larger-window,
full quality or arbitrary Hugging Face model qualification.
[Evidence](../benchmarks/results/2026-11-04-mac-hub-journeys-v1.json).

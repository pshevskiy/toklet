# Hugging Face models in Toklet Harness

## Download core — 23 October 2026

Toklet separates public Hugging Face search results from profiles qualified for
text chat. The first downloadable profile is
[mlx-community/SmolLM2-135M-Instruct-8bit](https://huggingface.co/mlx-community/SmolLM2-135M-Instruct-8bit/tree/0f0d9b8218915bc34d401e1a340b8c049d300d5e),
pinned to commit `0f0d9b8218915bc34d401e1a340b8c049d300d5e`.
Its six required files total 146,607,552 bytes. Their trusted sizes/SHA-256 values
come from the previously acquired, independently qualified snapshot; remote
search metadata cannot grant compatibility or replace these identities.

The cold Swift actor queries the public [Hub API](https://huggingface.co/docs/hub/api),
limits search to 20 MLX-tagged results and 2 MiB metadata, then admits the immutable
revision and complete required inventory before transfer. A streamed ephemeral
URLSession writes one file at a time, accepting HTTPS only within the official
Hugging Face service/CDN domains. The [official download documentation](https://huggingface.co/docs/hub/models-downloading)
explains those redirects. Standard TLS validation stays enabled; credentials,
cookies, repository scripts, pickle files and remote code are not used.

Downloads require a unified-memory GPU, at least 2 GiB physical RAM and a current
768 MiB available-process allowance supplied by the caller. This is conservative
admission for the selected configuration, not a device qualification or a promise
of fit. Final numerical loading retains its separate 512 MiB budget and 64 MiB text
allowance. Storage admission reserves two full snapshot sizes, metadata and 64 MiB
headroom before starting; checks repeat between files and during streaming.

Each operation writes to a fresh staging directory. HTTP failures, unsupported
redirects, dishonest lengths, hash mismatch, storage exhaustion and observed
cancellation prevent publication. Cancellation awaits transfer termination and
sink closure before cleaning staging. A synchronous same-root exclusive rename
publishes only a fully verified snapshot and receipt. Cancellation observed
before that point publishes nothing; cancellation arriving afterward can leave
a complete valid snapshot. Existing destinations are never replaced. A damaged
saved snapshot is preserved and rejected, requiring removal/re-download rather
than silently trusting its old receipt.

Before opening a saved model, the core rechecks its entire required inventory,
receipt and file hashes. Directory/file descriptors use no-follow flags; regular
file type and bounded length are checked with `fstat` on the descriptor used for
reading. Nonblocking opens reject FIFOs without waiting. Bounded reads and a
post-read change check prevent path replacement from redirecting validation to
another file. Protected Apple `/var` and `/tmp` aliases are recognized explicitly;
user-controlled links in the model-storage path reject before directory creation.
This validates the snapshot at that boundary; it is not protection against a
separate actor editing weights after validation.

The core and Models-screen integration are separate reviewed code tasks.
The integration and offline opening are described below.
See the [improvement record](improvements.md) for retained failures and decisions.

Strict Swift 6 compilation, deterministic offline checks and iOS type-check pass.
The real Mac URLSession test searches the public Hub, downloads the pinned six
files and verifies the saved snapshot again. Independent Python hashes also
match all six previously qualified local files. This establishes download and
storage behavior; it runs no numerical inference and establishes no performance
or physical-device limit. The [core receipt](../benchmarks/results/2026-10-23-model-hub-core-v1.json)
retains source copies, raw commands/results and earlier failed checks.

## Models screen and offline opening

In Toklet Harness, choose **Models** from the Chat header. Search public MLX
models by name, or use the **Available for chat** card without searching first.
Other results explain that their checkpoint/engine/text profile is not qualified;
they do not offer a misleading download/open action. The supported card shows the
repository, tested revision, 146.6 MB size and Apache-2.0 license.

Choose **Download model** to see received bytes and Cancel. Cancellation waits
for transfer shutdown; complete prior snapshots stay saved. The operation runs
while the app is open, cancels on background/Expert transition and does not
resume automatically. Transfers retain a 60-second request timeout and a bounded
two-hour resource timeout to allow slow links; cancellation remains available.
No resumable background service or process-kill recovery
is qualified. After a successful download, choose **Open downloaded model**.
Every opening rehashes the snapshot without contacting Hugging Face, then uses
the same FriendlyChatStore/native worker as local models. Downloading never
loads a second numerical model automatically. Model changes, unload and Expert
switches invalidate pending opening so late verification cannot replace a newer
choice. A fresh memory check follows verification, before opening.

Saved models live under **Files → Toklet Harness → TokletModels**. To repair a
damaged saved snapshot, remove its complete model folder there and download it
again. Existing folders are never silently overwritten. The normal local-folder
picker and Expert mode remain available independently.

The current iOS admission reads `os_proc_available_memory()` at the operation
boundary, separately from physical RAM and unified-GPU capability. The Mac runner
uses the plan’s configured process allowance `min(RAM/2,6GiB)` minus its current
TASK_VM_INFO footprint; that value is not free system memory. These checks remain
estimates, and the native loader still performs final numerical admission.

The app diagnostic `--hub-self-test` uses the fixed public profile and normal
model-download directory. It prints `TOKLET_HUB_CHECK_JSON`, includes snapshot
bytes and an explicit reused-snapshot flag, and proves signed bundle identity
before/after. It runs native text checks, then exercises the actual HubStore →
FriendlyChatStore → worker path with a temporary chat history and a network
handler that rejects every request. This demonstrates offline controller use;
it does not visually qualify physical taps, keyboard layout or every lifecycle
transition. Owner conversations are not read or replaced by that diagnostic.

The separate `--hub-download-self-test` diagnostic creates a new exclusive UUID
temporary directory, requires a genuinely fresh transfer and removes that entire
owned test directory before reporting success. It leaves the normal downloaded
cache intact. The distinction prevents an already saved model from being
misreported as a new phone download.

The final source passes strict Swift 6 controller/chat/raw CPU/Metal checks and
both iOS SDK builds. A real Mac search and download validates the pinned files;
final Mac controller checks reopen that verified snapshot. A signed iPhone 14 Pro
trial performs a genuinely fresh public download, verifies all files and matches
the Mac native and full 32-ID offline-controller replies. It exits naturally after
releasing resources and removing its test tree. The [app receipt](../benchmarks/results/2026-10-23-model-hub-app-v1.json)
preserves exact source/build identities and failed attempts. GUI taps and
keyboard/accessibility layouts are separate, unqualified boundaries; this adds
no performance or answer-quality claim.

## Qwen catalog extension — 23 October 2026

Models now selects between **SmolLM2-135M eight-bit (Llama family)** and
**Qwen2.5-0.5B eight-bit**. A supported search result can select its card; other
checkpoints remain visibly unsupported. Qwen is pinned to
`6474d4c1a05bd7bd634a9c993560e4e7272eda3d`; its six app files total 532,130,488
bytes, with exact identities taken from the previously qualified local snapshot.
The rejected Qwen four-bit candidate is not enabled.

Qwen checks a current 1.5 GiB process allowance and reserves 1 GiB for numerical
loading plus 128 MiB for text. SmolLM2 retains its 768 MiB admission, 512 MiB
numerical budget and 64 MiB text reservation. These are explicit conservative
admission policies; physical peaks and wider device fit are separate. Local
folder opening identifies a pinned configuration after acquiring Files access,
checks current memory again, and uses the same per-profile settings. The worker
queries the native stop set once when loading: Qwen’s token 0 remains ordinary
visible text, and its stops are 151643/151645.

Choosing another profile while idle refreshes that profile’s saved snapshot.
Selection stays disabled while an operation runs; Cancel remains available.
Old SmolLM2 snapshot receipts keep exactly the same serialized schema and remain
usable. Diagnostic commands optionally take the fixed key `qwen` or `smol`;
arbitrary repositories cannot enter that route.

The [Qwen app receipt](../benchmarks/results/2026-10-23-qwen-model-hub-app-v1.json)
records a genuinely fresh Mac transfer, full re-verification, native multi-turn
checks and offline-controller generation. Its first eight IDs exactly match the
committed native check, including visible token 0; all replies independently
decode with the pinned Python tokenizer. The same new package re-opens the old
Smol snapshot and passes native/controller checks. Qwen phone Chat remains
pending: installation refused an existing app session and did not stop it.

## Bounded larger-file admission — 23 October 2026

The download core now admits at most4GiB per file and6GiB per validated snapshot,
using shared explicit64-bit limits. This is a prerequisite for the selected
multi-gigabyte Mac candidate; the supported catalog is unchanged. Storage still
reserves two complete snapshot sizes plus headroom, and streaming still rejects
excess bytes before writing. Profile memory checks and all integrity, redirect,
cancellation and exclusive-publication gates remain in force.

The [limit receipt](../benchmarks/results/2026-10-23-model-hub-large-limits-v1.json)
records strict Swift compilation, the complete deterministic core suite and SDK
type checking. Added checks cover exact4GiB/6GiB boundaries, excess and extreme
signed sizes, a4GiB request reaching its owned sink before a mocked HTTP failure,
and an oversized request rejecting before creating a file. These use small
payloads and establish admission only. They do not qualify an actual multi-GB
URLSession transfer, new model support, physical memory fit or performance.

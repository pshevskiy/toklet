# Hugging Face models in Toklet Harness

## Download core — 23 October 2026

Toklet separates public Hugging Face search results from profiles qualified for
text chat. The first downloadable profile is
[mlx-community/SmolLM2-135M-Instruct-8bit](https://huggingface.co/mlx-community/SmolLM2-135M-Instruct-8bit/tree/0f0d9b8218915bc34d401e1a340b8c049d300d5e),
pinned to commit `0f0d9b8218915bc34d401e1a340b8c049d300d5e`.
Its six required files total146,607,552 bytes. Their trusted sizes/SHA-256 values
come from the previously acquired, independently qualified snapshot; remote
search metadata cannot grant compatibility or replace these identities.

The cold Swift actor queries the public [Hub API](https://huggingface.co/docs/hub/api),
limits search to20 MLX-tagged results and2MiB metadata, then admits the immutable
revision and complete required inventory before transfer. A streamed ephemeral
URLSession writes one file at a time, accepting HTTPS only within the official
Hugging Face service/CDN domains. The [official download documentation](https://huggingface.co/docs/hub/models-downloading)
explains those redirects. Standard TLS validation stays enabled; credentials,
cookies, repository scripts, pickle files and remote code are not used.

Downloads require a unified-memory GPU, at least2GiB physical RAM and a current
768MiB available-process allowance supplied by the caller. This is conservative
admission for the selected configuration, not a device qualification or a promise
of fit. Final numerical loading retains its separate512MiB budget and64MiB text
allowance. Storage admission reserves two full snapshot sizes, metadata and64MiB
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

The current task supplies the tested core. Models-screen integration, downloaded
model opening and physical app tests follow as a separate reviewed code task.
See the [improvement record](improvements.md) for retained failures and decisions.

Strict Swift6 compilation, deterministic offline checks and iOS type-check pass.
The real Mac URLSession test searches the public Hub, downloads the pinned six
files and verifies the saved snapshot again. Independent Python hashes also
match all six previously qualified local files. This establishes download and
storage behavior; it runs no numerical inference and establishes no performance
or physical-device limit. The [core receipt](../benchmarks/results/2026-10-23-model-hub-core-v1.json)
retains source copies, raw commands/results and earlier failed checks.

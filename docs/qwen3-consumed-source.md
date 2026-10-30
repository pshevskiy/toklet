# Qwen3 consumed-source identity

The opt-in Rust `mlx::qwen::Source::open_qwen3_audited` reader authenticates the
config, index and original container bytes consumed from the existing pinned
`mlx-community/Qwen3-0.6B-8bit` revision
`11de96878523501bcaa86104e3c186de07ff9068`.
[Saved verification](../benchmarks/results/2026-10-30-qwen3-consumed-source-v1.json).
Ordinary and existing Smol loading retain their previous behavior.

The reader checks the source budget before scanning the container. A bounded
provisional header supplies complete range boundaries; the retained descriptor's
full-file SHA-256 prepass authenticates those ranges. Verified prefix/header
redelivery precedes final parsing. Subsequent whole-tensor visitors rehash the
exact chunks supplied to their consumers. Errors, changed bytes, incomplete
coverage and consumer errors or panics prevent later identity publication.
Partial destinations must be discarded on failure. Qwen's existing logical
Whole upload path is supported; row-range requests explicitly reject.

The six-file canonical manifest is an **expected distribution anchor**. This
source receipt proves successful config/index/container delivery only. It does
not prove tokenizer consumption, ancillary-file verification, model assembly,
codec pairing, resident model identity or eligibility to restore preferences.
Copied receipts remain historical evidence after later disk changes or errors.
Paths, timestamps and instance IDs are not content authentication.

The cold audit retains a 64 KiB buffer and bounded range hashes, replacing the
opt-in source's old stream buffer. It performs one extra full-file hashing pass
and no per-token work or whole-model copy. The existing 64 MiB source metadata
allowance remains an estimate; retained Rust owner bytes are a separate metric.
Parent symlinks are permitted, and a retained handle is not a directory snapshot.

Verification on 30 October 2026 passes formatting, strict default/Metal Clippy,
486 default tests and 594 Metal-feature tests, including the same 17 new portable
controls in both suites. Their respective 21 and 187 ignored tests remain
explicitly separate. One release-mode cached source probe also passes:

| Observation | Result |
| --- | ---: |
| Original container bytes | 633,442,994 |
| Delivered tensor payload bytes | 633,364,480 |
| Tensor ranges | 704 |
| Ranges including prefix/header | 706 |
| Successful verified reads | 706 |
| Retained audit owner bytes | 99,840 |
| Identity absent after invalid public read | yes |

The runner independently reconstructs the canonical manifest from the existing
Swift catalog: `5e9aafa305e0e48182a856cafe3ae2b597599033e61b16664f42b09a4af5ff51`.
All eight owned commands exit successfully with closed process groups. These
ordinary source checks do not freeze the complete toolchain/runtime or qualify
GPU execution, generation, performance, rendered UI or iPhone behavior. The
Apple package, native factories, model/codec pairing and preferences are unchanged.

Next: attach source evidence after successful assembly, authenticate the actual
codec parser buffers and bind the exact model owner. Then restore untrusted
settings by accepted artifact/backend/load/policy/schema identity with fresh
context and memory admission.

The later [Qwen2 extension](qwen2-consumed-source.md) extracts the same concrete
cold reader for engine-owned Qwen profiles. The existing Qwen3 accessor remains
profile-filtered; the neutral accessor covers either admitted source.

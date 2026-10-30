# Shared Mistral consumed-source verification

The opt-in Rust `mlx::qwen::Source::open_mistral_audited` route uses one cold
`DenseAudit` for the pinned Mistral distribution and existing Qwen profiles.
`DenseProfile` selects an engine-owned manifest; the parsed architecture must
match before source inventory construction. Existing Mistral untied output,
bias-free projections and disabled sliding-window contract are preserved.

The shared source receipt is named `DenseSourceContentIdentity`. The existing
`QwenSourceContentIdentity` alias keeps its fields, methods and const behavior.
Qwen getters are explicitly filtered to their supported profiles; the neutral dense getter
and Mistral-specific getter expose the new source evidence. Mistral completion
cannot silently become Qwen identity. The codec/Metal `QwenProfile` remains its
original two-case type, with no changes to those consumers in this task.

FileAudit, one bounded prefix/header plan, complete contiguous physical coverage,
verified prefix/header delivery, repeated range checks and sticky invalidation
retain their existing mechanism. Source budget checks remain before metadata
opening and prepass. Whole logical reads are supported; arbitrary row reads
remain rejected. Existing128KiB header,4096range,4GiB file and6GiB manifest
limits are unchanged. No second full post-consumption pass, whole-model copy or
hot-loop dispatch is added.

The six-file manifest is an expected distribution anchor. Source completion
proves config/index/container delivery only: it does not prove tokenizer or
ancillary consumption, model assembly, codec pairing, resident model identity
or preference eligibility. Numerical kernels, native exports, selected37package,
Swift and app settings remain unchanged by this source-only task.

[Saved verification](../benchmarks/results/2026-10-30-mistral-consumed-source-v1.json)
passes all eight jobs: fmt, strict default/Metal Clippy,530default/638Metal-feature
tests and one exact release default-CPU cached stream. Seven new portable
controls pass in both suites;23default/192Metal ignored tests remain separate.
All eight current process groups close without endpoint errors.

| Stream observation | Result |
| --- | ---: |
| Original container bytes | 4,077,480,215 |
| Delivered tensor payload bytes | 4,077,395,968 |
| Tensor / physical ranges | 743 / 745 |
| Verified container reads | 745 |
| Retained audit owner bytes | 101,712 |
| All Qwen getters absent / identity poisoned after invalid read | yes / yes |

The runner independently reconstructs the canonical six-file catalog digest
`0f543bc8679995fe028857674e6e327d5baa6832453587adda92b0c45e4f4a26`
and checks the three delivered-file hashes. Its6GiB source allowance/context512
is an experiment setting, not a model/window or whole-device admission.

The first default-library attempt had338passes/seven failures at three stale
synthetic helper calls. Their artificial manifest correctly fails the filtered
Qwen getter. A test-only neutral-getter migration changes three calls in two
files; assertions, original eleven source paths and production filtering stay
exact. The failed five-group receipt is preserved. An earlier private draft
parser-name mistake was corrected before freeze, with no target/live impact.

Astra independently audits current source/inputs/logs, canonical raw evidence,
eight current and five failed closed process groups. Final publication review
precedes commit. The stream retains logs, not a
separate executable copy. Fresh computer-use inventory still reports the Mac
locked; no app was selected/launched or UI input sent. No device/performance
claim follows from this source-only check.

Next: successful Mistral model assembly, complete-codec verification and exact
owner-bound accepted identity, then shared Swift settings/package integration.

# Qwen2.5 consumed-source identity

The opt-in Rust `mlx::qwen::Source::open_qwen2_audited` reader verifies the
config, index and original container delivery for the existing pinned Qwen2.5
0.5B eight-bit model. It reuses Qwen3's concrete cold audit orchestration.
The engine selects the expected manifest and rejects a mismatched parsed
architecture before constructing the source inventory. No new per-token
selection or numerical path is introduced.

The public receipt type keeps its existing fields and methods. The neutral
`qwen_content_identity` accessor covers either admitted Qwen profile;
`qwen2_content_identity` and the existing `qwen3_content_identity` remain
profile-specific. Ordinary shape-compatible files do not receive audited
identity. Qwen2 retains tied output and QKV biases; its route does not acquire
Qwen3's q/k normalization.

The reader preserves budget checks before metadata opening and the container
prepass, bounded verified header delivery, exact contiguous full physical
coverage, repeated consumed-range verification and sticky invalidation after
errors or invalid public reads. Whole logical reads remain supported; arbitrary
row reads remain rejected. Existing file/range/header bounds are unchanged.
The audit retains a reusable64KiB reader and range hashes, with no whole-model
copy or second post-consumption whole-file pass.

The six-file manifest is an expected distribution anchor. Completion here
proves only config/index/container delivery. It does not prove model assembly,
tokenizer or ancillary consumption, codec pairing, resident ownership or
preference eligibility. Native exports, the selected36-export Apple package,
Swift integration and numerical behavior are unchanged by this task.

[Saved verification](../benchmarks/results/2026-10-30-qwen2-consumed-source-v1.json)
passes all eight commands: formatting, strict default/Metal Clippy,516default
and624Metal-feature tests, plus one exact release-mode cached source stream.
Six new ordinary controls pass in each suite;22default/190Metal ignored tests
remain separate. All eight process groups close without endpoint errors.

| Source-stream observation | Result |
| --- | ---: |
| Container bytes | 525,045,902 |
| Delivered tensor payload bytes | 524,976,896 |
| Tensor / physical ranges | 628 / 630 |
| Verified container reads | 630 |
| Retained audit owner bytes | 96,192 |
| Qwen3 accessor absent / identity invalidated after invalid read | yes / yes |

The runner independently reconstructs the six-file canonical catalog digest
`e92e6d359697d8ada4714ff4f5ff618ba8d9fc69d7abc24bfb4ec905eae6c59b`
and checks all three delivered-file hashes. Source allowance1GiB/context512
are this cold experiment's settings, not a new model/window admission.
Independent Astra saved-result audit verifies the source, logs, raw receipt and
all eight closed process groups. Final publication review precedes commit. No numerical,
performance, model-memory, rendered UI or iPhone result follows from this stream.

Next: attach evidence after successful model assembly, verify complete codec
assets through their actual parser buffers and bind the exact resident model
owner before adding shared accepted-load/preferences integration.

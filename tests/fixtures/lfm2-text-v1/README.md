# Pinned LFM2 codec reference

Byte-identical codec-only export/replay for
`mlx-community/LFM2-350M-4bit@18dc72abf3b2337f9123cfd6eeeb58dfa7947066`.

- [fixture.json](fixture.json): ten fixed chat cases, four plain inputs,
  actual token IDs/whole-prefix decodes, stop/mapped-ID controls and provenance.
- [fixture.raw.jsonl](fixture.raw.jsonl): all 539 raw observations, preserved
  before fatal checks, including 505 reserved-literal rejections.

Manual rendering is independently compared with the pinned Jinja template;
Tokenizers 0.22.2 supplies encoding and raw ByteLevel decoding. Astra independently
reconstructs 29 saved decode comparisons from the original vocabulary, without
importing tokenizer/template packages. Stable fixtures exclude per-run cache
paths and runtime observations. Source identity/runtime admission remain in
the fixture; this is not a second independent BPE implementation.

See the [report](../../../docs/lfm2-text-reference.md) and
[result](../../../benchmarks/results/2026-10-29-lfm2-text-reference-v1.json).
No model, Rust codec/app profile, native Chat or phone qualification follows.

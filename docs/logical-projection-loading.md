# Cold logical projection loading

B2b1 separates physical checkpoint tensors from the logical tensors consumed by
the ordinary decoder. Current production families receive identity mappings;
Phi3 admission remains B2b2. This is loading/planning work, with no new numerical
kernel, graph interpreter or token-time lookup.

One crate-private `qwen::logical` module describes `Whole` or original `Rows`
selections and an effective tensor spec. A row partition must cover its complete
rank-two parent in adjacent ordered nonempty ranges. Names must be unique;
shapes, sizes, count bounds and arithmetic are checked before publication.
Logical unpadded bytes must equal the unique physical payload bytes. The source
map retains original names and quantization overrides. No weight bytes are
copied into the plan. The map is built after redundant tied-head validation.

The existing scalar reader and load estimator, Metal planner and uploads, and
compatibility representation grouping consume that same plan. A `Whole` read
uses the unchanged original reader, preserving audited content identity. A
`Rows` read uses the [checked range reader](fused-row-source.md), retaining its
discard-on-error and audit-rejection contracts. Cold name/range clones are
discarded with the source; completed models retain concrete buffers.

Scalar admission counts all logical FP32 result owners and the largest selected
packed temporary. Metal counts every actual code and padded coefficient owner;
it does not round the physical parent once and reuse that incorrect total. A
synthetic four-bit/F16 example has80 source bytes and88 resident bytes after
splitting into2/1/1 rows. Source accounting remains80 bytes. Its exact budget,
one-byte-below and per-buffer boundaries are checked without allocating a GPU.

Formatting, strict default/Metal all-target Clippy,403 default tests,285 Metal
library tests and8 targeted logical tests pass; suites overlap. The existing
explicit tied/untied tiny Llama Metal graph regression also passes. The first
attempt could not expose a Metal device inside the sandbox; the preserved
GPU-only retry with hardware access passes against identical source. All eight
owned command groups close cleanly. No tolerance or numerical policy changed.

Pre-application inspection caught an undefined shorthand in a test initializer;
the private v1 proposal is retained and v2 fixes it before live application.
Astra authenticated the proposal, unchanged old readers/tests, actual source
and check evidence before commit. No real Phi payload execution, quality,
throughput or phone result is claimed. Next is the strict Phi3 parser, declared
window reporting, original fused-row inventory and independent head96 graph.

See [compact evidence](../benchmarks/results/2026-10-28-logical-projection-loading-v1.json),
[Phi contract](phi3-fused-source.md), [TODO](../TODO.md) and
[improvement record](improvements.md).

# Paired native Gemma loss block

This diagnostic pairs the frozen first1,025held-out IDs with the qualified
upstream BF16 block. Rows0–1,023 score IDs1–1,024; row1,024 is an unscored alignment
witness. It does not decide the full10,000-target loss or100-task gate.

An ignored real-model test reuses the existing native diagnostic session at
capacity1,025/chunk32 and the frozen compensated FP64 NLL helper. It emits exact
per-target order/identity, target FP32 bits, maximum, exponential mass, stable
NLL, lowest-ID greedy and raw row hashes. Only eight complete FP32 anchor rows
are saved, bounding capture to8MiB. Source identity, committed token history,
weight owners, allocation bytes and session buffer pointers are checked.

The independent checker verifies all1,024statistics records/aggregate arithmetic,
compares every anchor with the earlier native wrap capture and independently
replays complete scored-anchor distributions. Row1,024 remains unscored.
Two cold controls include uniform-row reductions and rejection of missing,
duplicate/misaligned targets, bad formulas, coherent false anchor statistics,
changed rows and truncation. Numerical fidelity failures remain rejected.

An initial strict Metal Clippy check rejected duplicate private test-harness
inclusion of the same frozen NLL helper. Astra recommended a reasoned local
`duplicate_mod` allowance, preserving one implementation without coupling the
harnesses. Initial logs are retained. Final fmt, strict default/Metal Clippy,
550default/658Metal tests and release build pass. Astra approves code and results.

All1,024targets and eight prior raw row hashes pass. Seven scored anchors
independently reproduce the complete distribution statistics; the last row is
unscored. Native mean NLL6.8268094585 versus upstream7.3745585369 gives a
**descriptive** one-block delta−0.5477490784. No loss threshold is applied to this
partial collection. On M1 Pro16GiB peak RSS is2,913,566,720bytes and physical
footprint2,907,704,008bytes, below6GiB. Source/binary/input endpoints, exact
session invariants and clean owned native/replay process exits pass.

Astra approves block-one reuse. Next collect the remaining nine frozen blocks,
reset positions per block and aggregate all10,000targets by target count before
applying the unchanged0.05nats/token loss-increase limit. The100-task gate and
all earlier fidelity rejections remain open.
[Evidence](../benchmarks/results/2026-11-04-gemma-native-loss-block-v1.json).

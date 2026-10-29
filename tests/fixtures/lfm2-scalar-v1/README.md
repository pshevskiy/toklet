# Independent tiny LFM2 scalar fixture

Synthetic original-packed Q4/group64/BF16 weights; no downloaded model.
H256/I256/V64/Q4/KV2/D64, capacity12, Conv→Attention→Conv.
The [protocol](protocol.json) freezes mixed FP64 reductions/FP32 stores and
the scalar gate before target outcomes. Four cases contain252 stage/state
groups and548,944 FP32 values. The manifest authenticates the data exports;
its own identity is recorded in the [result](../../../benchmarks/results/2026-10-29-lfm2-scalar-reference-v1.json).
Inactive source KV slots are+0; target checks observe active prefix rows, not
a tail-clearing requirement after reset. Source history suffix identities are
exact; cross-source numeric state comparisons use the same frozen gate.
No Rust graph, native Half, GPU, real model, text or quality qualification.

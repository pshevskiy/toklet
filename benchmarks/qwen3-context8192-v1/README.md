# Frozen8192 experiment inputs — no completed qualification

`manifest.json` and A/B/Q.u32le were frozen before the diagnostic run. Each array
contains8192 little-endian u32 IDs. A repeats the original512 structural fixture
sixteen times; B changes only its last ID. Both retain historicalA4096 as prefix.
Q is the disjoint pinned WikiText token span[14097,22289), preserving the previous
[0,10001) and[10001,14097) spans exactly.

The prospective quality contract scores target indices[7168,8192) using logits
[7167,8191),1024 targets, with unchanged inclusive mean-NLL increase0.05. Inputs
were independently reconstructed under a corrected, pinned runtime closure.
The preparation-time provenance fields remain immutable historical facts.

The human changed the execution goal before this quality comparison completed.
Complete input preparation is not numerical qualification. See the
[interrupted experiment](../../docs/qwen3-context8192-experiment.md).

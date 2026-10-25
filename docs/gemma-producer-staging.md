# Bounded multi-producer packed Gemma staging

The [receipt](../benchmarks/results/2026-10-25-gemma-producers-scalar-v1.json)
qualifies a bounded producer mechanism in the existing scalar packed graph. Its
public constructor still admits the original H64 / four-layer / 32-position
witness. The new private 35-layer test graph checks ownership and failure
behavior; it is not numerical qualification of the real E2B geometry.

## Ownership and execution

A private layout selects the producer prefix once during model construction.
It rejects zero producers, more than 15 producers or a prefix beyond the layer
count before allocating model metadata. Each producer owns persistent K/V
staging buffers in a session-owned vector. The plan resolves shared consumer
routes outside numerical loops. All staging owners are borrowed once per step;
no owner can be overwritten while a later consumer uses its proposed values.

A pinned `smallvec` 1.16.2 inline container retains up to 15 borrowed proposals.
The producer bound is checked again before numerical writes, preventing a heap
spill. Every proposal remains live through all producer and shared-consumer
layers, final normalization, tied head and softcap. All-cache validation precedes
commit; public KV, committed history and prior logits remain unchanged after
any earlier failure. Reset reuses persistent storage.

Persistent accounting includes the session vector header, actual staging-vector
capacity, every K/V capacity, cache owners and scratch capacities. Checked
arithmetic admits the planned session before its numerical allocations and
rechecks actual capacity before publication. The inline proposal container is
976 bytes of transient stack storage on the checked arm64 build, separate from
persistent model/session bytes and other call-stack frames.

## Verification

The existing four-layer test still compares every selected layer state and final
logits against both frozen numerical contracts under all three schedules. Its
separate allocation test covers 96 warmed decode steps and three resets without
Rust allocations. Current non-test-library logical totals are 155,076 model and
474,432 session bytes; earlier receipts retain their original totals.

Three new unit tests use 35 layers with 15 producers and 20 shared consumers,
LLLLG repeated seven times, and the doubled MLP beginning at layer 15. They use
small H64 matrices borrowed from the existing fixture and two deterministic PLE
tables. The plan routes later local/full consumers to owners 13/14. Checks cover
chunk equivalence, all committed cache slots, stable buffer identities, reset,
capacity rejection, 38 producer/consumer/final failure-and-retry boundaries,
zero allocations on every failure/retry and 32 single-token steps, independent
producer-count bounds, rejected budget before allocation and actual vector
capacity accounting. They reuse the existing test allocator; no second global
allocator is installed.

The private graph reports 1,217,592 logical model and 1,826,264 session bytes in
the test build. Borrowed fixture matrices repeat across layers, so the model
sum is not a measurement of unique physical tensor residency. Schedule agreement
is a structural check, not an independent E2B numerical oracle. Preliminary
source snapshots and outputs remain retained alongside final required checks.

Formatting, strict default/Metal Clippy, the CPU/Metal-feature suites and both
iOS compile targets must pass before Astra's final evidence review and commit.
Fixtures, numerical tolerances and existing test bodies remain unchanged; only
test-helper visibility and a new test-module declaration change outside the
packed implementation and its new structural tests.

## Next work

The [actual checkpoint](gemma4-checkpoint.md) still needs a named executable E2B
adapter, real-shape numerical witnesses and bounded generation/state checks.
Tokenization/chat, resident Metal, quality, representative performance, independent
phone admission and app integration remain open. No public architecture, ABI,
model eligibility or output/context limit changes here. The existing 15 packed
GPU, 15 dense GPU and eight long-rotary failures remain unwaived.

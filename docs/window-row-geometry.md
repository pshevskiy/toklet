# Shared local/full cache row geometry

29 October 2026. B3c3a extracts three crate-private functions used immediately
by the existing CPU cache: visible-view start, up to two old physical spans,
and up to two contiguous commit-copy runs. The future GPU cache can consume
the same row descriptors while checking its own element/byte/index contracts.
No new attention kernel, GPU state machine or model admission is added here.

## Distinct view and storage boundaries

A local owner retains W physical rows. At committed offset c, the first new
query is c; its visible old interval starts at `max(0,c-(W-1))` and ends at c.
Thus at most W−1 old rows are needed by any new query. Full attention starts at
zero. Each old descriptor contains an absolute position, physical slot and row
count, in increasing logical order, without copying or linearizing the ring.

Commit retains the final `min(new_tokens,W)` staged rows locally. Earlier rows
in a chunk larger than W remain available to attention but do not belong in the
final ring. Each copy descriptor contains the source row, destination slot and
row count. A wrapped local commit needs at most two disjoint runs; full retention
needs one. These are row units, not CPU/GPU pointers or byte offsets.

CPU views use the shared spans for both per-head slices and head-strided segments.
Commit computes runs once, copies contiguous ranges per head, then advances
offset/generation in the original order. This preserves floating-point bits,
including signed zero; it adds no numerical arithmetic or dtype conversion.

## Validation and ownership

The helpers accept prevalidated internal geometry. The existing public cache
constructor, prepare/view and all-owner batch validation still check nonzero
capacity, local window equality, admission and allocation arithmetic, finite new
data, lengths, owner identity and generation. These checks and their ordering
are unchanged. Debug assertions document internal helper preconditions; they
do not replace public validation. Backend-specific element strides and physical
buffer bounds stay with the consumer.

Cache fields, persistent buffers, allocated-byte accounting and frame lifetimes
are unchanged. Helper results are fixed arrays of small Copy descriptors with
no heap ownership, dynamic dispatch or per-element variant selection. Prepared
data remains immutable and old storage remains unchanged until batch commit.
Empty-span origins preserve existing observable behavior.

## Verification and limits

Three new pure geometry tests compare descriptors against independent absolute
row and per-token-copy oracles over small capacities, full/sliding modes,
singleton windows, wraps, chunks larger than the ring and near-usize boundaries.
A new cache integration test covers 64 combinations of retention, heads,
dimensions and four chunk partitions, completing 576 commits. It checks both
view representations against logical values, exact K/V physical bits after every
commit, unchanged allocation addresses/bytes and final partition equivalence.
Generation counts are not compared across different chunk partitions.

Formatting, strict default/Metal Clippy, all 424 default tests, 306 Metal-library
tests and 11 targeted window tests pass. Counts overlap. The new pure oracles
cover 678 old-span and 9,772 copy-plan cases. Existing independently frozen
cache fixtures, shared-KV routing and complete Gemma3 CPU graph/failure/allocation
tests also pass. These target states are asserted by the reviewed Rust tests;
they are not externally retained numerical captures. Astra reviews the actual
source and check evidence before commit.

This extraction makes no latency, throughput, GPU execution, real-model, app or
phone claim. See [machine-readable evidence](../benchmarks/results/2026-10-29-window-row-geometry-v1.json).

The [checklist](../TODO.md) continues with frozen Half old/new attention fixtures
and an optional native operation. Resident cache commit will use two completion
phases: finish and validate new math, then perform validated GPU copies and
publish. Uncertain submitted-copy failure must poison the entire cache set until
reset/rebuild and replay; a logical offset rewind cannot restore overwritten KV.

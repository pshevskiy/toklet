# Scalar gated-delta recurrence

This is the first native recurrent operator for E4/L4 of the
[active plan](engine-expansion-plan.md), implemented after the independent
[reference](gated-delta-oracle-preparation.md) was committed as `e6b253c`.
It is a checked CPU slice function, not a hybrid model or session.

## Layout and arithmetic

`cpu::gated_delta_step` reads normalized Q/K `[key_heads,key_dim]`, V
`[value_heads,value_dim]`, scalar decay/beta `[value_heads]`, and immutable
previous state `[value_heads,value_dim,key_dim]`. It writes a separate candidate
state in the same layout and output `[value_heads,value_dim]`. Key dimension
is contiguous. Value head `h` reads key head `h / (value_heads / key_heads)`;
divisibility is checked. No expanded key heads or alignment beyond `f32` occur.

Each state row is decayed, dotted with K to predict V, corrected by
`(V-prediction)*beta`, then read through Q. Products and sums use FP32 in increasing
key-coordinate order. The caller must supply normalized Q including the one
readout scale, and preconstructed gates in `[0,1]`. Raw normalization and gate
construction are separate mechanisms, not silently inherited from RMSNorm.
The separate [raw Q/K normalizer](gated-delta-normalization.md) is now checked;
the [scalar gate constructor](gated-delta-gates.md) is checked separately too.
Masks, vector gates and negative-eigenvalue updates are excluded.

All dimensions, checked products, slice lengths, finite inputs and gate ranges
are validated before any output write. Invalid admission preserves both output
buffers. Arithmetic overflow can leave partial candidate/output buffers; discard
both on error. The old state is immutable on every outcome and can be replayed.
There is no session commit or cancellation boundary inside this standalone step.

## Ownership and memory

The operator allocates no heap storage and copies no previous state into a
separate temporary. The caller supplies two state matrices, one read-only and
one candidate, plus output. Each matrix contains
`value_heads*value_dim*key_dim` FP32 elements. Q/K, V, gates and higher-level scratch
remain additional caller-owned storage. A future graph must include both state
owners and all other allocations in its memory plan.

A graph can swap the candidate into committed ownership only after the entire
multi-layer step succeeds. This function alone cannot establish a valid hybrid
prefix: convolution history, attention KV, positions and pending sampled tokens
still need session-level commit/replay rules. The extra candidate matrix does
not justify copying a whole transformer cache per token.

## Checks and limits

The ordinary offline suite authenticates the literal accepted receipt SHA before
reading the frozen manifest/binary hashes. It compares every recurrent state,
output and final state against independent FP64 math at the unchanged source
policy `1e-5 + 1e-4*abs(expected)`. Normalized mathematical inputs/gates are rounded
to FP32 at this operator boundary. The empty mathematical case makes no step call
and retains initial state; there is no zero-token batch API.

Six tests cover the frozen five geometries, hand-computed decay/correction and
contiguous grouped-head mapping, malformed/overflow dimensions, every nonfinite
input class, invalid beta/decay, pre-write preservation, prediction/correction/
candidate/readout overflow, and replay from an intact old state. The first check
run was stopped for Astra's fixture-binding correction; its source snapshots,
logs and interruption remain separate. Final checks run against corrected bytes.
Formatting, strict default/Metal Clippy, 197 CPU tests, 219 Metal-feature tests
and both iOS cross-checks pass (15/107 ignored respectively). The feature suite
does not imply a new GPU recurrence implementation or physical-device execution.

Completed results and exact scope are recorded in the
[receipt](../benchmarks/results/2026-10-24-gated-delta-scalar-v1.json).
The installed app and immutable native package are unchanged. No new Metal
operator, real checkpoint, graph admission, quality, performance, long-context
capacity or iPhone execution is qualified by this task.

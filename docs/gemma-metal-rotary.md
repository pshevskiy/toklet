# Standalone Metal proportional rotary

A fixed-shape resident operator now matches the frozen mathematical rotary
contract on this Mac. It is a building block for G4; it does not implement a
resident Gemma4 graph. The [receipt](../benchmarks/results/2026-10-25-gemma-metal-rotary-v1.json)
binds the independent witnesses, source, checked binaries and completed GPU
observations. The [scalar contract](gemma-rotary-scalar.md) remains the readable
CPU reference.

## Layout and numerical behavior

[`metal::gemma_rotary`](../src/metal/gemma_rotary.rs) accepts contiguous FP32
`[tokens, heads, head_dim]`, with 1–8 tokens, 1–8 heads and even head dimensions
2–512. Rotated dimensions must be even and no larger than the head dimension.
The immutable offset plus token index must stay within 0–131071. Base is bounded
to 1–1,000,000 and frequency factor to 0.125–8. Unsupported geometry rejects
before operator-buffer allocation.

Each pair joins coordinates `i` and `i + head_dim/2`. The inverse frequency is
`base^(-2*i/head_dim)/factor`, using the full head dimension in the exponent.
FP64 host phase/trigonometry prepares one `[tokens, active_pairs, 2]` table of
cosine/sine, each rounded once to FP32. The separate Safe/Precise Metal pipeline
performs noncontracting FP32 products and sums. Inactive coordinates copy their
bits, preserving negative zero and subnormal values. Active subnormal arithmetic
has no bitwise qualification. This mathematical contract does not reproduce
upstream approximate phase/trigonometry.

A frozen rational-series witness at angle one distinguishes staged products/
sums from either contraction of the first coordinate. Its second output is
`0x3f7fffff`, one FP32 value below one. The initial independent generator assumed
exact one, failed before target implementation, and was corrected using exact
rational rounding. Both versions and the expected failed execution are retained;
no numerical tolerance changed. Four further independently replayed witnesses
cover actual eight-query-head local/global dimensions at offsets 123 and 131069,
including position 131071. They supplement the seven nonempty scalar cases.

## Ownership, bounds and failure

A context owns the device, queue, existing default pipelines and one separate
precise rotary pipeline. Its 64 MiB preallocation allowance is compiler/context
policy rather than measured residency. Each operator borrows that context and
owns disjoint input, output, coefficient and status buffers. Storage uses u32
words for FP32 bits, preserving the existing bounded unsafe buffer inventory.
Seven sentinel words follow each logical buffer. Owned-byte accounting includes
the operator structure and actual buffer lengths; shared context is counted once.
Caller input/output, simultaneous owners and diagnostic storage need separate
admission.

Coefficients and buffers remain resident across repeated calls with the same
shape and offset. Changing the offset requires a new admitted owner/table;
efficient autoregressive graph scheduling is still open. Execution uses the
queried pipeline width, tracked hazards and retained command resources. No shared
CPU view is read until the command is confirmed completed. The standalone call
waits once for its own command; it grants no allocation-free or asynchronous
whole-graph decode claim.

Invalid lengths or nonfinite input reject before submission and leave the owner
reusable. Any later failure permanently poisons that owner, retaining the pending
command and preventing reuse. Caller output is copied only after completion,
guard and finite-result checks all pass, so it stays unchanged on failure.
A fresh owner is required after poison. The test captures raw completed overflow
results, status and guards before assertions. It never reads an unfinished or
failed-driver command's shared buffers. Driver failure injection is not qualified.

## Checked Mac result

On MacBookPro18,3, Apple M1 Pro with 16 GiB RAM and macOS 26.6.2, the release
ignored GPU method passes all eleven cases twice: 86,640 output comparisons under
the unchanged `1e-5 + 1e-4*abs(expected)` gate. Maximum absolute error is
`5.582909308188277e-7`. Repeated outputs are bitwise identical, and all twelve
single-token partitions equal their full-run slices. The rounding witness,
inactive bit copy, guard checks, pre-submission rejection/reuse and completed
overflow poison/recreate checks pass. Largest observed operator ownership is
100,116 bytes.

The completed test process reports 37,715,968 bytes maximum RSS and 26,460,760
bytes peak physical footprint, separately recorded within its 1 GiB diagnostic
ceiling. These include the test/compiler process and are not measurements of
whole-model residency or inference performance.

Formatting, both strict checks, all 312 CPU/335 Metal-feature tests, both iOS
targets and release geometry checks pass. Hardware tests remain explicit and
ignored in ordinary test runs. Sixteen controller corruption checks pass, with
both valid log prefix forms covered. The first actual child passed, but its
controller rejected the test runner's prefixed first record; that complete
attempt remains a failed controller run. A new reviewed controller and fresh GPU
run pass. Successful owned-execution wrappers are not claimed as invocation-
failure retention tests.

## Remaining qualification

This operator does not close earlier eight upstream long-position discrepancies,
fifteen dense and fifteen packed source-GPU intermediate failures, five FP64/
FP32 mathematical graph arrays containing 57 failed values, or the earlier 27
native ideal-FP64 logit failures per raw schedule. Those gates remain visible
in the [active plan](gemma4-implementation-plan.md).

Resident projections/graph/KV scheduling, original BF16/source semantics, broader
quality, representative end-to-end timing, mobile memory and maximum buffer
length, native ABI/app integration and 131072 numerical output remain open.
No catalog, model profile, numerical context limit or phone grant changes.

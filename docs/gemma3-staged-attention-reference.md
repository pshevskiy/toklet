# Frozen Half attention over old and staged KV

29 October 2026. B3c3b1 freezes an independent numerical and physical-state
reference for the next optional Metal operation. It does not run Toklet's
attention kernel, a complete graph, a model prefill or a cache transaction.
The [shared row descriptors](window-row-geometry.md) are already checked in CPU
use; the [remaining checklist](../TODO.md) tracks native execution separately.

## Layout and masks

Eight local/full cases use KV1, four query heads and head dimension 32 or 256.
Old Half K/V are separate head-major arrays with physical local retention W
or full capacity C. New Half K/V are immutable staging arrays; Q is
`[chunk_capacity,4,D]`. Chunk capacity is 32. All buffers have seven guards,
and unused storage is quiet NaN.

The ordered attention union is old `[max(0,c-(W-1)),c)` followed by new
`[c,c+n)` locally, or old `[0,c)` plus new globally. Each query sees keys no
later than itself and, locally, only distances below W. Physical retention W
and the W−1 old attention view are deliberately distinct. Score/probability
fixtures use logical `[4,n,keys]`; a native consumer must extract those rows
from its physical `[4,32,C]` scratch layout.

The cases cover a wrapped tiny W4 cache at c5 with seven new tokens; local/full
W512/C544 states at c512+n1 and c513+n1; and c500+n32. The latter preserves old
rows needed by early queries even though a later commit overwrites those slots.
The 513→514 cache states are chained exactly in the source. These are explicitly
seeded primitive states, not evidence of a 513-token model journey.

## Independent arithmetic and physical expectations

Main uses ordered FP32 dot products, division by FP32 sqrt(D), stable visible
max/exp/sum, FP32 probabilities and ordered FP32 context before the Half store.
Wide uses FP64 reductions with explicit FP32 score/probability and Half context
stores. Each of 24 case/stage groups keeps the frozen MAIN gate
`atol=max(1e-4,2*envelope), rtol=1e-3`; wide is diagnostic only.

All 104 arrays are retained: 48 numerical profile arrays and 56 raw physical
state arrays. Each profile contains 71,424 context, 145,012 score and 145,012
probability values. Exact uniform local controls distinguish excluded old keys,
retained keys and current staged values: first context features are
`[0,1,1,0]` at query512 and `[0,0,1,1]` at query513 for every head. Masked
probabilities are numerical zero.

The source also preserves immutable input hashes and expected post-copy raw
K/V slots. Its copy oracle writes only the final min(n,W) local staged rows in
at most two runs, leaving all untouched bytes and guards exact. These expected
arrays are future GPU-commit obligations; generating them does not qualify a
native copy, owner/generation validation or failure recovery.

## Checks and scope

Five portable controls pass. Owned source export and replay produce three
identical files, with finite deadlines, exact runtime/source bindings and clean
process-group closure. The committed fixture contains the manifest and binary;
progress logs remain in the private evidence directory.

Astra independently recomputes all 722,896 main/wide numerical values bit-for-bit,
all 24 envelopes/gates, 3,553,672 physical words, 10,120 masked-zero entries,
copy plans and chained states. No target code is imported by the producer or
the independent numerical audit. No Rust/Metal numerical execution, real model,
app, iPhone, quality or performance qualification is added by this source task.

See [machine-readable evidence](../benchmarks/results/2026-10-29-gemma3-staged-attention-reference-v1.json).
Next are the native reference operation and its own checks, then the resident
all-owner two-phase transaction. Submitted-copy uncertainty must poison the
complete cache set until confirmed completion and reset/rebuild plus replay.

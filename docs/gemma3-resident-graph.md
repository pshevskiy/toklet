# Resident Gemma3 graph

29 October 2026. B3c4c composes the existing packed projection/gather, FP32-scale
Half RMS, ordinary rotary, Half GELU/scaling, staged attention and checked cache
transaction into a concrete `metal::gemma3::{Model,Session}`. Three native
collection jobs and independent saved-byte audit pass in the bounded tiny scope.

## Shapes and ownership

Model owns original four-bit/group64/BF16 packed matrices, one tied embedding/head
owner, seven projections per layer, FP32 effective normalization coefficients,
zero bias and cached operations. The constructor explicitly consumes raw BF16
gamma and forms FP32 `1+gamma`. It authenticates no remote artifact and infers no
conversion history; the real checkpoint interpretation remains B3d work.
Caller-owned source storage can be released after construction. No expanded CPU
weight model remains in Model.

Session borrows Model and owns its cache set, resident scratch, token/history
capacity, candidate/published logits and cached input matrix views. Hidden rows
are `[32,H]`; query/context rows are `[32,4D]`, independently of H. K/V staging is
`[32,1,D]`; old local/full owners retain their separate head-major layouts.
Scores/probabilities use `[4,32,C]`. Queried Half matrix rows must be physically
tight before head reshapes are admitted. No new matrix multiplication kernel or
dynamic numerical dispatch is introduced.

The bounded leaves admit Q4/KV1, D32/256, H64/640, intermediate128/2048,
capacity up to544, chunks up to32 and up to32 layers. Packing, normalization,
rotary bases, buffer extents and device bounds are checked. These representable
combinations are not all qualified models. The initial complete graph fixture
is two-layer H64/D32 with local window4 and capacity12.

## Publication and readback

One math command starts with the cache set's NaN staging stamp, then encodes
embedding lookup/scaling, every layer and the FP32 tied output head. Each branch
preserves its pre-norm, operation, post-norm and residual order; Q/K normalization
precedes rotary, and GELU stores Half before gating. Final logits and all new KV
producers are validated after math completes. A separate all-owner GPU copy
phase finishes before nonfallible history copying and logits-owner swapping.
There is no host wait after each operator or layer.

Rejected inputs and completed nonfinite candidates preserve the prior committed
prefix and published output. Submitted-copy uncertainty poisons the whole cache
set; history/output/diagnostic access then rejects until terminal zero-prefix
reset and trusted prompt replay. Rewinding an offset cannot repair overwritten
KV. Sampled-token queues, chat rendering and cancellation between requests remain
outside this numerical forward API.

Ordinary sessions keep one FP32 vocabulary row per logits owner. All-row mode
allocates32 rows explicitly. Diagnostic mode requires all-row logits and captures
the48 tiny-graph stage identities before scratch reuse. Readback packs the actual
absolute query/key ranges from cached score storage into caller-owned slices.
Failed candidates invalidate overwritten diagnostic snapshots while preserving
the prior published logits after a completed nonfinite error.

## Memory and validation status

The ledger counts each packed owner once, FP32 norms, zero bias, the cache set
once, scratch, token/history capacity, both logits owners and actual metadata
capacities. Diagnostics count rounded physical GPU snapshots and reserved FP32
host readback separately. One explicit64MiB framework/compiler allowance is an
estimate, not a measured allocation. Caller-retained source/application memory
must be supplied explicitly. Checked size arithmetic and budget/device checks
precede numerical allocation. Planned construction/admission bounds are not
process-memory peaks.

Formatting, strict default/Metal Clippy and424 default tests pass. The corrected
core passes five debug/release controls and318 Metal-library tests. The final
Metal consumer adds two cold controls; all seven debug/release controls and320
Metal-library tests pass after a mechanical test-oracle lint correction. The
failed initial compile/lint receipts remain preserved.

Three frozen-release Mac jobs pass:16 full/chunk schedules,2400 stage captures
and428832 main-gated values, with maximum error/gate fraction0.41270. Wide is
diagnostic (maximum0.45602, zero exceedances). Exact native chunk equivalence,
causal prefixes, ordinary-last versus all-row head, reset/replay, original
weights/norms, nonzero guards and memory budget boundaries pass. Two actual GPU
NaN injections preserve the complete prior history/published state; the existing
submitted-first-copy failure poisons all output access until terminal reset and
trusted replay. All twelve warmed Rust forward allocation counts are zero in
release, including command creation/completion and excluding driver allocations.

The ordinary rotary job checks11520 values across32 commands, including D256
and positions511/512/513. Maximum fraction of the frozen main gate is0.80659;
the unchanged mathematical-envelope fraction is0.84023. This is component
qualification evidence, not a D256 full-model/context result.

Astra independently authenticates and rescores all saved stages, including
6549 main-profile bit differences within unchanged gates and maximum absolute
error0.001953125. It reconstructs170 physical states,122 successful commits,
129396 old-cache words,2352 masked probability zeros and59196 graph guards.
The original15 packed matrices,30 metadata owners and13 FP32 norms remain exact
and immutable. Complete publication/history preservation and the actual first-K
partial-copy mutation are reconstructed from raw records. Sixteen replays, four
ordinary-last comparisons and three causal equalities pass independently.

Budget/getter controls are passing reviewed Rust assertions where an individual
raw record is unavailable. Snapshot resource addresses are not exposed. The
independent reader initially needed index/swapped-role corrections, preserved
alongside the final audit; target outputs and numerical gates were unchanged.

All target raw arrays, physical state, guards, status and logs are retained.
Direct-process memory observations have35/4/1 samples;
the last job's single sample may precede numerical work. They establish no
process peak, real-checkpoint fit, latency or throughput. B3c4c closes only the
complete tiny Mac graph. Real-model/text/window, application and iPhone
qualification remain separate B3d/B5 work.

See [machine-readable evidence](../benchmarks/results/2026-10-29-gemma3-resident-graph-v1.json),
[independent Half source](gemma3-half-reference.md),
[cache transaction](gemma3-resident-cache.md),
[original-byte upload](original-matrix-upload.md) and [TODO](../TODO.md).

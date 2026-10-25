# Real Gemma projection admission

The pinned Gemma 4 E2B checkpoint's 278 text matrix codebooks fit the standalone
[FP32 packed projection](gemma-metal-affine.md) numerical domain. The latest
coordinate-correlated audit proves admission for 269 projection calls in the
frozen four-token staged-FP32 trace. Seven product envelopes remain inconclusive;
the complete tied head is separately rejected by the operator's output-row
limit. This is admission evidence, without projection execution, native activation
parity, a resident graph or app eligibility. The initial row-max audit and its
260/16/1 results remain preserved below.

## Authenticated inputs and scope

The checkpoint is `mlx-community/gemma-4-e2b-it-4bit` at
`238767527555cb75a05732a84dff5d6ba0dd6809`. Its original container SHA256 is
`038e39a37a7667373d2c3991375446b10c96ae1d717a68674870343db376b76e`.
The [existing mathematical capture](gemma4-text-math.md) supplies all 713 states
in each of two contracts, 103,403,520 raw bytes. This audit uses the complete
staged-FP32 states for input admission, with IDs `[2,100,257,100]` at positions
0..3. It does not substitute eight sampled values for a complete state.

All 556 BF16 scale/bias tensors are streamed in bounded chunks: 289,267,712
bytes and 72,316,928 paired coefficient positions. Every position checks its
original pair against a codebook covering all 16 possible codes, including
unused codes. Bounded memoization avoids repeating reconstruction for identical
pairs. Two retained metadata captures match byte for byte. The independent
verifier uses exact Fraction-to-binary32 rounding, without importing the producer.
Both replays agree with FP64 affine reconstruction followed by FP32 storage.
All codebooks admit; maximum reconstructed absolute weight is 1.2109375.

The two large tables have 262144 rows. Their codebooks pass, but whole owners
exceed the unchanged 16384-row limit. The per-layer embedding table is a gather
source, not an additional projection invocation. The tied token table supplies
the one head invocation, which remains geometry rejected. No chunked table/head
or mobile allocation qualification follows from this audit.

## Initial row-max classifications and unchanged limits

The fixed invocation map contains 277 calls:

| Input evidence | Calls | Admission result |
| --- | ---: | --- |
| Direct captured states | 172 | All numerical bounds pass; 171 admit geometry and the tied head does not |
| Exact two-rounded-add residuals | 35 | All 35 admit |
| Conservative GELU-product envelopes | 70 | 54 prove admission; 16 remain not proved |

Attention contexts retain their `[1, heads, tokens, coordinates]` layout while
per-token maxima reduce heads and coordinates. Residual inputs perform two
sequential FP32-rounded additions; reassociation is not allowed. Products are
not present in the original state capture. Their sufficient envelope uses
`abs(round_f32(GELU(g))) <= abs(g)` and the outward FP32-rounded product of the
captured per-token maxima. An overflowing or too-large envelope means not
proved, not that an actual uncaptured product is rejected.

The operator's bounds are unchanged: upward `S >= K*maxW*maxX`, with
`S <= F32_MAX/4`, and upward `U >= 2*K*2^-53*S`, with `U <= 1e-5/16`.
Both global and per-token bounds are
replayed, with exact rational checks on upward rounding. Maximum U for direct
inputs is 1.1434497592527358e-7; for derived residuals, 1.94289143211346e-8.
The largest coarse product envelope gives U=4.071010368811304e-6 at layer 4's
FF down projection, above the unchanged 6.25e-7 limit.

The 16 inconclusive FF down projections are layers
0, 1, 2, 3, 4, 5, 6, 7, 15, 24, 25, 26, 27, 28, 29 and 34. Finer independent
coordinate-correlated envelopes or authenticated actual product inputs are
required. No threshold is raised and no actual-input rejection is inferred.

## Checks, resources and retained failure

Twelve offline methods cover frozen codebooks and cutoff cases, exact binary32
rounding, token-axis/layout handling, residual order, inventory/byte accounting,
typed results and corrupt raw ledgers. Two additional methods execute the exact
owned retention function against oversized/growing reads and conflicting
recaptures, preserving the first buffer and streamed identity.

The first attempt passed its 12 methods, then the sandbox denied the resource
tool's system query. Its overall result remains failed with no resource credit.
A fresh directory with the required resource-query permission completes all
six owned commands: portable tests, retention tests, two producer captures and
two independent replays. Input/output guards and all six process/group absence
checks pass. No failed output is replaced.

Across those six completed processes, maximum RSS is 48,529,408 bytes and maximum
physical footprint is 39,142,096 bytes, each below the separate 512 MiB audit
policy. The policy is not measured whole-model residency. Scripts and interpreter
are pinned; the full standard-library/native-runtime closure is not claimed.
This reference-only task changes no engine numerical code and does not rerun
unrelated Rust suites or GPU benchmarks. The
[receipt](../benchmarks/results/2026-10-25-gemma-affine-admission-v1.json) binds
the exact source, raw captures, commands, resources and retained first failure.

## Remaining work after the initial audit

Retain the audited codebook and 260 admitted-call results. Resolve the 16
inconclusive products, table/head chunking and actual native activation fit
before graph composition. This short trace does not qualify arbitrary tokens
or longer histories. GPU projections, shared weights/KV/scratch and scheduling,
quality/performance, original-BF16 source parity, app/phone admission and 131072
output tokens plus prompt remain open. G2/G3/G4 and the earlier 57 cross-precision,
27 native ideal-logit, 15 dense/15 packed source-GPU and eight long-rotary
failures remain unwaived.

## Coordinate-correlated product refinement

The optional `coordinate-products` reference mode pairs each gate coordinate
with its own multiplier. It computes an outward FP32 product ceiling before
taking the token's maximum. This is a tighter sufficient envelope than the
product of independent row maxima. Any overflowing coordinate still makes its
token unproved. Complete equal `[1,4,width]` shapes and typed finite FP32 values
are required; neither GELU nor a projection is evaluated.

The new report format explicitly identifies the coordinate recipe. The verifier
requires that mode externally and rejects old row-max evidence presented as a
coordinate report. The default mode's complete 277 recipe/classification records
match the authenticated initial capture, including its 16 inconclusive cases.
Old evidence is preserved; the new mode supplies separate evidence.

Two new captures and independent rational replays prove 269 calls: 171 direct,
35 exact residual and 63 product envelopes. Nine former inconclusive envelopes
now pass: FF-down layers 7, 15, 24, 25, 26, 27, 28, 29 and 34. Each token's
coordinate envelope is no larger than its earlier row envelope. Layers 0..6
remain not proved; the maximum envelope U remains 4.071010368811304e-6 at layer
4. Layer 6's tighter U is 6.834568353764063e-7, still above 6.25e-7. The full
tied head and both whole table owners remain geometry rejected. All numerical
and geometry limits are unchanged.

Sixteen offline methods now cover correlation, joint versus one-sided coordinate
permutation, outward/subnormal rounding, overflow propagation, shape/type errors
and externally selected report modes, along with all prior offline methods.
The two retention methods and the actual default compatibility replay pass.
All seven owned commands, resources, endpoint guards and process/group closures
pass. Both 289,267,712-byte metadata captures retain the initial hash; both new
observation files match byte for byte. Maximum RSS/physical peaks across these
processes are 52,625,408 / 45,548,240 bytes under the same separate 512 MiB policy.
See the [coordinate receipt](../benchmarks/results/2026-10-25-gemma-affine-coordinate-v1.json).

Retain the tighter reference mode. Seven products still need actual-input
diagnosis; any stronger summation admission needs its own independent proof
before engine changes. Whole table/head storage, native activation parity,
resident graph/commands, quality/performance, app/phone fit and long output
remain open. No earlier source/precision failure is waived.

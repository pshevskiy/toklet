# Bounded FP64 matrix products for mathematical reference attention

This experiment addresses the measured scalar cost of the
[full-sequence tiny1025 oracle](gemma-long-witness.md) before cached reference
composition and the complete real quality workload. It changes reference
execution in Python/NumPy. Native Gemma's32-position executable ceiling,
production kernels, app eligibility and device qualification remain unchanged.

## Algorithm and arithmetic profile

The [candidate](../tools/reference/gemma_reference_attention.py) replaces only
QK and probability–V dot reductions with FP64 matrix products. Its explicit
profile is `fp64-blas-qk-pv-scalar-fsum-softmax-explicit-stage-boundaries-v1`.
FP64 BLAS reassociation is a distinct contract from scalar `math.fsum`; the
historical scalar evaluators remain available and byte-identical.

Inputs are batch-one Q`[1,query_heads,queries,D]`, K/V`[1,1,keys,D]`, with one
KV head, at most8 query heads,1025 absolute positions and D<=512. Contiguous
K/V must cover every query's entire visible interval. Full attention requires
the complete prefix. For example, local W512 queries1020..1024 can use keys
509..1024: the first query sees509..1020 and the last513..1024. Missing visible
keys, invalid origins/windows/shapes/dtypes and empty intervals reject before
numerical work.

Already-rounded staged-FP32 inputs convert exactly to FP64. Scores round to
FP32 before the existing scalar max/exp/fsum softmax. Probabilities round to
FP32 before their FP64 product with V, and context rounds to FP32 afterward.
The mathematical-FP64 path retains FP64 at those boundaries. Masking, scale1,
absolute positions and scalar normalization order are preserved.

Eight-query tiles bound temporary scores and probabilities; ordinary reference
graphs do not allocate a complete query-by-key score tensor. K/V conversions
are reused across heads/tiles. Diagnostic Q<=9 calls additionally expose all
scores and masked probabilities for independent replay. A conservative128 MiB
logical-array reservation includes inputs/output, conversion and old/new tile
overlap, finite checks, diagnostics and metadata. This is separate from the
512 MiB qualified-worker process policy and counts no model-fit capability.

Inputs and computed dots/casts/softmax shifts/outputs must stay finite. Computed
masked columns are checked too: an extreme finite input can overflow a masked
score and reject even when the visible-only scalar calculation would remain
finite. That stricter rejection behavior is explicitly tested. Qualified model
and fixture inputs must pass the actual candidate; no overflow is hidden.

The [graph comparison adapter](../tools/reference/gemma4_reference_graph.py)
keeps normalization, GELU, rotary, PLE, projection/gather batching and decoder
order as explicit arithmetic copies of the frozen scalar reference. WITNESS
admits1025 positions; E2B admits only the saved four-token diagnostic at this
stage. This adapter has no cache transaction or native backend.

## Frozen expectations and numerical acceptance

Before either candidate file existed, the
[scalar producer](../tools/reference/gemma_reference_attention_oracle.py) froze
eight standalone cases under both precision contracts. The
[independent replay](../tools/reference/check_gemma_reference_attention_oracle.py)
reconstructs rational/dyadic input recipes, sequential FP64 dots and validated
softmax/context stages. Both cold captures match across38356944 raw bytes.
All932968 score/probability/context values pass; the largest scalar-witness
error-to-bound ratio is`8.216144307792186e-10`. Seven coherent corruption controls
reject the intended failures. No new fast reference existed during preparation.

Cases cover full/local contexts through1025 keys, D256/D512, a W512 trimmed
origin, an8+1 query tile tail, W4, translated origins, single keys, mixed-sign
cancellation and near-tied scores. The compact fixture retains complete scores,
probabilities and contexts, with authenticated input recipes/identities. Its
expected values come directly from the accepted pre-target full capture.

Candidate acceptance uses the unchanged `1e-5 + 1e-4*abs(reference)` gate,
exact zero masked probabilities, complete matching-contract tiny and real
graph states, frozen greedy token IDs and unchanged prior cross-precision
failure indices. All outputs are serialized before independent numerical gates.
Original BF16 source authentication and its exact consumed-byte chain must
match the saved source plan. Its weights convert to FP32 rows for mathematical
reference evaluation; this does not execute an upstream BF16 runtime.

## Timing contract

Matched operator runs use the project's three warmups and ten raw measured
repetitions, medians and min/max dispersion. Each worker alternates scalar and
candidate order; the second cold worker reverses the initial order. The two
staged-FP32 workloads are full attention at H8/Q9/K1025/D512 and local W512 at
H8/Q5/K516/D256 with key origin509. Both return the same diagnostic outputs.
Every completed output is retained and numerically checked outside timing.
Input construction, serialization and comparison are excluded; operator
validation, conversion, allocation and synchronous matrix completion are included.

Before trials, retention requires candidate median<=90% of scalar median and
worst candidate sample<=95% of best scalar sample, within and across both runs.
This exceeds the usual5% improvement policy and separates the observed ranges.
Power/thermal state is recorded before the benchmark pair and after qualification checks. These
are reference-workload measurements, with production performance left open.

## Actual checks and retain decision, 27 October 2026

Fourteen offline methods pass, including complete frozen standalone outputs,
context-only parity, invalid geometry/dtype/origin/coverage, finite-input
dot/cast/shift failures, computed-masked overflow, exact W1 selection,
1025-query tiling, all frozen short graph states and sink-failure preservation.
The [independent candidate reader](../tools/reference/check_gemma_reference_attention.py)
passes932968 standalone comparisons, with111608 exact masked zeros. Maximum
standalone error-to-bound ratio is`2.634909055476384e-9`.

Two cold complete tiny1025 captures have identical files. Each passes all360
states/36350600 matching-contract values against the scalar trace; maximum
ratio is`5.112607738146951e-9`. All4100 per-position raw argmax choices match
per capture. These are raw distributions, not4100 generated tokens.

Real original-BF16-row execution passes2852 saved four-token states and34467840
matching-contract comparisons; maximum ratio is`4.336026313888867e-8`. All16
raw argmax choices match. The exact original read audit remains38182 reads /
18236693062 bytes with its unchanged hash chain. All eleven existing
cross-precision failures remain at the same array/value indices. Full upstream
BF16 execution, source/quality decisions and wider E2B contexts remain open.

| Matched staged reference operation | Run | Scalar median | Matrix-dot median |
| --- | ---: | ---: | ---: |
| Full H8/Q9/K1025/D512 | A | 11.5370 s | 23.0348 ms |
| Full H8/Q9/K1025/D512 | B | 11.5939 s | 22.8567 ms |
| Local W512/H8/Q5/K516/D256 | A | 1.6238 s | 6.1709 ms |
| Local W512/H8/Q5/K516/D256 | B | 1.6294 s | 6.2784 ms |

Both runs satisfy the frozen median and observed-range separation policy.
Each retains52 warmup/measured trials,156 arrays and6135584 independently
replayed values. Min/max dispersion and all ten measured samples are kept in
the evidence. This supports retaining the reference reduction for these
workloads. New complete tiny cold collection takes11.70/11.62 seconds including
imports, serialization and endpoint checks. The historical scalar357/359-second
collection is context; those differing collection overheads are not a matched
performance comparison. Production latency and throughput remain unqualified.

[Ten candidate/timing reader controls](../tools/reference/test_check_gemma_reference_attention.py)
pass. A masked1e-6 corruption passes the pointwise tolerance and then fails the
exact-mask requirement. A real-state corruption remains within0.8971 of its
matching-contract bound but changes a formerly passing cross comparison to1.1001,
and is rejected for changing the preserved discrepancy set. Coherent false
timing summaries and changed numerical arrays also reject.

On MacBookPro18,3 / Apple M1 Pro /16 GiB /macOS26.6.2(25G83), maximum measured
worker physical footprint is299303800 bytes; maximum RSS is399048704. Every
numerical/test/replay/control worker fits512 MiB. Source file authentication
and bounded row conversion provide reference ownership, without loading a
second complete FP32 model. Both benchmark endpoints report AC power, battery
100% and no recorded thermal/performance warning; CPU power status is unavailable.
All22 current owned groups and nine pre-oracle groups close; three runtime
inventories are rechecked. Protected release binaries and native source remain
unchanged. See the
[public evidence](../benchmarks/results/2026-10-27-gemma-reference-attention-v1.json).

## Remaining scope

The next mechanism is native scalar tiny capacity through1025, checked directly
against the accepted full-sequence oracle and independent KV witness, including
structural W512 wrap and uneven chunks. Astra confirms that a second cached
Python graph is not essential first. Keep real E2B/hybrid/GPU at32 until their
separate numerical and resource gates pass; complete real quality remains open. Keep WITNESS's W4 distinct from
real E2B's W512 and retain the eleven original real cross-precision failures,
earlier GPU/rotary gaps and independently required app/phone/full-window gates.

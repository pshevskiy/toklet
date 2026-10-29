# Tiny scalar LFM2 hybrid composition

B4c2 composes the existing scalar blocks into the frozen Conv→Attention→Conv
witness: H256/I256/V64, Q4/KV2/D64, capacity12, direct BF16 gains and three-tap
coefficients, epsilonFP32(1e-5) and ordinary theta1e6. This is a standalone tiny
numerical primitive, with append-only committed input and reset/replay. It does
not add a real-checkpoint factory, production chat/session route or native GPU
model.

`model::lfm2` keeps immutable borrowed original Q4/group64/BF16 matrices separate
from session state. One small layer route selects attention or convolution once
per layer. Packed `Scalar`, RMS, rotary, span attention, convolution and SiLU are
reused. KV2 projection output is explicitly staged head-major; B/C/x diagnostic
views retain their original row stride3H. There are no numerical trait objects,
per-token tensor-name lookups, full-context KV concatenations or new numerical
kernels. The tied embedding/output table is represented once. Direct gains get
no+1; embedding gets no scale; convolution gets no SiLU; residuals get no Gemma
clamp.

A session owns one KV cache, two convolution histories, per-owner candidates,
one reusable history work bank, fixed tokens/logits and scratch. Every layer,
final norm, head, finite check and fallible diagnostic callback finishes before
publication. One concrete boundary then validates both owner families, exact
inventory, IDs, generations, common offset/count, increments and token/logit
destinations before any copy. Its remaining work is infallible publication.
A late invalid history cannot publish an otherwise valid KV candidate. Reset
likewise validates both families before mutation, zeroes convolution history and
invalidates KV while retaining its stale physical tail.

The [independent source](lfm2-scalar-reference.md) was committed before target
outcomes. All four cases use whole, tokenwise and nonuniform partitions, with
57 mathematical stages per chunk and six actual state observations. Pre-chunk
history matches `history_before[start]`; post-chunk history matches
`history_after[end-1]`; KV compares active head-major prefixes. Tokenwise runs
cover every source state prefix. Internal states never committed by a larger
chunk are not fabricated observations, and inactive source-zero slots do not
impose clearing of stale target tails.

The target passes1,651,856 repeated comparisons across43 chunks/2,709
observations at the unchanged `1e-5 + 1e-4*abs(source)` gate. Unique coverage is
405,584 mathematical plus99,072 active state values. The44,288 inactive KV
source-zero entries are excluded explicitly. Maximum absolute error is
2.384185791015625e-7; maximum error/gate fraction is0.0016447079515116412.
The source's540 masked probabilities are checked as positive zero. Passing
intermediates are checked through borrowed in-process observations; no retained
passing target trace or independent external target rescore is claimed.

Five new tests check complete stage/state comparisons, exact final partition
KV/history and valid logits, future-token causality, ordinary/diagnostic equivalence, failures
and retries after branches/final stages, late mixed-owner validation failures,
callback failure, invalid input/admission, reset/generation exhaustion and warmed
zero Rust allocations. Full physical committed state—including stale tails—
remains exact on failure. Existing standalone history and KV APIs retain their
controls. Scratch, state, staging and token pointers remain stable after warmup.

Numerical accounting includes owner objects, actual Vec capacities, staging,
candidates, the shared work bank and21 scratch/publication vectors. Exact budget
and one-byte-under checks precede allocation, followed by actual-capacity
admission. Model bytes conservatively sum per-field borrowed matrix extents and
owned widened coefficients/metadata. Arbitrary caller aliases can be counted
repeatedly. Caller-owned whole containers, raw coefficient ranges, conversion
overlap, diagnostics and process/framework resources are separate; these are
not whole-process memory or unique-physical-allocation claims.

Formatting, strict default/Metal all-target Clippy and full tests pass:450 default
with20 ignored and531 Metal-feature with169 ignored. Counts overlap.

[Machine evidence](../benchmarks/results/2026-10-29-lfm2-scalar-hybrid-v1.json)
records checks, preserved compile-only failure, unchanged source and Astra review.
Resident Half leaves/composition, actual16-layer artifact, Mac process fit,
context controls, app/text, quality and iPhone remain later tasks in the
[LFM2 plan](lfm2-hybrid-plan.md). No latency or throughput improvement is claimed.

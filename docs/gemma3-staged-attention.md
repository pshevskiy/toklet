# Native Half attention over old and staged KV

29 October 2026. B3c3b2 qualifies an internal optional Metal attention leaf on
Apple M1 Pro. It reads resident old KV and immutable new staging, writes scratch
scores/probabilities/context and never copies or publishes cache state. The
[reference](gemma3-staged-attention-reference.md) and [shared row geometry](window-row-geometry.md)
define its semantics; the [checklist](../TODO.md) keeps the cache transaction and
complete graph separate. The module remains test-only until graph integration.

## Shape, layout and arithmetic

The checked structural domain is KV1/Q4, head dimension 32/256, capacity 1–544,
chunk 1–32 within the remaining capacity, and optional local window 1–capacity.
Source numerical qualification covers the eight frozen tiny/real-width cases;
extra exact controls exercise W1 and an empty old cache. This is not a new
model/backend admission or a claim of full declared model context.

Old K/V use separate Half arrays `[1,retention,D]`, where retention is W locally
or C globally. New K/V use `[32,1,D]`; Q/context use `[32,4,D]`. FP32 scores and
probabilities occupy `[4,32,C]`, stored in aligned `Buffer<u32>` words. Each
physical allocation has seven guard elements. The host uses shared old-span
descriptors to select at most two old intervals plus staging, in ascending
absolute order, without expanding KV heads or linearizing the ring.

One serial Reference thread owns each query/head. It computes ordered FP32 dot
products, divides by FP32 sqrt(D), then performs stable visible max/exp/sum,
FP32 probabilities and ordered FP32 context before the Half store. Safe/precise
Metal math and disabled contraction preserve the prescribed profile. Future
keys and locally expired keys have zero probability. Every active Q/new K/new V
row is explicitly checked; selected old inputs and arithmetic results also
check finiteness. Unused physical NaN storage is not treated as active input.

## Ownership and accounting

The operation owns one cached pipeline. Encoding borrows nine pairwise-disjoint
same-device buffers, copies eleven scalar arguments and performs no operand
upload, cache mutation, status reset, submission or wait. The caller retains
buffers and excludes CPU access/reuse until command completion. Status is sticky
and shared across operations; ordinary encoding can compose in one graph command.

Checked physical operand bytes include all guarded old KV, Q/context, new KV,
scores/probabilities and status. `capacity::KvOwner` counts old F16 KV once. At
C544/D256 this is 1,245,356 bytes for W512 or 1,278,124 bytes for full retention;
the largest individual buffer is 278,556 bytes. Immutable weights, the GPU
framework/context, compiler reserve, caller and diagnostics are separate. The
constructor checks one 64 MiB compiler allowance plus its Rust owner before
optional compilation; this reserve is not measured driver residency.

## Verification

Formatting, strict default/Metal Clippy, 424 default tests, 308 Metal-library
tests and two targeted debug/two release cold controls pass; counts overlap.
Both explicit release GPU tests pass, totaling 27 bounded submissions.

Astra independently checks every retained output and input hash. The eight
source cases contain 361,448 MAIN comparisons: one probability value differs,
maximum absolute error `7.450580596923828e-9`, maximum gate fraction
`4.077729383919762e-5`, zero failures. WIDE remains diagnostic, with 122,891
differing values, maximum absolute error `9.5367431640625e-7` and zero gate
exceedances. Gates are unchanged from the frozen source.

Eleven valid recovery/unused-row passes add 16,016 bit-exact MAIN comparisons
and zero WIDE diagnostic exceedances. Exact W1/c0 controls check 1,152 context
words and 204 probabilities separately. The audit also checks 6,908 masked zeros
across source/recovery cases, 32 uniform boundary words, all 135 before/after
input hash pairs and 797,172 inactive/guard words. Nine short operands and four
true equal-extent typed aliases reject before encoding. Five active-input errors
pass bad/sticky-valid/reset-valid phases; poisoning an unused old row leaves
outputs unchanged. Invalid active output bits are retained but are not a gate.

All eight measured ordinary encodes make zero calling-thread Rust allocations
in release. Objective-C/Metal driver allocations are outside that measurement.
Every raw record includes command status/error before assertions; successful
commands require completed status and no error. Both owned process groups close.
Memory observations are sampled direct-process charges, not execution peaks or
whole-model fit. No latency, throughput, graph, real model, app or phone claim
is made. Seeded 513/514 states are not a full-model prefill journey.

See [machine-readable evidence](../benchmarks/results/2026-10-29-gemma3-staged-attention-v1.json).
Next, the cache set must validate all owners and staged rows, complete Phase A,
then perform a separate validated GPU copy command before publishing. Uncertain
submitted-copy failure must poison the whole set until confirmed completion and
reset/rebuild plus replay. This numerical leaf does not implement that protocol.

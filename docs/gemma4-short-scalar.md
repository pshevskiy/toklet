# Named short Gemma4 E2B scalar adapter

The packed graph now has a named `Model::from_e2b(&PayloadStore)` constructor.
It resolves the admitted immutable E2B text payload into the existing scalar
implementation, with32 executable positions. It supplies raw logits. Tokenizer,
chat, generated continuations, resident Metal and app eligibility are separate
acceptance gates.

## Ownership, geometry and memory

The [cold payload owner](gemma4-payload-storage.md) must outlive its borrowed
model, and the model must outlive its sessions. Tensor names resolve once in
`src/model/gemma_packed/e2b.rs`; no names, maps, tensor ownership changes or
expanded full weight matrices enter numerical loops. The original tiny
constructor and scalar correctness fixtures remain available.

The adapter uses the exact validated35-layer plan: H1536/V262144, PLE256,
Q8/KV1,15 producers, local D256/R256 and global D512/R128. The MLP changes from
I6144 to I12288 at layer15. Local/full layers repeat LLLLG; shared consumers use
owners13/14. Direct FP32 norm slices and packed4/group64/BF16 matrices borrow the
same store. The embedding is also the output head, without a duplicate matrix.

The builder first validates all borrowed numerical shapes/values and sums their
logical bytes. It then replaces that sum with the same store's actual retained
tensor capacity plus its64MiB metadata reservation. This counts capacity excess
and the retained source layer-scalar vectors exactly once. Copied scalar fields
live in the new layer metadata. New model/layout/routing/unit-norm allocations
are added separately; load-only streaming and conversion overlap are excluded.
The store report cannot be supplied by a public caller independently of the
owner providing every tensor.

Physical local KV rings reserve512 slots; global caches reserve32. The internal
cache-plan ceiling is512, while the session checks its separate32-position
execution limit before numerical writes. Scratch and producer staging remain
bounded by32 tokens. The largest scratch buffers are two `[32,262144]` FP32
logit arrays; total scratch is79,396,864 bytes. Fifteen persistent stage owners
and inline borrowed proposals keep dependent consumers live through final
logits and atomic cache/history/logit publication. Session admission uses checked
model-plus-session accounting before numerical allocations and checks actual
capacities afterward.

## Verification scope

The offline named-layout test checks the real metadata plan and scratch without
allocating E2B weights. Independent frozen tensor inventories check backing
replacement, scalar retention, capacity excess, overflow, load-only overlap
exclusion and exact session admission. A private small35-layer graph with512-
slot rings checks late-failure/retry and token33 rejection while preserving all
15 caches, history, published logits and buffer addresses. Its weights are
synthetic and are not evidence of real E2B numerical parity.

Required formatting, strict default/Metal-feature checks, all297 CPU and319
Metal-feature tests, both iOS targets and the release probe build pass. The16
ordinary CPU and109 Metal-feature ignored tests remain explicit; these totals
do not imply every hardware or downloaded-model suite ran. Existing tiny
numerical bodies/gates and allocation-free checks remain unchanged.

The explicit `gemma4_text` example loads the [pinned real checkpoint](gemma4-checkpoint.md)
and executes raw IDs `[2,100,257,100]` in schedules `[4]`, `[1,1,1,1]` and `[1,3]`.
It writes every logit before independent comparison against both complete
[frozen mathematical contracts](gemma4-text-math.md). It also checks public
prefix preservation on a request ending at position33. Diagnostic concatenation,
logit snapshots and buffered output are outside numerical model/session byte
accounting and included in the measured process footprint. These observations
are controlled correctness probes rather than representative performance tests.

All three schedules produce bitwise-identical full logits. Each passes all
1,048,576 staged-FP32 comparisons under `atol=1e-5, rtol=1e-4`; the maximum
error-to-bound fraction is0.0013644315853741284. Each rejects27 FP64-ideal
comparisons, with maximum fraction1.2599434792233821, matching the preexisting
reference precision gap. Overall numerical acceptance remains false. Both
passing and failing comparisons are preserved; neither equal schedules nor
matching diagnostic argmax IDs waive that gap.

On MacBookPro18,3 / Apple M1 Pro /16GiB / macOS26.6.2 build25G83, logical model
accounting including its store is2,671,706,548 bytes; actual session capacity is
93,555,912 bytes. The fresh native process reports2,174,353,408 bytes maximum
resident memory and2,717,239,840 bytes peak physical footprint, within the6GiB
Mac allowance. Its67.96-second total includes checkpoint loading, all three
schedules, resets/rejection checks and diagnostic output. Files were hashed
before execution, with uncontrolled/warm filesystem cache. This is not a
qualified per-token, prefill/decode or user-visible latency benchmark.

Numerical results and resources are recorded in the
[receipt](../benchmarks/results/2026-10-25-gemma4-short-scalar-v1.json).
Original-BF16/source graph, earlier packed/dense GPU and long-rotary failures,
quality, generation/state semantics, representative performance, phone fit and
app integration remain open. No catalog family, ABI capability, context limit
or131072-output-token support is added.

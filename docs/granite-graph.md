# Granite full-model graph qualification

Status: required checks and the complete native capture pass, but the original
full-model numerical gate rejects the current affine4/8 Metal route. Astra
approves scoped tooling retention and the explicit non-adoption decision.
Quality, performance, observer overhead and managed adoption remain unqualified.
The [scalar decision report](../benchmarks/results/2026-11-13-granite-graph-v1.json)
binds the complete evidence and exact source hashes.

The [frozen prospective protocol](../artifacts/granite-graph-2026-11-12/prospective-graph-protocol-1.json)
has SHA256
`e4dec7ec0ca7e08e87bfbf538a71fca703d5553484e5a0e2eddc250418572c1d`.
Its [implementation interface](../artifacts/granite-graph-2026-11-12/parallel-interface-1.json)
binds the native/reference schemas and file ownership. The source is the
converted `MLX-Granite-4.2-3B-Heretic-affine4_8-g64-bf16-v1` child: data SHA256
`02f938b9f2a82d61c8609e19d2d65fcd7b32e52cc5b1cc7834b9843b9c4eec1e`
and config SHA256
`889ff79da95a5526811498cd47cee14c92a9d494fbd5dca70a3beb1c71b3df9b`.
The protocol also pins its conversion manifest and journal. Existing source,
conversion and configuration decisions retain their recorded scope in
[Granite configuration admission](granite-config.md) and the
[current checklist](../TODO.md).

The graph has 40 layers, hidden width 2,560, intermediate width 8,192,
40 query heads, eight KV heads, head dimension 64 and 100,352 vocabulary IDs.
It preserves default split-half RoPE with theta 10,000,000, RMS epsilon 1e-5,
attention multiplier 0.015625, unit embedding/residual/logit scalars and the
untied output head. The native diagnostic uses the existing `MlxAffine`,
`Compensated`, `Reference` and `FullLogits` choices; ordinary engine defaults
and the 512-position cap are unchanged.

The ten exact token cases cover one token, short and future-changed prompts,
offset five, a 17-token tile tail, three empty-cache keys at positions 509–511,
128/256/512-token contexts and a 128-token context starting at position 384.
The offset-509 case tests absolute positions with three keys; it does not
establish attention over 512 cached keys. Fully occupied cases have no spare
capacity for a pending sampled token. Each case exposes 643 arrays:
embedding/final norm/logits plus 16 stages in each layer. Attention-score and
probability internals are outside this native-visible array contract.

The independent producer imports admitted NumPy only, with a complete pinned
source closure and fresh `-s -B -X pycache_prefix` execution. It executes no
target engine, MLX or Torch code. FP32 BLAS and FP64 BLAS profiles reconstruct
the original packed affine codes and BF16 metadata in at most 256-row stripes;
floating norms cross the native Half storage boundary once. Activations and
KV use Half boundaries; scores, probabilities and logits have FP32 stores.
Each completed stage is written immediately, without retaining a full model
or dual-profile trace in memory. These are independent arithmetic profiles,
with no claim of scalar reduction order or exact GPU emulation.

For every case and stage separately, the offline comparison must derive
`atol = max(1e-4, 2 * max_abs(main - wide))` solely from the two authenticated
reference arrays and retain `rtol = 1e-3`. Every finite native coordinate must
pass against both profiles. Native results cannot enlarge the gate, and
errors cannot be pooled across cases or stages. Greedy differences require
near-tie investigation; exact native schedule/state comparisons remain binding.

The native worker loads immutable model weights once and borrows them from
serial sessions. Session-owned KV, scratch, diagnostic buffers and readback
storage are admitted together; immutable weights are counted once. It retains
source data/config descriptors and checks whole-file hashes plus matching
named/descriptor device, inode, length, mtime and ctime before and after the
run. Source directory identity and metadata must remain stable. The converted
single-file source has no index; no index repair or managed grant is created.

Requested chunk sizes 128, 32 and one use
`effective_chunk = min(requested_chunk, 512 - position_offset)`. Each primary
capture records every successful diagnostic chunk before the next evaluation,
concatenates rows into complete case arrays and retains the actual
`effective_chunks` schedule. Chunk128 writes original Half words only after
exact F32→Half→F32 bit round trips, with logits stored as F32. Fresh repeat
sessions hash and compare every actual byte. Authenticated baseline files and
sorted, nonoverlapping changed-block patches reconstruct each complete repeat;
identical repeats retain no duplicate payload. Saved reconstruction hashes are
verified independently of incoming diagnostic hashes. Chunk128 also retains
all-stage reset/replay evidence for every case.

Concrete state checks cover a fully cached prompt doing zero work with unchanged
logits, reset/replay, exact chunk schedules, full-logit versus GPU token-only
greedy selection, changed-prefix reuse versus a fresh session on the frozen
short/future-changed pair, capacity rejection preserving the committed prefix,
and pending sampled-token history/discard behavior where capacity permits.
The other two schedules retain final-logit reset checks. Numerical or schedule
mismatches remain explicit failures while collection continues; runtime or
resource failures leave retained, unqualified partial data. Cancellation and
device-failure injection have no public hook in this worker and are not claimed.
The existing tiny Granite GPU state fixtures remain independent evidence.

The [fourth reference capture](../artifacts/granite-graph-2026-11-12/reference-capture-4/manifest.json)
completed all 20 profile/case combinations: 12,860 arrays and 10,558,242,816
saved bytes. Its manifest SHA256 is
`0e7689ea27e68aa448272cd1d926b721e1e50c7ba6371f56b8962e5bf7e4e0d5`.
The [owned execution receipt](../artifacts/granite-graph-2026-11-12/reference-capture-command-4/result.json)
reports 593.901 seconds, a maximum observed charge including reserve of
409,568,672 bytes, successful cleanup and no live owned processes afterward.
This is diagnostic execution time, not an inference-performance result.

Captures [one](../artifacts/granite-graph-2026-11-12/reference-capture-command-1/result.json),
[two](../artifacts/granite-graph-2026-11-12/reference-capture-command-2/result.json)
and [three](../artifacts/granite-graph-2026-11-12/reference-capture-command-3/result.json)
remain failed and unqualified. The original 512 MiB reference/setup cap was
retained. Allocator-region diagnosis and
[49-array prefix comparisons](../artifacts/granite-graph-2026-11-12/allocator-diagnostic-closure-1.json),
followed by [560 identical fixture records and 1,120 comparisons](../artifacts/granite-graph-2026-11-12/allocator-policy-controls-2.json),
selected reference-subprocess-only `MallocLargeCache=0` and `MallocMediumZone=0`.
Engine allocator policy is unchanged. The sampled ownership guard includes a
32 MiB helper reserve; it does not establish a continuous allocator cap or
capture every short-lived peak. The 6 GiB engine ceiling, 3 GiB compiler-only
cap, 2 GiB disk floor and prohibition on compiler/model overlap remain binding.

The [integration cold checks](../artifacts/granite-graph-2026-11-12/integration-cold-checks-1.json)
record 51 portable tests in each normal/optimized Python mode and passing
formatting. Fresh [required Rust checks](../artifacts/granite-graph-2026-11-12/rust-checks-2.json)
pass strict CPU/Metal Clippy, 737 CPU/887 Metal tests and the four new helper
controls. The [explicit Granite GPU fixture](../artifacts/granite-graph-2026-11-12/granite-fixture-check-1.json)
and [eight admitted checker numeric controls per Python mode](../artifacts/granite-graph-2026-11-12/checker-numeric-checks-1.json)
pass. An initial launcher used an unsupported system Python version; its
[retained failure](../artifacts/granite-graph-2026-11-12/rust-launcher-failure-1.json)
and [verified original-process cleanup](../artifacts/granite-graph-2026-11-12/rust-launcher-cleanup-1.json)
precede the successful repository-runtime retry. The actual captures and offline
comparison have finished; Astra follow-up approves this code task in its scoped
tooling/rejection role. A completed comparison does not imply its gate passes. All full logs,
failed attempts, partial arrays, hashes and ownership observations stay in
artifacts; source/reference/native/checker work has separate file owners and
one coordinator queues resource-intensive jobs.

The [complete native capture](../artifacts/granite-graph-2026-11-12/native-capture-1.json)
passes all 30 case/schedule runs and 304 observed required state checks.
Its observed peak including reserve is 5,031,203,544 bytes under 6 GiB; owned
cleanup is empty and all 4,031 source/fixture inputs remain unchanged. Its
1,515.797-second clock includes export and state replay and is diagnostic.
The [complete comparison](../artifacts/granite-graph-2026-11-12/offline-comparison-1/manifest.json)
checks 19,290 primary plus 6,430 reset arrays against both profiles: 212 of
20,268,711,936 profile coordinates fail. These are 159 primary plus 53 reset
failures, representing 53 distinct case/stage/profile coordinates and 40 distinct
case/stage coordinates. All values are finite. Schedule and reset mismatching
bytes are zero; all 643 causal-prefix arrays match exactly. The required state
checks and final-logit receipts have no missing or failed observations. Other
unavailable hooks retain their stated limits above. The checker exits one for
the numerical rejection, completes its runtime endpoint and owned cleanup, and
observes 172,343,296 bytes including terminal accounting within 512 MiB. The strict
post-fix replay below observes 170,442,752 bytes within the same limit.

The [retained failure coordinates](../artifacts/granite-graph-2026-11-12/numerical-rejection-details-1.json)
start at `short/layers.4.attention_context[1,1704]`: native 5.31640625 versus
5.296875 in both profiles, error 0.01953125 above limit 0.017015625. The
[bounded localization](../artifacts/granite-graph-2026-11-12/attention-localization-result-1.json)
reproduces each original reference attention array exactly from its saved inputs.
Feeding captured native Q/K/V to either arithmetic profile reproduces this
native coordinate exactly; the maximum remaining local context difference is
0.0001220703125. This points to propagated upstream activation differences at
this first failure. It neither localizes every other failure nor creates an
independent reference: native-input diagnostics cannot qualify the primary graph
or enlarge its frozen gates.

Astra's [initial review](../artifacts/granite-graph-2026-11-12/astra-review-1.json)
requests two fail-closed corrections: required state/reset coverage and mandatory
raw-bound final-row receipts. Both are fixed. The updated checker passes 34
portable tests per mode, including 304 individual check deletions, 120 receipt-
field deletions, ten missing resets and twenty forbidden repeat resets. Combined
with 24 unchanged producer/reader controls, the portable total is 58 per mode.
Fresh admitted numeric controls pass in both modes. The
[strict genuine replay](../artifacts/granite-graph-2026-11-12/strict-replay-closure-2.json)
requires and verifies all 304 checks, ten full resets and forty raw final-row
receipts; every per-array numerical record is byte-identical to the initial
comparison. The same 212 failures remain rejected. No model/compiler rerun is
needed because all 4,031 relevant numerical/core/fixture inputs are unchanged.
Astra [follow-up approval](../artifacts/granite-graph-2026-11-12/astra-review-2.json)
resolves both findings with no remaining actionable issues. The
[publication record](../artifacts/granite-graph-2026-11-12/publication-checks.json)
tracks the commit, push and exact-head CI.

Decision: retain the reviewed tools/negative result and keep the current
candidate unqualified for adoption. No tolerance, corpus,
context or memory budget changes. Alternative arithmetic or artifacts need their
own qualification; codec/quality/performance work does not bypass this rejection.

The original quality campaign remains separate: both stock→modified and
modified→native comparisons require all 10,000 loss targets and 100 task cases,
maximum mean NLL increase 0.05, and maximum pass-rate loss two percentage points.
No truncation or corpus substitution is allowed. This graph campaign does not
establish quantization quality, every-workload physical fit, request latency,
performance retention, observer overhead, application adoption or iPhone support.

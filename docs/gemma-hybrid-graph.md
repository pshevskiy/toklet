# Gemma hybrid graph: tiny Mac qualification

The shared packed Gemma forward and generation code now has two concrete,
statically dispatched operation providers: the existing scalar reference and
GPU packed projections/table gathers. The complete **tiny** graph, cache and
selected failure/generation semantics pass on Apple M1 Pro. This is the first
G4.2 composition task; real E2B raw, formatted and streamed execution remain
open. No app profile, artifact eligibility, context ceiling or default changes.

Subsequent [real E2B hybrid execution/generation](gemma-hybrid-real.md) checks
the actual vector-copy/adapter and selected short schedules. The remaining
acceptance statements below describe this earlier tiny-only publication.

See the [public evidence](../benchmarks/results/2026-10-26-gemma-hybrid-tiny-v1.json),
[storage owner](gemma-resident-weights.md), [tiny independent graph](gemma-packed-graph-reference.md)
and [active delivery queue](gemma4-implementation-plan.md).

## Shapes, order and ownership

`model::gemma_packed::operations` is a sealed Gemma-specific interface with only
scalar and Metal implementations. Matrix layout and provider selection happen
outside numerical loops. Generic model/session types retain the CPU defaults;
there is no trait-object dispatch or general tensor framework. GPU weights keep
original row-major packed4/group64/BF16 coefficients; activations and KV are FP32.

The shared forward keeps normalization, rotary, attention, activation, PLE,
residual and softcap arithmetic and order. Only packed matrix products and
embedding/PLE row lookup use the alternate provider. GPU operations complete
before CPU consumers read their outputs. This intentionally includes dispatch,
synchronization and readback; it is neither a resident graph nor a speed claim.
KV/history and published distributions still commit only after a complete step.

Immutable `resident::E2bWeights`, a small `hybrid::Norms` CPU vector copy, the
borrowed graph and an external Context outlive each session. The real adapter
copies learned vectors bit-for-bit from their resident FP32 buffers, shares
all packed matrices, and resolves named roles and reusable geometry pools once.
Its public constructor still accepts only the named E2B owner. Its real vector
copy/adapter path is compiled but **not yet executed or qualified** by this task.
The private tiny adapter preserves the frozen FP32 synthetic norms, cannot
produce an E2B identity, and cannot enter the public E2B constructor.

Each session owns CPU KV/staging/scratch and pooled GPU projection/gather
workspaces. Context/pipeline policies remain separate from known allocations;
inline backend owners count in the session root, external Vec slots and guarded
buffers count once. The model charge includes both resident vector buffers and
CPU vector copies. Caller-held models, diagnostic data, codecs and other live
sessions need separate admission. A shared Context policy is charged once when
the diagnostic keeps two sessions alive together.

## Tiny numerical and state evidence

The optimized explicit test uses the existing four-layer frozen graph: hidden64,
17 vocabulary entries, two query heads, alternating64/128 head dimensions,
PLE64, two producer caches, four-slot local retention and32 total positions.
The upload Context is dropped; execution uses a fresh Context on the same device.

The frozen prefill, token-decode and mixed schedules produce3805 full raw array
records,3048704 bytes. Independent replay makes1524352 numerical comparisons
against both original mathematical and staged-FP32 arrays. Unchanged gates are
`1e-5 + 1e-4 * abs(reference)`. Maximum gate fractions are0.2890802583351034 and
0.0011368405538946377 respectively. All selected layer outputs/intermediates,
rotary/context layouts, staged K/V, committed ring/global slots, final norms,
PLE and full logits pass. Reset/replay retains the same storage addresses.
The observer writes attempted/completed descriptors and full arrays before
numerical assertions; partial failures retain their bytes and original errors.

Memory and failure-state results below also depend on the frozen native
assertions; their transcript is not an independent KV snapshot or allocation-
layout oracle. A separate controls test and independent reader validate35 original packed
matrices and105 distinct guarded-buffer hashes/identities before and after
repeated sessions. Exact admission passes; one byte less rejects without Rust
allocations on the measured test thread. A five-position request still charges
32-position physical storage and rejects the sixth token. Generation reuses the
existing stop-token, pending-token, output-callback failure, continuation and
pre-cancel/after-delivery cancellation checks.

The controls inject a projection error after12 earlier successful projections,
and a gather error after one successful gather. Each preserves the three-token
committed prefix, logits and KV; failed reset/reuse reject without publication.
Fresh scratch replays five tokens against the frozen contracts while the failed
owner remains live and charged. These are **pre-command injected failures**;
there is no in-flight Metal completion, driver recovery or during-prefill
cancellation qualification. A foreign norm owner is rejected before allocation;
a physically different Metal device was unavailable and is not qualified.

| Tiny ownership / observation | Bytes |
| --- | ---: |
| Model ownership | 171164 |
| Model plus raw session admission | 101744196 |
| Model plus generation session admission | 101744420 |
| Included Context plus two gather pipeline policies | 100663296 |
| Numerical test peak RSS / physical footprint | 26804224 / 10224216 |
| Controls peak RSS / physical footprint | 27901952 / 11354688 |
| Separate diagnostic process policy | 536870912 |

Policies are conservative admission allowances, not measured residency. These
small-model observations do not establish real E2B fit or throughput.

## Verification and remaining gates

Formatting, strict default/Metal Clippy and all356 CPU/408 Metal tests pass
(16/129 ordinarily ignored tests). Both iOS device and simulator targets
compile; this is compilation evidence only. All23 owned command groups close,
3533 common source/reference bindings match between hardware and final checks,
and protected release executables remain unchanged.

The publication binds exact source files, fixtures, copied release driver,
controller, raw transcripts/arrays and independent replays. Ten portable tiny
reader controls and eight ownership-reader controls pass. Failed pilotv1 is
retained as a harness expected-error mismatch with an empty array capture;
pilotv2 passed the earlier numerical core. A later strict-lint development
failure was corrected before the final frozen trial. No tolerance changed.

G4.2 still requires the original real E2B norm copy/adapter, whole-session memory,
full selected intermediate distributions and raw/formatted/streamed schedules.
G2.1/G5 precision and quality discrepancies remain unwaived. G4.3 resident
scheduling/performance and progressive context, app/native integration and phone
fit are separate open work. This tiny32-position diagnostic cannot satisfy the
[corrected model-driven window requirement](long-reply-engine-plan.md). Both Chat
and Expert must ultimately support the selected model's full supported window;
any smaller qualified/device-admitted capacity must be shown as partial support.

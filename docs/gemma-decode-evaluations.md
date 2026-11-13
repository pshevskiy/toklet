# Completed native decode-evaluation rows

The native observation API copies the existing completed
`generation::Report.decode_evaluations` counter into an additive record.
Fresh Gemma BF16/G2 requests verify count, output/state, source and resource
invariants. The unchanged **1% TTFT bound is unqualified**: both
cohorts' bootstrap upper bounds exceed 1%. Production summaries remain disabled
by default and opt-in. **Astra approves functional retention only**, with no
code findings; this does not qualify overhead or default adoption.
[Implementation contract](../artifacts/gemma-decode-rows-2026-11-13/implementation-contract-1.json).

## ABI and availability

The public ABI version remains **1**. Existing `TokletGenerationMetrics` stays
96 bytes; old generation symbols retain their signatures. New
`TokletGenerationMetricsV2` embeds that record at offset 0 and adds one u64
`decode_evaluations` at offset 96: total size 104, alignment 8. Its embedded
`base.struct_size` remains **96**, not 104.
`toklet_session_generate_observed_v2` accepts the V2 output. Private None/V1/V2
selection occurs at request boundaries, outside numerical loops.

Validated output is cleared before other inputs, preserving validation order.
Successful completed reports, including cancellation, supply the count.
Native validation/generation failure leaves `base.known_fields` zero and the
extension **unavailable**. Token state remains separate. Callback/state behavior,
generator arithmetic and counter updates are unchanged.

Managed summaries use V2 only on the opt-in observed path. The export manifest
adds `nativeGenerationMetricsVersion:2`. The
legacy Swift initializer retains `decodeEvaluations:nil`; old exports remain
readable with an unavailable count. New captures require V2. Completed counts
are validated against bounded native work, including `decode <= sampled`.

## What the count means

The counter sums successfully evaluated rows returned by
`evaluate_next_chunk` **inside the decode loop**. It does not count dispatches,
calls or emitted tokens. Pending rows evaluated at request start belong to
prefill, including a pending sampled ID from an earlier request.

For an ordinary completed two-output limit after prefill, decode rows are 1.
A hidden first stop samples one ID, emits none and has zero decode rows.
Cancellation after a successful decode evaluation but before the next sample
can produce decode/sample/emitted **1/1/1**. Equality is valid; neither
`decode = sampled - 1` nor `emitted - 1` is universal. Completed pre-cancellation
can have known zero. Native generation failure has no completed observation;
a later managed decoding/publication failure can retain the completed native
report. `nativeMetricStatus` distinguishes this boundary. The exact 384/128
reset workload now verifies **127** completed decode rows per enabled request.

The native copy adds no timers, counters, heap allocations, GPU commands/waits,
per-element events or diagnostic profiling. Existing overlapping timings are not
summed as exclusive durations. The 16 KiB collector/request, 1 MiB ring,
4 KiB summary, 6 GiB complete-engine ceiling, 512 MiB setup including helper,
3 GiB compiler cap and 2 GiB disk floor remain. Measured POD layouts: native
96/104 B, Swift metrics stride120 B, collector200 B, summary800 B, ring backing
25,600 B under its unchanged 1 MiB reserve. Native grows 8 B; the old Swift delta
is unmeasured. POD sizes do not establish total footprint.
[Layout observation](../artifacts/gemma-decode-rows-2026-11-13/layout-current-1.json).

## Checks and actual workload

Native CPU **13 tests** pass: layouts/write extents, V1/V2 state/logit parity,
pending-prefill/decode, hidden stops, cancellation, clearing and invalid/thread/
re-entry/poison boundaries. Fmt and strict CPU/Metal Clippy pass. Portable
**65 tests** (53 O1 +6 workload +6 publication) pass in normal and optimized modes.
An initial count overstated the dependent set; the complete command reran in
both modes before model work, without weakened validation.
[Count correction](../artifacts/gemma-decode-rows-2026-11-13/check-count-correction-1.json).
Fresh **794 CPU/954 Metal** tests pass (53/351 ignored), as do C layout/link,
tiny CPU/Metal smokes, strict Swift6 and21 offline controls per Python mode.
Six old cohorts (516requests/258native records) replay exact statistics with
decode counts unavailable.
[Current check snapshot](../artifacts/gemma-decode-rows-2026-11-13/checks-current-2.json),
[legacy replay](../artifacts/gemma-decode-rows-2026-11-13/legacy-captures-replay-1.json).

Post-decode cancellation uses a test-only wrapper around the real tiny CPU
session/core generator and the V2 mapper, not an injected FFI backend. Other
tiny tests exercise the FFI boundary.

Fresh conformance and an untimed disabled/summary preflight pair pass using the
new matching header/library and current strict Swift caller. The preflight's
frozen descriptor is byte-identical to the already fixed input; no output or
latency selection occurred. Two fresh processes then ran the original
**3 warmup + 10 measured pairs each**: 52 requests, 40 measured requests and
26 native summaries, in addition to the preflight's 2 requests/1 native summary.
Every request has exact 384 input IDs and hash, 128 actual output IDs,
history/committed/pending **512/511/1**, and every native summary has V2/count127.
[Fresh preflight](../artifacts/gemma-decode-rows-2026-11-13/workload-preflight-1/result.json),
[campaign](../artifacts/gemma-decode-rows-2026-11-13/workload-campaign-1/result.json).

The independent audit replays the saved rows/captures and unchanged statistics,
verifies all **526 protected endpoints**, and finds no semantic, source or
resource failure. All fresh native guard phases pass and have no owned live
processes after cleanup. Campaign processes take **1130.684/1131.325 s**; their
maximum observed charge including reserve is **4,837,249,704 B**. Including
the fresh preflight phases, the maximum is **4,837,659,520 B**, below 6 GiB.
Polling and task-lifetime accounting do not establish a continuous allocator
cap or a per-request peak.
[Preflight guards](../artifacts/gemma-decode-rows-2026-11-13/workload-preflight-guard-1/result.json),
[campaign guards](../artifacts/gemma-decode-rows-2026-11-13/workload-campaign-guard-1/result.json).

## Scientific result and limits

Both metrics in both cohorts must satisfy median-ratio overhead <=1% **and**
the original 10,000-resample/seed1 paired-median 95% upper bound <=1%.
All numbers below are percentages.

| Cohort | TTFT point | TTFT 95% upper | Full request point | Full request 95% upper |
| --- | ---: | ---: | ---: | ---: |
| 1 | 0.61177355 | **1.22978988** | 0.28633236 | 0.80652564 |
| 2 | -0.26915901 | **1.31341800** | -0.18174048 | 0.69677445 |

The full-request gates pass; both TTFT upper bounds fail the original bound.
`overhead_qualified:false` is the campaign's result. The independent verifier
reports `audit_passed:false`, `scientific_rejection:true` (exit1), with no
`failure` key. This is an unsuccessful scientific acceptance gate, not a broken
capture. The data do not prove true overhead >1%, and do not prove that noise
caused the rejection. No N escalation, gate change, collector change or further
timing run is used to rescue this result.
[Independent audit](../artifacts/gemma-decode-rows-2026-11-13/independent-decode-audit-1.json).

[Final check aggregation](../artifacts/gemma-decode-rows-2026-11-13/checks-final-1.json)
and [Astra approval](../artifacts/gemma-decode-rows-2026-11-13/Astra-review-1.json)
record functional retention only. The
[publication receipt](../artifacts/gemma-decode-rows-2026-11-13/publication-checks-1.json)
records the reviewed commit, push and exact-head CI. The original TTFT rejection
remains explicit.
PLAN permits an unqualified result with production summaries still disabled
and opt-in; no default adoption follows. The
[prior 384+128 workload report](gemma-workload-identity.md) covers an earlier
build/metrics path, whose successful gates do not qualify this V2 build.

Scope: headless official Gemma BF16/G2, Apple M1 Pro, 384 input +128 actual output,
shared capacity512/chunk32. Numerical512-input, UI/render, tails, other models/
devices and default adoption are unqualified. The generated XCFramework is
unchanged; the new API needs a matching header/library as this headless build
uses. iPhone packaging/runtime remain deferred.

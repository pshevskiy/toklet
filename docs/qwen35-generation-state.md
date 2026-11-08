# Qwen3.5 generation state bridge

Q1.I wraps an already admitted numerical Qwen3.5 session in the existing static
`InferenceSession` boundary. Generic generation retains its whole-output
admission, Stop, output-limit, continuation, callback-failure and cancellation
semantics. No ordinary model, native ABI or managed route becomes eligible.

The wrapper owns a preallocated `u32` history through the session capacity.
Numerical history contains only committed inputs; sampled and unprocessed input
tokens occupy a separate pending suffix. A failed numerical chunk leaves that
suffix and the valid committed prefix intact. Fully cached prompts retain their
last distribution. Exact extensions reuse the whole committed prefix; changed
or shorter committed prefixes reset and replay from logical position zero,
preserving the session's absolute position offset. Hybrid recurrence
cannot recover an arbitrary earlier prefix from KV alone. Replacing only pending
tokens preserves committed state. Reset and discard allocate nothing.

The caller supplies an aggregate numerical-plus-wrapper budget within 6 GiB.
Admission counts numerical owners once, then adds pending-history capacity and
the wrapper's additional metadata before allocation, and rechecks actual vector
capacity. Execution repair's numerical budget is narrowed to leave space for
these additional owners. It never widens the original numerical budget.
Tokenizer, stream, retained output, telemetry and application owners still
require separate admission and physical process measurement.

`GenerationMemory` retains distinct recurrent, convolution history, KV, scratch,
logits and numerical metadata categories, alongside wrapper history/metadata.
The legacy generic memory view combines recurrent/convolution banks with scratch;
use the detailed report for hybrid owner attribution. Its weight field includes
the numerical model's already combined weights/model ownership.

`GenerationState` is a bounded snapshot of history, committed and pending lengths,
the last prompt/reset prefix decision and the last numerical call's `Work`.
Sampling appends do not overwrite the prefix decision. Work describes one
numerical call, including partial completed work on failure; it is not a request
aggregate. No text, prompt arrays, clocks or physical footprint are inferred.
Generic generation supplies its existing completed host phase/event timings;
error outcome collection and request-summary overhead remain later gates.

Cancellation in generic generation remains at chunk/token boundaries. A chunk
already submitted completes before cancellation is observed. GPU workspace
poison requires explicit repair; reset does not repair it. No logits mask is
introduced: the numerical vocabulary remains padded, while the separately
qualified codec rejects padding at its text boundary.

Six portable CPU controls pass, including zero warmed allocations, complete-output
admission, no-mutation rejection, pending/Stop/callback/cancel cases and late
whole-chunk failure/retry. Fresh formatting, both strict all-target Clippy and
644 CPU/778 Metal tests pass (41/318 ignored). Native58's three arm64 slices each
export 46 symbols; existing CPU/Metal C smokes pass. Tiny Metal controls verify
poison/repair, pending preservation and exact-budget repair rejection.

The authenticated 17-input/offset-37 process reproduces the previously qualified
full numerical state fingerprint and all final logits bit for bit. Shortening,
editing, GPU failure/repair/retry and zero warmed Rust allocations pass. Its
lifetime footprint peaks at 2,764,769,992 B. The harness admits both sessions and
diagnostics in advance, counts shared weights once and reserves the other
session's nonshared owners during repair. These checks preserve the existing
raw-input numerical scope; they do not qualify deeper contexts or text requests.

Earlier context-namespace/unused-test-helper errors and source-changing check
attempts remain in the [correction record](../artifacts/qwen35-generation-state-2026-11-09/source-corrections.json).
The first tiny native run passes its test but its collector rejects the harness-
prefixed footer. A versioned collector accepts only the exact named-test prefix;
the fresh tiny run passes without a Rust or binary change. Original receipts
remain intact. Eight Python reader tests and 36 actual resealed negative controls
pass; original evidence is unchanged. The controls reject incorrect work/state,
hidden owners, double-counted shared weights, false allocator settings, changed
build/source/driver bindings, missing samples/environment coverage, duplicate
confirmation processes and unsupported scope claims.

Two fresh sequential release processes each run three warmup and ten measured
pairs, alternate execution order and reverse the initial role in confirmation.
Both roles use the same 17 raw inputs at absolute offset 37, resident weights and
reset numerical state. Reset, loading, comparison and diagnostic output stay
outside timing. The wrapper timer includes `set_tokens` and the pending-drain
loop. All state/final logits remain bit-identical and warmed Rust allocations are
zero. External 30-second power/thermal/pressure observations and endpoint device/
OS/configuration checks pass. No competing build or model computation runs during
timing; light documentation and metadata work is recorded separately.

| Cohort | Numerical median (s) | Generation-step median (s) | Lifetime peak (GB) |
| --- | --- | --- | --- |
| 0 | 2.968 | 2.985 | 2.765 |
| 1 | 2.961 | 2.950 | 2.765 |

These phase baselines use the explicit owned-child startup profile
`MallocLargeCache=0`, with the other studied allocator settings absent. Parent,
app and library defaults stay unchanged; inherited settings are recorded.
The measurements qualify no speedup, request TTFT, tail, sustained or summary-
instrumentation overhead bound. Aggregate engine-plus-codec/application fit still
requires its own actual composition and physical measurement.

[Required checks](../artifacts/qwen35-generation-state-2026-11-09/required-checks-3/checks.json),
[source state/recovery](../artifacts/qwen35-generation-state-2026-11-09/fidelity-final.json),
[phase baselines and dispersion](../artifacts/qwen35-generation-state-2026-11-09/baseline-final.json),
[reader controls](../artifacts/qwen35-generation-state-2026-11-09/reader-controls.json).
Astra approves scoped retention with no actionable findings;
[checks/review/commit](../artifacts/qwen35-generation-state-2026-11-09/publication-checks.json). The
[frozen contract](../artifacts/qwen35-generation-state-2026-11-09/acceptance-contract.json)
does not qualify deeper contexts, managed request metrics, summary overhead,
quality, adoption or iPhone runtime. The full selected goal stays active.

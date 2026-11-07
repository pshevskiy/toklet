# Proposal: observable, performance-first inference

7 November 2026. Research and design only, inspected at engine revision
`be28a3973614a13844463e02c8ff27244f11aeeb`. This proposes a bounded follow-on to
[M9](next-performance-milestone.md); it does not change selected kernels,
qualification, memory budgets or the deferred iPhone scope.

**Recommendation:** make every request explain its cost, then use that evidence
to remove unnecessary work and qualify the best existing optimization for the
actual Mac application. Start with a small request summary and one matched
baseline. Expand instrumentation only when it answers a concrete decision.

| ClickHouse idea | Toklet application | Priority |
| --- | --- | --- |
| Query history with resource counters | One correlated request summary across Swift, native generation and Metal | First |
| Inspectable execution pipeline | Extend the existing runtime plan with selected operations and fallback reasons | First diagnostic extension |
| Vectorized execution over contiguous arrays | Reuse weights across prefill rows; select variants outside numerical loops | Existing foundation; broaden only with evidence |
| Lazy materialization | Compute/read back only outputs required by generation, preserving scoring paths | Audit existing routes first |
| Sampling profiler | Bounded operator captures correlated with request summaries | Opt-in diagnosis |
| Performance regression testing | Matched before/after runs with raw samples, noise and correctness gates | Every retained optimization |

These are adaptations, not a proposal to embed a database. ClickHouse documents
request statistics and lifecycle events in
[query_log](https://clickhouse.com/docs/reference/system-tables/query_log),
pipeline inspection through
[EXPLAIN](https://clickhouse.com/docs/reference/statements/explain),
array-oriented execution in its
[architecture](https://clickhouse.com/docs/resources/develop-contribute/introduction/architecture),
and delayed column loading through
[lazy materialization](https://clickhouse.com/blog/clickhouse-gets-lazier-and-faster-introducing-lazy-materialization).
Their corresponding inference mechanisms need Toklet's own measurements.

## Evidence already available

| Existing capability | Source and implication |
| --- | --- |
| Generation phases and cache accounting | [`generation::Report`](../src/generation.rs) already records prefill, decode, sampling, total time, reuse, emitted/history/evaluated/pending counts. Extend this contract rather than add a competing statistics implementation. |
| Completed Metal timing | [`metal::Timing`](../src/metal/mod.rs) already separates command creation, encoding, commit, wait and readback, and exposes GPU start/end. One dense evaluation encodes dependent work into one command buffer. Per-layer host waits are not a demonstrated general defect. |
| Bounded GPU profiling | [`profile.rs`](../src/metal/profile.rs) has calibrated CPU/GPU clocks, stage counters, a 1,024-event maximum and a separate 64 MiB diagnostic allowance. It explicitly rejects MoE instrumentation; support must remain visible. |
| Application timing | [`NativeWorker`](../apple/Harness/NativeWorker.swift) records submit-to-callback timing including queueing and Chat tokenization; [`GenerationMetrics`](../apple/Harness/HarnessStore.swift) derives TTFT, decode rate and total time. Callback arrival alone does not measure displayed text. |
| Readback policy | The managed dense [`native session constructor`](../src/native.rs) selects FullLogits even though Metal also implements GreedyToken. This is a concrete adoption candidate; the current app planning/scoring contracts must be accounted for before changing it. |
| Memory ownership and process observations | Native context/owner reporting and [`NativeProcessMemory`](../apple/Harness/NativeProcessMemory.swift) distinguish admission, current footprint and conditional loading peaks. Reuse these meanings. |
| Runtime planning | [`toklet plan`](../src/bin/cli/plan.rs) already reports explicit MLX Metal planning. It initializes runtime/pipelines, reserves no numerical owners and does not establish execution or fit in practice. |
| Reproducible benchmark workers | [`benchmark.rs`](../src/benchmark.rs) and [`mlx_mac_baseline`](../examples/mlx_mac_baseline.rs) record completed fixed-token work; the latter separately checks profile/timing output parity. Its layer/role reconstruction is specific to the selected 24-layer dense graph. |

The gap is consistent attribution and comparison across these surfaces. In
particular, the ordinary Rust generator returns `Err` before returning its
`Report` on execution/output failure. A request wrapper can preserve partial
timing and terminal status without changing that public error contract. Existing
Swift terminal/state handling must be reused rather than duplicated.

Two recorded results guide priority, without predicting new gains:

- The retained affine8 four-row tile reduced 1,024-token prefill from 6.568 s
  to 5.669 s (13.69%), and the complete fixed-token request by 6.22%. It remains
  a Mac opt-in; managed-session policy, capacity transitions and rendered UI
  qualification are separate work. [Decision](m9-mac-affine-decision.md).
- Gemma BF16 G2 already removes unnecessary consumer-row work. Both primary
  runs improved prefill by about 61% and request time by about 28%. Its G3
  assistant experiment regressed the longer guard and remains test-only.
  [G2 evidence](../benchmarks/results/2026-11-06-gemma-bf16-g2-retention-v1.json),
  [G3 decision](../benchmarks/results/2026-11-06-gemma-assistant-mtp-retention-v1.json).

## 1. A compact request cost record

Introduce a versioned summary produced at the application/request boundary.
Use one request ID across UI submission, native work and optional operator
captures, plus a session ID and session-state generation to distinguish resets,
model switches and replaced owners. Reuse existing generation IDs where present.

| Field group | Required meaning |
| --- | --- |
| Identity | Schema version; request/session IDs; engine and native-package/build identity; model/tokenizer identity; device; selected backend, arithmetic, weight/KV format, chunk, context capacity and readback policy |
| Work | Submitted prompt count; reused committed prefix; newly evaluated prefill rows; decode evaluations; sampled, emitted, committed and pending tokens; cache action and replay reason |
| Latency | Submit→worker queue delay; formatting/tokenization; completed prefill and decode; host selection; output decoding; first token callback; first nonempty displayed text; complete request |
| Execution counters | Command buffers; dispatches by operation class; host waits; host-visible bytes copied/read; cache replay evaluations; unexpected pipeline builds and buffer growth after warmup |
| Memory | Unique weight owners, KV allocated capacity and valid positions, scratch, text/application allowance; measured current footprint and available lifetime peak with instrument/scope |
| Outcome | Success/stop/cancel/reject/error; failed phase; stable error category; last valid committed/pending state and replay-required status; unavailable fields with reasons |

Attach large immutable configuration manifests once per run/session and refer
to them by identity. Do not repeatedly hash weights or resolve tensor names.
Routine records contain counts and configuration, not prompts, response text,
token arrays or full logits. Exact token hashes belong in explicit reproducible
benchmark captures. Request IDs belong in records, not aggregate metric labels.

Record one terminal summary for every handled request outcome, including
rejection before execution. Preserve the existing transaction boundaries:
instrumentation must not publish KV positions, consume a pending token, suppress
an error or read state declared invalid. Process death can leave an incomplete
request; expose that instead of fabricating a terminal event.

Use monotonic durations with explicit clock origins. Separate overlapping
quantities: host wait includes GPU execution, so `GPU time + wait time` is not
request time. Likewise, native generation duration and submit-to-callback time
are nested. Do not subtract uncalibrated CPU/GPU timestamps to invent queue
latency. Keep unmeasured gaps explicit.

## 2. Give observability its own performance budget

Start with typed counters on the request/session owner, incremented at operation
or chunk boundaries. Use a fixed enum or struct with documented units and scope;
format names only during export. No per-element events, global hot-loop atomics,
string maps, synchronous disk writes or numerical-loop allocation.

Proposed modes and limits are design targets to qualify, not existing guarantees:

| Mode | Mechanism and budget |
| --- | --- |
| Disabled | Existing execution path remains available as the overhead baseline. |
| Summary | Proposed maximum 16 KiB collector per active request and 1 MiB retained process ring; at most 4 KiB per serialized summary, excluding separately stored manifests. Admission includes these bytes. |
| Diagnostic | Explicit, pre-admitted bounded token/operator capture, using current GPU counter facilities where supported. Every capture states covered offsets, omitted events and instrumentation mode. |

Keep summaries locally in memory initially; explicit export can write bounded
JSONL after execution. If continuous persistence becomes necessary, add a bounded
writer queue and a proposed 64 MiB rotation cap. On pressure, drop whole telemetry
records and increment a dropped-record counter; inference must not wait on the
exporter. Handle slow consumers, writer failure and disk-full conditions without
changing generation state. Never silently present incomplete telemetry as complete.

Target **at most 1% median TTFT and request-time overhead** for summary mode on
the selected workload, with no new steady-state numerical allocations, GPU
commands or waits. Ten samples cannot establish a 1% bound when noise is larger:
collect stronger paired evidence or keep the mode opt-in/unqualified. Count
native/application allocations separately from Rust numerical allocations.

This borrows ClickHouse's separation between
[event counters](https://clickhouse.com/docs/reference/system-tables/events) and
[sampling profiles](https://clickhouse.com/docs/concepts/features/performance/troubleshoot/sampling-query-profiler).
Detailed captures are for diagnosis; acceptance timing remains uninstrumented.
Existing strict GPU capture failures should retain their diagnostic meaning;
the new best-effort summary exporter must not alter those APIs.

## 3. Explain the selected execution and its cost

Extend `toklet plan` rather than create a second planner. Proposed additions:
operation sequence, tensor dimensions/layouts, chosen kernel variants,
prefill/decode differences, unique buffer ownership, intermediate lifetimes,
estimated bytes and fallback reasons. Distinguish a metadata prediction from the
actual selected runtime and from observed command/dispatch counts.

Keep this as a description derived from existing concrete plans. It does not
require a general graph interpreter, dynamic kernel registry or JIT. Cache
variant selection outside inner loops. Initially support the same dense MLX
route as the selected request-summary pilot; list other families as unsupported.

For profiles, attach stable operation role/layer/shape identifiers when encoding
instead of inferring them later from event order and a hardcoded layer count.
Preserve bounded counter storage and clock calibration. A trace exporter can then
show host encoding, GPU work and readback on aligned lanes, with request/chunk
IDs and a coverage manifest. Extend Gemma/MoE only with their concrete captures.

Add a small offline report over exported records: slowest comparable requests,
prefill versus decode cost, cache/replay cost, host-visible bytes, allocation
changes, memory peaks and missing instrumentation. A static HTML/trace export
can follow when it aids a real investigation; a live dashboard is not a first
dependency. ClickHouse ingestion or an OpenTelemetry adapter can remain optional
external consumers of the versioned export if multi-device collection is needed.

## 4. Performance experiments, selected by observed cost

The first performance candidate should be **managed adoption of the existing
affine tile for its qualified checkpoint**, after a current baseline and the
missing app/context checks. The earlier gain is evidence of potential, not proof
for the app's chunk-32 and capacity-transition configurations. The current
managed dense route selects FullLogits. Separately measure whether qualifying
GreedyToken for ordinary Chat is more valuable; preserve full-logit/scoring
consumers and update the actual resource/planning contract if that route changes.

| Observed bottleneck | Bounded experiment | Evidence needed to retain |
| --- | --- | --- |
| Prefill repeatedly reads/reconstructs the same weights | Qualify existing tile adoption; then consider another bounded row tile or an available Apple matrix primitive on representative shapes | Same reconstruction/accumulation contract and tail handling; independent numerical checks; actual app TTFT/request improvement; scratch and occupancy effects |
| Significant work produces unused intermediate/output rows | Audit last-row output-head selection, G2 consumer rows and token-only readback at actual callers; optimize only demonstrated remaining work | Full-position scoring remains correct; unchanged logits/selection semantics, cache state and stop behavior; lower measured work and end-to-end latency |
| Decode projection, normalization or attention dominates | Refresh shape/depth profiles, then choose one kernel experiment; preserve unsupported-shape fallbacks | Real-input arithmetic, quality and state checks plus independently repeated complete-request timing |
| CPU encoding or synchronization dominates | Reuse eligible descriptors/resources and reduce redundant commands or readback boundaries where measured | Buffer lifetimes/dependencies, failure and cancellation unchanged; completed-work timing, not submission rate |
| Prefix edits or continuation replay too much work | Instrument the existing exact-prefix mechanism and find avoidable replay at callers | Exact model/position/token-prefix identity; fully cached and pending-token controls; report reused/evaluated rows rather than a misleading binary cache hit |
| Visible text lags native token events | Measure worker queueing, decoding, polling and UI publication; batch only where beneficial | Better first-visible-text and inter-update latency, bounded text storage and preserved Stop responsiveness |

For each candidate, estimate its opportunity from measured phase share: if an
operation occupies fraction `f` of a request, even eliminating it saves at most
`f`; a local speedup `s` gives ideal request speedup `1 / (1 - f + f / s)`.
This is a screening estimate, not a forecast: changed memory traffic, scheduling
and overlapping work require a fresh end-to-end run.

Track logical weight/activation/KV bytes and arithmetic per shape. Label them as
modelled traffic, not measured DRAM bandwidth: cache reuse, compression, shared
owners and repeated reads change actual traffic. Use measured device counters
only when available and qualified. Preserve contiguous weights/activations,
static dispatch, fused dequantization and resident KV; these are already design
strengths. Do not assume a CPU SIMD width, cache size or GPU lane count.

## 5. Delivery and acceptance

These are proposed tasks, not implementation completion claims. Use the existing
[performance policy](../PLAN.md#validation-and-performance-policy) and
[improvement record](improvements.md) for all acceptance decisions.

| Order | Concrete delivery | Exit condition |
| --- | --- | --- |
| O1 | Freeze summary fields and adapt one managed Qwen2 Mac request path, with disabled/summary modes and bounded export | Success, pre-start rejection, execution/output failure, cancellation and replay summaries agree with authoritative session state; clock/memory meanings verified; overhead target qualified or mode remains opt-in |
| O2 | Matched baseline and small offline comparison report using existing workers | Separate app and numerical tracks; exact manifests, raw samples, profile parity and explicit missing observations; identify one actionable cost |
| P1 | One existing-route adoption or measured kernel/caller experiment | Frozen primary metric, unchanged correctness/quality/state/memory gates, independent confirmation; retain or reject explicitly |
| O3 | Extend runtime explanation and stable profile attribution for that route | Predicted versus actual dispatch/owner inventory checked; unsupported routes and bounded-capture overflow explicit |
| R1 | Reusable regression job for the selected route | Comparable baselines admitted by identity; failures/noisy results reported separately; artifact links and accepted revision retained |

O3 can move earlier if O2 cannot identify the cost with existing instrumentation.
This avoids making a broad observability rewrite a prerequisite for one useful
optimization. Later model families need explicit scope and owners.

Use the existing 128/512/1,024-input, 128-output numerical matrix only where
admitted. Keep the historical Qwen2 capacity-1,152/chunk-128/FullLogits benchmark
separate from the managed capacity-512/chunk-32/FullLogits Chat baseline. A
GreedyToken candidate must be explicitly identified. Freeze practical app
workloads that fit their actual capacity; never silently truncate larger cases.
Include exact append, fully cached, pending token, changed prefix, capacity
rejection and cancellation. Include cold loading separately; authenticated reads
may warm filesystem caches and do not establish cold-storage performance.

Use optimized builds, three warmups and ten measured samples for initial
screening, raw samples and median/dispersion, alternating order where useful and
a separate confirmation. Run competing model trials sequentially on the same
named Mac with power/thermal observations. Ten requests do not establish reliable
p95/p99; add a separately sized sustained/tail study when that metric matters.
Keep at least ten minutes of sustained evidence where required by the plan.

The default retain gate remains at least **5% improvement in the declared primary
metric**, exceeding noise, without unexplained required-workload regression.
Smaller effects need stronger evidence. Preserve frozen numerical and full quality
gates; reuse prior quality only under the project's existing exact-output/input
rules. No normalization/reassociation shortcut gets a relaxed tolerance.
ClickHouse's
[performance-testing practice](https://presentations.clickhouse.com/2022-highload-spb/clickhouse_performance_optimization_practices/)
supports collecting resource counters with timings and distinguishing a real
change from natural variation; Toklet's existing acceptance policy remains binding.

Preserve each route's numerical/process limits and the plan's initial 6 GiB Mac
ceiling, including telemetry and temporary storage; narrower trial caps still
apply. A logical allowance is not a measured process peak. Record unavailable
energy/device counters as unavailable. iPhone adoption requires its own hardware
qualification when the user resumes that scope.

Every implementation task needs formatting, strict Clippy, applicable CPU/Metal
tests and targeted model/session checks, one combined Astra diff/invariants/checks
review, an evidence record, concise TODO/improvement updates and a commit. This
proposal itself changes documentation only and has no new speedup claim.

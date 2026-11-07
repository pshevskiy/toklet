# O1: managed Gemma request summaries

7 November 2026. The pilot instruments accepted managed official Gemma 4 E2B
BF16 Mac Chat requests, with disabled mode as the default. It preserves the
native ABI, numerical source, weights, generation options and existing state
transactions. It does not adopt a new model or precision.

`NativeWorker(requestSummaryMode: .summary)` enables collection explicitly.
For a newly created app worker, `TOKLET_REQUEST_SUMMARIES=1` enables the same
pilot on Mac. While idle, `exportRequestSummaries()` returns bounded JSONL Data;
its caller chooses whether/where to persist it. There is no automatic file
writer or continuous exporter. Set `TOKLET_BUILD_ID` to the build/source receipt
identity (at most 256 bytes); missing identity is marked unavailable. The
reproducible runner supplies a hash of its source/archive manifest.

## Scope and metric meanings

The manifest records the exact official model/tokenizer manifest identity,
clock/memory instruments, field coverage and dropped/export-error/budget-skip
counts. Other model profiles are unsupported. Submit rejections are recorded
while the Gemma identity is visible; attempts without that identity are outside
this collection scope. Request IDs correlate records; stateGeneration is the
worker snapshot revision at submission, and before/after native owner IDs and
committed/pending counts identify authoritative session state. Capacity/chunk
refer to the session at submission and are unavailable if it had been suspended.

| Available | Interpretation |
| --- | --- |
| Request and queue milliseconds | Submission to worker terminal publication; submission to dequeue. Neither is terminal UI/render latency |
| Callback TTFT; native first/last emitted-event elapsed time | Callback TTFT includes worker queueing/tokenization. Native elapsed time has its own origin; no uncalibrated event→host subtraction |
| Native and callback decode token rates | `(N-1)/(last-first)` for more than one emitted token and positive interval. Hidden stop IDs are excluded; zero/single-token requests have no rate |
| Encoding/evaluation/output-decoding durations | Swift phase wall time. Evaluation includes callbacks, native scheduling, completed GPU work and hidden stop work; it is not isolated prefill or GPU execution |
| Prompt count, callback/native emitted count, output UTF-8 byte count | No routine token IDs, text or prompts retained in the summaries |
| Before/after native history/committed/pending totals | Session totals, not invented per-request sampled/prefill/decode evaluation counts. Chat uses native reset+prompt, not prefix reuse |
| Session admission, KV ownership, codec allowance | Admission/allowance and physical ownership remain separate; shared owners are not added again |
| Current process footprint and lifetime physical peak | TASK_VM_INFO observation at request completion. Lifetime peak is not a request peak; no isolated load-peak claim |
| Terminal outcome, failed phase/status, cancellation acknowledgment latency | Success/hidden stop, output limit, cancellation, rejection or failure. Queued lifecycle cancellation gets a terminal record; failed output decoding preserves evaluated prefix |

Unexposed native prefill/decode/sampling phases and work counts, kernel counters,
weight/scratch breakdown, memory pressure and UI render/final-answer timestamps
remain explicitly unavailable in the manifest. A native failure with unreadable
session state sets replayRequired=true; other cases remain unknown because the
ABI exposes no general poison/replay flag. O2 should add those probes with their
own acceptance checks, rather than infer them from overlapping durations.

## Bounds, failure and admission

Fixed 32-slot value-record ring; each active collector is below 16 KiB, each
serialized record below 4 KiB, total export below 135,201 bytes. The fixed ring is allocated when an opt-in worker is initialized. A 1 MiB
headroom check for collection/export is applied against the selected numerical
budget before collecting; this does not reserve bytes in native admission; an insufficient allowance skips telemetry and
increments budgetSkippedRecords without changing inference. Caller-owned
returned Data is separate. Ring overwrite drops whole records; counters
saturate. Export snapshots idle records under the worker lock, then serializes
outside it. Export errors increment a counter; generation/session state is
untouched. Mode changes require an idle provisioned worker.

No added native/GPU commands or waits, numerical allocations, per-element
instrumentation or token JSON. Callback collection updates only fixed scalar
fields under the existing worker lock. Optional phase observations stay on the
engine's owning thread. The native engine was rebuilt from current source for
acceptance; historical archives whose source pins differed were not reused.

## Reproduction and evidence

From the repository root, build the current release native library, then run:

```sh
cargo build --release --features metal --lib
python3 tools/experiments/o1_gemma_summary.py MODEL_FOLDER target/release/libtoklet.a artifacts/NEW_O1_RUN
```

The runner compiles strict Swift 6, tests formulas/bounds/ring pressure,
exact-fit/insufficient allowance and budget-skip behavior, and real
Gemma Chat/rejection/limit/cancellation/replay/recreation/queued cancellation/
output-failure/export-failure controls. It then runs two independent batches,
each with three warmup pairs and ten alternating measured pairs. Caller-side
submission→idle-acknowledgment timing includes collector setup, finalization and
publication in both modes; callback TTFT uses that same caller start. Output tokens
and native state must agree across modes. Raw samples and source/archive hashes
are retained. This short Chat workload does not qualify broader context, tail
percentiles, sustained performance or individual native phase overhead.

[Checks](../artifacts/o1-gemma-summary-2026-11-07/checks.json),
[current native build](../artifacts/o1-gemma-summary-2026-11-07/current-build.json),
[measurement result](../artifacts/o1-gemma-summary-2026-11-07/acceptance/result.json).
The accepted conformance JSONL lives in the same requested output directory;
reproduction never depends on or overwrites a previous run's evidence.
On Apple M1 Pro (AC power; nominal thermal state before/after), the two complete
caller-boundary batches measured median changes of −0.36%/+0.56% in callback
TTFT and +0.52%/+0.45% in request time. Raw outliers are retained. Paired bootstrap
upper bounds exceed 1% in required metrics, so **the overhead target is not
qualified**; summary mode stays opt-in/default-disabled. These differences are
not speedup claims. Observed lifetime physical peak stays below the 6 GiB cap.
Earlier worker-timestamp and historical-archive measurements are superseded.

Cargo fmt/strict Metal-feature Clippy, 594 CPU tests (32 ignored), 713 Metal-feature
tests (278 ignored), 16 portable Python checks, strict Swift controller/render/
iOS/simulator compilation and the explicit Gemma controls pass. Numerical and
full quality gates are reused only for unchanged numerical sources/artifacts;
mode comparisons require exact stock token/state agreement. No physical iPhone
qualification. [Result receipt](../benchmarks/results/2026-11-07-o1-gemma-summary-v1.json).
[Astra approved](../artifacts/o1-gemma-summary-2026-11-07/astra-final.md) the
scoped opt-in commit after Xcode source membership, isolated conformance output
and complete-boundary timing fixes. Unsigned simulator Xcode build passes.
The largest conformance JSON record was 1,198 bytes (1,199 including newline).

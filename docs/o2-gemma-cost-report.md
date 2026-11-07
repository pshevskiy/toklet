# O2.A: native phase metrics and matched warm cost reports

7 November 2026. This stage extends the managed official Gemma 4 E2B BF16 Mac
pilot with completed native generation phases and an offline cost report. It
preserves the legacy generate API and the numerical kernels. O2's cold-load,
continuation, sustained and UI-render studies remain separate follow-on work.

## Native observation contract

`toklet_session_generate_observed` is an additive entrypoint. Its 96-byte
`TokletGenerationMetrics` output copies the engine's existing completed
`generation::Report`: reused rows, evaluated prefill rows, sampled/emitted
counts, prefill/decode/host-selection/total seconds and optional first emitted
native-event time. No new timers, GPU commands, waits, retained tensors or
per-token collections are added. Backend selection remains at request scope.
The original 40-byte `TokletReport` and existing generate function stay intact.

`REPORT_KNOWN` covers a completed report, including cancellation; a separate
`FIRST_TOKEN_KNOWN` bit distinguishes no emitted event from a measured zero.
Hidden stop IDs are sampled but not emitted. A failed native generation has
unknown phase fields, even if its recoverable token state is available in the
ordinary report. Initialized output storage is cleared before request
validation; invalid output/error pointers follow the existing foreign-caller
contract. No stale measurements are retained in a session.

Swift summary schema 2 attaches validated `nativeMetrics` and a coverage status:
completed, notStarted, generationFailed or invalidRecord. Prefill throughput is
newly evaluated rows divided by completed prefill wall time; no rows/time means
unavailable. Callback/native clocks remain distinct. Failure during subsequent
text decoding retains the completed native report and its valid numerical prefix.
Kernel counters, decode-evaluation counts, isolated load peaks and UI render/
final-answer timestamps remain explicitly unavailable.

## Report and comparison scope

The existing pinned runner produces two independent batches with three warmup
pairs and ten alternating measured pairs each. The caller timer includes submit,
collector setup, finalization, publication and idle acknowledgment. Both modes
must emit identical token IDs and terminal session counts. Only summary mode
calls the new observed API; the disabled mode uses the legacy entrypoint.

The offline report joins raw requests to summaries by request ID and verifies
model/codec/build identity, complete record coverage, clocks, context/chunk,
requested output, token/state parity, hidden-stop accounting and phase sums.
It bounds input reads, rejects duplicate/nonfinite JSON and never treats a
missing metric as zero. It reports per-phase medians, prefill throughput,
unclassified native remainder, comparable caller timings, slow request IDs and
explicit coverage gaps. Native phase medians are instrumented observations;
medians of components need not sum to the median total. This does not establish
UI latency, cold storage, tail percentiles, sustained performance or DRAM traffic.

```sh
python3 tools/experiments/o1_gemma_summary.py MODEL_FOLDER NATIVE_ARCHIVE artifacts/NEW_O2_BASELINE
python3 tools/bench/request_cost_report.py artifacts/NEW_O2_BASELINE artifacts/NEW_O2_REPORT
```

Use a freshly verified archive with all 46 exports. The local Mac/iOS/simulator
package is built by `tools/experiments/m7_apple_package.py`; header/symbol/link
inventory is checked per slice. The previous generated package selection is
preserved in the selection receipt. Cross-compilation does not qualify iPhone.

## Validation and decision

Native tests cover prefill, pending continuation, hidden stop, zero output,
cancellation, callback failure, invalid boundary, reentrancy and wrong-thread
access. The C smoke checks both old/new APIs and layout. Strict Swift validation
checks unknown and nonfinite records, real Gemma counters and pre-start rejection;
existing lifecycle/failure/collection controls remain required. Reporter fixtures
independently check phase arithmetic, work/identity/owner mismatches, dropped
records, duplicate JSON and bounded reads.

All required checks pass: fmt/strict Metal-feature Clippy, 599 CPU tests (32
ignored), 718 Metal-feature tests (278 ignored), 46-export package header/symbol/
link checks on all three Apple slices, C CPU smoke, strict Swift UI/controller/
render/iOS/simulator checks and unsigned simulator Xcode build. Five reporter
fixtures also pass under optimized Python; three measurement-runner tests pass.
[Astra approved](../artifacts/o2-gemma-metrics-2026-11-07/astra-final.md) after the
reporter gained request/callback/caller duration reconciliation and native/wrapper
clock enclosure checks. No numerical quality gates were relaxed.

Two matched warm batches observe about 395/393 ms prefill, 1,794/1,796 ms decode
and 9 ms host sampling. Decode is the dominant observed native phase; operator
attribution is needed before selecting a kernel change. Prefill rate is about
38 evaluated rows/s for this 15-token prompt. Median summary-mode changes were
+0.79%/+0.07% callback TTFT and −0.51%/+0.39% caller request time. The 1% overhead
bound remains unqualified, so collection stays opt-in/default-disabled. This is
an attribution delivery, not a speedup or broader workload qualification.

[Result receipt](../benchmarks/results/2026-11-07-o2a-gemma-phase-report-v1.json),
[validated report](../artifacts/o2-gemma-metrics-2026-11-07/report-validated/report.md),
[check ledger](../artifacts/o2-gemma-metrics-2026-11-07/checks.json),
[native package](../artifacts/o2-gemma-metrics-2026-11-07/native46/build.json).

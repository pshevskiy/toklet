# Observability and performance outcomes

This records the engine/CLI delivery and explicit model decisions. The user
chose to finish Qwen work with rejection decisions. Numerical, quality and
memory gates are preserved; positive Qwen performance/adoption is not claimed.
UI/screenshare and further physical iPhone work remain user-deferred. Larger
models remain research options.

| Requirement | Delivered outcome and evidence |
| --- | --- |
| O1 request metrics | Managed and ordinary Gemma CLI summaries expose completed phases, tokens/work, cache state, outcomes and bounded telemetry. Unknown observations retain nulls with reasons. Summaries stay disabled by default: three of eight original CLI overhead comparisons fail the 1% bound. [CLI and measurements](gemma-cli.md) |
| O2 baselines and offline reporting | Matched Chat/raw/CLI workloads, state/boundary checks, independent cohorts, sustained runs and phase reports. Additive native ABI metrics include completed decode rows. [Workloads](o2-gemma-workloads.md), [reports](o2-gemma-cost-report.md), [decode counters](gemma-decode-evaluations.md) |
| O3 explanations and attribution | Concrete execution plans and unique owners; stable source/layer/role/shape identifiers; predicted/observed work and overflow/unsupported reporting. Bounded profiling preserves logits and output state. [Runtime](o3-gemma-runtime.md), [console profile](gemma-profile-cli.md) |
| R1 regression jobs | Reusable Chat and ordinary CLI comparators bind source/model/workload/publication identities and independent owner cohorts, with separate regression/noise/incomparable exits. Actual baseline admission and both Python modes pass. [Chat](gemma-regression-job.md), [CLI](gemma-regression-job.md#ordinary-cli-extension) |
| Gemma 4 E2B | Official BF16 single-row submission is retained as a Mac opt-in. Two complete caller cohorts improve 7.4005%/7.1650%; both paired lower bounds exceed 5%. Numerical, complete 10k-target/100-task quality, state, allocation and memory gates pass. Preferred Heretic Q4 retains its explicit quality rejection. [Measured retention](gemma-single-row-submission.md), [Heretic decision](gemma-heretic-quality-probes.md) |
| Qwen3.5-4B | Reject selected Avan adoption at its original 128/256 gate. The separate authenticated Huihui candidate also rejects: 372 incompatible reference intervals. Existing scoped diagnostics remain retained. [Original decision](qwen35-context256.md), [Huihui decision](qwen35-huihui4b-envelope.md) |
| Qwen3.5-9B | Reject the selected Lukey route at the original dual-profile 128/256 gate: both captures have 45,332 failed comparisons; 14,847 reference intervals are disjoint. The newer header-only candidate is not pursued under the user's closing direction. [Decision](qwen35-9b-context256.md) |
| MiniCPM5-2B | Retain qualified diagnostic/CLI and measured RMS-store opt-in scopes. Reject current managed adoption: the 1,025-position comparison retains 82 numerical failures. Full quality remains unqualified. [Decision](minicpm5-precision-audit.md) |
| LFM3-1.7B | Retain the authenticated BF16 CPU/CLI route and finite widening after full quality, state, memory and request measurements. Managed/GPU/Q4 and all-attempt summaries remain conditional extensions. [Retention](lfm3-finite-widening.md) |
| Granite 4.2 3B | Retain converted-source and diagnostic tooling; reject current converted Metal adoption at its full-model numerical gate. Dependent quality/performance remains unqualified. [Decision](granite-graph.md) |
| Ministral 3 3B Reasoning | Retain bounded conversion/diagnostics; reject current packed Metal adoption with 78 failures in 244,809,728 comparisons. Dependent quality/performance remains unqualified. [Decision](ministral3-graph.md) |
| Qwen3.5-2B | Retain source/conversion/raw-ID diagnostics; reject wider adoption at context17 with three failures in 680,841,024 comparisons. Conditional fixed255 and downstream campaigns do not advance. [Decision](qwen35-2b-real-source.md) |

Each outcome applies to its named artifact, route, device and workload. Retained
diagnostics do not grant model adoption. Optional ordinary GPU/host/footprint
and cancel-signal probes remain explicit coverage gaps permitted by the
[metric contract](model-observability-plan.md#metric-contract). They are not
reported as measured zero. No unmatched Ollama/vLLM speed ratio is qualified;
the [external observations](lfm3-engine-comparison.md) retain input/output limits.

The 6 GiB Mac ceiling, immutable weights/session ownership, exact cache/state
rules, independent numerical fixtures and original complete quality gates remain
binding. A rejected prerequisite stops downstream qualification.

The final diagnostic and CI correction are published as `155c6718` and
`60415234`, with Astra approval and all seven code-head workflows/15 jobs green.
[Publication](../artifacts/qwen35-alternative-admission-2026-11-15/envelope-publication-1.json),
[requirement audit](../artifacts/qwen35-alternative-admission-2026-11-15/whole-goal-final-audit-1.json)
and [current checklist](../TODO.md) retain the exact scope and evidence.

# MiniCPM5 exact-reduction RMS store experiment

This Mac engine experiment preserves the scalar Half RMS normalization arithmetic
and distributes independent output stores across one queried SIMD group per row.
It is selected explicitly for the bounded MiniCPM5 geometry. The scalar reference
and ordinary factories remain available. Full model/adoption gates stay separate.

## Mechanism and ownership

The input/output layout is contiguous `[rows, hidden]`; weights are `[hidden]`.
Lane zero performs the original ordered FP32 reduction (including compensated
addition), epsilon/division/square-root/inverse. A SIMD broadcast shares that
inverse; lanes write disjoint columns with the original multiplication order and
checked Half conversion. In-place input/output is supported. There is no extra
scratch, GPU command, operator wait or host readback.

The pipeline queries the execution width, uses precise Metal compilation and
caches once per model GPU owner. Selection checks the existing dense MLX
MiniCPM5 geometry, Compensated/MlxAffine/Reference modes, fresh session, capacity
at most 512 and chunk at most 32. Other experimental schedules cannot compose.
Admission reserves another 64 MiB before lazy compilation; it is an allowance,
not a claim that physical use increased by that amount. Ordinary timing/profile
APIs still report completed work; the new kernel has a distinct trace name.

## Console use

Build the existing `minicpm5_probe` release example with the `metal` feature.
The explicit commands take the pinned model directory, independent reference
manifest, its SHA-256, and a fresh output directory:

```text
minicpm5_probe bench-rms-baseline MODEL REFERENCE SHA NEW-OUTPUT full-logits
minicpm5_probe bench-rms-parallel MODEL REFERENCE SHA NEW-OUTPUT full-logits
minicpm5_probe capture-rms-parallel MODEL REFERENCE SHA NEW-OUTPUT
```

The two request commands record the implementation and separate session-setup
seconds, alongside load/conversion/compilation, token-decision TTFT, completed
prefill/decode, selection, command counts, cache/state and memory metrics. Setup
includes lazy compilation for the candidate and stays outside warm request time.
No tokenizer, chat template, display capture or user-visible TTFT is involved.
Use `tools/bench/minicpm5_rms_performance.py` with the raw report, owned-process
receipt and frozen contract to validate a cohort; the ordinary request reader
keeps its existing schema. Numerical captures require the explicit
`--rms-parallel-store` checker flag. Raw evidence and receipts are never rewritten.

## Qualification

The [prospective mechanism contract](../artifacts/minicpm5-rms-store-2026-11-10/prospective-contract.json)
and [six-cohort request contract](../artifacts/minicpm5-rms-store-2026-11-10/request-contract.json)
freeze correctness, workload and retention criteria before candidate outcomes.
The primary metric is ordinary complete fixed-work request time; both comparisons
must improve at least 5%, beyond observed dispersion and between-run variation,
without unexplained TTFT/decode/state/memory regressions. The new scalar control
is also compared with the frozen previous release.

Isolated GPU screening passes 240 cases and 1,861,392 finite Half comparisons,
with independent uniform-row checks, tails, status flags and invalid controls.
Two reversed-order micro runs reduce completed RMS kernel time about 59–62% for
rows 1/4/17/32 at hidden size 2048. Eight encoders share each command; these are
operator measurements, not complete-request results.
[Raw samples and summary](../artifacts/minicpm5-rms-store-2026-11-10/micro-summary-1.json).

Fresh formatting, strict all-target CPU/Metal Clippy and all 670 CPU/811 Metal
tests pass (46/334 explicit tests ignored respectively). Two explicit GPU tests
pass, including in-place output, actual tiny-graph dispatch, all-stage bitwise
parity and reset/replay. All 34 portable tests pass.
[Required checks](../artifacts/minicpm5-rms-store-2026-11-10/required-checks-3/checks.json),
[GPU checks](../artifacts/minicpm5-rms-store-2026-11-10/gpu-integration-1/checks.json),
[portable checks](../artifacts/minicpm5-rms-store-2026-11-10/portable-1/checks.json).
The real-model capture passes all 52,896,768 original dual-profile comparisons
with zero failures (maximum gate fraction 0.9722063365). All 2,614 captured arrays
and their journal are bitwise identical to the original scalar capture; native
selections and generated IDs match. The 512 boundary is a functional state gate.
One-byte-under-budget rejection preserves state/owners; the extra pipeline
allowance is explicit. Maximum instrumented admission is 2,945,971,432 bytes,
including the application allowance.
[Numerical gate](../artifacts/minicpm5-rms-store-2026-11-10/native-comparison-1.json),
[exact parity/admission](../artifacts/minicpm5-rms-store-2026-11-10/native-exact-parity-1.json).
All three fresh native65 slices pass compile/export/strict C link/package checks
with 46 exports each; CPU and Metal C smokes pass. This does not qualify iPhone
runtime.
[Package](../artifacts/minicpm5-rms-store-2026-11-10/native65-publication.json),
[C smokes](../artifacts/minicpm5-rms-store-2026-11-10/native-smoke-65/checks.json).
The six eligible release cohorts retain 60 measured requests and 18 warmups:

| Variant | Run 1 request / TTFT | Run 2 request / TTFT | Decode tokens/s, runs 1 / 2 |
| --- | --- | --- | --- |
| Frozen previous release | 499.94 / 192.48 ms | 499.93 / 192.36 ms | 9.758 / 9.759 |
| New scalar control | 500.60 / 192.57 ms | 500.17 / 192.59 ms | 9.740 / 9.753 |
| Parallel stores | 392.92 / 165.45 ms | 392.66 / 165.29 ms | 13.193 / 13.197 |

Complete warm request time falls **21.51% / 21.50%**, internal token-decision
TTFT falls **14.08% / 14.17%**, and decode rate rises **35.44% / 35.31%**.
The largest request IQR is 0.469%; between-run median variation is at most
0.087%, so both gains clear the frozen 5% screen. Scalar fallback differs from
the previous release by +0.132% / +0.049%. All four per-depth completed walls
improve; completed command count remains four per request.

These are loaded-model, warm-filesystem requests: four prompt IDs and four
fixed teacher decisions at context depths 4–7. Loading is separate (7.20–7.42 s
across these processes). Observed scalar session setup is 8.37 / 7.32 ms and
parallel setup 9.47 / 8.70 ms. Those are single observations per process with
existing Metal/compiler caches, not uncached compilation distributions. No text
streaming, long-context performance, sustained stability or tail claim follows.
AC power, Low Power Mode off and thermal state zero pass at every cohort boundary.
[Samples, dispersion, environment and decision](../artifacts/minicpm5-rms-store-2026-11-10/performance-summary-1.json).

Request admission is 2,785,878,866 bytes for scalar and 2,852,987,730 for parallel,
before the same 32 MiB application allowance. Across all six cohorts, exact final
wait4 RSS is at most 2,743,615,488 bytes and live-PID lifetime footprint samples
at most 2,732,771,776. The diagnostic capture observes 2,595,913,728 RSS and
2,791,213,552 sampled physical bytes. Final physical peaks remain unavailable;
these observations establish no memory-saving claim or unrestricted physical peak.

All 34 portable tests, three actual request positives/38 resealed negatives and
11 capture negatives pass; original evidence bytes remain unchanged. The updated
reader also revalidates the original 52.9M-value capture with identical results.
[Request controls](../artifacts/minicpm5-rms-store-2026-11-10/request-controls-1/checks.json),
[capture controls](../artifacts/minicpm5-rms-store-2026-11-10/capture-controls-1/checks.json),
[legacy revalidation](../artifacts/minicpm5-rms-store-2026-11-10/legacy-capture-revalidation.json).

Astra approves retaining the explicit opt-in for this qualified workload with
no actionable findings. Reviewed code/contracts stay frozen; administrative
documentation closure is permitted. Ordinary defaults, model adoption and
ordinary telemetry's unqualified 1% overhead gate remain separate.
[Review](../artifacts/minicpm5-rms-store-2026-11-10/astra-review-result.json),
[commit and integration receipt](../artifacts/minicpm5-rms-store-2026-11-10/publication-checks.json).

## Interrupted checks and limits

The first required-check attempt passes formatting and CPU Clippy, then stops
at its disk guard during compilation. An inactive generated debug-check cache
is removed; models, source, raw logs and published binaries are preserved. A
fresh run with check-only debug information disabled passes CPU tests but finds
an autorelease closure arity error in Metal setup. That error is fixed and fresh
checks rerun. An initial portable-test controller guesses a nonexistent filename
after three successful suites; the inspected numerical test name corrects it,
and all 34 portable tests pass. Original logs remain.

An earlier post-micro snapshot observes disk free space below the floor; its
cause is unverified. Verified byte-identical APFS clones recover fixture storage;
no uninterrupted floor claim is made. This storage work establishes no engine
memory improvement. Final physical memory peaks remain unavailable unless a
separate final counter supplies them; live-PID footprint and final wait4 RSS have
different scopes. The 6 GiB Mac ceiling, 512/32 bounds, 512 MiB codec limit and
all original numerical/quality gates remain binding. Physical iPhone, full
MiniCPM5 lineage/codec/task quality/deeper numerical qualification and managed
adoption remain open.

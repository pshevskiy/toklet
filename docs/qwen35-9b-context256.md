# Qwen3.5-9B 128/256 evaluation

Q2.G rejects deeper coupled fidelity for the pinned 9B artifact. Both complete
native captures have 45,332 failures in 79,353,856 comparisons, with maximum gate
ratio 39.3934. Their bodies match bitwise and greedy IDs agree with both references.
Functional, partition and repair checks pass under 6 GiB. Descriptive performance
measurements and evidence controls are complete. Astra approves scoped diagnostic
retention and commit with no actionable findings;
[review/commit record](../artifacts/qwen35-9b-context256-2026-11-09/publication-checks.json).

## Numerical result

The frozen workload repeats the existing 17-token pattern to 256 inputs at
position offset 37. Checkpoints 128 and 256 observe absolute positions 164 and 292.
Each includes all 1,051 final-token stages, recurrent states, committed K/V and
full logits. The native body is 158,707,712 B; both independent arithmetic profiles
occupy 476,123,136 B under the new explicit 512 MiB capture bound. Older readers
retain their 256 MiB bound. Original scalar/vector graphs, the approved exact
helper, source weights, profiles and tolerances are unchanged.

| Checkpoint | Reference profile | Compared values per capture | Failures | Maximum gate ratio |
| --- | --- | ---: | ---: | ---: |
| 128 | Ideal FP64 | 18,773,504 | 4,116 | 18.0125 |
| 128 | Staged FP32 | 18,773,504 | 6,321 | 25.8518 |
| 256 | Ideal FP64 | 20,903,424 | 19,782 | 29.8906 |
| 256 | Staged FP32 | 20,903,424 | 15,113 | 39.3934 |

An independent interval audit finds 14,847 saved coordinates with disjoint
original reference tolerances: 3,776 at 128 and 11,071 at 256. It expands tolerance
radii and endpoints outward before checking separation; the largest conservative
gap is 0.00142625. Counts include committed-prefix coordinates saved in both
checkpoints. No scalar can satisfy both original intervals at those coordinates.
This explains an incompatible coupled gate; it does not prove a particular
implementation correct or authorize a policy change. Matching greedy IDs is
insufficient for adoption. Prior one-input/context17 qualification and the
separate 4B 128/256 rejection retain their original scopes.

[Decision](../artifacts/qwen35-9b-context256-2026-11-09/decision.json),
[fidelity 1](../artifacts/qwen35-9b-context256-2026-11-09/fidelity-1.json),
[fidelity 2](../artifacts/qwen35-9b-context256-2026-11-09/fidelity-2.json),
[interval audit](../artifacts/qwen35-9b-context256-2026-11-09/reference-intervals-2.json).

## Functional and memory evidence

Both native runs compare every logit bit and both checkpoint states across
32/17/1-token partitions, with zero warmed Rust numerical allocations. From a
128-token prefix they test partial observer failure, cancellation, stale
nonfinite future K/V, GPU-head poisoning, blocked execution, minus-one repair
admission, repair and exact retry. Empty, oversized and invalid-token requests
preserve the prefix. Generation admission accepts 128+128 and rejects 512+128
and 1024+128 at the fixed 512 capacity. These are functional/admission witnesses,
without request, codec, quality or managed-route qualification.

| Memory observation | Bytes |
| --- | ---: |
| Planned numerical session admission | 5,538,789,612 |
| Repair admission peak before diagnostic reserve | 5,580,214,680 |
| Separate diagnostic reserve | 335,544,320 |
| Requested fixed diagnostic owners | 254,345,296 |
| Measured native lifetime peaks | 5,666,820,656 / 5,666,804,272 |
| Measured independent reference lifetime peak | 733,218,208 |
| Mac ceiling | 6,442,450,944 |

One external vector holds all 256 logit rows. Weights retain no second full CPU
matrix copy after Metal upload. Planned admission, requested owners, diagnostic
allowance and measured physical footprint remain distinct. The independent
reference completes in 19,816.804 seconds under its frozen eight-hour deadline;
this is a diagnostic collection cost, without an engine speed claim.

Three positive copies and 39 damaged-evidence controls pass, including resealed
finite/nonfinite body changes, work/state/clock scope, live counters and original
primary-result linkage. A finite rewritten embedding remains structurally valid
and then fails both numerical profiles. Thirty original evidence files are
unchanged after these controls.

[Closed evaluation](../artifacts/qwen35-9b-context256-2026-11-09/closed-evaluation-1.json),
[reference validation](../artifacts/qwen35-9b-context256-2026-11-09/reference-validation-1.json),
[controls](../artifacts/qwen35-9b-context256-2026-11-09/evidence-controls-1.json).

## Descriptive performance baselines

Four release cohorts collect 40 measured samples and 12 warmups on a
MacBookPro18,3: Apple M1 Pro, 16 GiB RAM, 16 GPU cores, macOS26.6.2/25G83.
Each repetition resets an empty numerical session at offset37, capacity512 and
chunk32, with resident weights and complete logits for every input. Both depth
outputs repeat exactly;128 matches the256 prefix, full256 logits match the native
witness, and raw final rows match both checkpoints.

| Inputs | Cohort | Median completed-forward time | IQR | Sample standard deviation | Input tokens/s |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 1 | 32.0677s | 1.1726s | 0.7307s | 3.9916 |
| 256 | 1 | 68.3990s | 11.6626s | 6.3220s | 3.7427 |
| 256 | 2 | 63.3298s | 0.1932s | 0.1852s | 4.0423 |
| 128 | 2 | 30.8483s | 0.1853s | 0.1588s | 4.1493 |

The first256 cohort varies widely. All raw samples are retained without trimming;
these same-code observations do not establish an optimization gain. The metric is
the sum of completed chunk forwards. Reset, comparisons, hashing and output writes
are excluded. Context, payload load, upload, binding and session setup have
separate clocks; configuration reads and remaining bookkeeping are outside those
named setup clocks. Across new processes, payload load takes21.625–23.251s,
upload9.183–9.643s, context38.898–395.712ms, bind1.230–1.319ms and
session35.318–119.587ms. New-process setup does not establish cold filesystem or
driver caches. Both probes use release optimization, strict warnings and190pinned
inputs each, with explicit native62 production-source reuse.

All eight before/after boundaries report AC power, nominal thermal state and Low
Power Mode off, with stable named hardware/OS/GPU. System free-memory observations
range63–70%; system swap use is about3.963–4.005GB and is not attributed to this
engine. Maximum child lifetime footprint is5,665,608,096B; maximum conservative
owned-group-plus-collector charge is5,675,734,296B. Planned owner accounting stays
separate. Continuous thermal stability, reliable tails, sustained throughput,
energy, request TTFT, ordinary summary overhead, quality and adoption remain
unqualified. A valid descriptive baseline cannot rehabilitate numerical rejection.
One positive and17damaged-performance-evidence controls pass;36original files are
unchanged after controls.

The first attempt rejected its environment before model startup, with no model
output folder, prelaunch inputs or sample. Its exact boundary was not persisted,
so its original cause is not asserted. Later unsandboxed observations showed
battery power/Low Power Mode on. After the user connected power, a fresh check and
all actual boundaries were eligible. The retry saves raw preflight before
admission under unchanged conditions, with12preflight/metric tests passing.
Original source, contract and failed logs remain preserved.

[Retry contract](../artifacts/qwen35-9b-context256-2026-11-09/performance-contract-2.json),
[complete summary and raw samples](../artifacts/qwen35-9b-context256-2026-11-09/performance-summary-2.json),
[performance controls](../artifacts/qwen35-9b-context256-2026-11-09/performance-controls-2.json),
[retry preparation](../artifacts/qwen35-9b-context256-2026-11-09/performance-retry-preparation-checks.json).

## Checks and retained failures

Formatting, both strict Clippy configurations and 667 CPU / 803 Metal tests pass
(46/330 ignored). The portable Qwen sweep passes 190 tests. Two owned-cleanup and
two primary-receipt controls pass. Native capture v2 saves its original guard/
cleanup receipt before parsing; the reader binds primary/final/raw data, checks
typed group and peak observations and bounds lifetime. The original layout
collector remains unchanged. Fresh optimized native build and ABI checks pass;
native62/package/Csmoke reuse is explicitly audited against unchanged production,
ABI and dependency inputs. These test/diagnostic additions imply no new package
or application qualification.

Earlier checks caught a corrected test-only greedy-ID type mismatch and the
unchanged 2 GiB disk guard. The first reference cleanup masked its primary error
with PermissionError; that original cause remains unknown. All old groups were
verified absent. Cleanup now checks actual process-group membership when macOS
denies a group probe and preserves primary errors. Independent APFS clones of
1,368 identical tracked fixtures recover 3.70 GB while preserving 948 protected
hashes. No source bytes, model, golden, raw artifact or cache file is deleted.

Keep the 6 GiB Mac ceiling, 512/32 pilot, 512 MiB codec budget and original
numerical/quality gates. Whole-GPU execution, default allocator/codec/application
fit, requests, summary overhead, quality, managed adoption and physical iPhone
remain separate gates. The full improvement goal stays active.

[Acceptance contract](../artifacts/qwen35-9b-context256-2026-11-09/acceptance-contract.json),
[required checks](../artifacts/qwen35-9b-context256-2026-11-09/required-checks-3/checks.json),
[Python tests](../artifacts/qwen35-9b-context256-2026-11-09/python-qwen35-all-4.log),
[primary receipt controls](../artifacts/qwen35-9b-context256-2026-11-09/native-primary-receipt-checks.json),
[native build](../artifacts/qwen35-9b-context256-2026-11-09/native-build-2/checks.json),
[compiled profile](../artifacts/qwen35-9b-context256-2026-11-09/compiled-profile-contract.json),
[ABI](../artifacts/qwen35-9b-context256-2026-11-09/abi-contract.json),
[package reuse](../artifacts/qwen35-9b-context256-2026-11-09/production-reuse-audit.json),
[probe builds](../artifacts/qwen35-9b-context256-2026-11-09/probe-build-2/checks.json),
[failed-run recovery](../artifacts/qwen35-9b-context256-2026-11-09/failed-attempt-recovery.json),
[fixture recovery](../artifacts/qwen35-9b-context256-2026-11-09/fixture-space-recovered.json),
[retry source freeze](../artifacts/qwen35-9b-context256-2026-11-09/reference-retry-1/adapter-freeze.json).

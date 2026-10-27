# Qwen3 Mac archived output screen — 27 October 2026

**The screen completed; the candidate is rejected under this versioned contract.**
Worst normalized-log-probability error is **0.2022337512**, above the frozen
inclusive **0.05** limit. All 1,024 greedy IDs agree. The eight untouched numerical
validation inputs remain unopened. No precision implementation or app limit changes.

## Fixed scope and results

The [contract](../tests/fixtures/qwen3-output-screen-v1/contract.json) was frozen
and reviewed by Astra before comparison. It selects the existing original and
future-tail 512-token cases, native chunk128 captures, vocabulary 151936 and the
matching independent FP64-arithmetic/Half-storage CPU reference. These two cases
share their first 511 input tokens; their failures are overlapping evidence.
The other native schedules already have byte-identical complete-state evidence
in the [historical trial](qwen3-context512-native.md).

Every saved FP32 logit is widened exactly to FP64. Per row and profile, subtract
the raw maximum, apply NumPy FP64 exp, accumulate every entry with `math.fsum`,
take `math.log`, and subtract that normalizer. Compare all log probabilities
without rounding or clipping; also require exact lowest-ID raw-logit argmax.
Nonfinite values reject. This is a new output-fidelity screen; it leaves the
historical canonical FP32 stage rejection unchanged.

| Measure | Original | Future tail |
| --- | ---: | ---: |
| Compared positions |512|512|
| Failed positions |11|11|
| Vocabulary-entry errors above 0.05 |56034|56034|
| Greedy disagreements |0|0|
| Maximum log-probability error |0.20223375120124842|0.20223375120124842|

There are 155,582,464 entry comparisons in total. Both cases fail at positions
29, 33, 41, 45, 49, 65, 136, 145, 217, 305 and 384. The worst position is 41,
vocabulary ID 93515: native log probability −21.134241072765853 versus reference
−21.3364748239671. Both choose greedy ID 1879. Small probability mass or matching
tokens do not waive a uniform error bound.

## Execution and checks

No model graph, new calibration or numerical validation ran. The reader consumes
saved logits only. Raw parent files, frame extents/hashes, complete reference
logit arrays and contiguous row coverage all authenticate. Its 1,024-row journal
is retained before aggregate classification, SHA
`1c8ecda6750e1f76e2dcf7c66307ea32f64654b6445d0ce242cf1511b2df9ad4`.

The final pre-measurement contract SHA is
`0df7e712e7d397d8f334a194a187c0c26938656ab86867fe3a1a3ce624e21721`.
It distinguishes the historical native binary/source/model identity from current
HEAD. Only two production source files differ from the historical capture;
those identities are recorded, without claiming current engine qualification.
The diagnostic1025 loader and app are outside this screen's scope.

Thirty cold Python checks pass: analytic distributions, constant shifts, tails,
ties, nonfinite/extreme rows, typed metadata, interruption reporting, exact empty
runtime markers and the reused frame/array owner controls. Actual origins for
90 numerical/helper modules and 18 non-system native images are checked before
and after comparison. All 1,996 parent/runtime audits pass their endpoints.
Both finite process groups are absent. Signal interruption retains partial
structured evidence and restores handlers. Rust numerical code is unchanged;
the preceding full CPU, formatting and default/Metal Clippy checks carry forward.

On the named Apple M1 Pro Mac, the cold reader's peak physical memory is
47,104,672 bytes and maximum RSS is 93,995,008 bytes, below its 512 MiB limit.
These are reader resources, not model performance measurements.

The initial sandbox unit attempt passed its tests but could not obtain macOS
memory counters. A later admission attempt rejected legitimate empty installed
package markers before any row comparison. Both failures remain archived. The
localized marker fix preserves nonempty numerical capture admission and was
reviewed before the successful run. No numerical result or criterion was changed.

## Decision and next work

Astra verified the outcome and independently reconstructed journal totals.
Reject this contract/candidate pairing; keep numerical qualification open and
the eight validation inputs untouched. There is no supported erroneous-operator
hypothesis, so no dtype switch, blanket FP32 conversion or layer walk follows.
The [full relative-quality pass](qwen3-full-quality.md) remains separate, with
weak absolute scores and no task-margin claim.

Proceed with one bounded functional audit of the existing experimental C512 Mac
Chat/Expert and pinned Hugging Face/offline journey. Preserve limits and truthful
experimental/partial labels. Functional acceptance cannot close qualified
Q1–Q5 delivery. See the [queue](qwen3-mac-delivery-plan.md) and
[public receipt](../benchmarks/results/2026-10-27-qwen3-output-screen-v1.json).

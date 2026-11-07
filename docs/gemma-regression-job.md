# R1: Gemma offline regression screening

7 November 2026. Delivered in the scoped offline Gemma Chat screening task.
Nine new tests and the exact named CI commands (14 offline +3 contracts) pass
locally; Astra approves after duplicate-cohort, missing-suite and malformed-input
receipt fixes. Hosted GitHub execution is not claimed.
[Checks/review/commit](../artifacts/r1-gemma-regression-2026-11-07/checks.json).

Run `tools/bench/gemma_regression.py` over two independently collected
[O2.A measurement bundles](o2-gemma-cost-report.md). It validates both bundles
before comparing **uninstrumented caller request time**, with callback TTFT as a
guard. Native phases are used for attribution, not acceptance timing. The job
does not download models or run another GPU process.

```sh
python3 tools/bench/gemma_regression.py BASELINE_BUNDLE CANDIDATE_BUNDLE NEW_OUTPUT \
  --baseline-commit BASELINE_40_HEX_REFERENCE \
  --candidate-commit CANDIDATE_40_HEX_REFERENCE
```

Collect each input with the same pinned
`tools/experiments/o1_gemma_summary.py MODEL MAC_ARCHIVE NEW_BUNDLE` driver.
Its two batches each contain three warmups and ten measured disabled/summary
pairs. Preserve the native package/check/quality receipts alongside the bundles.
The supplied Git references locate the accepted baseline and candidate work;
exact measured source/build/archive hashes remain authoritative when timing
precedes the final reviewed commit. A reference is not inferred provenance.

The comparator requires the exact official Gemma BF16/G2 model/codec identity,
512-position/chunk-32/12-output Chat workload, eight frozen emitted IDs and
24/23/1 history/committed/pending counts, matching hardware/OS/AC power and
unchanged measurement drivers. It reuses the strict cost-report checks for
request/phase/callback clock enclosure, owner/counter consistency, complete
summary coverage and matched per-mode work. Identical bundles cannot confirm
themselves. Incomparable inputs produce a failure receipt, not a speed ranking.

Each batch reports both raw ranges and medians, change percentage and a
deterministic independent-sample bootstrap interval. Requests from distinct
process runs are not treated as pairs. The existing 5% screening threshold
applies; a positive change is slower.

| Exit | Result | Meaning |
| --- | --- | --- |
| 0 | pass | No screened regression; this alone does not accept an optimization |
| 1 | incomparable | Invalid, incomplete, mismatched or self-compared inputs |
| 2 | regression | At least one required latency has >=5% median regression and its interval excludes zero in either batch |
| 3 | noisy | A >=5% point change is unresolved, a broad interval crosses both thresholds, or the primary improvement is not confirmed in both batches |

A repeated >=5% decrease in total latency with intervals below zero is only a
`repeatable_primary_speed_signal`. The output always says
`accepted_optimization=false`: independent numerical/task quality, other admitted
workloads, memory, UI behavior and source review are still required. No p95/p99,
thermal-equivalence or 1% summary-overhead qualification follows from ten samples
or from the legacy bundle's limited environmental observations.

The GitHub `observability-reports` job checks three named portable suites under
optimized Python, including rejection fixtures. Named imports fail when a suite
is absent. The separately reviewed O2.B1 suite can be added after its commit. Model/GPU suites remain explicit
local hardware jobs. New reports use fresh output directories and retain links to
both bundles and the caller-supplied revision references.

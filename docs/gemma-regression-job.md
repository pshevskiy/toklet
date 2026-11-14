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
512-position/chunk-32/maximum-12-output Chat workload, eight frozen emitted IDs and
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

The GitHub `observability-reports` job uses named portable suites under
optimized Python, including rejection fixtures. Named imports fail when a suite
is absent. The separately reviewed [O2.B1 suite](o2-gemma-workloads.md) adds
six workload tests to the original three R1 suites, for 23 local CI checks. Model/GPU suites remain explicit
local hardware jobs. New reports use fresh output directories and retain links to
both bundles and the caller-supplied revision references.

## Ordinary CLI extension

`tools/bench/gemma_cli_regression.py` supports the official BF16/G2 route at
384 input tokens, 128 output tokens, capacity 512 and prefill chunk 32, with
two independent cohorts and two fresh disabled-mode
owners per cohort. Each owner supplies three warmups and five measured requests.
G17's A/B labels mean instrumentation disabled/summary. Engine revision roles
are declared separately; the adapter selects disabled mode for both revisions.

```sh
python3 -B tools/bench/gemma_cli_regression.py BASELINE_INPUT CANDIDATE_INPUT NEW_OUTPUT \
  --policy per-tile
```

Each input JSON contains `capture_pin`, `revision_proof` and `observer_proof`.
Pins record an absolute path, byte length and SHA-256. Select `per-tile` or
`batched-single-row` explicitly and keep the policy equal between revisions.
The tool reads bounded local evidence and hashes the recorded engine binary;
it does not execute the engine or read model assets. It writes `comparison.json`
and `comparison.md`, or `failure.json` for incomparable inputs, into a new
directory. Existing output directories are refused.

The accepted baseline binds actual published source, build, model, tokenizer,
raw capture and CI evidence. A candidate may be an uncommitted frozen snapshot:
its actual base commit, diff, changed-content archive, source hashes, binary and
checks identify the measured variant. Its commit stays null until publication.
Late publication binding produces a new receipt after matching the measured
content; earlier reports and samples stay unchanged. Baseline admission alone
produces no comparative pass or speed signal.

```sh
python3 -B tools/bench/gemma_cli_regression.py --bind-publication \
  COMPARISON_JSON ACTUAL_PUBLISHED_REVISION_PROOF_JSON NEW_RECEIPT_JSON
```

Snapshot proof reconstructs the source freeze from a pinned base inventory,
ordinary UTF-8 unified diff and changed-file content archive. This first version
rejects binary patches and renames. Source archives are bounded to 32 MiB,
metadata to 2 MiB per file, raw receipts to 8 MiB and binaries to 64 MiB.
Published proof requires an actual commit, exact-head CI and a source-content
mapping. A reviewed cold admission/provenance change in the capture controller
is allowed only when the retained timed syntax and common observer match.

The new format preserves each complete five-request owner block. For every
revision and cohort, independent bootstrap draws select whole owner blocks,
then compute the pooled ten-request median. It retains the existing 5% screening
and confirmation policy and the same four exits above. It never pairs requests
across independent runs or pools confirmation cohorts. The legacy Chat bootstrap
keeps its original independent-request sampling unit.

Both formats require exact work and observation boundaries, compatible recorded
identities, complete clean owners and independent observations. Relabelled or
overlapping clock/work fingerprints cannot confirm themselves. Named hardware
configuration does not establish unique physical-host identity. The output
continues to set `accepted_optimization=false`; the separate 1% observer gate,
numerical/task quality, memory and other acceptance requirements keep their scope.

Twenty-eight independent controls pass in normal and optimized Python modes,
including a correlated-owner case with a -10% point change and a [-55%, +80%]
interval that must remain noisy. The nine legacy controls also pass in both
modes. Both real G17 policies admit two cohorts and four disabled owners each,
with model reads and child launches blocked. This validates baseline admission;
no real candidate comparison or speedup is claimed. Eight CLI controls pass for
all four exits, publication binding and overwrite refusal. Required engine checks
are reused against unchanged numerical/native/build inputs; the verification
helper's initial inventory error is retained with its corrected passing result.
Astra requested owner-local power freshness, malformed-input receipts and
bounded initial metadata reads. Six independent methods demonstrated the old
gaps, then passed after the fixes in both modes. Astra approves scoped retention
with no remaining findings. Code is published as `ae984fda`; the
[live publication receipt](../artifacts/gemma-cli-regression-2026-11-15/publication-status.json)
tracks the final head and exact-head CI result.
[Astra approval](../artifacts/gemma-cli-regression-2026-11-15/astra-review-1/verdict-2.json).
[Portable checks](../artifacts/gemma-cli-regression-2026-11-15/affected-checks-2.json),
[CLI controls](../artifacts/gemma-cli-regression-2026-11-15/cli-controls-2/result.json),
[required-check reuse](../artifacts/gemma-cli-regression-2026-11-15/required-reuse-2.json),
[real admission](../artifacts/gemma-cli-regression-2026-11-15/real-baseline-admission-2.json).

Named `cli-regression` CI jobs run both new and legacy portable controls with
`PYTHONOPTIMIZE=0` and `1`. The input/proof API and independent test controls are
frozen in the [implementation contract](../artifacts/gemma-cli-regression-2026-11-15/implementation-contract-1.json)
and [API schema](../artifacts/gemma-cli-regression-2026-11-15/api-schema-1.json).

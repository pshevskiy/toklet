# Gemma workload identity and untimed preflight

Both original 1% summary-overhead criteria pass for callback TTFT and complete
caller time in two headless **384-input/128-output** cohorts on official Gemma
4 E2B BF16/G2, capacity 512 and chunk 32. Independent raw replay reproduces every
statistic exactly. Astra approves scoped code retention and this qualification
with no findings. Production summaries remain
disabled by default and opt-in. The earlier
[short France workload result](gemma-summary-overhead.md) retains its own scope.
This does not qualify a numerical 512-token prompt, UI/render, p99 or default
adoption.

## Descriptor and interface

The existing runner adds `--workload PATH` and `--preflight-only`. Omitting them
preserves the France prompt, 15 encoded input tokens, reply cap 12, eight emitted
IDs, three warmup pairs and the existing conformance path. Example interfaces,
to be invoked only through the admitted serial guard:

```text
python3 -B tools/experiments/o1_gemma_summary.py MODEL ARCHIVE PREFLIGHT_OUT \
  --workload INPUT_MANIFEST --preflight-only
python3 -B tools/experiments/o1_gemma_summary.py MODEL ARCHIVE TIMING_OUT \
  --workload PREFLIGHT_OUT/frozen-workload.json --measured-pairs 10
```

The version-1 manifest has exactly eleven fields: `schema_version`,
`workload_id`, `conversation_json`, `conversation_sha256`, `input_token_count`,
`input_tokens_sha256`, `maximum_output_tokens`, `capacity`, `chunk`,
`cache_policy` and `expected_output`. It is bounded to 16 KiB; the exact UTF-8
conversation value is bounded to 8 KiB and is passed to Chat without reserialization.
Python rejects duplicate/unknown fields, nonfinite numbers, booleans in integer
fields, wrong types, altered hashes and requests exceeding the shared window.
Its private `_sha256` binds the raw manifest and is never serialized as a wire field.
Input count and hash must already be known even for preflight.

`expected_output` may be null only for preflight. Timing requires exact frozen
IDs, terminal reason/stop and history/committed/pending counts. The checker
validates reset operation 3, base history zero, owner identities, native work,
request-ID joins, nested clocks and complete bounded exports. Hidden stop
sampling is distinguished from emitted callbacks and the requested cap.
[Contract](../artifacts/gemma-workload-identity-2026-11-13/implementation-contract-1.json).

Swift parses the bounded manifest before model creation. After the existing
end timestamp, it verifies admission, context/configuration, the saved prefix
and SHA256 of the authoritative input IDs serialized as little-endian u32 words.
The untimed disabled/summary pair must agree. Python validates the enabled raw
native summary and writes `frozen-workload.json`; it does not infer expectations
from timing results. Existing native/production numerical code, ABI and ring are
unchanged. Larger captures still harvest raw summaries after each complete pair,
outside both timed intervals.

## Actual boundary proof

The public fixture asks for a Rust code review. Preparation used the existing
audited official model/Chat codec owner under the 6 GiB engine budget, selecting
the first complete 384-token rendering from the prospectively fixed ordered
0..512 note-word prefixes. The selected prefix contains 253 note words. No
output or latency influenced selection, and no encoded IDs were truncated.
[Preparation policy](../artifacts/gemma-workload-identity-2026-11-13/input-preparation-prospective-1.json),
[selection attempts](../artifacts/gemma-workload-identity-2026-11-13/input-workload-1.json.attempts.json),
[guarded preparation](../artifacts/gemma-workload-identity-2026-11-13/input-prepare-1/result.json).

The [input manifest](../artifacts/gemma-workload-identity-2026-11-13/input-workload-1.json)
binds 384 input IDs with u32LE SHA256
`4ef45f4b84d488ee83a5e3c3ba701b13e1afc25e676ce076d55bd97b42a0f566`.
The requested reply cap is 128, capacity 512 and chunk 32. Both untimed modes
actually emit all 128 IDs, reach output limit and agree at state **512/511/1**.
The preflight has two requests, zero measured requests and
`overhead_qualified:false`.
[Preflight and frozen expectations](../artifacts/gemma-workload-identity-2026-11-13/workload-preflight-1/result.json),
[raw pair](../artifacts/gemma-workload-identity-2026-11-13/workload-preflight-1/preflight.json),
[native export](../artifacts/gemma-workload-identity-2026-11-13/workload-preflight-1/preflight.json.summaries.jsonl),
[guard/source endpoints](../artifacts/gemma-workload-identity-2026-11-13/workload-preflight-guard-1/result.json).
This is the actual 384+128 shared boundary, not a 512-token prompt or the separate
numerical 512-input matrix. It makes no task-quality claim.

## Completed measurement

Campaign 3 uses three warmup pairs and ten measured pairs in each of two fresh
processes, alternating disabled/summary order: **52 requests, 40 measured and
26 native summaries**. Every request matches the frozen 384-token input and 128
output IDs, reaches output limit and retains state 512/511/1. Neither native
ring drops a record. Both the median ratio and paired-median bootstrap upper
bound must be <=1% for each metric in each cohort.

| Metric | Cohort 1 median overhead | Cohort 2 median overhead | Cohort 1 paired-median 95% upper | Cohort 2 paired-median 95% upper |
| --- | ---: | ---: | ---: | ---: |
| Callback TTFT | -0.02834665% | 0.11528843% | 0.41935036% | 0.79466248% |
| Complete caller request | 0.00279417% | 0.17926247% | 0.82056575% | 0.72482787% |

Bootstrap seed 1 and 10,000 resamples are unchanged. The small negative point
estimate is not a speedup claim; ten measured pairs do not qualify tails.
[Completed result](../artifacts/gemma-workload-identity-2026-11-13/workload-campaign-3/result.json),
[cohort 1 raw samples](../artifacts/gemma-workload-identity-2026-11-13/workload-campaign-3/samples-1.json),
[cohort 2 raw samples](../artifacts/gemma-workload-identity-2026-11-13/workload-campaign-3/samples-2.json).

The resumed launch passes ten stable AC/Low-Power-off/thermal-zero checks.
Cohort processes finish in 1,151.0411 and 1,119.2658 seconds; those process walls
are not request clocks or a separate sustained qualification. Maximum observed
owned-tree charges including helper are 4,843,475,672 and 4,843,328,216 bytes,
below 6 GiB. All owned processes are cleaned up and all 489 recorded input pins
match before/after. Setup stays within 512 MiB including helper, compiler within
3 GiB, with a 2 GiB disk floor. Sampled conservative process observations are not
a continuous allocation cap.
[Runtime/resource/endpoints](../artifacts/gemma-workload-identity-2026-11-13/workload-campaign-guard-3/result.json),
[stable launch and closure](../artifacts/gemma-workload-identity-2026-11-13/timed-campaign-closure-3.json).

## Checks and retained failures

Strict Swift 6 compilation and fresh default outcome conformance pass.
The initial portable run passed 52 controls and failed on two incomplete
synthetic native-summary fixtures. Those fixtures were repaired, a regression
was added, and validation was not weakened. A subsequent writer-boundary fix
checks the derived frozen manifest before exclusive file creation. All 57
controls pass in normal and optimized Python modes.
[Strict build](../artifacts/gemma-workload-identity-2026-11-13/harness-build-1.json),
[retained fixture failure](../artifacts/gemma-workload-identity-2026-11-13/portable-normal-1/stderr.log),
[normal checks](../artifacts/gemma-workload-identity-2026-11-13/portable-normal-3/result.json),
[optimized checks](../artifacts/gemma-workload-identity-2026-11-13/portable-optimized-2/result.json).
Fresh package fmt, both strict Clippy configurations and 786 CPU/946 Metal tests
pass (53/351 ignored). Four saved default cohorts replay exactly, covering 464
requests/232 native rows; this does not qualify a new workload or build. Genuine
preflight data replays through unchanged validators and the corrected writer;
its original source/build identities are preserved.
[Current checks](../artifacts/gemma-workload-identity-2026-11-13/checks-current-2.json),
[default replay](../artifacts/gemma-workload-identity-2026-11-13/default-compatibility-replay-1.json),
[writer fix and phase reuse](../artifacts/gemma-workload-identity-2026-11-13/writer-fix-revalidation-2.json).
The repair changes only the cold bounded canonical/exclusive Python writer;
its exact inverse restores the previous module. The original preflight keeps
its source/build identity; campaign 3 actually executes the current caller
build and [revalidated frozen descriptor](../artifacts/gemma-workload-identity-2026-11-13/frozen-workload-revalidated-2.json).
The check receipt predates campaign 3, whose completed result supersedes only
its earlier unavailable-timing status.
The native archive's
[lineage](../artifacts/lfm3-tokens2-2026-11-13/gemma-current-staticlib-1.json) and
[preservation receipt](../artifacts/lfm3-tokens2-2026-11-13/gemma-qualified-staticlib-preservation-1.json)
are retained; they do not replace changed-code checks.

Attempt 1 remains **failed**: the power guard stopped its first cohort after
339.62 seconds, with no complete timing cohort and clean owned cleanup. Its last
saved successful gate cannot identify which later condition failed. Attempt 2
also remains **failed**: Battery Power/Low Power Mode admission failed, and a
launcher dependency bug still let the outer recovery path proceed. The inner
guard rejected before `Popen`; no model child started. Neither attempt's timings
are reused.
[Attempt 1](../artifacts/gemma-workload-identity-2026-11-13/workload-campaign-guard-1/overhead-1/result.json),
[first closure](../artifacts/gemma-workload-identity-2026-11-13/timed-campaign-closure-1.json),
[attempt 2 conditions](../artifacts/gemma-workload-identity-2026-11-13/recovery-conditions-2.json),
[pre-launch refusal](../artifacts/gemma-workload-identity-2026-11-13/workload-campaign-guard-2/conformance/result.json),
[second closure](../artifacts/gemma-workload-identity-2026-11-13/timed-campaign-closure-2.json).
Nine offline dependency controls cover failed/throwing prerequisites preventing
child launch; the integrated corrected entry is the one used by campaign 3.
[Dependency controls](../artifacts/gemma-workload-identity-2026-11-13/launch-dependency-checks-3.json),
[integration check](../artifacts/gemma-workload-identity-2026-11-13/launch-integration-checks-3.json).

The two original statistical criteria pass for this workload; the independent
audit verifies all raw work, 489 source endpoints and original statistics. Power
checks are enforced by polling; only the last successful phase endpoint is
retained, so these records do not form a complete continuous power timeseries.
[Final audit and verifier](../artifacts/gemma-workload-identity-2026-11-13/independent-workload-audit-3.json),
[final check closure](../artifacts/gemma-workload-identity-2026-11-13/checks-final-3.json).
[Astra approval](../artifacts/gemma-workload-identity-2026-11-13/Astra-review-2.json)
resolves the original missing-cohort hold. The
[publication receipt](../artifacts/gemma-workload-identity-2026-11-13/publication-checks-3.json)
records commit, push and exact-head CI closure. This is solely headless managed
official BF16/G2 Chat at the actual
384+128 boundary in a 512-token shared window: no numerical 512-input, UI/render,
p99, other-model/device or default-adoption qualification follows.

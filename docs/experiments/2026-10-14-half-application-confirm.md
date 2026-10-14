# Independent compensated Half application repeats

Both selected models repeat the unchanged full scripted three-turn application trace
with three warmups and ten measured scenarios. All39turn response signatures per
model match the initial collection exactly: visible token IDs/text, stop reasons,
reused/prefill/sample/emitted/history/evaluated/pending counts. This is independent
repeatability of the same practical configuration, not an alternating CPU/external
experiment or an identical-output comparison with the historical FP32 trace.

| Checkpoint | Initial / confirming scenario median, s | Median change |
| --- | ---: | ---: |
| Small |20.445239 /20.456533|+0.055240%|
| Larger |43.550813 /43.572448|+0.049678%|

These changes are descriptive run-to-run variation, not new optimization gains.
The analysis retains all samples, ranges, MAD and per-turn token/text/finalization
boundaries. Do not infer tail percentiles or confidence intervals from ten requests.
The same timing scopes and nullable/phase-bounded separate-text limitations in the
[first full trace](2026-10-14-half-application-full.md) remain. Driver scenario
includes validation/memory instrumentation; native components overlap and are not
summed. Each turn starts empty, ends reset, and uses scripted assistant history.

All78turns/60measured turns and3,589/3,576raw events replay with the exact accepted
[application audit](../../benchmarks/results/2026-10-14-half-application-full-analysis-v2.py.txt).
No engine, qualified collector, audit, source model, precision, prompt IDs, output
cap128, greedy/EOS rule or default changes. Both workers naturally exit0 with empty
cleanup errors/complete transport and independent finalized text. Both collectors
exit0; source/binary/input endpoints match without errors. Current observations prove
recorded modelPIDs13885/15339 absent, their owned groups empty, application commands
absent and machine stopped/not-starting. Historical collector IDs were not retained.

Complete admissions remain1,443,266,720/3,577,266,480bytes. Maximum sampled joint
worker-plus-driver native lifetime charges are1,332,742,160/3,464,810,024bytes,
below6GiB. Persistent inventories are unchanged; native counters are charges, not
complete shared physical allocation totals. Oracle tokenization precedes model
launch and independent final decode follows owned model exit. No full FP32 model
copy is retained. Source hashing warms storage; setup is not cold-storage evidence.

Measured scenario totals remain below600seconds, so these runs do not add separate
ten-minute application qualification. Start power shows AC/low-power modezero;
start/end thermal observations record no warning. End power, continuous sensors,
energy and full system pressure are unavailable. No builds/tests/other owned
numerical/VM work or large hashes overlap timed requests.

Both launches use parent`dee229f` and a clean tree observed at launch; the adapter
itself has no dirty field. The immutable qualified CLI/source remains separately
bound. Device is Apple M1 Pro/MacBookPro18,3/16GiB/macOS26.6.2(25G83)/Rust1.90.0.
Exact model/tokenizer/transcript/precision/cache/compiler/build identities remain.

[Small confirmation](../../benchmarks/results/2026-10-14-small-half-application-confirm-v1.json),
[small raw](../../benchmarks/results/2026-10-14-small-half-application-confirm-v1.raw.jsonl),
[larger confirmation](../../benchmarks/results/2026-10-14-larger-half-application-confirm-v1.json),
[larger raw](../../benchmarks/results/2026-10-14-larger-half-application-confirm-v1.raw.jsonl),
[full replay](../../benchmarks/results/2026-10-14-half-application-confirm-analysis-v1.json),
[comparison](../../benchmarks/results/2026-10-14-half-application-confirm-comparison-v1.json)
and [qualification receipt](../../benchmarks/results/2026-10-14-half-application-confirm-qualification-v1.json)
bind exact copies, repeated signatures, arithmetic, memory and idle observations.
Content/link/whitespace and unchanged protected CPU binaries pass. No new engine
suite is required for this unchanged-source data task. Astra accepts after all98
bindings, byte-identical audit replay, repeated signatures, arithmetic and saved
idle observations pass, with no findings.

The small configuration still emits128/14/128versus historicalFP32128/18/128;
the larger standard FP32 application remains overbudget. These repeats do not
remove those comparison limits. Frozen Half full loss/task gates separately pass
both models; both Q4 full-loss failures still block the explicit M5 exit. Final
comparison/release decision remains; no default or external speed adoption here.

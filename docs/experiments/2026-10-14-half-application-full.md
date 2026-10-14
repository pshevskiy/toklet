# Full compensated Half application traces

Both fixed checkpoints complete the frozen authored three-turn trace with three
warmups and ten measured scenarios. Each turn starts with an empty session and
ends with reset; model/session buffers remain resident. Scripted assistant replies
and tool-result text are used in subsequent prompts. This is application latency
and state evidence, not live tool execution or task scoring. Actual mode is
HalfCompensated / Original normalization / FullLogits, capacity2,048/chunk128.

| Checkpoint | Scenario median, s | Measured range, s | Scenario MAD, s | Measured total, s |
| --- | ---: | ---: | ---: | ---: |
| small | 20.445239 | 20.441034–20.457573 | 0.002533 | 204.468590 |
| larger | 43.550813 | 43.499204–43.559388 | 0.004455 | 435.410454 |

| Checkpoint / turn | Prompt tokens | Emitted tokens | Stop | Median first token / visible text, s | Median finalized response, s |
| --- | ---: | ---: | --- | ---: | ---: |
| small / 1 | 36 | 128 | output_limit | 0.148793 / 0.148793 | 7.679671 |
| small / 2 | 76 | 14 | stop_token | 0.292206 / 0.292206 | 1.041563 |
| small / 3 | 117 | 128 | output_limit | 0.443052 / 0.443052 | 11.722983 |
| larger / 1 | 36 | 128 | output_limit | 0.598541 / 0.598541 | 17.457782 |
| larger / 2 | 76 | 128 | output_limit | 1.221819 / 1.221819 | 22.250734 |
| larger / 3 | 117 | 13 | stop_token | 1.861743 / 1.861743 | 3.844846 |

All measured responses within each configuration reproduce the same token IDs,
text, stop reason and state. The small Half second turn emits14tokens; the
historical [qualified FP32 tiled trace](2026-10-12-application-tiled.md) emits18.
Its scenario median129.141379s is context for a practical configuration comparison;
these outputs are different, so no identical-output or pure backend speed gain
is claimed. The larger standard FP32 CLI cannot admit simultaneous weights,
tokenizer/application state and these session buffers under6GiB. The bounded raw
task reference is quality evidence and is not an application latency baseline.
The two checkpoint results are different-model workloads, not a speed ranking.

Driver timings run from submission through receipt of token/nonempty text/done/
reset JSON events. Finalization includes trailing Unicode text and reset. Per-token
receipt timestamps are retained; the first visible token timestamp reconstructs
exactly. A separate final-text event retains only its saved boundary and phase
bounds, not an independent full event receipt vector. No complete per-event receipt
timing claim is made. Native numerical/sampling/formatting/tokenization/text/
publication components overlap totals and are not added. Scenarios include driver
validation and memory instrumentation between turns. Summed turn medians are not
the scenario median. Setup and independent oracle detokenization are untimed.

There are78total turns including warmups,60measured turns and30measured turns per
model. Small/larger raw streams retain3,589/3,576events. Both workers naturally
exit0 with empty cleanup errors, finish owned transports, and independent final
tokenizer decoding agrees. Endpoint inventories match with no errors. Current
observations prove recorded model PIDs96640/97850 absent and both owned groups
empty; application commands are absent and the machine stopped/not-starting.
Historical collector IDs were not retained and no individual collector PID
reconstruction is claimed.

| Checkpoint | Complete admission, bytes | Maximum sampled worker-plus-driver lifetime charge, bytes |
| --- | ---: | ---: |
| small | 1,443,266,720 | 1,331,447,848 |
| larger | 3,577,266,480 | 3,467,005,408 |

Complete weights/KV/scratch/history/logits/metadata/framework/tokenizer/application
inventories fit6GiB and remain fixed. Native charges are not complete shared
physical allocation totals. Oracle tokenizer preparation precedes model launch;
independent final decoding follows owned model exit. No duplicate FP32 model is
retained. Source hashing warms storage; startup is not cold-storage qualification.

Both measured scenario totals are below600seconds; these traces do not add
independent ten-minute application qualification. The separately accepted
[small](2026-10-14-small-half-controlled-full.md) and
[larger](2026-10-14-larger-half-controlled-full.md) controlled sequences carry their
named sustained-duration scopes. Start power shows AC attached and AC low-power
mode zero. Start/end thermal queries record no warning. End power, continuous
temperature/clocks, system pressure and energy are unavailable. No builds/tests,
other owned numerical work/VM runs or large hashes overlap native requests.

Dispatch parent is `9a6e7a4`; actual launch checks show a clean tree for both
collections, though this adapter does not store a working-tree-dirty field.
Immutable CLI SHA-256 remains
`82c2e36b573d9fe62883c3d46e1c546acd5a96e87af44d3a9ffbabd2ae00b4ee`.
Compiled snapshot/source identity is separately qualified. Hardware is Apple M1
Pro / MacBookPro18,3 /16GiB/macOS26.6.2(25G83)/Rust1.90.0. Fixed model/tokenizer/
transcript/compiler/build/precision/cache identities are retained. No engine or
qualified collector change occurs in this data task.

The exact audit initially lacked complete timestamp validation and nullable
summaries. Astra identified those gaps; the corrected source verifies finite
nonnegative timestamps, required-null semantics, exact token receipt mapping,
phase-bounded separate text and observed-only nullable summaries. Fifteen invalid
timing variants reject; valid immediateEOS/empty output, mixed/all-null summaries
and pre-done/reset-tail visibility fixtures pass. Original preliminary analysisv1
is retained and excluded; native data is unchanged. Corrected full replay and
content/link/whitespace checks pass. Astra accepts the full evidence after exact
published replay, all104bindings, turn/scenario arithmetic, repeated response
signatures and saved idle observations, with no findings.

[Small collection](../../benchmarks/results/2026-10-14-small-half-application-full-v1.json),
[small raw](../../benchmarks/results/2026-10-14-small-half-application-full-v1.raw.jsonl),
[larger collection](../../benchmarks/results/2026-10-14-larger-half-application-full-v1.json),
[larger raw](../../benchmarks/results/2026-10-14-larger-half-application-full-v1.raw.jsonl),
[corrected analysis](../../benchmarks/results/2026-10-14-half-application-full-analysis-v2.json),
[exact audit](../../benchmarks/results/2026-10-14-half-application-full-analysis-v2.py.txt)
and [qualification receipt](../../benchmarks/results/2026-10-14-half-application-full-qualification-v1.json)
bind completed evidence and byte-identical published copies.

Both selected Half configurations separately pass unchanged frozen full loss/task
quality. This completes first full application collection only; confirming
comparisons, release decisions and the unchanged Q4-blocked M5 exit remain open.
CPU/scalar remains default. No quality gate, model, quantizer or corpus changes.

Sequential native reproduction requires fresh output paths:

```sh
.venv/bin/python tools/experiments/metal_application.py --model small --output artifacts/half-small-application-rerun.json
.venv/bin/python tools/experiments/metal_application.py --model larger --output artifacts/half-larger-application-rerun.json
.venv/bin/python benchmarks/results/2026-10-14-half-application-full-analysis-v2.py.txt artifacts/half-small-application-rerun.json artifacts/half-larger-application-rerun.json > artifacts/half-application-rerun-analysis.json
```

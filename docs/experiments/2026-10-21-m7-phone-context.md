# M7 exploratory iPhone context matrix

Both selected models complete repeated raw-ID workloads on the wired iPhone14Pro
(iPhone15,2, iOS27.0.1). All156 requests,19,968 visible token events, exact prefix/
pending state, same-workload output equality and whole-process trial budgets
pass saved-data replay. This is descriptive heated-device evidence, not full
quality, supported limits or an overall M7 release.

The source revision is b2dee4a with signed prepared-v11 and native package-v4.
The small Qwen2.5-0.5B uses compensated Half/Original normalization; the larger
Qwen2.5-1.5B uses frozen group8 Q4v2 compensated tiled execution/Compensated norm.
Both retain Online attention, FP16 KV and FullLogits. Model/checkpoint/packed
identities remain the frozen M6 closure. No weights, kernels, numerical modes,
defaults or gates change for this matrix. Swift receives only IDs/timings/statistics. The inherited native FullLogits GPU
readback remains unchanged.

Six rows combine small/larger with frozen128/512/1024-token prompts. Each row
runs twice from the same admitted transfer receipt, with three warmups and ten
measured requests per run. Requests explicitly reset, then emit128 raw IDs with
no hidden-stop set. Capacity1152 and prefill chunk128 stay fixed. Every completed
state has history prompt+128, evaluated prompt+127 and one pending sample. Each
row retains all128 output IDs and requires exact equality within and across both
runs. These raw prompts do not establish chat formatting, EOS/task quality or
arbitrary prompt equivalence.

The table reports medians over ten measured completed requests in each run.
GB denotes decimal bytes/1e9. Native TTFT includes prefix preparation/prefill and
first sampling; callback TTFT additionally includes Swift scheduling and is saved
separately. Decode is127/(last-native-event minus first-native-event); completed
request time brackets submit through returned completion. No isolated prefill
clock or tail percentile is inferred from this sequence.

| Model | Prompt | Run | Native TTFT, s | Decode, IDs/s | Completed request, s | Peak, GB |
|---|---:|---:|---:|---:|---:|---:|
| small | 128 | 1 | 1.356 | 21.378 | 7.296 | 1.114 |
| small | 128 | 2 | 1.512 | 20.739 | 7.637 | 1.114 |
| small | 512 | 1 | 10.846 | 15.638 | 18.962 | 1.114 |
| small | 512 | 2 | 15.058 | 12.756 | 25.017 | 1.114 |
| small | 1024 | 1 | 39.929 | 8.844 | 54.288 | 1.114 |
| small | 1024 | 2 | 39.880 | 8.835 | 54.255 | 1.114 |
| larger | 128 | 1 | 11.768 | 4.049 | 43.157 | 1.928 |
| larger | 128 | 2 | 13.062 | 3.722 | 47.180 | 1.945 |
| larger | 512 | 1 | 49.883 | 3.613 | 84.940 | 1.928 |
| larger | 512 | 2 | 48.466 | 3.649 | 83.332 | 1.928 |
| larger | 1024 | 1 | 95.401 | 3.427 | 132.851 | 1.928 |
| larger | 1024 | 2 | 95.627 | 3.426 | 132.635 | 1.928 |

Load/session creation is excluded from request timing and saved separately:
small3.263–3.360s, larger37.926–42.723s. File/config/installation proofs and transfers
also occur outside the request clocks. The analysis retains each raw timing,
median/min/max/MAD, three warmup observations and all ten measured observations;
rounding in the table is for reading only.

Conditions heat across the sequence: the first small128 run records thermal
states0/1, its confirmation1/2, and every later row state2 (serious). No cooldown
is imposed between rows. All recorded request observations have foreground0,
low-power mode false, battery state3/full and battery level1.0 while wired.
Small128/512 timing changes therefore cannot establish controlled performance
improvement or regression. Even close confirmation medians on the long rows
support repeatability only, not a general speedup or external-runtime ranking.
Energy instrumentation is unavailable in these records; there is no joule or
battery-drain estimate. The values characterize this Pro phone and OS/build.

Observed peak physical footprint is1,114,245,536bytes small and1,944,701,920bytes
larger, including load/proof/request/unload/close checkpoints and per-request
lifetime peaks. Actual numerical admissions are1,155,124,440 and1,968,452,528bytes,
within the explicit1.5/2GiB native budgets. Both whole-process trial caps are
2.5GiB (2,684,354,560bytes). These are successful trial budgets, not a permanent
OS allocation guarantee or qualified supported model/context limit. Memory-
pressure behavior, quality at admitted depths and base iPhone14 evidence remain.

The larger512/1024 runs each complete more than600 native seconds in their ten
measured requests:839.172/827.973 and1,344.731/1,329.965s. This supplies long-sequence
raw thermal/memory observations. The small1024 measured totals545.932/541.958s
remain below600s; warmups are not counted toward that measured duration. No
minimum-duration/full-device/sustained-release acceptance is inferred here.

The driver completes24 ordered stages: six fresh tiny-CPU storage proofs, six
preparations and twelve no-copy model measurements. Every full payload plus2GiB
reserve fits its same-device/app, dual-clock/boot-bound fresh observation; every
copy has its own saved freshness check. Ready-receipt hashes stay fixed for both
measurements, and no measurement copies or reinstalls an app. Exact installed
signed-file proofs bracket each of18 application invocations; the selected device,
profile/team/bundle, current-install guards, exits and endpoints all replay.

There is exactly one owned-app preflight reset: the first storage-proof stage
finds an existing verified harness, signals that exact fresh PID with SIGTERM and
observes absence before copying inputs or launching even the tiny proof. Its
final absence and all other47 cleanup observations are natural with no signals.
No signal interrupts or follows a measured request. The original analyzer
incorrectly required all48 observations natural and refused this first reset;
that failed attempt and its exact analyzer/check sources remain. Astra inspected
the raw selected-device/current-install/process/signal records and cleared a
narrow exception for only this first record. The corrected analyzer checks every
raw outcome/device/executable/PID, exact SIGTERM/name15 and command order, while
retaining natural-only requirements elsewhere. Twelve analyzer regressions cover
the exact exception, wrong stage/device/PID/command/order/absence and previous
source/data/metric constraints. Six driver checks also pass.

The [qualification receipt](../../benchmarks/results/2026-10-21-m7-phone-context-qualification-v1.json)
binds1,654 input/output paths and147 exact public copies, including24 redacted
stage receipts,18 raw native consoles, fixed job inputs, analysis/driver logs and
58 exact collection-source snapshots. Private signing/account/device metadata
remain ignored with hashes; public copies are checked against the actual private
identifiers. Original raw private command/metadata inventories and their redacted
summaries are replayed, not substituted by absence or success flags. Engine, app and
collection-controller sources stayed frozen through collection, analysis and
this evidence review; the analyzer correction is retained as described above.

Astra cleared the driver and corrected analyzer source, then accepted completed
code/evidence/index/privacy/source/metric review with no findings. Retain these
observations and unchanged candidate paths. Full10,000-
target loss and100-task phone quality, full native state/lifecycle/pressure
qualification, remaining sustained scope, supported limits and base iPhone14
qualification remain before M7.3/overall M7 completion.

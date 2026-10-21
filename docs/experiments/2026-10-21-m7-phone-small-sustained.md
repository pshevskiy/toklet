# Dedicated small-model iPhone sustained sequence — 21 October 2026

The wired iPhone14Pro (iPhone15,2), iOS27.0.1, completes the dedicated small-model
512-prompt/128-output sequence: three warmups followed by24measured requests and
620.5660416229999 completed native seconds, excluding warmups. Completed measured
caller time is620.567968376seconds. All27requests produce the identical128IDs,
with exact640-token sampled history,639committed positions and one pending token.
This closes the small-model duration observation for this Pro; it does not establish
constant throughput, controlled speedup, energy or overall M7 acceptance.

| Observation | All24 measured median | First5 measured median | Last5 measured median |
| --- | ---: | ---: | ---: |
| Native TTFT, seconds |13.650255312|10.875849875|23.226907084|
| Decode, IDs/second |13.474095705|15.510536009|8.386331920|
| Completed caller request, seconds |23.077777167|19.063901667|38.370678167|

Performance slows over this sequence. Observed thermal states are1(fair) and2
(serious), every sample is foreground, Low Power Mode is off, and the phone is
wired with full battery. These observations are descriptive; no controlled causal
attribution to thermals is claimed. The early/late windows and all raw samples
remain available, rather than summarizing this as one steady guaranteed rate.
Native TTFT includes numerical prefill and first-token work; isolated prefill and
visible-text timing are not measured here. Energy remains unset.

Maximum observed whole-process physical footprint is1,114,180,024bytes. Numerical
admission is1,155,124,440bytes, within the explicit1.5GiB numerical trial budget;
whole-process observations stay below the separate2.5GiB trial cap. Allocated
capacity1152 and actual history640 are distinct. This is a qualified workload
observation, not an OS memory-allocation guarantee or evidence for base iPhone14.

## Collection, replay and retained attempts

Collection revision `38c900c` uses accepted native packagev5 and signed app
preparationv14. The driver consumes the exact Astra-accepted full phone task-quality
qualification before its first stage. Three stages—fresh tiny CPU storage proof,
sealed preparation and no-copy sustained measurement—complete with42 clean owned
host commands. The immutable source snapshot preserves83files:81tracked sources
and the two driver/check files. All model/session/kernel/default and duration
policies remain unchanged.

The initial storage-proof preflight finds one exact installed harness process,
sends verified successful SIGTERM, then observes absence before copying and the
fresh free-space observation. Preparation and measured execution both exit
naturally. Report5natural endpoint absences and1owned preflight reset. Independent
raw replay checks exact command/PID/device/executable/SIGTERM and install→query→
signal→absence→copy ordering. No other stage admits a reset exception.

The replay verifies same-byte source/prerequisite/stage receipts, installed bundle,
launch termination, device/boot endpoints, storage freshness/copy seals, all3456
native token events, completed clocks, exact reset/state/output policy, footprint
chronology, conditions and measured duration. Six driver and16analyzer regressions
pass. First/last-five metrics and raw measured/warmup samples remain separate.

Two earlier queued drivers ran zero physical stages: the first was cancelled after
its predecessor analysis failed, and the second refused its failed quality gate.
Their ledgers/source remain. The completed analyzer's three failed attempts also
remain: an incorrect all-natural assumption, a task-payload-derived four-guard
assumption for plain preparation (correct counts are4/3/2), and Python3.12 builtin
`sum` compensation differing by one rounding bit from Swift's ordered binary64
`+=`. The final replay reproduces the producer's exact ordered total without a
tolerance, while retaining the unchanged≥600-second gate. No numerical run was
repeated and no quality/duration threshold was loosened to resolve these audits.

The [qualification](../../benchmarks/results/2026-10-21-m7-phone-small-sustained-qualification-v1.json)
pins360 inputs and155 exact public copies, including83source copies. Private
signing/installation/device evidence remains local and pinned; actual private
identifiers are scanned from public copies. The
[analysis](../../benchmarks/results/2026-10-21-m7-phone-small-sustained/m7-phone-small-sustained-analysis-v2/analysis.json)
retains the complete sequence and summaries. Astra accepted final source and completed evidence with no findings. Retain the small duration/memory/thermal
observation, with its measured slowdown. Physical state/OS lifecycle, energy/final
supported limits, original base-device scope and M7.3 remain open.

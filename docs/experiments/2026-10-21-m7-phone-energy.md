# Bounded phone energy instrumentation observation

Energy is unavailable in this tested setup and remains unset. The local Xcode
Power Profiler template is listed, but two actual bounded attachment attempts on
the wired iPhone14Pro/iOS27.0.1 fail with instrument exits21 (PID) and19 (name). The first reports
"Cannot find process for provided pid"; the second reports "Cannot find process
matching name: TokletHarness". This records observed tool failure, without claiming
a hardware limitation or its cause. No trace, attributable units, joules, watts,
battery-life estimate or relative-power result is produced.

Each attempt runs a separate bounded small-model state diagnostic using the exact
signed prepared-v16 app and previously admitted transfer. The sidecar first checks
current installation and queries one exact owned executable/PID; the completed
native launch metadata matches that PID. The first attachment selects that PID;
the second uses the documented process-name selector after the same identity
guard. A name selector alone could not establish trace attribution if it succeeded.
Both attempts request10seconds, suppress interactive prompts and have bounded
host timeouts with owned child drainage/reaping. The parent joins the sidecar and
native child, then independently rechecks installation and natural process absence.

Both native diagnostics pass and exit0. All non-instrument host commands exit0;
the instrument failures are preserved, and no trace exists in either attempt.
Private command/device evidence stays local by digest. Both child receipts and
complete raw-evidence closures are bound even on failure; trace inventory is empty.
These instrumented jobs do not replace the accepted uninstrumented correctness,
throughput or memory results. The [qualification](../../benchmarks/results/2026-10-21-m7-phone-energy-qualification-v1.json)
accepts the bounded availability observation, with `energy_qualified=false` and
`energy_joules=null`; it does not turn instrument failure into an energy pass.

Astra accepted both collectors and completed evidence with no findings. Eleven regressions per version include target
identity, no broad target selection, success, timeout, interruption, independent
cleanup failure, dangling outputs, same-buffer parse/hash and conflicting evidence
bindings. Earlier reviewed sources and the initial fake-redaction assertion failure
remain retained. The production engine, signed app and numerical/quality policy
are unchanged. Base-device qualification and overall M7 remain separate.

# Physical lifecycle activation argument boundary

The first phone lifecycle attempt reached active-generation READY and then failed
at Settings activation: `devicectl` reported missing `--device` (exit64). Its
launch subcommand treats all arguments after the bundle identifier as remote app
arguments. The controller had placed device/output options after that boundary.
The failed run, source snapshot and exact receipt remain local and are pinned in
the [qualification](../../benchmarks/results/2026-10-21-m7-lifecycle-cli-fix-qualification-v1.json).

The controller now places shared device/output options immediately after the
two-part subcommand, before command-specific arguments. Both activation bundle
identifiers remain last. The regression fake enforces the actual positional
boundary and success assertions check Settings and original-harness argv.
Same-PID admission, installed identity, process ownership, console drainage and
cleanup requirements remain unchanged. No app, native library, kernel or quality
policy changes are needed.

Astra accepted source and completed component evidence with no findings.
All217 portable methods across22 isolated modules and18 focused lifecycle tests
pass. Saved source and test output are included with the qualification. These
checks execute no device commands and establish controller behavior only. The
failed physical run is not accepted lifecycle evidence; a fresh sealed preparation
and physical retry remain required. M7.3 remains open.

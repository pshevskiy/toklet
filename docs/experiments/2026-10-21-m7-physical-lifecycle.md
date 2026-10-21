# Sealed physical lifecycle controller

The physical controller prepares canonical 512-token lifecycle jobs for the small
and larger selected model paths, pinned to the accepted app's eight code files.
Preparation alone reuses existing exact storage admission/copy/installed-signature
checks; execution has its own controller because it must act while a console
child is still running. Conflicting diagnostic modes and lifecycle execution
through the ordinary blocking measurement controller are rejected before children.

After fresh natural app absence, a directly owned CoreDevice console child waits
for an active-generation READY marker. The controller observes the exact app PID,
activates Settings, waits two seconds and activates the same Toklet installation.
No original launch or foreground activation uses terminate-existing. The final
console termination receipt must name the original PID; any replaced PID fails
closed. Multi-GiB input hashes bracket the round trip rather than delaying the
short active-generation interruption window. Full source/artifact/install/profile,
seal/storage and device endpoints are still required.

Trace admission accepts direct foreground return or the extra return-inactive
phase, requires actual scene-phase provenance, foreground active-generation READY,
original callback clocks, matching interruption/saved/rebuilt IDs, and exact
sampled IDs at every replay/reference/fresh stage. Memory checkpoints and events
must fit the explicit trial cap with monotonic lifetime peaks. Final native state
must be closed, zero and released. Synthetic warning events stay explicitly
synthetic; real OS pressure is not inferred from this test.

Astra found a late-launch cleanup race: querying remote absence before reaping a
still-live launcher could leave a subsequently created app. Cleanup now drains
and independently reaps the owned host launcher first, then performs exact-path
remote cleanup. Drain/reap/capture-close errors are independently retained so one
failure cannot suppress later cleanup/evidence. Astra also found an unbound
observed-prefix relation; READY/interruption IDs now bind to the saved/rebuilt
prefix and each settled output array. Both fixes retain exact previous sources.

Seventeen focused checks cover the complete mocked success/new-PID/timeout/late
launch/drain-error/capture-close paths as well as provenance, prefix, checkpoint,
marker and configuration mutations. All190 methods across20 isolated M7 modules
pass. Two owned local commands pass: CoreDevice launch help confirms activation
and console semantics, and both complete real-model job inventories admit without
loading a native model or using a device. This establishes the controller, not
actual CoreDevice same-PID behavior; physical execution is still required.

The initial17-method run failed one obsolete mock assertion: it required SIGTERM
after host reap had already ended the mocked app. The test now requires verified
absence and does not require an unnecessary signal. Failed output and test source
are retained. An initial local-help wrapper used a relative output path and threw
after the help child, before saving its owned receipt; raw output is retained,
with no invented PID or owned exit status. A fresh absolute-path invocation owns
both successful checks. No kernel, ABI, Swift app or numerical gate changed.

The [qualification](../../benchmarks/results/2026-10-21-m7-physical-lifecycle-qualification-v1.json)
retains current/parent/failed source and exact evidence, building on the accepted
[lifecycle app](2026-10-21-m7-lifecycle-app.md). Astra accepted the source fixes and completed staged component evidence with no
findings. Retain for physical collection; actual OS
lifecycle, pressure, supported limits and M7.3 remain open.

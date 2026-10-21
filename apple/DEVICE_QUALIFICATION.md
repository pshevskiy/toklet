# M7 physical-device acceptance protocol

Status,21 October2026: signed tiny CPU and Metal integration passes on the wired
physical iPhone14Pro (`iPhone15,2`, iOS27.0.1). See the
[physical integration report](../docs/experiments/2026-10-20-m7-physical-tiny.md).
Developer Mode, development signing/profile, actual installation and process
release are verified. M7.3 remains open: no real-model iPhone memory budget,
model/context limit, sustained rate, thermal behavior, energy or actual OS
lifecycle result is accepted. The earlier
[setup observation](../benchmarks/results/2026-10-19-m7-physical-setup-v1.json)
records the historical pre-signing state and remains unchanged. The
[physical measurement controller](../docs/experiments/2026-10-21-m7-physical-measurement.md)
also passes26 raw tiny timing/memory samples with exact IDs/state and cleanup;
this does not establish real-model memory/performance or sustained behavior. Qualify each
device under its exact name; do not transfer Pro results to a base iPhone14,
ProMax, iPad or another Mac. Base iPhone14 evidence remains separate.

The [bounded hashing confirmation](../docs/experiments/2026-10-21-m7-hash-memory.md)
adds two identical small-model short runs (26 requests), preserving exact IDs,
state and cleanup while reducing process-lifetime peak from2.098GB to about
1.017GB. These observations validate the diagnostic memory fix under unchanged
trial budgets; full quality, supported model/context limits, sustained behavior
and actual OS lifecycle qualification remain open. The
[larger short oracle](../docs/experiments/2026-10-21-m7-larger-oracle.md) also
passes two runs/26 requests within the same trial process policy, with exact
independent IDs/state/proofs/release and about1.825GB observed lifetime peak.
These short checks leave the same qualification gates open.

## Admission and identity

Before a physical run, enable Developer Mode and establish a valid signing
identity/team. Keep account credentials, phone passcodes, serials and UDIDs out
of published results. Record the signed app executable/Info.plist/resource
hashes, engine commit, compiler/flags, OS/device identity, provisioning scope
without private identifiers, and instrument versions. Recheck the installed
app identity and exact model/config/token/packed payloads before and after the
run. Preserve failed runs and incomplete output; a simulator or unsigned app
must never count as a physical pass.

Use the selected M6 configurations: small compensated Half/Original norm or
larger frozen group8 Q4v2 compensated tiled/Compensated norm, with Online
attention and FP16 KV. Record all choices explicitly and retain exact checkpoint,
packed-format and corpus identities. Start with tiny CPU and actual-device
Metal self-tests, then real-model numerical/state checks using independent
qualified references. Do not loosen tolerances or interpret a near tie without
its original gap and numerical evidence.

## Memory and lifecycle

Choose and record an independent trial process budget and smaller numerical
admission cap. The native report is logical admission; add Swift input/history/
output storage, command metadata, framework/pipeline resources and temporary
load/conversion allocations. Measure whole-process/native footprint and peak
through loading, warmup, maximum-context requests and release with a named
instrument. Record system pressure and actual allocation failures. Total device
RAM and Mac6GiB admission are not an iPhone app budget. Publish supported limits
only after measured peak and steady state fit the explicit phone policy.

Exercise exact-prefix continuation, fully cached prompts, pending sampled IDs,
changed prompts, hidden stops, cancellation in prefill/decode, reset, replay and
full-capacity overflow preservation. During active work test inactive/background
transitions and resume; separately exercise memory-warning handler integration
and real memory pressure where safely observable. Distinguish synthetic handler
notifications from actual OS pressure. Wait for native work completion before
releasing resources. Confirm prefix retention or an explicit replay/reset
requirement, same-model replay, orderly close and release. Do not assume cleanup
completed while iOS suspended the host. Record termination/jetsam/incomplete
runs as failures or unavailable evidence, never successful recovery.

## Measurement and sustained behavior

Use optimized release builds, three warmups and ten measured initial repetitions
under the same policy as PLAN.md. Retain raw samples and median/dispersion;
ten requests do not establish tail percentiles. Record power source, battery,
thermal state, foreground/cache state, capacity, prompt/stop IDs and emitted
counts. Separate loading/conversion/compilation from warm inference. Tokenization
is outside this raw-ID harness. TTFT runs from request submission to first token;
decode rate is `(N-1)/(last-first)` for N>1. Time completed GPU work. The UI's
100ms polling is unsuitable as an engine timer; use native events or named
instrumentation and state the timing boundary.

Run a documented repeated-request sequence for at least600 completed seconds,
with resident weights and explicit session resets. Include128/512/1024 prompt
plus128 output cases and qualified capacity boundaries where the independently
admitted phone budget permits. Preserve the sequence, raw intervals, memory,
thermal states and power conditions; observe degradation across the sustained
run and confirm any claimed change in a second comparable run. Record energy
only when supported instrumentation actually supplies it, with units and scope;
leave unavailable values unset. Application transcripts are separate from raw
throughput. Do not infer a Mac speedup or external-runtime lead on the phone.

## Exit record

Publish reproducible physical receipts, exact workload/configuration, numerical
and lifecycle outcomes, whole-memory evidence, at least600s sustained samples,
thermal behavior and energy availability. List passing device/model/context
limits and failed/rejected cases. Keep M7.3 and M7 pending until this evidence
exists and is reviewed. Signed tiny integration satisfies the initial physical
integration gate only. Pro qualification cannot by itself close a base iPhone14
requirement.

The [score API component](../docs/experiments/2026-10-21-m7-score-api.md) adds a
read-only final-committed-row diagnostic and passes independent CPU/Metal and
three-target build/link checks. It is preparation for phone teacher forcing,
not evidence of a completed phone corpus/task run. Score before appending each
target, verify exact committed prefix after cancellation, and retain the frozen
corpus/context/aggregation/gates when implementing the collector.

The [Swift score adapter](../docs/experiments/2026-10-21-m7-worker-score.md)
now exposes validated scalar diagnostics on the owning thread. Mac CPU/Metal,
deterministic close/interruption publication and both SDK builds pass. This
component neither runs a phone corpus nor qualifies actual OS lifecycle recovery.

The [teacher-forcing coordinator](../docs/experiments/2026-10-21-m7-teacher-forcing.md)
passes independent tiny CPU loss, Metal sequencing and strict Mac/device/simulator
module compilation. Its10000-target check is simulated protocol coverage only.
Full phone collection must still admit exact corpus/model/config inputs, enforce
foreground interruption policy and observe process-memory peaks and cleanup.

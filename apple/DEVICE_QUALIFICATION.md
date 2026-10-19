# M7 physical-device acceptance protocol

Status: not executed. M7.1 packaging and M7.2 simulator integration do not close
M7.3. No iPhone memory budget, model/context limit, sustained rate, thermal
behavior or energy result is accepted. The current wired target is a physical
iPhone14Pro (`iPhone15,2`, iOS27.0.1), with Developer Mode enabled and developer
services available. The older wireless iPhone14ProMax (`iPhone15,3`, iOS26.6.1)
still reports Developer Mode disabled. Neither is the Inference Lab iPhone14
simulator. Qualify each device under its exact name; do not transfer Pro results
to a base iPhone14, ProMax, iPad or another Mac. Base iPhone14 evidence remains
separate. The [setup observation](../benchmarks/results/2026-10-19-m7-physical-setup-v1.json)
records successful owner account sign-in and an available Personal Team, but
zero valid signing identities and no signed or installed physical app.

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
exists and is reviewed. Development signing remains a prerequisite for
the wired Pro's first physical execution; no desktop or simulator result waives
this gate. Pro qualification cannot by itself close a base iPhone14 requirement.

# M6 producer-typed quality validation

The separate typed collector restores the Rust producer's four FP32 winner
values per comparison row before the existing numerical checks. Raw events,
FP64 losses, margins and vocabulary error statistics remain unchanged. No
inference kernel, representation, request, tolerance or quality gate changes.

## Reproduced rejection and correction

The original larger-model 32K attempt emitted all 128 scored rows, but its
collector rejected row 92: shortest FP32 JSON spellings `30.187262` and
`30.160664` subtract to approximately `0.026598` when interpreted directly as
FP64. Restoring their declared FP32 values gives
`30.1872615814208984375 - 30.160663604736328125 = 0.0265979766845703125`, exactly the
recorded full-vocabulary maximum. Both modes selected token 24 on this row.

That original attempt remains **failed**: the validation exception prevented
natural successful worker completion, and no successful exit code was recorded.
Its complete emitted rows cannot replace the clean lifecycle requirement. The
original summary, raw events and empty stderr are preserved. Its calculated
loss difference is diagnostic only; it does not qualify the missing 32K case.

The independent offline winner checker also exposed the same type distinction
in larger 4K row 127's top-two margin. The new shared restoration helper changes
only the two two-element winner arrays in copied row containers. It rejects
invalid pair shapes, booleans, nonfinite values and FP32 overflow, and preserves
signed-zero bits. FP64 fields and the original saved structures stay untouched.
Changed winners remain unresolved until separately reviewed investigation.

## Separate collection and replay

The typed collection uses an explicit new format and the original reviewed
native paired worker, frozen input IDs, capacity, chunk 128, mode selection,
shared weights and separate session KV. The existing bounded worker owner,
12-hour per-event default, 24-hour lifetime, native charge sequence, EOF,
cleanup and prelaunch/endpoint identity checks are retained.

Replay verifies raw coverage before creating its typed view, then invokes the
unchanged numerical/state/memory replay checks. It additionally verifies the
complete consumed-file inventory, executable command and wait metadata. The
mixed-format dispatcher keeps successful original records on their original
replay route; it never relabels a failed original as a typed success.

Twenty-five targeted portable tests pass, including both real serialization
regressions, raw-buffer immutability, unchanged FP64 values/loss rejection,
incorrect winner/error rejection, missing/failed natural exit, request/state
coverage, endpoint failure preservation and mixed-route identity conflicts.
The broader 40-test selection, including the separately reviewed winner capture
code, also passes independently under Astra. Root formatting passes.

A partial compatibility replay verifies seven original successes, 896 scored
rows, 917 raw events and 334 bindings. It rejects the failed larger 32K attempt.
The two larger 4K winner changes stay open. This is incomplete eight-role quality
coverage; a fresh larger 32K typed run remains required.

The fresh Small4K typed probe v2 completes 128 scored rows with natural exit 0,
empty cleanup/endpoint failures and unchanged identities. All native request and
comparison events, including full-distribution hashes, and its numerical quality
summary exactly match the earlier original Small4K run. Its mean NLL difference
is -0.00024724887088295867 with zero changed winners. The typed replay verifies
131 events and 112 bindings. Complete paired admission is 1,520,375,500 bytes;
maximum sampled joint native lifetime charge is 1,353,697,152 bytes, within 6 GiB.

The [qualification receipt](../../benchmarks/results/2026-10-16-m6-typed-quality-qualification-v2.json)
binds code, preserved failures, five newly completed original quality cases,
probe events and replay helpers. Astra accepted code and evidence after
independent 40-test execution, byte-identical seven-record and probe replays,
all six source pins, 26 evidence pins and 26 exact published copies, with state,
memory, natural exit and failure preservation independently checked. All review
execution and hashing finished before the next native worker. The prior probe
and candidate publication remain preserved: a final whitespace check removed
one extra source EOF newline, and fresh build/probe evidence binds that exact
source correction. No numerical behavior changed.

The fresh larger32K typed retry, winner investigations, remaining profiles and
boundary runs, and final KV/M6 decisions remain open. Runtime defaults and
numerical gates are unchanged.

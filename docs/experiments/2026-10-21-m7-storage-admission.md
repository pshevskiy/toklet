# M7 observed storage admission and first small-model phone oracle

The host controller now rejects new real-model transfers unless a fresh,
same-device signed tiny run reports enough filesystem space for every copied
file plus an explicit2GiB reserve. It also verifies original admission when
retrying without copies. Astra cleared source,55 portable checks pass, and the
small-model preparation and actual short oracle pass. Larger packed preparation also passes. Astra accepted the code and evidence. M7.3 remains open.

The observation receipt binds the current private device selection, signed app,
prepared installation, native inputs, all controller/policy/test sources and the
tiny fixture. The host replays the bounded raw console, launch metadata and
private boot observation against the saved result. Old observation receipts
without the new clock/boot fields cannot admit a transfer.

Before launching the tiny diagnostic, the host records wall and monotonic times
before the installation guard and input hashing. These are conservative lower
bounds for its later filesystem read. An owned `sysctl kern.bootsessionuuid`
probe supplies the host boot identity; the raw UUID stays private and only its
SHA binding is public. A changed boot, regressed clock or age of five minutes
or more fails closed. Receipt completion cannot make an earlier read fresh.

After every expensive guard/hash and immediately before each actual Model,
Packed or job/config copy child, both ages are rechecked. Fresh installed-path
checks also precede every copy. Source failure or expiration still permits
independent exact-install process cleanup. The recorded original copy decisions
are replayed for a no-copy retry; old ready receipts without complete original
admission are rejected. No-copy reuse needs no fresh capacity measurement.

Payload accounting includes the exact staged model, packed and job files. Byte
count and SHA256 come from one streamed descriptor in256KiB chunks, avoiding a
separate size/hash race. Individual counts, total and reserve addition use
checked UInt64 bounds. Exact-capacity acceptance and one-byte-short rejection
remain tested. The policy reserves no physical space and cannot prevent another
filesystem user from consuming space after observation; this is a trial policy,
not guaranteed allocation capacity or a model memory budget.

The first source review found that separate stat/hash reads could undercount a
transient smaller file. The final implementation streams both together. A real
replacement regression restores five admitted bytes only when hashing opens a
one-byte file: old code would count three total bytes including the job, while
current code correctly counts seven. Further regressions cover omitted/changed
bindings, wrong installation, replay changes, boot/clock/freshness failures,
missing original decisions and every copied file. Main-path tests expire a
proof after hashing but before Model and job children, verify zero affected
copies, and retain two clean process-absence checks. Missing initial proof starts
zero children. No model, GPU, SDK or phone is required by these55 tests.

The small preparation admits988,098,991 payload bytes including the job, plus
2,147,483,648 reserve, requiring3,135,582,639 bytes against a reported
32,768,000,000-byte observation. Copy ages are54.413s and87.013s by wall time;
both clocks pass. Remote inventory and cleanup pass. The no-copy run completes
three warmups plus ten measured requests on iPhone14Pro (`iPhone15,2`, iOS27.0.1),
prompt `[9707,11,358,498]`, capacity32/chunk3 and maximum3. Every request emits
`[1410,1492,752]` with history7/committed6/pending1, matching installed/model
proofs, foreground conditions, unload/close/release and clean process absence.
These are short raw-ID/state checks, not full numerical/quality qualification.

The small run's lifetime physical peak reaches2,097,663,088 bytes within the
provisional2.5GiB process policy. Before→after artifact proof grows from
21,251,744 to1,013,664,064 bytes; loaded reaches2,009,615,400. This localizes an
unexpected near-file-sized memory increase before loading. Foundation chunk
autorelease retention is a hypothesis for a separate bounded-memory fix and
same-workload confirmation. Larger inference is deferred until investigated;
transfer-only preparation does not hash/load model data in the phone app.
No numerical gate or declared phone limit is changed.

The first larger preparation copies Model and Packed inputs under valid161.301s
and260.953s observations, then refuses the job write when the observation expires.
No inference starts, and endpoint/cleanup checks pass. The immediate fresh-proof
retry passes all three copy checks at57.080s,67.470s and83.187s. It admits
4,397,778,750 payload bytes plus2GiB reserve, requiring6,545,262,398 bytes
against27,443,200,000 reported free bytes. The copied inventory, job,
identity endpoints and process absence pass. No larger-model inference or mobile
throughput/memory-limit qualification is accepted here.

The [qualification receipt](../../benchmarks/results/2026-10-21-m7-storage-admission-qualification-v1.json)
binds current controller/policy/test and frozen runtime/model inputs, private
evidence hashes, exact safe copies and the successful retry driver. Raw boot UUID,
signing and device inventories remain private. The sandbox-denied initial boot
probe and successful privileged read are preserved as setup observations, with
no automatic approval rejection. No new Rust/Swift numerical code or SDK build
is attributed to this host-only task; the signed app remains the reviewed
filesystem-observation version. Astra accepted this task with no remaining findings.

The native small path uses compensated Half projections, Original normalization,
Online attention, FP16 KV and FullLogits readback. The larger transfer binds the
frozen group8 Q4v2 payload; its inference is deferred. Native package receipts
retain exact engine/toolchain inputs, with no numerical default changed.

Astra verified978 bound paths,30 exact published/staged copies, all tiny proofs,
small oracle/state and both original ready-admission replays, privacy, links and
whitespace. Review was entirely offline, with no concurrent native/device work.

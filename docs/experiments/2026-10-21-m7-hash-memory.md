# M7 bounded artifact hashing

Per-chunk autorelease pools remove the near-file-sized diagnostic memory growth
observed before loading a real model. Astra accepted code and evidence;84 portable checks,
17 actual Mac cases, both iOS SDK builds and two same-workload iPhone14Pro runs
pass. This fixes artifact hashing memory, without qualifying a supported phone
model/context limit or completing M7.3.

Foundation's `FileHandle.read(upToCount:)` returns bridged `Data`. Although the
loops read only256KiB at a time, those buffers can remain autoreleased until
the diagnostic's long-running queue closure returns. Both model/config proofs
and the installed-bundle proof now wrap each read, byte-limit check and SHA256
update in a local pool. Only a Bool continuation flag escapes the pool. Handles,
byte counters and hash state retain their existing ownership; file/type/path,
exact inventory, SHA, size limits and thrown errors remain checked. Numerical
kernels, sampling, device budgets and model defaults are unchanged.

An independent regression creates a256MiB sparse zero file outside Swift,
hashes it in Python, and hard-links it into the tiny model's admitted inventory.
The loader ignores this extra non-safetensors file; both artifact proofs must
hash it exactly. A192MiB whole-process budget is deliberately smaller than the
file, with the existing128MiB numerical policy. The old implementation fails
before loading: current/peak footprint rises from10,912,344 to280,724,248bytes
after proof, then550,159,320bytes after the failure-path final proof. Earlier
CPU/Metal/failure cases share this process, so the pre-proof peak is preserved.
The first fixed run passes all13 requests with after-proof9,716,312bytes and
lifetime peak9,749,080bytes. The final checker additionally requires all six
memory phases, all13 samples, integer positive observations under192MiB and the
independent padding SHA. Missing/reordered phases and oversized sample peaks
fail portable regressions. The hardened actual Mac run also passes.

Configuration admission now tests valid JSON padded to exactly512KiB and the
same JSON plus one byte. The boundary succeeds and the excess preserves the
streamed file-limit error. Existing changed SHA, extra file, symlink, endpoint
mutation, output failure, cleanup and independent tiny ID/state checks pass.
Actual Mac CPU/Metal and bounded-hash CPU cases retain39 raw request samples.
Both SDK Release builds retain strict Swift concurrency and warnings-as-errors;
existing Xcode orientation/AppIntents tool warnings remain. The84-test M7 suite
first encountered sandbox process-group EPERM in its deliberate timeout test;
the unchanged rerun outside that restriction passes all84. Targeted measurement
checks pass22 tests. No Rust numerical source changed, so this task adds no
unrelated Cargo or engine performance claim.

The first physical preparation stops before building because an existing
harness process is outside that invocation's ownership. A separate stop-only
check verifies the exact historical signed bundle/current installed path and
terminates only that app, with process absence confirmed. The retry signs,
verifies and installs the new app. A first storage-query invocation then refuses
reuse because required exact controller snapshots were missing, before any
child starts. Adding the unchanged snapshots permits authenticated reuse; both
failed attempts remain recorded.

A fresh tiny filesystem observation admits the small transfer under the existing
payload-plus2GiB policy. Both subsequent short runs use the exact earlier job:
three warmups, ten measured requests, prompt `[9707,11,358,498]`, capacity32,
chunk3, maximum3, selected compensated Half/Original normalization/Online
attention/FP16 KV/FullLogits. Every request returns `[1410,1492,752]` with
history7/committed6/pending1. Model and installed-bundle proofs, foreground
conditions, close/weak release and two independent process-absence checks pass.

| Own-process observation, bytes | Previous app | Fixed run1 | Fixed run2 |
| --- | ---: | ---: | ---: |
| Before artifact proof | 21,251,744 | 19,105,416 | 18,941,576 |
| After artifact proof | 1,013,664,064 | 18,990,728 | 18,826,888 |
| Loaded | 2,009,615,400 | 1,014,909,296 | 1,014,778,224 |
| Lifetime peak at final proof | 2,097,663,088 | 1,016,990,064 | 1,016,858,992 |
| After final proof, current | 2,097,663,088 | 22,218,400 | 110,577,408 |

The two fixed runs confirm the removed file-sized retention. Final current
footprints differ; resource close/release does not promise immediate OS
reclamation. TASK_VM_INFO observes this application's physical footprint and
process-lifetime peak, without counting separate system GPU-service allocations.
These short raw-ID trials do not establish full quality, sustained throughput,
thermal behavior or energy, and no loading-speed claim follows from their times.
The provisional2.5GiB process policy is unchanged.

The [qualification receipt](../../benchmarks/results/2026-10-21-m7-hash-memory-qualification-v1.json)
binds executed sources, before/after raw evidence, exact safe copies, historical
source mappings, private signing/device hashes and unchanged protected CPU
binaries. Raw identity/signing/boot observations remain private. The exact
baseline remains attached to the prior storage-admission report. Larger-model
inference, full phone quality/state/context/lifecycle/sustained qualification and
base iPhone14 evidence remain pending. Retain the pools; M7.3 stays open.

Astra accepted622 distinct bound paths,130 exact staged copies, seven historical
mappings, the unchanged workloads and all stated memory values, raw replay,
privacy, links and whitespace, with no remaining findings.

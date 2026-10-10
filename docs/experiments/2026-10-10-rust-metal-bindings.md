# Rust Metal binding qualification, 10 October 2026

This standalone [Rust probe](../../tools/experiments/rust_metal_probe.rs)
qualifies the public `objc2`/`objc2-metal` boundary before an engine backend is
introduced. It uses a fresh utility package generated from a pinned
[manifest](../../tools/experiments/rust_metal_probe.Cargo.toml) and
[lockfile](../../tools/experiments/rust_metal_probe.lock); the engine remains one
package with its existing CPU dependencies and execution. The release probe
passes all 39 completed dispatches, structured shader-error propagation and
pre-device budget rejection on the current Mac. M4 remains pending.

The [shader](../../tools/experiments/rust_metal_probe.metal) adds one to an FP32
array. Counts 1, 17 and 257 exercise complete and irregular threadgroup tails.
Inputs are dyadic quarter increments, so addition is exact in FP32. The native
probe and the independent [Python validator](../../tools/experiments/rust_metal_probe.py)
compare every output bit, every input bit and seven sentinel values at the end
of both allocations after every completed command. Each case has three warmups
and ten samples, with buffers reused across all thirteen dispatches.

## Layout, ownership and completion

Two contiguous shared allocations each hold `[count + 7]` FP32 elements.
The shader receives read-only input at buffer 0, mutable output at buffer 1 and
a copied four-byte UInt32 count at buffer 2. Offsets are zero. Length,
`isize` bounds and actual pointer alignment are checked before typed access;
zero initialization covers every allocated slot. No alignment beyond FP32 is
assumed. A rounded dispatch returns immediately for every index outside count;
checked rounding also prevents UInt32 index wraparound.

A private owner holds each retained Metal buffer. Native handles and borrowed
slices stay inside the synchronous probe. CPU input initialization and output
sentinel fill happen before submission; CPU readback occurs only after
`waitUntilCompleted` and successful status/error checks. There is no fallible
host work between commit and the completion wait. Lexical owners remain alive
through that wait, and command buffers must retain their native resource
references. Shared allocations use explicit tracked hazards.

The queue and pipeline are created once per process. Each case creates its two
buffers once, reuses them across repetitions and releases them before the next
case. Each repetition has its own autorelease pool. Device/pipeline limits
determine the threadgroup width. macOS 15 and Apple Silicon unified memory are
checked before qualified APIs are used. Metal 2.3, safe math and precise
functions are set explicitly and read back before compilation.

Admission counts both shared buffers once, the host input and both diagnostic
readbacks, plus a 64 MiB allowance for runtime/metadata and the fixed small
diagnostic record. This is five data arrays, with maximum admitted total
`67,114,144` bytes. It preserves the 6 GiB ceiling. The supplied smaller budget
is checked before device discovery; a one-byte budget must fail admission.

## Driver and evidence

The driver refuses existing build or evidence paths and disallows evidence
inside the generated package. Every Cargo command explicitly pins
`CARGO_TARGET_DIR` inside that package, overriding inherited target settings.
Locked offline formatting, Clippy with warnings denied, two pure Rust bounds
tests and a release build precede native checks. Compilation has an owned
process-group watchdog; each native run has a separate bounded group.

The native process reports an explicit completed-work idle record and waits for
an acknowledgement. The driver reads the physical peak charge of that child
while it remains alive, then acknowledges and reaps it. Normal execution must
exit zero; invalid shader compilation must preserve its structured NSError
failure; insufficient budget must preserve its admission failure. Malformed or
unfinished protocol output kills/reaps only the driver's owned group.

Source, shader, manifest, lockfile, copied inputs, binary, driver and helper
identities are recorded. The driver rechecks sources and binary after native
qualification. Raw native JSONL and build/native stderr have separate hashes.
Total external GPU memory, GPU trace and energy remain unmeasured.

Five offline Python checks pass. They reject changes in values, guards, phases,
sample counts, completion, launch limits and memory accounting; check evidence
overwrite/alias refusal and inherited Cargo output isolation; and exercise
actual child acknowledgement, memory-read timing and failed-protocol cleanup.
Astra found the inherited target-directory issue during code review; the
explicit override and regression check resolve it.

## Native results and checks

The [successful attempt](../../benchmarks/results/2026-10-10-rust-metal-bindings-v2.json),
[native checkpoint](../../benchmarks/results/2026-10-10-rust-metal-bindings-v2.json.native.jsonl)
and [build log](../../benchmarks/results/2026-10-10-rust-metal-bindings-v2.json.stderr.txt)
are exact raw copies. All 3,575 active output values and 273 output guards match
bits. The 3,848 checked input slots include all values and 273 input guards.
Each of the three cases preserves thirteen complete rows and reuses its buffers.
Shader failure returns the compiler's actual NSError text; one-byte admission
fails before any device API call. Both expected failures exit one after the
completed-work acknowledgement; normal execution exits zero. No watchdog fires.

The [first build attempt](../../benchmarks/results/2026-10-10-rust-metal-bindings-v1.json)
stopped at Clippy's two `manual_is_multiple_of` findings, before native execution.
Its [diagnostics](../../benchmarks/results/2026-10-10-rust-metal-bindings-v1.json.stderr.txt)
remain preserved. Both checks now use the standard integer method, and the
fresh second attempt passes formatting, Clippy, both Rust bounds tests and the
locked release build. The original failed attempt was not overwritten.

Device is Apple M1 Pro with unified memory, MacBookPro18,3, 16 GiB RAM and macOS
26.6.2 (25G83). Rust/Cargo are 1.90.0; macOS SDK is 26.5. Builds explicitly set
`MACOSX_DEPLOYMENT_TARGET=15.0`; the release target is `aarch64-apple-darwin`.
Dependencies pin `objc2=0.6.5`, `objc2-foundation=0.3.2`, `objc2-metal=0.3.2` and
`serde_json=1.0.149`, with exact transitive versions/checksums in the lockfile.
The pipeline reports execution width 32 and maximum group size 1,024; the
device reports `[1024,1024,1024]` thread limits, maximum buffer length
`9,534,832,640` and recommended working set `12,713,115,648` bytes. These queried
capabilities do not replace admission. Groups are 1, 1 and 9 for the tested counts.

Normal execution's own-process physical charge peak is `6,177,368` bytes.
Shader-error and admission-error processes peak at `3,768,800` and `1,474,896`
bytes, respectively. All are below 6 GiB. Total GPU/system physical memory and
energy remain unavailable. Device discovery, library compilation and pipeline
creation take 42.199, 132.537 and 26.453 ms in this process; system caches are
uncontrolled. These are functional observations without performance acceptance.
The per-row timer covers command/encoder creation, encoding, submission and
successful completion; it excludes allocation, sentinel reset and readback.
Full-engine latency is unqualified.

AC power is attached, AC low-power mode is disabled, and no thermal/performance
warning is reported at either native boundary. Native checks and all root Rust
checks run after the first primary scalar numerical worker exits normally.
The original scalar worker remains idle at CPU time `112:18.61`; only its
transport and the comparison coordinator are held. The full offline Python
suite overlapped that first primary scalar run, so the run is preserved but
excluded from comparisons and replaced by a complete clean run. No compiler,
model test or GPU dispatch overlaps the clean replacement, which starts after
all native and root Rust checks finish.

Formatting, Clippy with warnings denied, all 75 ordinary engine Rust tests,
the two standalone Rust bounds tests and 54 offline Python checks pass.
The qualified CPU source hashes and frozen scalar worker/CLI binary hashes
remain unchanged. Astra accepted the complete binding task without remaining
findings, independently verifying source/copied inputs, lockfile, binary/helper
hashes, exact evidence copies, all dispatch bits/guards, failure stages and exit
codes, memory accounting and power. Review used lightweight reads and validation;
it ran no GPU, compilation or model workloads.

Successful native source SHA-256 is
`74eee0413c11284336d7c019f3522d09ba71120135648ecdb19dae1548d1277f`,
shader `7f1364605048cc5fe83099cf7520bd1b83f49e46b7d3f2cc1c440ee2000ee850`,
driver `880b12d8abd664695516ee88d30de888d045fcd28c209df77fbff5f27baca1b4`,
and binary `8c95b55b4d65cdc567017345d23ab5a4d067a91b60b12fe4cd155e9676149966`.
Base engine revision is `0f122d7`; source/manifest/lockfile identities are bound
into the result, with all generated copies rechecked before and after native work.

```sh
.venv/bin/python tools/experiments/rust_metal_probe.py --output benchmarks/raw/rust-metal-bindings.json --build-dir artifacts/rust-metal-bindings
.venv/bin/python -m unittest discover -s tools/experiments -p 'test_*.py'
```

Run native checks separately from timed CPU workloads. These small exact
dispatches establish binding/ownership behavior only; complete FP16 numerical
operations, resident model/session state, GPU failure recovery, Qwen generation,
quality, GPU trace and mobile qualification remain pending. M4 remains open.

The completion and buffer contracts follow Apple's
[command completion API](https://developer.apple.com/documentation/metal/mtlcommandbuffer/waituntilcompleted%28%29?changes=_5&language=objc)
and the pinned generated bindings for
[MTLBuffer](https://docs.rs/objc2-metal/0.3.2/objc2_metal/trait.MTLBuffer.html) and
[MTLComputeCommandEncoder](https://docs.rs/objc2-metal/0.3.2/objc2_metal/trait.MTLComputeCommandEncoder.html).

# Standalone Metal toolchain check, 10 October 2026

The [Swift probe](../../tools/experiments/metal_probe.swift) qualifies a small
public Metal dispatch on the current Mac before Rust backend work. Runtime
source compilation succeeds, shared-buffer FP32 addition matches CPU bits at
1, 17 and 257 elements, and seven sentinel values after every admitted output
remain untouched. Completed command status is checked before CPU readback.
This prepares toolchain evidence only; M4 and the dependency chain remain open.

The [record](../../benchmarks/results/2026-10-10-metal-toolchain.json) and
[stderr](../../benchmarks/results/2026-10-10-metal-toolchain.json.stderr.txt)
come from the [bounded driver](../../tools/experiments/metal_probe.py), which
always compiles a fresh binary and binds its source/binary hashes to the run.
It refuses existing evidence/build paths, kills/reaps only its new process group
on timeout, and retains incomplete status on failure. Build timeout is five
minutes; native execution timeout is one minute. Two offline tests exercise
overwrite admission and actual timeout/child reaping without Apple APIs.

The device is Apple M1 Pro with unified memory, macOS 26.6.2 (25G83). The compiler
is Apple Swift 6.3.3, macOS SDK 26.5, command-line-tools developer directory,
`-O -warnings-as-errors -target arm64-apple-macosx15.0`. Shader language is Metal
2.3 with explicit safe arithmetic and precise FP32 math functions. The device
reports Apple families 1–7 and Mac2, pipeline execution width 32, maximum 1,024
threads per threadgroup, maximum buffer length 9,534,832,640 bytes, and recommended
working set 12,713,115,648 bytes. These are queried values for this device;
the engine's 6 GiB ceiling is unchanged and no portability claim follows.

Shared buffers have checked lengths/alignment and explicit Float binding.
CPU initialization finishes before submission, retained command references and
lexical owners preserve resources, and CPU access resumes after successful
completion. Threadgroup width uses pipeline/device capabilities; rounded grids
are guarded by the admitted element count in the shader. Logical buffer payloads
are 96, 288 and 3,168 bytes including input/output sentinel padding. Own-process
physical peak, complete GPU/system working memory, and energy are unavailable.

MPS reports device support and recommended row strides. FP16/FP32 stride is 2/4
bytes for one column, 16/16 for three, 16/32 for eight, 1,792/3,584 for 896, and
9,728/19,456 for 4,864. Future matrix descriptors must use their actual stride;
this inventory does not evaluate MPS matrix multiplication or select a GEMM.

The standalone `metal` compiler and `xctrace` tools are unavailable. Runtime
library compilation works through Metal, but the GPU trace-document capture
destination is unavailable in this run. No actual capture is recorded. Swift
compilation takes 1.294 seconds and new-process wall time 0.553 seconds. Library
and pipeline creation use warmed system caches from exploratory invocations;
their sub-millisecond values are not uncached compilation costs. Single command
timings are functional diagnostics without warmups or repetitions, establishing
no throughput or performance acceptance.

AC power was attached with AC low-power mode disabled and no reported thermal
warning at either boundary. Native checks and ordinary suites finished during
the scalar benchmark's excluded third warmup, before measured requests. Source,
binary, driver and raw-copy identities were checked. Formatting, Clippy with
warnings denied, all 74 ordinary Rust tests and 33 offline Python checks pass.
No decoder, FP16 execution, MPS multiplication, full-model memory, GPU trace,
iPhone integration or milestone completion is qualified by this preparation.
Astra accepted the completed task without findings after independently checking
resource/dispatch safety, provenance, exact raw copies, power, logical accounting,
report links and both offline tests. Its review ran no GPU/performance jobs.

```sh
python3 tools/experiments/metal_probe.py --output benchmarks/raw/metal-toolchain.json --build-dir artifacts/metal-toolchain
.venv/bin/python -m unittest discover -s tools/experiments -p 'test_*.py'
```

Run hardware checks separately from measured performance jobs. The Rust engine
and frozen release benchmark binaries remain unchanged.

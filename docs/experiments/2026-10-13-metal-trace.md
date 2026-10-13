# M4 Metal scheduling and cost attribution

M4 is functionally complete: the resident FP16 decoder runs full tiny/Qwen
forward passes and complete Qwen generation, preserves transactional session
state, bounds repeated-token allocations, and now explains scheduling costs
with real GPU captures and calibrated stage counters. CPU remains the default.
FP16 held-out 10,000-target loss / 100-task quality and mature-engine speed
parity remain separate, unqualified gates. No M5 optimization is included.

## Mechanism

Profiling is explicitly enabled on a diagnostic session. Production sessions
keep the same numerical graph, one command and one completion wait per chunk.
The profiler preallocates timestamp slots, compute-pass descriptors and a marker
sink; capability checks require stage-boundary timestamp sampling. Each custom
kernel gets encoder start/end counters. MPS owns its encoders, so two dependent
one-lane atomic OR-zero markers enclose each multiply without changing any
result bits. Tracked read/write resources preserve those dependencies.

Counter resolution occurs only after successful command completion and final
logit readback. Reset/failure cannot expose a stale successful profile. Each
raw GPU tick is converted through device CPU/GPU timestamp pairs before and
after the command, subtracting integer origins before floating-point conversion.
The conversion follows [Apple's paired-clock method](https://developer.apple.com/documentation/metal/converting-gpu-timestamps-into-cpu-time).
Zero/error samples, reversed or out-of-calibration intervals and unexpected
resolved byte lengths are rejected. CPU readback and clock calibration are
timed separately. Profile export allocations and stdout occur outside GPU work.

Custom intervals measure encoder processing. Enclosed MPS intervals include
internal kernels and scheduling. Transition intervals are time in the observed
command span without attributed custom/MPS/marker work; they do not establish
global physical GPU idleness. Interval unions account for overlapping work;
individual category medians should not be summed into a total.

## Workload and evidence

Apple M1 Pro, 16 GB, macOS 26.6.2, engine parent revision `876a5f7`, isolated
optimized release build with `--features metal`, precise shader arithmetic.
Collectors record rustc/build flags, power state, source/binary/model identities
before and after the run. Sources are dirty-task hashes, not a claim that the
parent commit already contains profiling. The qualification record binds the
final tested binary and checks.

A single immutable model is borrowed by ordinary and profiled sessions with
separate KV/scratch, capacity 32 and prefill chunk 4. Tiny uses exact source FP32
fixture weights; Qwen uses the pinned Qwen2.5-0.5B-Instruct artifact. Each of
14 runs resets both sessions, evaluates four fixed prompt tokens, then evaluates
one greedy token. Runs 0–2 warm up, 3–12 are uncaptured samples, and run 13 is
captured separately. Every final logit bit matches the ordinary session, every
first greedy ID matches its frozen reference, and repeated logit hashes/IDs
remain stable. No source tolerance was changed.

Tiny exports 82 stages per operation, 2,296 total; Qwen exports 918 per operation,
25,704 total. Actual `.gputrace` bundles capture four completed commands apiece,
with hashes for every retained bundle file. Raw JSONL and large capture bundles
remain local ignored artifacts; committed summaries retain their identities.

- [Final Qwen collector](../../benchmarks/results/2026-10-13-metal-trace-qwen-v3.json) and [replayed analysis](../../benchmarks/results/2026-10-13-metal-trace-qwen-analysis-v3.json).
- [Final tiny collector](../../benchmarks/results/2026-10-13-metal-trace-tiny-v3.json) and [replayed analysis](../../benchmarks/results/2026-10-13-metal-trace-tiny-analysis-v3.json).
- [Resident decoder/session qualification](2026-10-13-metal-engine.md), including complete generation, cancellation/EOS/output failure, prefix reuse, finite overflow/replay and allocation checks.
- [Trace qualification record](../../benchmarks/results/2026-10-13-metal-trace-qualification.json).

## Measurements

Medians of ten uncaptured Qwen samples, milliseconds. Prefill is four tokens;
decode is one token at committed position four. These small diagnostic shapes
are attribution evidence, not representative generation throughput acceptance.

| Completed work / host stage | Ordinary prefill | Ordinary decode | Profiled prefill | Profiled decode |
| --- | ---: | ---: | ---: | ---: |
| Completed GPU command | 40.659 | 105.545 | 60.993 | 126.486 |
| Host encoding | 1.455 | 1.443 | 5.667 | 5.770 |
| Command creation | 0.014 | 0.011 | 0.007 | 0.010 |
| Commit | 0.015 | 0.016 | 0.013 | 0.016 |
| Completion wait | 41.187 | 106.225 | 61.599 | 127.003 |
| Final logits readback | 0.083 | 0.086 | 0.082 | 0.086 |
| Paired-clock calibration | 0.000 | 0.000 | 0.002 | 0.002 |

| Profiled GPU attribution | Prefill ms | Decode ms |
| --- | ---: | ---: |
| Observed GPU span | 60.993 | 126.486 |
| Custom encoder union | 15.904 | 15.940 |
| Enclosed MPS union | 33.512 | 98.630 |
| Marker union | 1.439 | 1.506 |
| Unattributed transitions | 11.146 | 11.401 |
| Overlapping attribution | 1.056 | 1.068 |

MPS projections dominate this observed decode workload; custom-kernel cost stays
similar while four-row prefill reuses weights. This suggests measuring decode
projection bandwidth/dispatch in M5. It does not establish the private MPS
microkernel bottleneck. Instrumentation materially changes host/GPU costs and
is retained only as an opt-in diagnostic. Tiny ordinary/profiled GPU medians
are 0.898/2.853 ms prefill and 1.255/3.094 ms decode, further showing that overhead.

Qwen load is 3.486 s, including 3.205 s conversion and 0.042 s compilation;
session/counter allocation is 4.268 ms. Weight storage is 988,069,596 bytes,
counted once across sessions. Combined diagnostic admission is 1,198,520,036
bytes, including native metadata/counter allowances. Persistent memory records
remain unchanged at every idle handshake. Native peak charge is 1,120,814,520
bytes after warmup and 1,120,912,824 after uncaptured samples, a 96 KiB increase.

Before capture, the worker admits a conservative additional resource-version
estimate of 5,024,182,624 bytes: four command versions plus one working copy and
native overhead. Total estimated admission is 6,222,702,660 bytes, below 6 GiB
(6,442,450,944). This is an estimate, not a guaranteed maximum of Apple's private
capture allocator. Observed capture peak charge is 2,136,459,400 bytes. The Qwen
capture occupies 3,991,454,167 disk bytes; tiny occupies 5,012,447 bytes. Public
libproc charge counters are sampled only while the owned child is idle and do
not claim complete shared/clean physical-page accounting.

## Checks and decision

Default/Metal-feature formatting and strict all-target Clippy pass; ordinary
CPU suites pass 76/80 tests without requiring GPU/model availability. All 137
Python experiment tests pass, including eight collector artifact/failure tests,
four independent clock/interval analysis tests and the existing bounded pipe/
native ABI tests. The current release GPU suite with profiling enabled passes
all tiny decoder/session contracts and Qwen frozen comparisons: 612 arrays over
12 prompt/chunk cases, with unchanged source gates. Native measurements run
without overlapping builds/tests or large artifact hashing.

Retain the profiler for diagnosis. Astra accepted the task after independent code,
protocol, capture/source/binary hash and numerical evidence review, closing
functional M4. Astra identified and the regression fixes stale-profile access after command
creation failure; real overflow rejection and completion replay also pass. The
earlier integration failure/publication evidence remains valid; profiling does not add waits or
change commit boundaries. Protected CPU release binaries remain unchanged.

# Seven-tap scalar convolution history

Status: revised concrete entrypoints pass fresh checks and both original timing
runs. The first generic public boundary is rejected and archived. Astra approves
scoped FP32 convolution/history retention with no findings. Complete LFM3 runtime
and adoption remain unqualified.

## Shapes, arithmetic and ownership

`model::short_conv3` preserves its concrete `History`, `Prepared`, `commit_batch`
and `reset_batch` API. `model::short_conv7` exposes the corresponding seven-tap
API. Both use one shared implementation with bounded static specializations for
three and seven taps. Other kernel sizes reject during construction. Concrete
prepare methods instantiate the shared implementation inside the engine crate;
backend/profile selection does not occur in per-element loops.

Input/output are caller-owned `[tokens, channels]` FP32 slices. Immutable weights
are `[channels, kernel]` with oldest-to-current taps. History and each candidate
are `[kernel-1, channels]`, oldest-to-newest. Input is already raw `u = B*x`;
this component performs neither source BF16 rounding nor the separate `C` gate,
output projection or SiLU. The scalar numerical leaf is unchanged: increasing-tap
FP64 products/sums with one FP32 output cast. The original FP32 bound stays
`1e-5 + 1e-4 * abs(reference)`; committed raw history is checked bitwise.

Each history owns its fixed committed bank, identity, position and generation.
The caller supplies one candidate per owner and one reusable work bank. Prepare
validates the whole chunk, lengths, checked arithmetic, capacity, generation and
finite operands before writes; it never publishes committed state. An arithmetic
cast failure may leave scratch partial and returns no frame. Discard scratch and
retry from the valid prefix. Dropping an uncommitted frame leaves state intact.

The common identity allocator distinguishes owners across both profiles, including
matching physical bank sizes. A frame authenticates its owner, generation and
position. Batch commit validates every frame before any publication; missing,
duplicate, stale, foreign or mixed steps cannot partially commit. Reset retains
storage and invalidates old frames. Empty owner batches are no-ops; zero-token
prepare rejects, so an empty sequence makes no prepare/commit call.

A seven-tap bank at width 2,048 contains `6 * 2048 * 4 = 49,152` bytes. Count the
committed bank plus owner object, per-owner candidate and shared work separately;
weights, input/output and collections remain separate owners. No capacity grows
with position. Both concrete profiles report zero Rust allocations for 64
prepares, 32 batch commits, eight resets and eight rejected zero-token calls after
warmup. Seven-tap allocation checks use the model's actual width 2,048.

## Independent numerical evidence

The [exact-rational oracle](../tools/reference/short_conv7_fixture.py) decodes
FP32 bit patterns into fractions, sums exact products and rounds directly to
nearest-even FP32, including subnormal/tie controls. It imports no numerical
runtime or engine arithmetic. Five [frozen cases](../tests/fixtures/short-conv7-v1.json)
cover asymmetric/nonzero history, impulses, FP32 rounding, empty input and actual
BF16 coefficients widened exactly to FP32. The source subset is seven taps from
channels 0, 1, 1,023 and 2,047 of authenticated LFM3 layer0 convolution.

A separate [tensor reference](../tools/reference/short_conv7_tensor.py) uses
admitted Torch 2.9.1 CPU Float64 depthwise convolution, with explicit initial
history and `[channels,1,7]` layout. All 132 outputs match the exact oracle's FP32
bits; final raw histories also match. The runtime binds 20,228 distribution files
and versions. Its owned process closes below the separate 512 MiB audit guard
(256,016,384-byte conservative final RSS sum); this is not engine memory
admission. [Versioned reference receipt](../benchmarks/results/2026-11-10-short-conv7-reference-v1.json),
[raw tensor output](../artifacts/lfm3-seven-tap-2026-11-10/tensor-report-1.json),
[resources](../artifacts/lfm3-seven-tap-2026-11-10/tensor-checks-1.json).

Rust checks compare every scalar output/raw history and exercise 288 schedules,
including all 256 partitions of the authenticated nine-token case. They cover
whole-chunk prevalidation, late overflow/discard/retry, reset/replay, exact capacity,
foreign/cross-profile/stale frames, atomic mixed/duplicate/missing batches,
generation exhaustion and checked allocation arithmetic. The established
three-tap numerical fixtures and scalar convolution function remain unchanged.

## Performance experiment and console probes

Release probes measure 13 independent owners at width 2,048 with chunks 1/32/128.
The timed boundary is all prepares plus atomic batch commit; reset/six-row seeding,
loading, construction and printing are outside. Each cohort has three warmups and
ten measured samples; samples aggregate `max(1,64/tokens)` individual requests.
This is an operator baseline, not model throughput, TTFT or telemetry overhead.

The initial generic public prepare method passes correctness but misses the
unchanged 5% three-tap regression screen. Primary single-token comparisons change
from -9.71% to +13.78%; A/A variation reaches 11.19%. A frozen counterbalanced
confirmation still shows several 10–12% candidate regressions. It is rejected;
[source/binary snapshots](../artifacts/lfm3-seven-tap-2026-11-10/source-epoch-1/manifest.json)
and both [primary](../artifacts/lfm3-seven-tap-2026-11-10/performance-1/checks.json)
and [confirmation](../artifacts/lfm3-seven-tap-2026-11-10/performance-2/checks.json)
results are retained.

Symbol inspection finds a separate `finite` helper in the consuming crate only
for that generic boundary. The revised implementation keeps the loop private and
exposes ordinary concrete three/seven-tap prepare methods inside Toklet. Its
[symbol inspection](../artifacts/lfm3-seven-tap-2026-11-10/symbol-inspection-2.json)
no longer shows that separate helper. This is a measured hypothesis about the
compilation boundary. Both revised primary runs pass all six unchanged 5%
compatibility screens, with three-tap changes −2.33% to +1.26%. Scalar math,
fixtures, quality gates, warmups, samples and timed code are unchanged.

| Chunk | Seven-tap median, run1 / run2 | Three-tap change, run1 / run2 |
| --- | --- | --- |
| 1 | 0.3569 / 0.3575 ms | +0.54% / +1.26% |
| 32 | 9.3908 / 9.2370 ms | −0.60% / +0.38% |
| 128 | 36.7357 / 38.1361 ms | +1.09% / −2.33% |

These M1 Pro/16 GiB/macOS 26.6.2 operator measurements use matched release
Rust 1.90 builds, AC, Low Power Mode off and nominal Foundation thermal state at
every endpoint. Raw samples/dispersion and process resource records are retained.
Seven-tap final RSS peaks stay below 5.2 MB; allocated numerical array counters
are reported separately. Neither is full-model memory admission. No optimization
gain, 1% telemetry overhead or whole-model speed claim is made.
[Revised capture](../artifacts/lfm3-seven-tap-2026-11-10/performance-3/checks.json),
[frozen revision policy](../artifacts/lfm3-seven-tap-2026-11-10/performance-contract-3.json).

Run the probes with `cargo run --release --example short_conv3_history -- --tokens 32`
or `cargo run --release --example short_conv7_history -- --tokens 32`.
The baseline harness uses the previous reviewed engine, identical timed source,
flags and exact matching registry package versions/checksums. Early direct-cargo
and lock-graph preflights rejected before timing; six cached newer dependencies
were explicitly pinned to the candidate's versions. Initial failures remain in
[lock checks](../artifacts/lfm3-seven-tap-2026-11-10/baseline-lock-checks-1.json)
and the earlier build logs. Final measurements record AC/Low Power Mode/thermal
state, raw samples, named hardware and observed process memory.

## Qualification limits and next work

Fresh fmt, both strict Clippy configurations, CPU/Metal suites, explicit existing
Conv3 GPU regressions, native three-slice export/link/package and CPU/Metal C
smokes pass for the final code: 688 CPU, 834 Metal and nine example tests,
both strict Clippy configurations, two explicit existing Conv3 GPU tests and
native71 three slices with 46 exports each. Ordinary CPU/Metal suites retain
49/337 ignored explicit/hardware/real-model tests; this component does not claim
those broader qualifications. The independent oracle has four portable tests.
A sandboxed GPU attempt reported no device; the privileged revised-build repeat
passes both existing Conv3 tests. iPhone compile slices do not
qualify device execution. Initial moved-test imports/empty-slice inference errors
were corrected before targeted runtime checks.

This supplies FP32 raw convolution/history only. No seven-tap Metal kernel,
source BF16 activation profile, full LFM3 graph/codec/quality, 6 GiB model fit or
managed route is admitted. Reuse it in the independently qualified graph, then
validate sandwich/QK norms, per-head gates, partial RoPE and rotating attention
with absolute offsets. Keep full source/model/session checks and both original
10,000-target/100-task comparisons/caps before adoption. Engine/CLI work leads;
screen sharing and iPhone runtime work remain deferred.

[Fresh required checks](../artifacts/lfm3-seven-tap-2026-11-10/required-checks-2/checks.json),
[native package](../artifacts/lfm3-seven-tap-2026-11-10/native71-publication.json),
[C smokes](../artifacts/lfm3-seven-tap-2026-11-10/native-smoke-71/checks.json),
[benchmark CLI rejections](../artifacts/lfm3-seven-tap-2026-11-10/benchmark-cli-negative-1.json).

Astra independently regenerates the exact fixture, reconciles all 132 tensor
outputs, verifies current source/package/performance pins and recomputes all 18
cohorts/six screen comparisons. [Review](../artifacts/lfm3-seven-tap-2026-11-10/astra-review-result-1.json),
[commit/evidence](../artifacts/lfm3-seven-tap-2026-11-10/publication-checks.json).

The [next-task inspection](../artifacts/lfm3-seven-tap-2026-11-10/next-rotary-inspection-1.json)
identifies reusable full-head proportional rotary geometry and generic window
ownership. Their mathematical profile does not qualify LFM3's FP32 frequency/
angle construction and BF16 trig/activation rounding; independent source-profile
fixtures are the next mechanism.

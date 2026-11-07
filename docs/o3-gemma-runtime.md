# O3: concrete official BF16 runtime explanation and attribution

The selected Mac implementation passes the correctness, memory, native build
and measurement checks below. Astra approves after the native temporary-lifetime
fix and targeted worker stress test. Detailed collection stays opt-in; no1%
overhead bound or performance improvement is qualified.

`toklet plan --model PATH --route official-gemma-bf16` now selects the pinned
official Gemma 4 E2B BF16/G2 Mac route explicitly. The default MLX planner remains
available. The BF16 planner reads bounded, pinned source metadata and queries the
actual device/precise projection pipeline. It allocates no numerical weights,
KV, activation scratch or logits. Payload and codec authentication remain loading
requirements. MLX-specific numerical-policy flags are rejected on this route.

The report describes the gather/projection sequence using stable role/layer IDs,
physical source slots and shapes. Embedding gather and tied output-head projection
have distinct operation IDs and the same source owner. The recipe preserves CPU
normalization, rotary, shared-KV attention, GELU/residual/PLE work and commit rules.
G2 processes every row through producer layers and only the final row through
consumers/head. Reported prefill rows describe one full configured chunk, rather
than an entire request, cache action or observed dispatch sequence.

Command/dispatch/wait counts are predictions from concrete tiling. Active-output
host-byte estimates describe FP32 copy payload; they exclude status/guards and
are not measured DRAM or DMA traffic. Driver and codec allowances remain policies,
distinct from physical residency. Memory minima use the same accounting helpers
as construction, including KV proposals, generation wrapper and backend scratch.
Allocator overreservation can increase actual admission. Loading and session
creation revalidate; insufficient minima produce completed reports with false
fit flags. Session budget cannot exceed the load owner's budget.

Current release checks on the M1 Pro reconcile all 279 operations, 277 GPU
matrix owners, 349 tile owners and 262 vectors with live immutable owners.
Minimum and actual session admission both equal 5,814,073,992 bytes; whole-test
process physical/lifetime peak is 4,919,284,008 bytes, below 6 GiB. This includes
inspection, loading and session creation, rather than an isolated loading peak.
The earlier inventory receipt retains its earlier metadata layout and numbers.
[Current live reconciliation](../artifacts/o3-gemma-runtime-2026-11-07/inventory-validation-fixed/result.json),
[earlier CLI/direct equality controls](../artifacts/o3-gemma-runtime-2026-11-07/inventory-validation-final/checks.json).
The inventory describes the selected managed G2 policy; the raw public session
constructor also retains its full-row reference policy.

## Bounded committed-evaluation capture

`OfficialBf16::new_profiled_session` is explicit and selects G2. It charges 64 MiB
before counter allocation and preallocates at most 1,024 GPU events for one selected
committed evaluation. The opaque request ID is supplied by the caller; evaluation
ordinal is bounded 1..512. Each tile registers role/layer/source slot/tile/shape
before encoding, with actual projection command/dispatch/wait and active-copy
counts. Two CPU gather scopes and exclusive host phases accompany calibrated
GPU stages. GPU time overlaps host waits and is never added to exclusive host time.
Active-copy bytes exclude guard fill/status traffic and are not DRAM measurements.

Reset/history replacement invalidates a used capture. Failed or unselected work,
overflow and a poisoned session cannot export a successful frame. The ordinary
constructor keeps profiling absent; numerical shaders, precision and command/wait
sequence are preserved. No new steady-state event allocation is introduced there.
These Rust diagnostic APIs do not add native ABI/UI counters or claim other routes.

Two complete captures, prefill 15 and first decode 1, reconcile 349 events each with
stable identities/shapes and two CPU gathers. Every logit bit matches plain G2.
LatePLE/head failures, event overflow, reset, pre-start cancellation and a failure
after an earlier capture preserve the valid prefix and reject invalid/stale export.
Whole-test process peak 4,948,234,560 bytes; admission includes 64 MiB diagnostic and
16 MiB caller allowances. [Hardware control](../artifacts/o3-gemma-runtime-2026-11-07/profile-correctness-fixed/checks.json).

Native temporaries in `Trace::new` and `Trace::resolve` are scoped by local
`autoreleasepool` calls, so the public Rust API works without an app/thread pool.
A dedicated Rust worker performs 512 constructions and 16,384 exports after
warmup, with no observed footprint growth at its sampled checkpoints; lifetime
peak is 4,963,520,856 bytes. All 349 counter records remain valid after constructor
pools drain. This is observed lifetime coverage on the pinned Mac, rather than a
continuous allocation or general driver-memory guarantee.
[Worker lifetime control](../artifacts/o3-gemma-runtime-2026-11-07/profile-lifetime-fixed/checks.json).

## Checks and observed overhead

Fresh Rust 1.90 formatting and strict all-targets Clippy pass with CPU and Metal.
CPU tests pass 604, ignore 32; Metal tests pass 726, ignore 285. Explicit real-model
checks are separate. [Required gates](../artifacts/o3-gemma-runtime-2026-11-07/required-checks-3/checks.json).

Two independent paired cohorts each use three warm pairs and ten measured pairs,
with reversed starting order, identical 15-token prompt, 8 visible IDs plus stop 106,
context 512/chunk 32 and fresh sessions. They verify 24/23/1 history/committed/pending
state; pending stop makes logits unavailable. Full numerical parity is checked in
the separate evaluated-frame control above. Generation caller time is measured
separately from constructor/setup and profile export. The selected capture is
evaluation 2 only: first-token latency precedes that decode-frame instrumentation.
Observed median request changes are −0.28% and +0.98%; maximum lifetime peak across
these requests is 4,966,256,960 bytes. These observations do not establish a 1%
bound, other evaluation-ordinal overhead, summary mode, UI or tail qualification.

A second experiment compares prior `c3c1ecdb` and current default None execution
with an identical injected console harness: three warmups/ten measured requests
per process, two independent cohorts per version, counterbalanced order. Both
versions preserve exact IDs/text/state. Request-median changes are +0.22%/−0.09%;
TTFT changes +0.62%/+0.14%. These short observations establish no1% bound or
optimization gain. Across both experiments 104 requests pass, 80 measured, all
recorded AC/thermal 0. Profiling remains explicit/diagnostic.
[Raw inputs/samples/dispersion and checks](../artifacts/o3-gemma-runtime-2026-11-07/overhead-report-pooled.json).

The first shared release-target build reused the baseline executable for current
source; identical binary hashes/test registry exposed this before inference.
Separate targets and verified binaries fix it. The first paired test unwrapped
unavailable logits with a pending stop; it was corrected to assert that journal
invariant. Both failures and corrections remain in linked raw evidence.

Mac/iOS/arm64-simulator static libraries compile and link with all 46 C exports
each. Mac CPU and Metal C smoke tests pass on the authenticated tiny fixture.
[Native checks](../artifacts/o3-gemma-runtime-2026-11-07/native-checks-pooled.json),
[package receipt](../artifacts/o3-native48-2026-11-08/build.json). Existing generated
app links remain on their previous package; no device runtime qualification.
Document/evidence checks and final Astra approval pass.
[Review/commit receipt](../artifacts/o3-gemma-runtime-2026-11-07/publication-checks.json). No new kernel optimization, model adoption, UI/scanout,
sustained or device runtime qualification is claimed.

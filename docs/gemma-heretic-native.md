# Native mixed-format Heretic Gemma diagnostic

G1.B delivers an explicit Mac Rust diagnostic for the selected uncensored MLX
checkpoint. Required checks and Astra review pass for this implementation and
scoped G2 retention. Managed profile, codec/template qualification and model
adoption remain separate required work.

The source is the exact artifact in the [G1.A cold recipe](gemma-heretic-mlx.md):
`deadbydawn101/gemma-4-E2B-Heretic-Uncensored-mlx-4bit`, revision
`cde01efcb11f8a12e88a953b752fc16062aa1486`. Its 277 original affine4/group64/
BF16-metadata matrices and one original BF16 PLE projection remain unchanged.
No tensor is requantized or expanded into a second full FP32 model.

`resident::heretic::HereticContext`, `HereticWeights` and `Norms` separate
immutable weights from model references and session state. The existing
two-pass loader authenticates the complete captured container, then matches
every selected component while consuming it. The BF16 projection has its own
matched component and immutable GPU owner. Excluded media/unused-KV arrays are
hashed but not retained. Identity describes the historical whole-file anchor
and matched selected bytes; it asserts no later pathname or excluded-byte state.
This fixed single-file route consumes no shard index.

All names and logical-to-physical matrix slots resolve during loading. Each
session owns projection scratch for twelve affine shapes, two gather workspaces
and one BF16 workspace. Selection is a small match at a projection boundary;
there is no new indirect scalar/head/tile dispatch. Original precise Metal
kernels, guards, CPU graph, shared KV, causal masking, rotary positions, tied
head, pending-token and failure-prefix behavior are reused. Raw scoring returns
all rows. The G2 candidate computes only required last consumer/head rows during
generation, preserving producer work and committed cache state.

The public diagnostic limit is 512 shared positions, chunk capacity at most 32.
The 512-input/128-output and 1,024/128 cases exceed this limit and remain
unsupported; inputs are never truncated to fit the benchmark.
Load, norm and session budgets preserve the 6 GiB ceiling; caller/codec/app and
other live sessions require separate reserves. Per-session admission includes
the shared model once. It must not be summed as unique physical ownership across
sessions. Initial controls observe approximately 2.624 GB requested resident
source ownership and 2.954 GB single-session admission, apart from the 64 MiB
test caller reserve. The two-session fault/replay control's observed process
lifetime peak is 3.104 GB. These are scoped observations, not general model-fit
or memory-saving claims.

Load reporting separates metadata preparation, authenticated affine loading,
and BF16 read/validation/upload. The affine phase includes hashing, coefficient
validation, upload and vector conversion; it is not an isolated I/O timer.
Context device/library/pipeline constructor time is separate. The four final
new-process loads take 33.4–33.7 seconds. A new process does not establish cold filesystem storage.
Conditional isolated load peaks are reported only when that interval establishes
a new process lifetime high-water mark.

Request captures retain completed caller time, event TTFT, prefill/decode/
sampling phases, authoritative work/state and memory. Event decode rate is
`(emitted_events - 1) / (last_event_time - first_event_time)` when defined;
numerical decode evaluation rate separately includes hidden-stop work. Neither
is visible-text latency. Tokenization, final text decoding and renderer timings
are outside this bare-token track. Operator profiles, command/wait/copy counters,
steady-state allocation probes, energy and UI visibility are explicitly
unavailable rather than zero-filled. Captures are explicit public benchmarks,
with no routine per-token JSON or content logging in the engine.

Initial all-row and G2 controls authenticate all 1,094 selected components. All
15 prompt-logit rows are bit-identical across 15 versus 3/5/7 chunk schedules;
G2 last logits match the all-row reference. Fully cached prompts, pending samples,
hidden stop, pre-start/callback cancellation, output failure and admission checks
pass. An injected mid-graph failure preserves the committed prefix, poisons that
session and permits a fresh session to replay using the same immutable weights.
Independent shared-BF16 fixtures match the original projection and CPU oracle,
including guard corruption, reuse and exact budget rejection.

Final captures use the same source, FP32 activation/KV state, context/chunk,
prompt IDs, output work, stop policy and Reset semantics. On Apple M1 Pro with
16 GiB RAM, macOS 26.6.2, Rust 1.90 release/precise Metal and AC power, both
counterbalanced cohorts (three warmups, ten measured requests each) pass:

| Cohort | All caller | G2 caller | Caller reduction | All event TTFT | G2 event TTFT |
| --- | --- | --- | --- | --- | --- |
| 1 | 1.849 s | 1.365 s | 26.18% | 0.793 s | 0.305 s |
| 2 | 1.871 s | 1.381 s | 26.20% | 0.794 s | 0.306 s |

Observed cross-cohort median variation is 1.16%. All 52 requests preserve exact
15-token Reset input, eight emitted tokens and hidden stop 106. The final event
rate is roughly 7.45–7.59 tokens/s; no decode speedup is established. Each process
stays below 2.944 GB lifetime physical footprint. Earlier preliminary captures
remain separate after adding the last-token event metric.

Eight deeper trajectories include 128-input/128-output and 384/128 cases;
512 matched decode distributions and final prefill distributions are bit-identical.
Numerical prefill/decode time combined improves 16.9–19.0% and about 34.6%,
respectively; prefill improves 62.7–66.3%. One 128-input decode phase is 1.8%
slower, then 1.0% faster in confirmation. Single-row operations and shapes
coincide; the isolated variation remains in the report. No decode or tail bound
follows. Both sustained modes then pass at least ten minutes after three warmups:
322 All requests over 601.1 seconds and 434 G2 requests over 600.6 seconds.
Median caller time falls from 1.854 to 1.369 seconds (26.18%). Every output/state
check and all ten periodic host checks per mode pass; AC power and nominal
thermal state persist. Lifetime physical peaks remain below 2.947 GB. These
captures retain bounded diagnostic samples and do not establish allocation-free
decode or memory savings. The unchanged 5% performance gates pass and Astra
approves this scoped retention. No cross-engine ratio follows from different cache,
KV precision or timers.

Formatting, strict all-target CPU and Metal Clippy, 610 CPU tests (33 ignored)
and 742 Metal-feature tests (294 ignored) pass. Seven explicit GPU regressions
cover the new shared-BF16 fixture and existing affine/BF16/profile paths. All
three Apple arm64 library slices expose the same 46 C symbols and pass strict
link/package checks; Mac tiny CPU/Metal C execution passes. This package check
leaves the generated app link unchanged and provides no iPhone runtime evidence.

Full source fidelity and original/modified/quantized quality/adoption remain
required. Preserve the frozen 10,000-target NLL and 100-task gates, earlier
rejections, and distinct modified-model identity. iPhone remains deferred.

[Contract](../artifacts/gemma-heretic-runtime-2026-11-08/acceptance-contract.json),
[G2 hypothesis](../artifacts/gemma-heretic-runtime-2026-11-08/g2-reuse-hypothesis.json),
[initial all-row control](../artifacts/gemma-heretic-runtime-2026-11-08/control-initial/checks.json),
[G2/fault/replay control](../artifacts/gemma-heretic-runtime-2026-11-08/control-g2/checks.json),
[shared BF16 GPU fixture](../artifacts/gemma-heretic-runtime-2026-11-08/gpu-shared-initial/checks.json),
[final primary timing](../artifacts/gemma-heretic-runtime-2026-11-08/primary-report.json),
[deeper checks](../artifacts/gemma-heretic-runtime-2026-11-08/secondary-checks.json),
[phase assessment](../artifacts/gemma-heretic-runtime-2026-11-08/secondary-assessment.json),
[final release control](../artifacts/gemma-heretic-runtime-2026-11-08/control-final/checks.json),
[required checks](../artifacts/gemma-heretic-runtime-2026-11-08/required-checks-2/checks.json),
[sustained checks](../artifacts/gemma-heretic-runtime-2026-11-08/sustained-checks.json),
[GPU regressions](../artifacts/gemma-heretic-runtime-2026-11-08/hardware/checks.json),
[native package](../artifacts/gemma-heretic-native50-2026-11-08/build.json),
[full evidence](../artifacts/gemma-heretic-runtime-2026-11-08/final-report.json),
[checks/review/commit](../artifacts/gemma-heretic-runtime-2026-11-08/publication-checks.json).

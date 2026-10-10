# Selected Qwen Half Metal residuals, 10 October 2026

The [driver](../../tools/experiments/rust_metal_qwen_residual.py) applies the
unchanged [in-place residual primitive](2026-10-10-metal-residual.md) to width
896 using complete saved operands and unchanged frozen source gates. All 20
selected stages pass native qualification after acceptance/commit of the tiny
residual task. This adds no kernel, engine dependency or execution path.

## Selected evidence

The four frozen short cases have 4, 3, 4 and 1 token rows, including changed
future tokens and a nonzero position offset. Five saved operations per case
have complete operands:

| Output | Mutable state baseline | Immutable delta |
|---|---|---|
| Layer 0 attention residual | Embedding | Layer 0 attention projection |
| Layer 0 output | Layer 0 attention residual | Layer 0 down projection |
| Layer 1 attention residual | Layer 0 output | Layer 1 attention projection |
| Layer 1 output | Layer 1 attention residual | Layer 1 down projection |
| Layer 23 output | Layer 23 attention residual | Layer 23 down projection |

Layer 23 attention residual is excluded because the selected artifacts omit
layer 22 output. Other layers and composed forward/cache/session execution are
unqualified. Twenty stages supply 53,760 active values. Greedy bounded grouping
creates eight native inputs, each at most 8,192 values, without splitting rows
or losing cases. Three warmups and ten samples per stage give 260 positive
dispatches, 698,880 active outputs and 1,820 state guards.

The shared provenance checker pins all four full/committed Half/FP32 manifests
and config. Complete array shapes, payload hashes, original FP32 source anchors
and source gates must match the committed probes. Before target execution, each
stored Half output must equal independent FP32 addition of its Half operands
followed by one Half RNE. Native bits must match exactly, and complete original
FP32 arrays must also pass unchanged pre-GPU gates. No gate or reference changes.

No weights are needed for this operation. The driver reads only saved arrays,
small manifests and config; it calls no checkpoint hashing or loading function.
The frozen source checkpoint identity is retained as reference provenance,
without asserting the contents of a currently installed checkpoint. No complete
model weights, CPU/GPU model pair or Transformer execution is retained.

## Ownership and checks

State and delta remain separate contiguous `[rows,896]` Half allocations.
One lane owns each element, adds in FP32 and stores Half once. State restores
its complete baseline before every repetition outside GPU access. Queue,
pipeline and both allocations are reused, retained resources stay alive through
completion, and host readback follows successful status/error checks. Immutable
delta plus both buffers' seven physical guards are checked. Nonfinite results
are invalid/discarded; this task adds no rollback or session recovery claim.
Logical traffic is 5,376 bytes per width-896 row before cache effects.

At four rows, shared storage is 14,364 bytes and host baselines/readbacks are
28,728 bytes. Conservative native admission is 67,151,956 bytes including the
fixed 64 MiB runtime/fixture/diagnostic allowance. Unified allocations count once;
the engine's 6 GiB ceiling is unchanged. The preparation driver additionally
retains bounded selected arrays; native child physical peaks do not establish
preparation-process/system physical memory.

Fresh build, generated inputs and evidence must be disjoint and outside source
fixtures/model paths. Full native diagnostics are checked before compaction,
and full raw JSONL remains local/hash-bound. Final validation rechecks all
manifests/config, qualification helpers/drivers and all eight inputs before
declaring completion. The existing 39 add/tail dispatches and structured
shader/admission failures repeat. The native source is unchanged from tiny
residual qualification, whose RMS/embedding/overflow regressions remain evidence.

Five offline synthetic tests require no downloads or GPU. Distinct saved values
exercise each pairing, all 20-stage/eight-batch shapes and unchanged gates.
Source anchors, shape/hash/sum corruption, exact outputs versus original gates,
checkpoint-free execution/final helper/input checks and path preflight are
covered. All five tests pass. The isolated native package also passes formatting,
Clippy with all targets and warnings denied, eleven Rust tests and release
compilation. The unchanged engine passed formatting, Clippy and 75 Rust tests
in the preceding tiny task; its 77 offline Python checks plus these five give
82 covered checks, without rerunning the unchanged suites. Astra accepted the
completed task without findings after independently checking exact Half results,
unchanged FP32 gates, identities, memory and final evidence copies.

## Native evidence

The [native record](../../benchmarks/results/2026-10-10-qwen-metal-residual-v1.json)
contains all eight batches: 260 completed residual dispatches, 698,880 active
outputs and 1,820 state guards. Every Half bit matches the independent result.
The maximum unchanged FP32 gate fraction is `0.4831672705365675`; the maximum
absolute error is `0.854278564453125`. These are maxima over different stages;
the latter remains within its original scale-dependent gate. No numerical
policy changes or failed native attempts are needed. The 39 add/tail dispatches
pass, and shader/admission failures retain the required structured rejection.
No watchdog fires.

The largest selected-residual child physical peak is 15,630,960 bytes; the
maximum conservative native admission is 67,151,956 bytes. Native peaks for
add, invalid shader and insufficient budget are 5,898,840, 3,703,288 and
1,491,280 bytes. These are child-process measurements, not whole-system or
preparation-process peaks. This is a correctness check; diagnostic
encode/commit/wait intervals establish no performance qualification.

The binary SHA-256 is
`a83e6bc465e9cb7b274669d82d4a3eadd36f842541e426ce6a5667bb30795966`,
identical to the qualified tiny residual binary. The result SHA-256 is
`57e31c6894417d5280748db26ecd28be2d691a9b92cfb8762b1e29334c7814de`.
The full local native JSONL SHA-256 is
`0c61ec940a7253e6364c46faa63bd308a04fce061bc30a139a8ada7ab5fd29e3`;
the retained [stderr](../../benchmarks/results/2026-10-10-qwen-metal-residual-v1.json.stderr.txt)
SHA-256 is `affa72463bf0e83e7c3b11207b565f3e604e8df76e147feac59beb3f8f0a278d`.
The Qwen driver SHA-256 is
`4ee6564d1613fa98cadb78bc7ac82ff6076d7a36717d9d04af4feed7f0272009`.
The record additionally pins native source, all qualification helpers, all eight
input files, the four frozen manifests and config. Checkpoint identity describes
the frozen source, with no current checkpoint-payload verification claimed.

The [hold](../../benchmarks/results/2026-10-10-cpu-residual-check-window.json)
and [idle snapshot](../../benchmarks/results/2026-10-10-cpu-residual-check-window-idle.json)
apply to this following task too. Tiny acceptance and commit `8c89d68` preceded
Qwen execution. The final numerical driver had exited naturally; original
scalar worker 17701 remained asleep with unchanged CPU time `112:18.61`.
The [completion](../../benchmarks/results/2026-10-10-cpu-qwen-residual-check-window-completed.json)
rechecks qualified CPU source/receipt and frozen worker/CLI binaries. Only
orchestration process 34871 was then
[resumed](../../benchmarks/results/2026-10-10-cpu-qwen-residual-resumption.json)
at `2026-10-10T03:48:17.009446+00:00`, after all native and test work finished.
It completes the primary series and resumes the original scalar transport.
Numerical benchmark workers were never suspended. Documentation and final
review perform light reads/calculations only while CPU timing resumes.

Full residual coverage, composed decoder/cache/session failure behavior,
held-out quality, performance, GPU traces and M4 completion remain pending.

```sh
.venv/bin/python tools/experiments/rust_metal_qwen_residual.py --output benchmarks/raw/qwen-metal-residual.json --build-dir artifacts/qwen-metal-residual --input-dir artifacts/qwen-metal-residual-inputs
```

# M5.4 named mechanism evaluations complete

The four named evaluations and their task decisions are checked, reviewed by
Astra and committed. This closes the evaluation scope; quality and performance
release acceptance remain pending. Both fixed Q4 loss failures and the remaining
M5.6 evidence remain open.

| Mechanism | Decision | Evidence |
| --- | --- | --- |
| GPU greedy token ID | Retain explicit functional API and 1,215,448-byte host-admission reduction; reject latency adoption at 0.2890% slower. FullLogits stays default. | [Selection](2026-10-13-metal-greedy.md). |
| MLP SiLU/gating | Exact intermediate Half boundary and state checks pass; 0.006032% complete-request gain is below noise/5%. Remove prototype. | [Negative experiment](2026-10-14-mlp-gating.md). |
| Rotary/cache writes | Exact placement/rotary/storage/state checks pass; 0.014593% gain is below the 5% gate and Split dispersion. Remove prototype. | [Negative experiment](2026-10-14-rotary-cache.md). |
| Attention residual/post-RMS | Both Half boundaries and reduction modes are preserved; complete requests are 0.553034% slower. Remove prototype. | [Experiment](2026-10-14-residual-rms.md). |

Each experiment retains matched execution choices at its integration boundaries,
fixed model/token identities, bounded ownership/admission, raw completed-work
samples, numerical/refinement/state checks and Astra review. Unpromising
candidates do not merit confirming runs; removed prototypes retain bound local
sources/binaries for reproduction. No fusion delivers a retained request-latency
improvement on the named larger 512-input/128-output workload. Component traffic
or dispatch reductions do not override complete-request evidence.

## Bounded dispatch and scope

Existing operation-level enums choose projection, normalization and readback
outside numerical inner loops. Compiled pipelines are cached. Device/SIMD widths
are queried rather than assumed; the fixed eight-row compensated Q4 prompt tile
handles tails and explicit admitted shapes. There is no general autotuner or
unused fusion dispatch table. Rejected fusion variants leave live dispatch.
CPU/scalar, Original projection/normalization, FullLogits and chunk 128 remain
defaults. Compensated projection/RMS and Q4 tiles remain explicit experimental
options with the qualification limits in their reports.

Stochastic sampling is conditional in the plan and currently unnecessary;
greedy semantics, lowest-ID ties, finite validation and history/KV ownership
remain qualified. Full release workloads, application/task comparisons and the
remaining M5.6 acceptance evidence are separate. M5.2, M5.6 and overall M5 remain
open under the unchanged quantizer, preselected checkpoints, corpus and gates.

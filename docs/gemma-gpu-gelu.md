# Full-finite FP32 GPU GELU feasibility

15 November 2026. Artifact-only prerequisite for the [resident GPU block](mlx-gpu-performance-plan.md).
Astra approves scoped retention with no findings. No production activation or
ordinary model-route change is included.

## Contract and fixed layout

The unchanged [CPU reference](../src/cpu/gelu.rs) evaluates ordered FP64 cubic and
host tanh, rounds activation to FP32, then performs a separate FP32 multiplication.
The direct FP32 tanh candidate fails three amplified negative-tail families at
the original bounds. Its failed driver/numerical evidence remains linked through
[previous progress](../artifacts/mlx-gpu-gap-2026-11-15/g2-next-mechanism-progress-1.json).

The fixed successor has 138,241 records / 3,317,784 bytes (about 3.16 MiB).
Coarse raw-bit bins cover both signs; 8,193 fine records preserve the approved
negative-tail table byte-for-byte. Compact indexing assigns -8 exactly once.
Observed AFFINE / CONSTANT / IDENTITY / REFERENCE_ONLY modes have
60,565 / 32,820 / 32,056 / 12,800 records. Identity/constant modes avoid invalid
extreme-value margins. Unsupported or subnormal-sensitive arithmetic requests
reference replay. Native identity binds source, compiler, host/OS and math profile;
this is a certificate against observed native behavior, not ideal real tanh.

Generation and an independent verifier stream all **4,278,190,080 finite FP32
bit patterns**, including both signed zeros and subnormals. They retain bounded
records/digests rather than the entire grid. Actual generation/verification take
198.395 / 199.890 seconds under separate fixed 600-second, 512 MiB guards.
Those are certification timings; inference latency is unmeasured here.

## Admission, failure and replay

The precise GPU shader evaluates conservative intervals for the actual gate/up
pair, preserving the materialized activation and separate product boundary.
Uncertain arithmetic requests exact whole-case native replay after completion;
raw approximate values, pre-placeholder diagnostics and flags remain separate
from final publication. Genuine fatal input/output and guard failures preserve
caller output. A future block must replay its entire transaction before any KV
or token publication, including the measured fallback cost.

All gate/up pairs are not enumerated. GPU arithmetic assumptions, per-pair
admission and reference fallback remain explicit; GPU-only execution is not
qualified. A production owner must bind the approved profile/table cold and
reuse pipelines/buffers. Draft owners and the complete block remain unqualified.

## Checks and scope

All ten serial stages pass at unchanged limits. Original 42 cases retain
1,498,575 reference comparisons. GPU controls cover 616,546 mode/index/FMA points
in ten bounded chunks, 14 rounding points and 484 multiplication pairs; 201
subnormal-sensitive pairs require reference. Certificate and raw-reader
corruption controls reject.

Normal execution replays five boundary/adversarial cases: 139 trigger lanes
cause whole-case replay of 1,736 elements. All 24 saved original/future model
input cases avoid fallback. Forced replay checks all 34 successful nonempty
cases / 499,405 elements; publications match frozen FP32 activation/product bits
exactly. These are saved four-position prefill/decode inputs, not fresh deeper
own-input or long-generation qualification.

Largest observed runtime charge is 323,321,856 bytes sampled and 323,944,448
bytes for the separate terminal direct-pair calculation, including helper.
These observed scopes are not a continuous allocator cap or whole-model peak.
No request-speed, full-model quality/state, summary-overhead or external-engine
win follows from this artifact.

[Handoff/source pins](../artifacts/mlx-gpu-gap-2026-11-15/g2-gelu-full-domain-1/handoff.json),
[native certificate](../artifacts/mlx-gpu-gap-2026-11-15/g2-gelu-full-domain-1/certificate.json),
[independent native verification](../artifacts/mlx-gpu-gap-2026-11-15/g2-gelu-full-domain-1/independent-verification.json),
[all guarded checks](../artifacts/mlx-gpu-gap-2026-11-15/g2-gelu-full-domain-1/root-progress-1.json),
[normal raw replay](../artifacts/mlx-gpu-gap-2026-11-15/g2-gelu-full-domain-1/replay.json),
[forced raw replay](../artifacts/mlx-gpu-gap-2026-11-15/g2-gelu-full-domain-1/forced-reference-1/replay.json),
[combined review packet](../artifacts/mlx-gpu-gap-2026-11-15/g2-gelu-full-review-checks-1.json),
[Astra verdict](../artifacts/mlx-gpu-gap-2026-11-15/astra-gelu-full-review-1/verdict.json).

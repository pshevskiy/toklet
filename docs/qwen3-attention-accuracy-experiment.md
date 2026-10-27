# Qwen3 arithmetic diagnosis and opt-in attention experiment

Astra found **arithmetic-profile sensitivity, not a demonstrated native operator
bug**. A bounded same-input check then identified one concrete accuracy mechanism:
ordered FP32 attention reductions can cross Half rounding boundaries. The new
opt-in compensated attention reduces that local error on the GPU. It has not yet
passed the whole-model output contract; defaults and qualification stay unchanged.

## Why the previous gate did not identify a bug

The unchanged uniform0.05 normalized-log-probability bound was applied to all512
rows of the already-captured original case, with exact lowest-ID greedy agreement.

| Comparison | Failed rows | Entries above0.05 | Maximum error |
| --- | ---: | ---: | ---: |
| CPU FP32 / CPU FP64 |24|96620|0.21534818258089317|
| Native / CPU FP32 |19|72487|0.1526226727748039|
| Native / CPU FP64 |11|56034|0.20223375120124842|

All three greedy streams agree at every position. CPU FP32 itself fails the
chosen FP64/Half trajectory-fidelity requirement. That does not waive native
rejection or prove every operator correct; it shows this gate cannot distinguish
a bug from legitimate arithmetic-profile differences on its own.

The uniform bound constrains every possible target's NLL, including very unlikely
events. It is stronger than the unchanged0.05 **mean held-out loss increase**
policy. For the same reference, a uniform bound would also bound mean target-loss
change, but the converse does not follow. Here the quality test additionally uses
a different, unquantized baseline. Existing whole-model failures and weak absolute
task quality remain in their original reports.

Source review confirms native normalization, rotary/transcendentals and attention
use FP32 arithmetic where the independent FP64 profile widens operations before
the declared stores. Original MLX has further BF16/kernel-shape distinctions.
Switching storage to BF16 is not a demonstrated repair. For this checkpoint's
finite metadata range, BF16 scale times an eight-bit code is exactly representable
in FP32; ordinary affine product rounding is not the explanation either.

## Concrete mechanism and implementation

Saved native layer0 Q/K/V inputs at positions0,41,65 were authenticated and replayed
without model execution. At65, native exactly matches the ordered FP32 emulation,
yet differs from same-input FP64 after Half rounding. The emulation uses NumPy
transcendentals and is explicitly not a bit-exact Metal oracle.

`AttentionMode::CompensatedReference` now selects one additional kernel at the
operation boundary. It keeps the readable Reference implementation and default
factory selection. Layouts, causal mask, GQA mapping, KV/context Half storage,
FP32 score/probability stores, scratch admission and error handling are preserved.
The changes are Neumaier score and softmax-mass sums, an explicit product residual
for probability-times-value, compensated context summation, and the existing
round-to-odd intermediate before the Half store. FP32 sqrt/division/exp remain.
The added pipeline cannot lower the legacy scalar launch width.

The compact282624-byte fixture contains three Q rows and66 K/V rows from existing
captures. A separate accurate FP64 test oracle preserves the FP32 score/probability
boundaries. The fixed original primitive tolerance remains1e-4+1e-3|expected|.
Exact integer squared errors use2^-48 units; Half-bit mismatches also distinguish
signed zero. Neither metric is fitted after GPU execution.

## Measured local GPU result

All2048 context values per position are compared against the same frozen oracle.
The baseline is the actual unchanged GPU Reference kernel, not its CPU proxy.

| Position | Half mismatches, Reference→candidate | Squared-error units, Reference→candidate |
| --- | ---: | ---: |
|0, exact control|0→0|0→0|
|41|9→2|27744→272|
|65|7→1|1329169→1024|

Both positions improve both metrics; the pooled result is16→3 mismatches and
1356913→1296 squared-error units. Independent raw-record readback reconstructs
those integers and matches the expected Half bits against the separate NumPy
FP64 diagnostic.

Three explicit GPU tests pass: causal/GQA/offset/chunk/scratch guards; exact zero
and one-key controls; cancellation and signed-midpoint witnesses; nonfinite status;
and the own-input comparison. Raw actual/expected Half records precede numerical
assertions, so failed trials retain values even if later metrics cannot print.
Required formatting, strict default/Metal Clippy, CPU suite and two targeted CPU
oracle/planning checks pass. Model execution has not occurred in this task.

The largest GPU-test process physical peak is24592984 bytes/RSS37928960 bytes.
These are process resources, not a benchmark or whole-phase compiler footprint.
The initial saved-logit triangle has no measured lifetime peak and is diagnostic
evidence only; the separate attention reader does have measured resources. No
resource qualification is inferred from that missing observation.

## Decision and next bounded task

Retain the candidate as an **opt-in accuracy experiment** with demonstrated local
benefit. This neither proves attention caused the final logit failures nor grants
whole-model qualification. The ordinary app stays on its existing experimental
profile and selected512 positions.

Next capture only complete logits for the already-used original/future-tail512
cases and apply the unchanged0.05/exact-greedy screen once. Review and freeze that
capture before execution. Rejection stops this experiment; it does not trigger
another automatic layer investigation. The eight untouched validation inputs
remain closed unless the frozen candidate passes and receives review. Full
quality must rerun before adoption; the old result belongs to old candidate bytes.

See the [delivery plan](qwen3-mac-delivery-plan.md),
[historical rejection](qwen3-output-screen.md) and
[evidence](../benchmarks/results/2026-10-27-qwen3-compensated-attention-v1.json).

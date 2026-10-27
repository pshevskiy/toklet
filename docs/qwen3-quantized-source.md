# Qwen3 original quantized source: completed experiment and decision

Q1/Q2 of the [Mac delivery queue](qwen3-mac-delivery-plan.md) are complete.
Astra's decision is to **retain native Half unchanged as an experimental
candidate and reject its current C512 qualification**. There is no BF16 switch,
capacity increase, adoption or tolerance expansion. Historical numerical
failures and the rejected calibration envelope remain rejected.

## Original source and capture

The original `mlx-community/Qwen3-0.6B-8bit` checkpoint at
`11de96878523501bcaa86104e3c186de07ff9068` runs through the installed, unchanged
MLX-LM Qwen3 loader/graph: strict loading, no dtype override or adapters, GPU,
batch1, offset0, no persistent cache, evaluation mode. Its704 parameters and
197 quantized modules are inventoried. Original checkpoint, config, runtime,
implementation and input hashes are retained. No downloaded repository code runs.

An isolated MLX-only environment copies5270 authenticated files/322305302 bytes
from existing packages, omitting Torch code and metadata. The original environment
is unchanged. The first cold import rejected an unadmitted Rich dependency;
the second detected that Transformers5.7 ignores the legacy Torch-disable flag
and imported Torch. Both failed attempts remain retained and executed no model.
The successful environment uses exact copied libraries rather than a patched
availability check or modified numerical implementation.

Two fresh processes execute all six frozen32/128/512-token calibration cases.
Each produces204201984 **BF16** logits,1225211904 bytes including exact FP32
widening. Every raw and widened array agrees bit-for-bit between processes.
Two separate bounded readers verify every value's finite status, bit-derived
widening, complete hashes and descriptor/path endpoints. BF16 output storage is
observed; this does not assert every internal accumulation dtype.

On Apple M1 Pro/16GiB/macOS26.6.2(25G83), capture peak physical footprints are
1380353512 and1494926824 bytes, below6GiB. The complete readback peaks are
314786632 and315179848 bytes, below512MiB. Cold source tests run23 cases;
comparison runs4 targeted tests. Every owned capture/readback/comparison group
exits and its process group is absent. This is correctness/extraction evidence,
not a latency benchmark.

## Saved-data comparison

The comparison authenticates the source and both unchanged CPU Half-storage
references. It evaluates1344 full distributions and1338 next-token targets
using stable FP64 log-sum-exp/reductions. The last row of each case has no
next-token target. Nested prefixes overlap; these authored calibration inputs
are not independent held-out quality samples. No native graph is executed here.

| Calibration case | Source diagnostic NLL | CPU FP32-arithmetic NLL | CPU FP64-arithmetic NLL | Source/FP32 mean TV | Argmax differences |
| --- | ---: | ---: | ---: | ---: | ---: |
| prose32 |4.826557|4.832249|4.832041|0.021654|0|
| prose128 |2.804595|2.803867|2.803884|0.012095|0|
| prose512 |0.721545|0.720007|0.720015|0.002888|0|
| code32 |2.142257|2.152109|2.151489|0.011658|0|
| code128 |1.782495|1.776988|1.776860|0.008050|1|
| code512 |0.452579|0.454263|0.454270|0.002564|2|

Target-weighted diagnostic NLL is1.0452698915 source,1.0450938619 CPU FP32
and1.0450699205 CPU FP64. Row-weighted source/CPU mean total variation is
0.0047885872/0.0047896952. Both KL directions and every row's greedy/runner-up
margin, target loss and maximum-error coordinate remain in the raw journal.

The three source/CPU argmax differences cover two unique code positions.
Position69 is repeated across nested cases and has an exact BF16 source tie.
At position60, the source itself changes greedy choice between128- and512-token
executions of the same prefix. It cannot simply become an exact schedule-independent
token oracle. CPU FP32/FP64 argmax agrees on all1344 rows.

At prose position317, source/CPU maximum logit differences reach2.03912 and
2.85782, while total variation is about1.2e-6 and2.5e-6. Large absolute maxima
alone poorly describe distribution effects. Mixed-sign diagnostic NLL differences
establish neither quality loss nor quality preservation. Correctly rounded
selected primitives and previous exact cache schedules do not establish complete
native numerical qualification either.

## Decision and finite next step

Astra reviewed the actual source and comparison receipts. There is no evidence
here justifying conversion of the native graph to BF16 or widening its numerical
gates. Keep the Half implementation as an experimental candidate, preserve the
failed C512 qualification, and stop layer localization/envelope derivation.

The next code task is explicit **diagnostic-only1025-position Qwen3 execution**,
with bounded memory and independent boundary/state checks. App limits stay
unchanged. The same quality milestone includes pinning the unquantized baseline,
checkpoint-specific tokenization/rendering and the complete10000-target/100-task
suite under existing0.05-nat/2-percentage-point rules. Freeze the candidate and
quality policy before scoring. Quality determines retention/rejection; it cannot
fit tolerances. Any subsequent independent numerical contract requires review,
freeze and untouched numerical validation before qualification.

The [frozen source metadata](../tests/fixtures/qwen3-quantized-source-v1/README.md)
and [public receipt](../benchmarks/results/2026-10-27-qwen3-quantized-source-v1.json)
retain exact identities, commands, source provenance, memory and calibration
metrics. Large captures/runtime inventories remain in ignored artifacts.

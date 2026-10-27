# Qwen3 compensated attention: complete C512 screen rejected

The one frozen whole-model screen **rejects** the opt-in compensated-attention
candidate. Local attention accuracy improves, but that benefit does not establish
complete-model fidelity. No default, reference, threshold, context limit or
qualification changes. This experiment stops here.

## Complete result

Apple M1 Pro,16GiB, macOS26.6.2; pinned
`mlx-community/Qwen3-0.6B-8bit` revision
`11de96878523501bcaa86104e3c186de07ff9068`. The candidate uses MlxAffine projection,
Compensated normalization and CompensatedReference attention, Half activations/KV
and FP32 logits. Engine revision `261f836`; the receipt pins the added capture
test and exact debug binary. This is accuracy evidence, not a latency benchmark.

All151936 vocabulary entries at all512 positions are compared for both the
already-used original and future-tail cases. The independent CPU FP64/Half
reference and inclusive0.05 normalized-log-probability bound remain unchanged.
Exact lowest-ID greedy agreement is also required.

| Per-case result | Previous Reference attention | CompensatedReference |
| --- | ---: | ---: |
| Rows above the bound |11/512|20/512|
| Vocabulary entries above the bound |56034|137559|
| Maximum error |0.20223375120124842|0.1644533777242394|
| Greedy disagreements |0|0|

Both cases have the same aggregate failure counts. The candidate's worst entry
is position45, vocabulary27921. Its smaller maximum accompanies more violations;
it is not an overall fidelity improvement or an accepted replacement. Matching
all1024 greedy selections does not waive distribution errors.

## Capture and verification

One explicitly selected ignored GPU test captures four128-row logit frames per
case:311164928 bytes each,622329856 bytes total. Existing layer-capture helpers
and old tests are unchanged. Only logits are persisted; the existing diagnostics
also allocate embedding/final-norm snapshots. Exact frozen histories,512 committed
positions, zero pending tokens, and the unchanged first511 rows across the future
perturbation pass. Full parent/frame/consumed-reference hashes and2004 reader
endpoint checks pass. Partial results are retained on failures as well.

Astra reviews the capture and frozen runner before the single model execution,
then reviews its results. Formatting, strict Metal Clippy,369 CPU tests and
integration compilation pass. Six owned preparation/capture/reader groups are
absent afterward. Native process physical peak971916464 bytes/RSS983875584 bytes
fits the existing6GiB bound; the reader records56394424/119242752 bytes within
512MiB. These are process observations, not full-window admission or throughput.
The first preparation attempt correctly rejects a runner edit during checking;
the successful frozen preparation and narrow reviewed argument fix are retained.

## Astra's resolution of the hard question

CPU FP32 also fails this uniform FP64/Half trajectory requirement despite matching
greedy IDs. The evidence demonstrates arithmetic-profile sensitivity; it does not
prove a native defect or excuse this rejection. Against the same reference, a
uniform loss bound implies a mean target-loss bound. The converse is false, and
the old full-quality comparison additionally uses a different unquantized baseline.

The present Q4 promise requires every token probability ratio to remain within
`exp(±0.05)`, including negligible tails. Astra recommends considering a separately
specified native precision profile for the Chat/Expert product, qualified through
independent operator/state conformance and the unchanged full-quality policy,
while reporting uniform fidelity separately. That is a proposed versioned product
contract, **not an applied waiver**. The present Q4 requirement remains rejected.

No further operator search is justified solely by this failure. The eight unused
validation inputs remain unopened; no new full-quality result, default adoption,
larger context or iPhone qualification is claimed. The ordinary experimental
C512 Mac app remains usable. See the [delivery plan](qwen3-mac-delivery-plan.md),
[local experiment](qwen3-attention-accuracy-experiment.md) and
[frozen evidence](../benchmarks/results/2026-10-27-qwen3-compensated-output-screen-v1.json).

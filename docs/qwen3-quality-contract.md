# Complete Qwen3 quality contract and execution worker

The [active Q3 task](qwen3-mac-delivery-plan.md) now has frozen complete inputs,
an explicit native diagnostic execution worker, and a verified unquantized
baseline payload. Subsequent [complete scoring](qwen3-full-quality.md) passes the
unchanged relative policy: loss increase0.00400406 nats/token and tasks3/100
versus5/100, exactly the two-case limit. Absolute strict quality is weak.
Numerical C512 qualification stays rejected; app and ordinary-loader limits stay512.

## Candidate and baseline

The candidate remains `mlx-community/Qwen3-0.6B-8bit` at
`11de96878523501bcaa86104e3c186de07ff9068`, packed8/group64 weights, Half
activations/KV, Compensated normalization and Reference attention. The frozen
manifest authenticates actual Rust/shader/Cargo/worker source bytes. Scoring
must check those exact bytes, the built executable, model payload and endpoints;
a changed implementation cannot silently inherit this contract.

The unquantized baseline is official `Qwen/Qwen3-0.6B` revision
`a9c98e602b9d36d2a2f7ba1eb0f5f31e4e8e5143`, the official revision immediately
preceding the converted weight upload. Its1503300328-byte BF16 file has SHA-256
`f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b`.
The downloaded file matches the [pinned official inventory](https://huggingface.co/api/models/Qwen/Qwen3-0.6B/revision/a9c98e602b9d36d2a2f7ba1eb0f5f31e4e8e5143?blobs=true).
Its digest also matches the later official revision. The converter's card names
the source repository but omits a source revision: contemporary history is
supporting evidence, not a revision explicitly recorded by the converter.

Candidate and baseline configs are exactly equal after removing the candidate's
two quantization dictionaries. The tokenizer JSON is identical. Both profiles
will consume the same frozen IDs/rendering/reset/stops, bypassing differing
tokenizer-config wrapper defaults. File acquisition is not reference execution
or model-fit qualification. Original quantized-source parity remains distinct
from quality against this unquantized model.

## Unchanged full workload

The original pinned WikiText-2 raw test corpus supplies the first10001 Qwen3
token IDs without added BOS. Score all10000 targets in ten independent blocks:
nine1024-target blocks and one784-target block. Each block includes its preceding
token, resets positions to zero and overlaps only that preceding token. Aggregate
by targets, not an unweighted mean of block means. Capacity1025 is required.

The100 equally weighted cases are first40 HumanEval, first30 TruthfulQA and
first30 English LAMBADA rows. Code/prose use the exact pinned plain Qwen3
`enable_thinking=false` template, with the empty thinking prefix and no invented
system/BOS. LAMBADA is raw. Output caps remain256/32/16. Greedy ties select the
lowest token ID; EOS151645 is hidden but consumes a sampled output slot.
Code is appended exactly without fence repair, prose uses exact trimmed answer
letters, and continuation uses the first exact case/punctuation-sensitive word.
Generated code requires the existing isolated grader, after collection exits.

The largest request is HumanEval/32:289 input+256 output=545 shared positions.
Inputs and requested outputs are not shortened to fit the512-position app.
All selected sources/cases stay excluded from calibration/tuning. Mean loss
increase must be at most0.05 nats/token and task pass-rate loss at most2 percentage
points against the unquantized baseline. These are frozen before scoring;
held-out results cannot fit tolerances or numerical contracts.

## Implementation and actual checks

[Preparation](../tools/bench/prepare_qwen3_quality.py) authenticates datasets,
checkpoint/tokenizer identities, baseline config equivalence and complete
workload geometry. A model-independent policy module preserves existing recipes
while the historical Gemma producer remains frozen. Pinned Jinja and a manual
plain renderer must agree. The [independent checker](../tools/bench/check_qwen3_quality.py)
reconstructs case selection, answers, rendering, IDs, execution requests and
all loss targets, reading Parquet in different batches. It separately reopens
baseline config/tokenizer assets and checks the entire candidate descriptor.

[Rust replay](../examples/qwen3_quality_contract.rs) checks independent plain
rendering and raw tokenization for all100 prompts/10001 loss IDs. All70 code/
prose prompts also pass the actual app Chat codec. The full request budgets
still belong to the diagnostic worker; app limits are unchanged.

[Native execution](../examples/qwen3_quality.rs) supports `loss` and `tasks`.
Loss rejects any input count other than10001 before opening the model; it reads
every causal row and scores every target once using the existing stable FP64
reduction. Tasks require all100 cases with40/30/30 categories, original output
caps,1025 shared capacity and fixed EOS, resetting state between cases. Numerical
kernels remain unchanged. Snapshots and bounded CPU copies are charged by the
existing diagnostic planner. Ready records preserve known C512 rejection and
do not declare quality qualification.

An explicit `probe` mode accepts only the original four-token witness and labels
its results non-quality. Its three target positions/logits and losses agree
within checks derived from the original short full-logit gate. The loss bound
is twice its maximum absolute logit allowance, from log-sum-exp's Lipschitz bound.
This is short target/loss consistency, not a full native-logit comparison, new
numerical contract or held-out quality measurement.

Four cold input controls, two portable Rust protocol tests and the five-test
Metal example suite pass. Twelve coherent corruptions reject changed IDs,
source tests/answers, output/quality limits, candidate capacity/precision,
baseline descriptor and baseline assets. Two independent Rust corruptions
reject changed loss/prompt IDs. Full loss rejects the short probe before model
loading. Cold runtime/source/model endpoints and all11 process groups close.

On Apple M1 Pro/16GiB/macOS26.6.2, preparation/checker/controls peak physical
memory is255968648/222168312/222168408 bytes; maximum RSS493469696/522092544/
657915904 bytes. Rust codec peaks191677040 physical/251215872 RSS. The short
Metal probe peaks946259096 physical/430686208 RSS, below6GiB; its logical session
plus separate application allowance admits1144712236 bytes. These are bounded
input/probe observations, not full-quality memory or latency benchmarks.

See [frozen metadata](../benchmarks/qwen3-quality-v1/README.md) and
[public evidence](../benchmarks/results/2026-10-27-qwen3-quality-inputs-v1.json).
The subsequent complete comparison executes both profiles on every target/task,
resolves all80 isolated code runs and restores the VM before retaining the
experimental candidate. Frozen preparation flags remain historical facts and
are not rewritten. A subsequent precision contract still needs review and
untouched numerical validation before C512 or app-window qualification.

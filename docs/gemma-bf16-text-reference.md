# Original BF16 short mathematical graph reference

This reference composes the admitted
[original BF16 row-backed weights](gemma-bf16-reader.md) with the existing
[independent graph operators](gemma4-text-math.md). It prepares complete short
graph evidence for the unquantized baseline. Independent full-graph agreement,
upstream BF16 runtime behavior, model fit, quality and app/full-window support
remain separate requirements.

## Arithmetic and ownership

The [weight adapter](../tools/reference/gemma4_bf16_weights.py) resolves all 278
matrices and 262 vectors from the fixed official layout. Matrices borrow one
authenticated original-file owner; only 1143948 bytes of direct finite FP32
vectors stay expanded. Sorted vector loading and every later gather/projection
read are included in a source-read chain frozen before execution. No complete
expanded model is retained, and no numerical operator is changed.

The arithmetic profile is
`original-bf16-weights-fp32-storage-fp64-reductions-decimal80-rotary-v1`.
BF16 weights expand exactly into FP32. The two contracts use either FP64
activations or explicit staged-FP32 operator boundaries, both with existing
FP64 reductions and independently defined Decimal80 rotary angles. This is
distinct from upstream BF16 arithmetic and does not waive its remaining gaps.

The [producer](../tools/reference/gemma4_bf16_text_reference.py) streams all 713
states for each contract at IDs `[2,100,257,100]`, then repeats with only the last
ID changed to 101. Original and future files each contain 103403520 bytes;
the complete four-forward capture is 206807040 bytes and 2852 typed states.
States emitted by the graph are saved before worker comparison gates. The
unchanged graph itself rejects nonfinite internal states before its sink;
capture of such an internally rejected state is not promised.

## Admission and independent checks

The [stdlib checker](../tools/reference/check_gemma4_bf16_text.py) uses only the
old immutable fixture's state names, order, shapes and dtypes. Packed-model
values and historical failure counts are not original-weight expected values.
It prepares the complete original-source consumption chain before graph launch,
then verifies saved coverage, hashes, finiteness, prediction/tie behavior, all
causal prefixes and every cross-precision statistic. Cross-precision failures
remain visible diagnostics under the unchanged `1e-5 + 1e-4*abs(reference)` gate.

Twenty-four selected scalar checks derive embedding/PLE rows, embedding
scaling, first input RMSNorm and first K projection directly from official bytes.
The softcap checks independently recompute the operation from saved head values;
they do not independently reconstruct the head. The 1060 consumed
source excerpts occupy 3493888 bytes, saved before their comparisons. These
checks strengthen selected boundaries without claiming an independent oracle
for the entire graph. Two cold captures must have identical complete array files.

The 512 MiB diagnostic ledger reserves 320 MiB for admitted runtime/BLAS,
64 MiB for tile/conversion overlap, 32 MiB for active graph state, 48 MiB for
head/logit/capture overlap, 24 MiB for streamed comparison, 4 MiB for vectors/views,
8 MiB for metadata/authentication and 12 MiB margin. The source reader's bounded
raw tile, FP32 expansion, FP64 projection conversion and prior tile can coexist.
Sink `tobytes()`, prediction partitioning, saved comparison arrays and runtime
endpoint verification are included. Diagnostic arrays are released before the
next graph or comparison. Runtime files and all actual loaded origins are
admitted separately. Measured physical and RSS peaks must both fit the policy;
this is not whole-model or phone fit qualification.

Eight existing independent graph methods and three new
[composition/failure tests](../tools/reference/test_gemma4_bf16_text_reference.py)
cover complete tiny graphs, softcap tails, geometry, ownership, source-free
composition and retention of raw states on error/interruption/endpoint failure.
Seven [coherent saved-capture controls](../tools/reference/test_check_gemma4_bf16_text.py)
exercise order, shape, read audit, prediction, nonfinite values, causal prefixes
and a selected independent source boundary. They modify copied captures rather
than source checkpoint files.

## Actual reference preparation, 26 October 2026

All 11 offline methods pass. Two cold captures produce identical complete raw
files: original SHA `0e175fccc78fd76e02234a35fe72d0afafd5ad61a05a231047ed5e663ec67793`,
future SHA `8e2e2b33918c6a9e78f291f2db5b957b36601b7d743304d4808f17bca43a4b88`.
Both match the pre-execution 38182-read/18236693062-byte source audit, chain
`0734f8d8b7813aeb726890eb04bf25d7b663a19a101c9e26b5c49dcef1100b67`.
The 707199-byte plan SHA is
`04be1a0f1134441aabacb14de6b688fe4c81e97a01cf5d0919e2d0cd37397dd3`.

Each independent replay checks all 2852 states, 1426 causal prefixes and 24
selected scalar boundaries. All prefix ratios are zero; the largest selected
boundary ratio is approximately `1.31e-9`. All seven coherent controls reject
their intended errors. These are capture, causal and selected numerical results;
independent full-graph agreement remains false.

Under the unchanged cross-precision gate, each capture retains 11 failed values
across seven sequence/state arrays. The original sequence has two failures:
one each in layer-14 post-feedforward norm and layer-31 Q projection. The changed
sequence has nine failures across layer-14 post-attention, post-feedforward and
post-per-layer norms, and layer-16/31 Q projections. The worst error-to-bound
ratio is `2.106351505638157`. Complete original values and statistics remain
available; no gate is widened and earlier packed/BF16/GPU/rotary failures remain
unwaived. Head/logit comparisons pass only for these frozen four-position cases.

Both arithmetic contracts predict `[236761,106,1018,106]` for the original raw
sequence. This is per-position argmax evidence, without Chat, generated replies
or quality scoring. It is separate from the earlier four-bit model's values.

On Apple M1 Pro / MacBookPro18,3 / 16 GiB / macOS 26.6.2 (25G83), the two graph
workers peak at 295470136 and 291013640 physical bytes, with RSS301760512 and
381485056 respectively. Both fit 512 MiB, as do the offline, plan, independent
replay and corruption-control workers. Timings include reference instrumentation
and do not qualify generation performance. The plan and complete traces remain
bound by the [public record](../benchmarks/results/2026-10-26-gemma-bf16-text-reference-v1.json).

## Remaining scope

This four-position reference does not qualify the whole 32-position diagnostic,
the frozen 1025-position quality workload or the model's full shared context.
Next separate numerical context capacity from bounded projection/chunk scratch,
then execute the complete frozen quality requests without shortening prompts,
targets or output caps. Resident execution, existing source/precision failures,
model-specific full input/output windows, app integration and independent phone
feasibility remain open under the [active plan](gemma4-implementation-plan.md).

The subsequent [native host scratch mechanism](gemma-chunk-scratch.md) addresses
the Rust allocation prerequisite separately. This full-prefix Python reference
and the native total context still retain their existing32-position bounds.

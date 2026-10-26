# Full-sequence tiny context oracle through 1025 positions

This prepares an independent mathematical reference before implementing a
cached reference or increasing native Gemma's32-position executable ceiling.
It follows [session context/cache/history ownership](gemma-session-capacity.md)
and supports the [unchanged quality-input demand](gemma-quality-contract.md).
The tiny four-layer WITNESS retains its local window4. It is not numerical
evidence for real E2B's window512, original BF16 graph, quality, app or device fit.

## Independent algorithm and bounds

The historical [batch evaluator](../tools/reference/gemma4_text_math.py) remains
byte-identical, SHA256 `6c04a81b93c95edfad4a4de646da3731b71a6dd175163fb856d2009bd103b096`.
Existing authenticated head/replay tools continue using that source. The
[long batch producer](../tools/reference/gemma4_long_witness.py) is a small,
explicit copy of its arithmetic, with named tiny weights and total positions
restricted to1–1025. It has no incremental KV session, native cache dependency,
tokenizer or checkpoint loading path.

Projection and gather calls remain at most32 rows through the existing bounded
row helpers; larger full-sequence activations are batched without materializing
a complete expanded weight matrix. Casts, epsilon, FP64 products and `math.fsum`
reductions, stable softmax, direct normalization weights, GELU, softcap and
Decimal80 rotary conventions remain unchanged. Computing `max(scores)` once per
query removes repeated identical work without changing arithmetic.

The complete sequence is evaluated at absolute positions0..1024. Its first32
IDs are the original frozen tiny sequence; subsequent IDs follow `(7*p+3)%17`.
A second sequence changes only the final token. Each has90 states in each of
the mathematical-FP64 and staged-FP32 contracts. Both use reconstructed FP32
weights. Long calls require a streaming sink rather than retaining every state.

The [stdlib plan/checker](../tools/reference/check_gemma4_long_witness.py) freezes
shapes, precision, offsets, token IDs and scope before execution:109051800 bytes
per sequence,218103600 bytes total per capture,360 complete state records.
Source data comes from the existing independently frozen packed tiny fixture.
The512 MiB lifetime ledger reserves320 MiB runtime/BLAS,64 MiB conversion tiles,
64 MiB active graph arrays,16 MiB capture,16 MiB comparisons and32 MiB for
metadata, immutable tiny weights and margin. File authentication streams64 KiB
blocks; it does not load each109 MB capture into memory.

## Verification policy

The [offline tests](../tools/reference/test_gemma4_long_witness.py) compare every
32-token state against the existing frozen contracts; a33-token call checks
projection/gather tails and its unchanged causal prefix. Invalid bounds, real
profiles, missing sinks and sink failure are checked before long collection.

Two cold full captures must have identical files. Independent replay checks
complete state extents/hashes/finiteness,360 frozen first32-prefix comparisons
and180 unchanged1024-position future-prefix comparisons. Cross-precision
differences are reported with the original numerical gate and never waived by
a capture-preparation pass.

Forty-eight selected scalar checks independently reconstruct complete embedding,
embedding scaling and PLE lookup rows from the frozen source, Decimal80 rotary
and causal/window attention at positions511/512/1024 for all four layers and both
contracts/sequences, and complete head-to-softcap behavior. Attention checks
resolve shared producers explicitly and preserve W4 masks. These are selected
operator boundaries: rotary uses captured q_norm, attention uses captured Q/K/V,
and softcap uses the captured head. Embedding/PLE/scaling derive from frozen
source weights. Full independent end-to-end1025-position graph agreement remains
separate from these local recomputations.

The compact fixture retains complete original/future logits and original
producer K/V states, selected original intermediate positions, and final future
intermediates. Its descriptors retain original shape and selected positions.
The complete captures remain separate immutable local evidence. Future cached
reference/native code must consume these frozen values rather than supply its
own expected outputs.

[Eight coherent corruption controls](../tools/reference/test_check_gemma4_long_witness.py)
cover shape, missing state, false execution claims, short and future prefixes,
late rotary/attention and softcap. Numerical controls update state/file/manifest
hashes before replay, exercising numerical gates rather than checksum rejection.
Exact word patches and resulting identities preserve reproducibility without
keeping another218 MB private copy for every intended failed control.

## Actual collection, 27 October 2026

All12 offline methods pass: eight existing graph checks and four new admission,
parity, batch-tail and sink-failure checks. Two cold full captures have identical
original/future files. Each independent replay checks360 states,1134848 frozen
short-prefix values,18157568 unchanged causal-prefix values and1662532 selected
scalar boundary values. All48 boundaries have zero measured difference. The
largest short-prefix error-to-bound ratio is`7.260697463525155e-10`; the changed
future token leaves its complete preceding prefix exactly equal numerically.

Across18175300 cross-precision comparisons, neither tiny sequence has a failed
value under the unchanged gate. The maximum ratio is`0.9539444607715041`, at
`layer.3.context`. This is retained evidence, without widening a tolerance or
closing the eleven separate real original-BF16 discrepancies. All eight coherent
controls reject with their intended errors, including last-position corruptions
that pass short-prefix and future-prefix checks before failing scalar boundaries.

The [compact fixture](../tests/fixtures/gemma-long-witness-v1/README.md) is346994
manifest bytes plus6974448 binary bytes. The complete captures remain109051800
bytes per sequence. Source snapshots establish that native code stayed at
commit`8ff5499` and no incremental implementation existed during collection.

On MacBookPro18,3 / Apple M1 Pro /16 GiB /macOS26.6.2(25G83), capture-A/B
physical peaks are277922656/294896504 bytes, with339574784/320651264 RSS.
Their complete cold collection wall times are357.21/359.07 seconds, including
four full forwards, serialization and runtime endpoint checks in each process.
These are reference collection costs, not engine latency or throughput results.
Offline tests peak294814608 physical bytes; independent replay/freeze and
controls stay below37 MiB physical and125 MB RSS. All measured workers fit the
512 MiB diagnostic allowance. All11 owned process groups close, both runtime
inventories are rechecked and protected binaries remain unchanged. Rust/core,
shaders and examples are unchanged; an unrelated engine test rerun is not part
of this reference-only task. See the
[public evidence](../benchmarks/results/2026-10-27-gemma-long-witness-v1.json).

## Remaining work

The measured scalar attention cost justifies a separate bounded FP64 matrix-dot
reference experiment under the unchanged gates. Name its distinct reduction
profile, qualify standalone and complete graph outputs against this frozen
scalar oracle, and preserve all discrepancies. Then implement a bounded cached
mathematical reference and compare it against
this pre-target oracle across uneven chunks and absolute/window boundaries.
Then qualify native larger-context execution separately. Real E2B needs its own
bounded source/capture policy and measured runtime before executing the entire
1025-position quality workload. Keep all eleven original BF16 cross-precision
failures and earlier packed/GPU/rotary gaps. No unchanged quality workload,
full model input/output window or iPhone requirement is closed by tiny data.

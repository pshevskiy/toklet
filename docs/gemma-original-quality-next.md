# Next Gemma original-quality experiment

Astra recommends one measured four-token original-BF16 source experiment after
real 1,025-position wrap qualification, before building the full scorer. Reuse
existing readers, bindings and saved outputs; avoid another full-weight cache.
This is the next task boundary, not a completed quality result.

The [verified text layout](gemma-quality-contract.md) contains 9,257,138,758
BF16 weight bytes. Widening even the non-PLE portion to FP32 exceeds the 6 GiB
process policy. Gather only requested PLE rows and stream projection/head tiles
through the authenticated descriptor. The existing
[weight adapter](../tools/reference/gemma4_bf16_weights.py) already resolves the
278 matrices and 262 vectors; the [short reference](gemma-bf16-text-reference.md)
retains original-checkpoint results for `[2, 100, 257, 100]`.

The code task should:

1. Keep the established FP64 projection reductions and explicit staged-FP32
   operator boundaries. Compare all four logit rows, the three available next-token NLLs, greedy IDs
   and selected intermediate states with the saved original reference.
2. Pin a short upstream text comparison and freeze its policy before outcomes.
   Keep the prior eleven cross-precision failures and 27 packed-native
   mathematical failures rejected. FP32 accumulation would be a separately
   qualified arithmetic profile, not a silent speed change.
3. Measure descriptor read/hash, BF16 widening, GEMM and tied-head time separately,
   plus actual RSS and physical footprint. Use Llama's profiling method rather
   than its complete widened-weight cache. Preserve the 6 GiB limit.
4. Use measured bottlenecks to choose one follow-up: tile reuse across multiple
   at-most-32-row teacher-forced blocks, or a bounded cache justified by prefill
   and several single-token measurements. Qualify a 1,025-position original
   pilot before collecting all 10,000 loss targets and 100 tasks.

The independent packed-artifact wrap oracle qualifies different behavior and
cannot substitute for this original-BF16 baseline. No full-workload ETA follows
until the source experiment measures its actual cost. Each code task retains
required checks, one combined Astra diff/results review, and a separate commit.

## Profiling preparation, 4 November 2026

The [standalone adapter](../tools/reference/gemma4_bf16_profile.py) now times
authenticated raw reads/guards/hash, BF16 decoding/validation/guard, and inclusive
projections. The remainder is explicitly FP64 conversion, computation and
framework overhead; tied-head operations are separate. Inclusive and remainder
intervals overlap. Byte counters measure traffic rather than resident memory.
Two tiny controls preserve bit-identical full graph states under both contracts,
source read chains, borrowed ownership and rejected requests. Real-model
execution, timing qualification and upstream agreement remain pending.
[Preparation evidence](../benchmarks/results/2026-11-04-gemma-bf16-profile-v1.json).

## Prospective upstream comparison policy

Astra identified the cached Transformers5.7.0 implementation, independently
checked against SHA-256
`fd93159380556e46029260ae38e5ec53ffd750ecb9872d0e89f57b334a25ed4e`.
Installed PyTorch is2.9.1. The [existing layout audit](gemma-unquantized-layout.md)
pins the shared-KV source; execution additionally needs imported helpers, native
runtime, CPU backend, configuration and thread count pinned before any outputs.

Use the official checkpoint revision
`70af34e20bd4b7a91f0de6b22675850c43922a03`, CPU eager attention, eval/inference
mode and no generation cache at `[2,100,257,100]`. Preserve upstream BF16 dtype
transitions, including FP32 norm/rotary/softmax work. Load one BF16 layer at a
time, gather only requested embedding/PLE rows, retain required shared K/V and
tile the tied head. Do not call whole-checkpoint `from_pretrained`.

Before the real run, require a tiny fully resident upstream model and streamed
harness to agree on greedy IDs, head tiling and outputs under the same dtype and
operators. Freeze their exact acceptance contract before inspecting outcomes.
The real cross-arithmetic comparison is **descriptive**, prospectively: save all
four full logits, three next-token NLLs, greedy IDs/top-two margins and selected
embedding, attention and final-normalization states. Fatal gates are finite
outputs, exact input/weight identity, shapes, causal alignment and measured peak
under6GiB. There is no established per-value equivalence gate between upstream
BF16 and this reference's stagedFP32/Decimal80 arithmetic. Establishing one is
a separate prospective decision. This experiment does not waive any of the
earlier11/27 failures or the provisional four wrap-logit rejections, and cannot
establish full quality or app eligibility.

The [short source reproduction/profiling](gemma-bf16-source-experiment.md) and
[tiny upstream streaming control](gemma-upstream-tiny.md) now pass their stated
scopes. Next admit the official540selected/60excluded tensor mapping and read
coverage, then run the descriptive real four-token comparison.

## Real upstream result and next boundary

The [real four-token run](gemma-upstream-real.md) passes execution/resource
checks with descriptive arithmetic differences. Astra selects pinned official
upstream CPU eager BF16 as the prospective original-quality baseline; the
independent graphs retain their fidelity role and all prior rejections. Next
check full-width versus tiled stock tied-head bits at the saved real activations,
then qualify a representative loss block before the1,025 original pilot.
